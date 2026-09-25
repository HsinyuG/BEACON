# Copyright (c) OpenMMLab. All rights reserved.
import copy
import io
import json
import os
import random
import warnings
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from mmengine import print_log
from mmengine.fileio import get
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoTokenizer

from xtuner.utils import IGNORE_INDEX
import heapq    
import time

import re
from xtuner.perception_modules.oracle_bev import BEVMaskGenerator

# Base directories for the optional / experimental data paths used further below
# (oracle-BEV mask generation and the freespace-loss occupancy cache). They default
# to the original HPC container layout; override via env vars on a local machine.
# NOTE: the freespace-loss branch is disabled by default (use_freespace_loss=False),
# so these paths are only touched if that experimental path is re-enabled.
_BEACON_DATA_ROOT = os.environ.get("BEACON_DATA_ROOT", "/data")
_BEACON_FALCON_DATA = os.environ.get("FALCON_DATA_ROOT", "/data")
_BEACON_CACHE_ROOT = os.environ.get("BEACON_CACHE_ROOT", "./beacon_cache")

# Referenced from InternVL
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(
    image, min_num=1, max_num=6, image_size=448, use_thumbnail=False
):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def total_image_token(
    orig_size, min_num=1, max_num=12, image_size=448, use_thumbnail=True
):
    orig_width, orig_height = orig_size

    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if max_num >= i * j >= min_num
    }
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    if use_thumbnail:
        blocks += 1

    return blocks


def load_json_or_jsonl(json_path):
    if json_path.endswith(".json"):
        with open(json_path) as f:
            data = json.load(f)
    elif json_path.endswith(".jsonl"):
        with open(json_path) as f:
            data = [json.loads(line) for line in f]
    else:
        raise ValueError(
            f"Unsupported file format: {json_path}, " f"only support .json and .jsonl."
        )
    return data


class InternVL_V1_5_Dataset(Dataset):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
    IMG_START_TOKEN = "<img>"
    IMG_END_TOKEN = "</img>"

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_path,
        template,
        data_paths,
        image_folders=None,
        repeat_times=1,
        max_length=8192,
    ):
        self.template = template
        self.max_length = max_length

        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # The following modifications are only to ensure full
        # consistency with the official template,
        # without investigating the impact on performance.
        if self.cfg.llm_config.architectures[0] == "Phi3ForCausalLM":
            self._system = "You are an AI assistant whose name is Phi-3."
            self.template["INSTRUCTION"] = "<|user|>\n{input}<|end|><|assistant|>\n"
        elif self.cfg.llm_config.architectures[0] == "InternLM2ForCausalLM":
            self._system = "You are an AI assistant whose name " "is InternLM (书生·浦语)."
            self.template["SYSTEM"] = "<|im_start|>system\n{system}<|im_end|>"
            self.template["INSTRUCTION"] = (
                "<|im_start|>user\n{input}" "<|im_end|><|im_start|>assistant\n"
            )
        else:
            raise NotImplementedError

        self.min_dynamic_patch = self.cfg.min_dynamic_patch
        self.max_dynamic_patch = self.cfg.max_dynamic_patch
        self.downsample_ratio = self.cfg.downsample_ratio
        self.image_size = self.cfg.force_image_size
        self.use_thumbnail = self.cfg.use_thumbnail
        patch_size = self.cfg.vision_config.patch_size
        self.patch_token = int(
            (self.image_size // patch_size) ** 2 * (self.downsample_ratio**2)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.transformer = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )

        if not isinstance(data_paths, (list, tuple)):
            data_paths = [data_paths]
        if not isinstance(image_folders, (list, tuple)):
            image_folders = [image_folders]
        if not isinstance(repeat_times, (list, tuple)):
            repeat_times = [repeat_times]
        assert len(data_paths) == len(image_folders) == len(repeat_times)

        print_log("Starting to loading data and calc length", logger="current")
        self.data = []
        self.image_folder = []
        self.group_length = []
        self.conv2length_text = (
            {}
        )  # using dict to speedup the calculation of token length

        for data_file, image_folder, repeat_time in zip(
            data_paths, image_folders, repeat_times
        ):
            print_log(
                f"=======Starting to process {data_file} =======", logger="current"
            )
            assert repeat_time > 0
            json_data = load_json_or_jsonl(data_file)
            if repeat_time < 1:
                json_data = random.sample(json_data, int(len(json_data) * repeat_time))
            elif repeat_time > 1:
                int_repeat_time = int(repeat_time)
                remaining_repeat_time = repeat_time - repeat_time
                if remaining_repeat_time > 0:
                    remaining_json_data = random.sample(
                        json_data, int(len(json_data) * remaining_repeat_time)
                    )
                    json_data = json_data * int_repeat_time
                    json_data.extend(remaining_json_data)
                else:
                    json_data = json_data * int_repeat_time

            self.data.extend(json_data)
            self.image_folder.extend([image_folder] * len(json_data))

            # TODO: multi process
            for data_item in json_data:
                if "length" in data_item:
                    token_length = data_item["length"]  # include image token
                else:
                    conversations = "\n".join(
                        [temp["value"] for temp in data_item["conversations"]]
                    )
                    str_length = len(conversations)

                    if str_length not in self.conv2length_text:
                        token_length = self.tokenizer(
                            conversations,
                            return_tensors="pt",
                            padding=False,
                            truncation=False,
                        ).input_ids.size(1)
                        self.conv2length_text[str_length] = token_length
                    else:
                        token_length = self.conv2length_text[str_length]

                    if "image" in data_item and data_item["image"] is not None:
                        if (
                            "image_wh" in data_item
                            and data_item["image_wh"] is not None
                        ):
                            # more accurate calculation of image token
                            image_wh = data_item["image_wh"]
                            if isinstance(image_wh[0], list):
                                image_wh = image_wh[0]
                            image_token = total_image_token(
                                image_wh,
                                self.min_dynamic_patch,
                                self.max_dynamic_patch,
                                self.image_size,
                                self.use_thumbnail,
                            )
                            image_token = self.patch_token * image_token
                        else:
                            # max_dynamic_patch + use_thumbnail
                            image_token = self.patch_token * (
                                self.max_dynamic_patch + self.use_thumbnail
                            )

                        token_length = token_length + image_token
                    else:
                        token_length = -token_length

                self.group_length.append(token_length)
            print_log(
                f"=======total {len(json_data)} samples of {data_file}=======",
                logger="current",
            )

        assert len(self.group_length) == len(self.data)
        print_log("end loading data and calc length", logger="current")
        print_log(f"=======total {len(self.data)} samples=======", logger="current")
        self._max_refetch = 1000

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def __len__(self):
        return len(self.data)

    @property
    def modality_length(self):
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length

    def prepare_data(self, index):
        data_dict: dict = self.data[index]
        image_folder = self.image_folder[index]

        out_data_dict = {}
        if data_dict.get("image", None) is not None:
            image_file = data_dict["image"]
            if isinstance(image_file, (list, tuple)):
                assert len(image_file) == 1
                image_file = image_file[0]

            try:
                image = self.get_image(os.path.join(image_folder, image_file))
            except Exception as e:
                print(f"Error: {e}", flush=True)
                print_log(f"Error: {e}", logger="current")
                return None

            images = dynamic_preprocess(
                image,
                self.min_dynamic_patch,
                self.max_dynamic_patch,
                self.image_size,
                self.use_thumbnail,
            )
            pixel_values = [self.transformer(image) for image in images]
            pixel_values = torch.stack(pixel_values)
            out_data_dict["pixel_values"] = pixel_values

            num_image_tokens = pixel_values.shape[0] * self.patch_token
            image_token_str = (
                f"{self.IMG_START_TOKEN}"
                f"{self.IMG_CONTEXT_TOKEN * num_image_tokens}"
                f"{self.IMG_END_TOKEN}"
            )
            token_dict = self.get_inputid_labels(
                data_dict["conversations"], image_token_str
            )
            out_data_dict.update(token_dict)
        else:
            token_dict = self.get_inputid_labels(data_dict["conversations"], None)
            out_data_dict.update(token_dict)
            out_data_dict["pixel_values"] = torch.zeros(
                1, 3, self.image_size, self.image_size
            )
        return out_data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.data))

    def get_image(self, path):
        if "s3://" in path:
            img_bytes = get(path)
            with io.BytesIO(img_bytes) as buff:
                img = Image.open(buff).convert("RGB")
            return img
        else:
            return Image.open(path).convert("RGB")

    def get_inputid_labels(self, conversations, image_token_str) -> dict:
        input = ""
        out_conversation = []
        while conversations and conversations[0]["from"] == "gpt":
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for msg in conversations:
            if msg["from"] == "human":
                if image_token_str is None and "<image>" in msg["value"]:
                    warnings.warn(
                        f'The current data << {msg["value"]} >> is '
                        f"in plain text mode, but "
                        "there are <image> tags present in the data. "
                        "We need to remove the <image> tags."
                    )
                    msg["value"] = msg["value"].replace("<image>", "")
                if "<image>" in msg["value"]:
                    msg["value"] = msg["value"].replace("<image>", "").strip()
                    msg["value"] = image_token_str + "\n" + msg["value"]
                    msg["value"] = msg["value"].strip()
                input += msg["value"].strip()
            elif msg["from"] == "gpt":
                out_conversation.append(
                    {"input": input, "output": msg["value"].strip()}
                )
                input = ""
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get("input", "")
            if input is None:
                input = ""
            input_text = self.template.INSTRUCTION.format(input=input, round=i + 1)

            if i == 0:
                system = self.template.SYSTEM.format(system=self._system)
                input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True
                )
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False
                )
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get("output", "")
            if self.template.get("SUFFIX", None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
            print_log(
                f"Warning: input_ids length({len(input_ids)}) "
                f"is longer than max_length, cut to {self.max_length}",
                logger="current",
            )
        return {"input_ids": input_ids, "labels": labels}


from os import PathLike

class InternVL_V1_5_Dataset_Multiview(InternVL_V1_5_Dataset):
    def __init__(
        self,
        model_path,
        template,
        data_paths,
        image_folders=None,
        repeat_times=1,
        max_length=8192,
    ):
        self.template = template
        self.max_length = max_length

        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        # The following modifications are only to ensure full
        # consistency with the official template,
        # without investigating the impact on performance.
        if self.cfg.llm_config.architectures[0] == "Phi3ForCausalLM":
            self._system = "You are an AI assistant whose name is Phi-3."
            self.template["INSTRUCTION"] = "<|user|>\n{input}<|end|><|assistant|>\n"
        elif self.cfg.llm_config.architectures[0] == "InternLM2ForCausalLM":
            self._system = "You are an AI assistant whose name " "is InternLM (书生·浦语)."
            self.template["SYSTEM"] = "<|im_start|>system\n{system}<|im_end|>"
            self.template["INSTRUCTION"] = (
                "<|im_start|>user\n{input}" "<|im_end|><|im_start|>assistant\n"
            )
        else:
            raise NotImplementedError

        self.min_dynamic_patch = self.cfg.min_dynamic_patch
        self.max_dynamic_patch = self.cfg.max_dynamic_patch
        self.downsample_ratio = self.cfg.downsample_ratio
        self.image_size = self.cfg.force_image_size
        self.use_thumbnail = self.cfg.use_thumbnail
        patch_size = self.cfg.vision_config.patch_size
        self.patch_token = int(
            (self.image_size // patch_size) ** 2 * (self.downsample_ratio**2)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.transformer = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )

        if not isinstance(data_paths, (list, tuple)):
            data_paths = [data_paths]
        if not isinstance(image_folders, (list, tuple)):
            image_folders = [image_folders]
        if not isinstance(repeat_times, (list, tuple)):
            repeat_times = [repeat_times]
        assert len(data_paths) == len(image_folders) == len(repeat_times)

        print_log("Starting to loading data and calc length", logger="current")
        self.data = []
        self.image_folder = []
        self.group_length = []
        self.conv2length_text = (
            {}
        )  # using dict to speedup the calculation of token length

        for data_file, image_folder, repeat_time in zip(
            data_paths, image_folders, repeat_times
        ):
            print_log(
                f"=======Starting to process {data_file} =======", logger="current"
            )
            assert repeat_time > 0
            json_data = load_json_or_jsonl(data_file)
            if repeat_time < 1:
                json_data = random.sample(json_data, int(len(json_data) * repeat_time))
            elif repeat_time > 1:
                int_repeat_time = int(repeat_time)
                remaining_repeat_time = repeat_time - repeat_time
                if remaining_repeat_time > 0:
                    remaining_json_data = random.sample(
                        json_data, int(len(json_data) * remaining_repeat_time)
                    )
                    json_data = json_data * int_repeat_time
                    json_data.extend(remaining_json_data)
                else:
                    json_data = json_data * int_repeat_time

            self.data.extend(json_data)
            self.image_folder.extend([image_folder] * len(json_data))

            # TODO: multi process
            for data_item in json_data:
                if "length" in data_item:
                    token_length = data_item["length"]  # include image token
                else:
                    conversations = "\n".join(
                        [temp["value"] for temp in data_item["conversations"]]
                    )
                    str_length = len(conversations)

                    if str_length not in self.conv2length_text:
                        token_length = self.tokenizer(
                            conversations,
                            return_tensors="pt",
                            padding=False,
                            truncation=False,
                        ).input_ids.size(1)
                        self.conv2length_text[str_length] = token_length
                    else:
                        token_length = self.conv2length_text[str_length]

                    if "image" in data_item and data_item["image"] is not None:
                        # new added
                        image_files = data_item["image"]
                        if not isinstance(image_files, (list, tuple)):
                            image_files = [image_files]
                        # end of new added
                        if (
                            "image_wh" in data_item
                            and data_item["image_wh"] is not None
                        ):
                            raise NotImplementedError
                            # # more accurate calculation of image token
                            # image_wh = data_item["image_wh"]
                            # if isinstance(image_wh[0], list):
                            #     image_wh = image_wh[0]
                            # image_token = total_image_token(
                            #     image_wh,
                            #     self.min_dynamic_patch,
                            #     self.max_dynamic_patch,
                            #     self.image_size,
                            #     self.use_thumbnail,
                            # )
                            # image_token = self.patch_token * image_token
                        else:
                            # max_dynamic_patch + use_thumbnail
                            # image_token = self.patch_token * (
                            #     self.max_dynamic_patch + self.use_thumbnail
                            # )
                            # new added
                            # Fallback: assume each image uses the max patch count + thumbnail flag
                            image_token = len(image_files) * self.patch_token * (
                                self.max_dynamic_patch + self.use_thumbnail
                            )
                            # end of new added

                        token_length = token_length + image_token
                    else:
                        # token_length = -token_length
                        raise NotImplementedError

                self.group_length.append(token_length)
            print_log(
                f"=======total {len(json_data)} samples of {data_file}=======",
                logger="current",
            )

        assert len(self.group_length) == len(self.data)
        print_log("end loading data and calc length", logger="current")
        print_log(f"=======total {len(self.data)} samples=======", logger="current")
        self._max_refetch = 1000

    def prepare_data(self, index):
        data_dict: dict = self.data[index]
        image_folder = self.image_folder[index]

        out_data_dict = {}
        if data_dict.get("image", None) is not None:
            image_files = data_dict["image"]
            # if isinstance(image_file, (list, tuple)):
            #     assert len(image_file) == 1
            #     image_file = image_file[0]
            if not isinstance(image_files, (list, tuple)):
                image_files = [image_files]

            pixel_value_chunks = []
            image_token_blocks = []

            for single_image_file in image_files:

                try:
                    # image = self.get_image(os.path.join(image_folder, single_image_file))
                    # new added, handle different types of image file inputs
                    if isinstance(single_image_file, (str, PathLike)):
                        folder = image_folder or ""  # handle None
                        image = self.get_image(os.path.join(folder, single_image_file))
                    else:
                        image = self.get_image(single_image_file)  # e.g., NumPy array
                except Exception as e:
                    print(f"Error: {e}", flush=True)
                    print_log(f"Error: {e}", logger="current")
                    return None

                images = dynamic_preprocess(
                    image,
                    self.min_dynamic_patch,
                    self.max_dynamic_patch,
                    self.image_size,
                    self.use_thumbnail,
                )
                # pixel_values = [self.transformer(image) for image in images]
                # pixel_values = torch.stack(pixel_values)
                # out_data_dict["pixel_values"] = pixel_values
                transformed = [self.transformer(image) for image in images]
                pixel_value_chunks.extend(transformed)

                # num_image_tokens = pixel_values.shape[0] * self.patch_token
                num_image_tokens = len(transformed) * self.patch_token
                # image_token_str = (
                #     f"{self.IMG_START_TOKEN}"
                #     f"{self.IMG_CONTEXT_TOKEN * num_image_tokens}"
                #     f"{self.IMG_END_TOKEN}"
                # )
                image_token_blocks.append(
                    f"{self.IMG_START_TOKEN}"
                    f"{self.IMG_CONTEXT_TOKEN * num_image_tokens}"
                    f"{self.IMG_END_TOKEN}"
                )
            # token_dict = self.get_inputid_labels(
            #     data_dict["conversations"], image_token_str
            # )
            pixel_values = torch.stack(pixel_value_chunks)
            out_data_dict["pixel_values"] = pixel_values

            token_dict = self.get_inputid_labels(
                data_dict["conversations"], image_token_blocks
            )
            out_data_dict.update(token_dict)
        else:
            token_dict = self.get_inputid_labels(data_dict["conversations"], None)
            out_data_dict.update(token_dict)
            out_data_dict["pixel_values"] = torch.zeros(
                1, 3, self.image_size, self.image_size
            )
        return out_data_dict
    

    def get_inputid_labels(self, conversations, image_token_blocks) -> dict:
        input = ""
        out_conversation = []
        # print("[InternVL_V1_5_Dataset_Multiview] Debug: ", image_token_blocks)
        while conversations and conversations[0]["from"] == "gpt":
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        image_block_iter = iter(image_token_blocks or [])
        for msg in conversations:
            if msg["from"] == "human":
                if image_token_blocks is None and "<image>" in msg["value"]:
                    warnings.warn(
                        f'The current data << {msg["value"]} >> is '
                        f"in plain text mode, but "
                        "there are <image> tags present in the data. "
                        "We need to remove the <image> tags."
                    )
                    msg["value"] = msg["value"].replace("<image>", "")
                if "<image>" in msg["value"]:
                    # msg["value"] = msg["value"].replace("<image>", "").strip()
                    # msg["value"] = image_token_str + "\n" + msg["value"]
                    # msg["value"] = msg["value"].strip()
                    # Replace each <image> with the next block in order
                    parts = msg["value"].split("<image>")
                    new_msg = parts[0].strip()
                    for part in parts[1:]:
                        block = next(image_block_iter, "")
                        if block == "":
                            warnings.warn(
                                "Not enough image_token_blocks provided for <image> tags."
                            )
                        if new_msg:
                            new_msg += "\n"
                        new_msg += block
                        if part.strip():
                            new_msg += "\n" + part.strip()
                    msg["value"] = new_msg.strip()
                input += msg["value"].strip()
            elif msg["from"] == "gpt":
                out_conversation.append(
                    {"input": input, "output": msg["value"].strip()}
                )
                input = ""
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get("input", "")
            if input is None:
                input = ""
            input_text = self.template.INSTRUCTION.format(input=input, round=i + 1)

            if i == 0:
                system = self.template.SYSTEM.format(system=self._system)
                input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True
                )
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False
                )
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get("output", "")
            if self.template.get("SUFFIX", None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[: self.max_length]
            labels = labels[: self.max_length]
            print_log(
                f"Warning: input_ids length({len(input_ids)}) "
                f"is longer than max_length, cut to {self.max_length}",
                logger="current",
            )
        return {"input_ids": input_ids, "labels": labels}
    


import os, math, pickle, tempfile
from typing import Dict, Iterator, List, Optional, Tuple

import cv2


# --------------------------
# Loader
# --------------------------
class SingleFrameLoader:
    def __init__(
        self,
        ann_file: str,
        dataset_root: str,
        # snap_thresh: float = 0.5,
        fov_deg: float = 90.0,
        out_hw: Tuple[int, int] = (512, 512),
        use_dynamic: bool = False,
        use_version: Optional[str] = None,
        verbose: bool = False,
        max_distance_horizontal: float = 6.4,
        max_distance_vertical: float = 0.5,
        split: str = "train", # train or val
        return_imgs=False,
    ):
        with open(ann_file, "rb") as f:
            ann = pickle.load(f)
        self.data_list: List[Dict] = ann["data_list"]

        self.dataset_root = dataset_root
        # self.snap_thresh = float(snap_thresh)
        self.fov_deg = float(fov_deg)
        self.out_hw = out_hw
        # Backwards-compat: keep `use_dynamic`, but prefer `use_version` if provided.
        self.use_dynamic = bool(use_dynamic)
        self.capture_versions = self._normalize_capture_versions(
            use_dynamic=use_dynamic, use_version=use_version
        )
        self.verbose = bool(verbose)
        self.max_distance_horizontal = float(max_distance_horizontal)
        self.max_distance_vertical = float(max_distance_vertical)

        self.navmesh_root = os.path.join(dataset_root, "scene_datasets", "mp3d")
        self.split = split
        if 'overfit' in ann_file:
            self.split = 'train' # in overfitting we should keep it same
        # self.split = 'val' # BUG: in overfitting we should keep it same
        print(f"[SingleFrameLoader] Dataset split: {self.split}")
        # self.pf_cache: Dict[str, habitat_sim.PathFinder] = {}
        self.return_imgs = return_imgs
        self.ann_file = ann_file

    @staticmethod
    def _normalize_capture_versions(*, use_dynamic: bool, use_version: Optional[str]) -> List[str]:
        if use_version is None:
            return ["dynamic" if bool(use_dynamic) else "static"]

        v = str(use_version).strip().lower()
        if v in ("dynamic", "static"):
            return [v]
        if v == "both":
            return ["dynamic", "static"]
        raise ValueError(
            f"Unsupported use_version={use_version!r}; expected 'dynamic', 'static', or 'both'."
        )

    def __iter__(self) -> Iterator[Dict]:
        for e in self.data_list:
            for capture_version in self.capture_versions:
                out = self._proc_one(e, capture_version=capture_version)
                if out is not None:
                    yield out

    def __len__(self) -> int:
        return len(self.data_list) * max(1, len(self.capture_versions))

    # --------------------------
    # small math helpers
    # --------------------------
    def wrap_to_pi(self, a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def yaw_to_rad_auto(self, yaw_val: float) -> float:
        """If someone stored degrees by accident, convert; else treat as radians."""
        y = float(yaw_val)
        if abs(y) > (2.0 * math.pi + 1e-3):  # likely degrees
            return math.radians(y)
        return y

    def yaw_rad_to_deg_wrapped(self, yaw_rad: float) -> float:
        return math.degrees(self.wrap_to_pi(float(yaw_rad)))

    # --------------------------
    # pano crop (equirect -> pinhole)
    # Habitat convention: yaw=0 faces -Z; positive yaw turns LEFT.
    # --------------------------
    def pano_to_perspective(
        self,
        pano_bgr: np.ndarray,
        fov_deg: float = 90.0,
        out_hw: Tuple[int, int] = (512, 512),
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        interp: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        H_p, W_p = pano_bgr.shape[:2]
        H_o, W_o = out_hw

        j, i = np.meshgrid(np.arange(W_o), np.arange(H_o))  # x=j, y=i
        x = (j + 0.5) / W_o * 2 - 1
        y = (i + 0.5) / H_o * 2 - 1

        s = math.tan(math.radians(fov_deg) / 2.0)
        x_cam = x * s
        y_cam = -y * s
        z_cam = -np.ones_like(x_cam)  # forward is -Z when yaw=0

        dirs = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float32)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-9

        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)

        # yaw about +Y (left-positive), then pitch about +X
        Ry = np.array(
            [[math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0,          1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)]],
            dtype=np.float32,
        )
        Rx = np.array(
            [[1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch),  math.cos(pitch)]],
            dtype=np.float32,
        )
        R = Ry @ Rx

        d = dirs @ R.T

        theta = np.arctan2(d[..., 0], -d[..., 2])  # [-pi, pi], 0 at -Z
        phi = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))  # [-pi/2, pi/2]

        u = (theta + np.pi) / (2.0 * np.pi) * W_p
        v = (np.pi / 2.0 - phi) / np.pi * H_p

        crop = cv2.remap(
            pano_bgr,
            u.astype(np.float32),
            v.astype(np.float32),
            interpolation=interp,
            borderMode=cv2.BORDER_WRAP,
        )
        return crop
    
    def pano_to_linear_slice(
        self,
        pano_bgr: np.ndarray,
        fov_deg: float = 90.0,
        out_hw: Tuple[int, int] = (512, 512),
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        interp: int = cv2.INTER_LINEAR,
    ) -> np.ndarray:
        """Equirect pano -> linear yaw/phi crop (no rectilinear projection)."""
        H_p, W_p = pano_bgr.shape[:2]
        H_o, W_o = out_hw

        yaw_center = math.radians(yaw_deg)
        yaw_span = math.radians(fov_deg)
        pitch_center = math.radians(pitch_deg)

        # Output pixel grid
        j, i = np.meshgrid(np.arange(W_o), np.arange(H_o))

        # Linear yaw per column
        yaw = yaw_center + ((j + 0.5) / W_o - 0.5) * yaw_span  # radians
        u = (yaw + math.pi) / (2.0 * math.pi) * W_p  # equirect x

        # Linear phi (elevation) per row around pitch_center; span proportional to output height
        phi_span = math.pi * (H_o / H_p)  # adjust if you want a tighter/looser vertical crop
        phi = pitch_center + (0.5 - (i + 0.5) / H_o) * phi_span
        v = (math.pi / 2.0 - phi) / math.pi * H_p  # equirect y

        u = u.astype(np.float32)
        v = v.astype(np.float32)

        return cv2.remap(
            pano_bgr,
            u,
            v,
            interpolation=interp,
            borderMode=cv2.BORDER_WRAP,
        )


    def _find_pano(
        self, scene: str, token: str, vhash: str, *, capture_version: str
    ) -> Optional[str]:
        img_prefix = str(capture_version).strip().lower()
        base_dir = os.path.join(
            self.dataset_root,
            f"captures_v3_{self.split}_{scene}",
            f"{img_prefix}_mp3d_viewpoints",
            "rgb",
        )

        candidates = [
            f"{vhash}.png",          # common: viewpoint-id only
            f"{token}.png",          # if you stored token name
        ]
        for fn in candidates:
            p = os.path.join(base_dir, fn)
            if os.path.exists(p):
                return p
        return None

    def _find_depth(
        self, scene: str, token: str, vhash: str, *, capture_version: str
    ) -> Optional[str]:
        img_prefix = str(capture_version).strip().lower()
        base_dir = os.path.join(
            self.dataset_root,
            f"captures_v3_{self.split}_{scene}",
            f"{img_prefix}_mp3d_viewpoints",
            "depth",
        )

        candidates = [
            f"{vhash}.npy",
            f"{token}.npy",
        ]
        for fn in candidates:
            p = os.path.join(base_dir, fn)
            if os.path.exists(p):
                return p
        return None
    
    def _cam2img(self):
        H_out, W_out = self.out_hw
        fx = fy = W_out / (2.0 * math.tan(math.radians(self.fov_deg) / 2.0))
        cx = W_out / 2.0
        cy = H_out / 2.0
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                     dtype=np.float32)
        K4 = np.eye(4, dtype=np.float32)
        K4[:3, :3] = K
        return K4
    
    def _cam2ego(self, yaw_deg):
        yaw = math.radians(yaw_deg)
        c, s = math.cos(yaw), math.sin(yaw)
        T = np.eye(4, dtype=np.float32)
        R_front = np.array([[0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0]], dtype=np.float32)
        Rz = np.array([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]], dtype=np.float32)
        T[:3, :3] = Rz @ R_front
        return T

    def _wrap_to_pi_np(self, a: np.ndarray) -> np.ndarray:
        """Vectorized wrap to [-pi, pi)."""
        return (a + np.pi) % (2 * np.pi) - np.pi

    def rectify_patchwork_pano_depth_to_range_ring( 
        self,
        pano_depth_patch: np.ndarray,
        max_depth: float = 10.0,
        ring_vfov_deg: float = 90.0,
        face_yaws_deg=(0.0, 90.0, 180.0, -90.0),
        invalid_to: float = 10.0,
    ) -> np.ndarray:
        """
        Convert patchworked 4-face *plane z-depth pano* -> *ray-range pano*.
        Correct in vfov ring (|phi|<=ring_vfov/2). Outside ring: replicate boundary rows.
        Assumes pano center column is yaw=0, and the 4 faces are at 0/90/180/-90.
        """
        # BUG: this ring we didn't use the top and bottom surface so if we crop in the middle it will have leaked corners. refer to debug_outputs/debug_intermediate_lss_attn.ipynb
        d = pano_depth_patch.astype(np.float32).copy()
        Hp, Wp = d.shape

        # 0 -> 10, clip to [0, max_depth] (your desired behavior)
        d[d <= 0] = invalid_to
        d = np.clip(d, 0.0, max_depth)

        # yaw theta per column (matches your pano_to_perspective convention)
        theta = ((np.arange(Wp, dtype=np.float32) + 0.5) / Wp) * (2 * np.pi) - np.pi  # (Wp,)

        # elevation phi per row
        phi = (np.pi / 2.0) - ((np.arange(Hp, dtype=np.float32) + 0.5) / Hp) * np.pi  # (Hp,)
        cosphi = np.cos(phi).astype(np.float32)

        ring_half = np.deg2rad(ring_vfov_deg * 0.5)
        ring_mask = np.abs(phi) <= (ring_half + 1e-6)
        ring_rows = np.where(ring_mask)[0]
        if ring_rows.size == 0:
            raise ValueError("No ring rows found; check pano height / ring_vfov_deg.")
        v_top = int(ring_rows[0])
        v_bot = int(ring_rows[-1])

        # face assignment per column: nearest yaw center
        face_yaws = np.deg2rad(np.array(face_yaws_deg, dtype=np.float32))[:, None]  # (4,1)
        delta = self._wrap_to_pi_np(theta[None, :] - face_yaws)  # (4,Wp)
        face_id = np.argmin(np.abs(delta), axis=0)               # (Wp,)
        delta_sel = delta[face_id, np.arange(Wp)]
        cosd = np.cos(delta_sel).astype(np.float32)

        # z_face -> range: r = z / (cos(phi)*cos(delta))
        denom_cols = np.maximum(cosd, 1e-6)  # (Wp,)
        denom_ring = (cosphi[ring_rows][:, None] * denom_cols[None, :])  # (Nr,Wp)
        denom_ring = np.maximum(denom_ring, 1e-6)

        out = np.empty_like(d, dtype=np.float32)
        r_ring = d[ring_rows, :] / denom_ring
        r_ring = np.clip(r_ring, 0.0, max_depth)
        out[ring_rows, :] = r_ring

        # replicate ring boundary rows to fake top/bottom
        out[:v_top, :] = out[v_top, :]
        out[v_bot + 1 :, :] = out[v_bot, :]

        return out

    def range_to_plane_depth_z(self, range_hw: np.ndarray) -> np.ndarray:
        """
        Convert ray-range depth (distance along ray) -> plane z-depth
        consistent with your pano_to_perspective camera rays:
        x_cam=x*s, y_cam=-y*s, z_cam=-1, then normalize.
        zdepth = range * (-dir_z) = range / sqrt(x_cam^2+y_cam^2+1)
        """
        H_o, W_o = self.out_hw
        j, i = np.meshgrid(np.arange(W_o), np.arange(H_o))
        x = (j + 0.5) / W_o * 2.0 - 1.0
        y = (i + 0.5) / H_o * 2.0 - 1.0

        s = math.tan(math.radians(self.fov_deg) / 2.0)
        x_cam = x * s
        y_cam = -y * s

        norm = np.sqrt(x_cam * x_cam + y_cam * y_cam + 1.0).astype(np.float32)
        zdepth = range_hw.astype(np.float32) / norm
        return zdepth
    
    @staticmethod
    def bounded_geodesic_gaussian_cells(
        Ai_cpu: np.ndarray,
        si: int,
        sj: int,
        rmax_cells: float,
        sigma_floor: float = 1e-6,
        return_dist: bool = False,
    ):
        """
        Ai_cpu: (H,W) bool numpy array. True = traversable.
        si,sj: seed indices (row=i, col=j)
        rmax_cells: max radius in cell-units (cardinal=1, diagonal=sqrt(2))
        sigma: derived as rmax_cells/3 (with floor)
        Returns:
        - gs (H,W) float32, gaussian weights in [0,1]
        - optionally dist (H,W) float32 (INF for unreachable/outside rmax)
        """
        H, W = Ai_cpu.shape
        INF = 1e9
        dist = np.full((H, W), INF, dtype=np.float32)
        visited = np.zeros((H, W), dtype=bool)

        # If seed out of bounds or not traversable, return all zeros (and INF dist if requested)
        if si < 0 or si >= H or sj < 0 or sj >= W or (not bool(Ai_cpu[si, sj])) or rmax_cells <= 0:
            gs = np.zeros((H, W), dtype=np.float32)
            return (gs, dist) if return_dist else gs

        dist[si, sj] = 0.0
        pq = [(0.0, si, sj)]

        SQ2 = math.sqrt(2.0)
        moves = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2),
        ]

        while pq:
            d, y, x = heapq.heappop(pq)
            if visited[y, x]:
                continue
            visited[y, x] = True
            if d > rmax_cells:
                break

            for dy, dx, w in moves:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                if not bool(Ai_cpu[ny, nx]):
                    continue

                # strict diagonal gating (no corner cutting)
                if dy != 0 and dx != 0:
                    if not (bool(Ai_cpu[y, nx]) and bool(Ai_cpu[ny, x])):
                        continue

                nd = d + w
                if nd <= rmax_cells and nd < float(dist[ny, nx]):
                    dist[ny, nx] = float(nd)
                    heapq.heappush(pq, (float(nd), ny, nx))

        # Gaussian: sigma = rmax/3 (with floor)
        sigma = max(float(rmax_cells) / 3.0, float(sigma_floor))
        inv_2s2 = 1.0 / (2.0 * sigma * sigma)

        gs = np.zeros((H, W), dtype=np.float32)
        m = dist < INF
        if np.any(m):
            d = dist[m].astype(np.float32)
            gs[m] = np.exp(-(d * d) * inv_2s2).astype(np.float32)

        return (gs, dist) if return_dist else gs


    def _proc_one(self, e: Dict, *, capture_version: str) -> Optional[Dict]:
        instr = str(e.get("sub_instruction", "")).strip()
        if not instr:
            if self.verbose:
                print("Skip: empty sub_instruction")
            return None

        fixed_local = e.get("fixed_waypoints", [])
        if not fixed_local:
            if self.verbose:
                print("Skip: empty fixed_waypoints")
            return None
        p_goal = np.asarray(fixed_local[-1], dtype=np.float32).reshape(3) # (x, y, z)
        horiontal_dist = np.linalg.norm(p_goal[[0, 1]])
        cam_h = float(e["habitat_cam_height"])
        vertical_dist = abs((-cam_h) - p_goal[2])
        if horiontal_dist > self.max_distance_horizontal:
            if self.verbose:
                print(f"Skip: horizontal distance {horiontal_dist:.3f} > {self.max_distance_horizontal:.3f}")
            return None
        if vertical_dist > self.max_distance_vertical:
            if self.verbose:
                print(f"Skip: vertical distance {vertical_dist:.3f} > {self.max_distance_vertical:.3f}")
            return None

        # --- upstream filter: target must be visible in npz ---
        strict_filter = True
        npz_rel = e.get("navigable_mask_path", "")
        if npz_rel:
            scan_id = e.get('scene_token')
            npz_path = os.path.join(os.path.dirname(self.ann_file), 'annotations', scan_id, npz_rel)
            if os.path.isfile(npz_path):
                ann = np.load(npz_path)
                vis = ann.get("visible_mask")  # bool array HxW
                aff = ann.get("affordance_mask")  # bool array HxW
                trav = ann.get("traversable_mask")  # bool array HxW
                if strict_filter:
                    x, y = float(p_goal[0]), float(p_goal[1])  # use local p_goal
                    x_min = y_min = -6.4
                    res = 0.1
                    i = int(round((x - (x_min + 0.5 * res)) / res))
                    j = int(round((y - (y_min + 0.5 * res)) / res))
                    if i < 0 or i >= vis.shape[0] or j < 0 or j >= vis.shape[1] or vis[i, j] == 0:
                        if self.verbose:
                            print(f"[SingleFrameLoader] skip: target not visible token={e.get('token')}")
                        return None
                    # Per-waypoint (and goal) geodesic masks within trav & vis.
                    # Proof-of-concept: if seed is not traversable, force it traversable (no snapping).
                    region = (vis.astype(bool) & trav.astype(bool))
                    region_goal = region
                    if 0 <= i < region.shape[0] and 0 <= j < region.shape[1] and bool(vis[i, j]):
                        if not bool(region[i, j]):
                            region_goal = region.copy()
                            region_goal[i, j] = True

                    affordance_gt_prob, dist = self.bounded_geodesic_gaussian_cells(
                        region_goal,
                        i,
                        j,
                        rmax_cells=10.0,
                        return_dist=True,
                    )
                    affordance_gt_bin = (dist <= 10.0)

                    # # waypoint masks (num_wp=6): (6,H,W) uint8
                    # wps_xy = np.asarray(fixed_local, dtype=np.float32)[:, :2]
                    # num_wp = 6
                    # if wps_xy.shape[0] < num_wp:
                    #     pad = np.repeat(wps_xy[-1:, :], repeats=(num_wp - wps_xy.shape[0]), axis=0)
                    #     wps_xy = np.concatenate([wps_xy, pad], axis=0)
                    # elif wps_xy.shape[0] > num_wp:
                    #     wps_xy = wps_xy[:num_wp]

                    # Hm, Wm = region.shape
                    # wp_mask = np.zeros((num_wp, Hm, Wm), dtype=np.uint8)
                    # for k in range(num_wp):
                    #     xk, yk = float(wps_xy[k, 0]), float(wps_xy[k, 1])
                    #     ik = int(round((xk - (x_min + 0.5 * res)) / res))
                    #     jk = int(round((yk - (y_min + 0.5 * res)) / res))
                    #     if ik < 0 or ik >= Hm or jk < 0 or jk >= Wm:
                    #         continue

                    #     region_k = region
                    #     if bool(vis[ik, jk]) and not bool(region[ik, jk]):
                    #         region_k = region.copy()
                    #         region_k[ik, jk] = True

                    #     _, dist_k = self.bounded_geodesic_gaussian_cells(
                    #         region_k,
                    #         ik,
                    #         jk,
                    #         rmax_cells=10.0,
                    #         return_dist=True,
                    #     )
                    #     wp_mask[k] = (dist_k <= 10.0).astype(np.uint8)
                    # # debug
                    # try:
                    #     assert (affordance_gt_bin == aff * vis).all(), "Affordance GT mismatch with ann mask"
                    # except AssertionError as err:
                    #     print(f"AssertionError: {err}")
                    #     token = str(e.get("token", ""))
                    #     if not hasattr(self, "_debug_count"):
                    #         self._debug_count = 0
                    #     else:
                    #         self._debug_count += 1
                    #     save_path = f'debug_outputs/0_pipeline_{self._debug_count}.npz'
                    #     np.savez_compressed(
                    #         save_path,
                    #         affordance_gt_prob=affordance_gt_prob,
                    #         affordance_gt_bin=affordance_gt_bin,
                    #         dist=dist,
                    #         vis=vis,
                    #         trav=trav,
                    #         aff=aff,
                    #         token=token,
                    #     )
                    #     print(f"Saved debug npz to {save_path}")
                    #     # exit(0)
                    #     # end of debug
                    
                else:
                    raise NotImplementedError
                    visible_affordance_mask = vis & aff
                    if not np.any(visible_affordance_mask):
                        if self.verbose:
                            print(f"[SingleFrameLoader] skip: target not visible token={e.get('token')}")
                        return None
            else:
                raise FileNotFoundError(f"NPZ file not found for navigable_mask_path: {npz_path}")

        scene = e.get("scene_token") or e.get("scene_name") or e.get("scene_idx")
        if not scene:
            if self.verbose:
                print("Skip: missing scene")
            return None
        
        meta_action = e.get("meta_action")
        if not meta_action:
            if self.verbose:
                print("Skip: missing meta_action")
            return None

        # --- must exist (saved by you) ---
        if ("habitat_base_position" not in e
            or "habitat_base_yaw" not in e
            or "habitat_fixed_waypoints_position" not in e):
            if self.verbose:
                print("Skip: missing baked habitat_* keys in pkl sample")
            return None

        # --- NEW: trust PKL creator's feasibility check ---
        if not bool(e.get("all_navigable", False)):
            if self.verbose:
                token = str(e.get("token", ""))
                vp = str(e.get("viewpoint_id", ""))
                print(f"Reject: all_navigable=False | scene={scene} token={token} vp={vp}")
            return None

        # pf = self._pf(scene)

        base_pos = np.asarray(e["habitat_base_position"], dtype=np.float32).reshape(3)

        # magic fix (keep)
        base_yaw_rad = self.yaw_to_rad_auto(e["habitat_base_yaw"] + math.pi / 2)
        base_yaw_deg = self.yaw_rad_to_deg_wrapped(base_yaw_rad)

        # # snap base (keep current behavior)
        # base_snap = snap_or_fail(pf, base_pos, self.snap_thresh)
        # if base_snap is None:
        #     if self.verbose:
        #         d = np.linalg.norm(pf.snap_point(base_pos) - base_pos)
        #         print(f"Snap base failed: dist={d:.3f} > {self.snap_thresh:.3f}")
        #     return None
        # base_pos = base_snap

        # if not pf_is_navigable(pf, base_pos, 2.0):
        #     if self.verbose:
        #         print("Base not navigable (even after snap)")
        #     return None

        # waypoints in Habitat (NO waypoint snapping / nav checks; trust all_navigable)
        wps = np.asarray(e["habitat_fixed_waypoints_position"], dtype=np.float32)
        if wps.ndim != 2 or wps.shape[1] != 3:
            if self.verbose:
                print(f"Bad waypoint shape: {wps.shape}")
            return None

        # pano path
        token = str(e.get("token", ""))
        vhash = token.split("_yaw")[0] if "_yaw" in token else (str(e.get("viewpoint_id", "")) or token)
        pano_path = self._find_pano(scene, token, vhash, capture_version=capture_version)
        if pano_path is None:
            if self.verbose:
                print(f"Missing pano for scene={scene} token={token} vhash={vhash}")
            return None

        pano = cv2.imread(pano_path, cv2.IMREAD_COLOR)
        if pano is None:
            if self.verbose:
                print(f"Failed to read pano: {pano_path}")
            return None
        depth_path = self._find_depth(scene, token, vhash, capture_version=capture_version)
        if depth_path is None:
            if self.verbose:
                print(f"Missing depth for scene={scene} token={token} vhash={vhash}")
            return None
        depth = np.load(depth_path)  # in meters
        # fix the wrong panoramic depth
        depth_range_pano = self.rectify_patchwork_pano_depth_to_range_ring(
            depth, max_depth=10.0, ring_vfov_deg=90.0, invalid_to=10.0
        )

        # Crop 4 directions relative to base yaw
        dirs = {"front": 0.0,  "left": 90.0, "back": 180.0, "right": -90.0}
        imgs: Dict[str, str] = {}
        depths: Dict[str, np.ndarray] = {}
        cam2imgs: Dict[str, np.ndarray] = {}
        cam2egos: Dict[str, np.ndarray] = {}
        ego2global = np.asanyarray(e['ego2global'], dtype=np.float32)
        for n, off in dirs.items():
            if self.return_imgs:
                imgs[n] = self.pano_to_perspective(pano, self.fov_deg, self.out_hw, base_yaw_deg + off, 0.0)
                # depths[n] = self.pano_to_perspective(depth, self.fov_deg, self.out_hw, base_yaw_deg + off, 0.0, interp=cv2.INTER_NEAREST)
                # 1) sample corrected *range* pano
                depth_range = self.pano_to_perspective(
                    depth_range_pano, self.fov_deg, self.out_hw, base_yaw_deg + off, 0.0, interp=cv2.INTER_NEAREST
                )
                # 2) convert range -> plane z-depth (so your downstream expects the same "depth type" as before)
                depths[n] = self.range_to_plane_depth_z(depth_range)
            cam2imgs[n] = self._cam2img()
            cam2egos[n] = self._cam2ego(off) # ego is already base_yaw aligned

        if self.verbose:
            print(f"[OK] scene={scene} vhash={vhash} base_yaw_deg={base_yaw_deg:.2f} pano={os.path.basename(pano_path)}")

        return {
            "instruction": instr,
            "images": imgs, # dict of 4 views, each 512, 512, 3, BGR order
            "capture_version": str(capture_version).strip().lower(),
            "scene": scene,
            "token": token,
            "viewpoint_hash": vhash,
            "agent_pos_hab": base_pos.astype(np.float32),
            "agent_yaw_hab_rad": float(base_yaw_rad),
            "cam_height": cam_h,
            "waypoints_hab": wps.astype(np.float32),
            "pano_path": pano_path,
            "depths": depths,  # dict of 4 views, each 512, 512, float32 meters
            "depth_paths": depth_path,
            "cam2imgs": cam2imgs,  # dict of 4 views, each 4x4 float32
            "cam2egos": cam2egos,  # dict of 4 views, each 4x4 float32
            "ego2global": ego2global,
            "p_goal": p_goal, # (3, ) float32 in ego frame
            "waypoints": np.array(fixed_local, dtype=np.float32),  # (N, 3)
            "ep_idx": int(e["episode_idx_global"]),
            'meta_action': meta_action,
            'navigable_mask_path': e.get('navigable_mask_path', None),
            'affordance_gt_prob': affordance_gt_prob if 'affordance_gt_prob' in locals() else None,
            'affordance_gt_bin': affordance_gt_bin if 'affordance_gt_bin' in locals() else None,
            # 'wp_mask': wp_mask if 'wp_mask' in locals() else None,
            'instruction_id': e.get('instruction_id', None),
            'sub_index': e.get('sub_index', None),
        }

from typing import Callable, Dict, Iterable, List, Optional
from tqdm import tqdm

class InternVL_V1_5_Dataset_Multiview_PKL(InternVL_V1_5_Dataset_Multiview):
    def __init__(
        self,
        model_path,
        template,
        pkl_path,
        dataset_root,
        *,
        max_length=8192,
        min_dynamic_patch_override: Optional[int] = 1,
        max_dynamic_patch_override: Optional[int] = 1,
        use_thumbnail_override: Optional[bool] = False,
        loader_kwargs: Optional[Dict] = None,
        output_mode: str = "VQA",
    ):
        self.template = template
        self.max_length = max_length

        self.output_mode = output_mode  # "RoboPoint" or "Affordance" or "VQA"

        self.cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if self.cfg.llm_config.architectures[0] == "Phi3ForCausalLM":
            self._system = "You are an AI assistant whose name is Phi-3."
            self.template["INSTRUCTION"] = "<|user|>\n{input}<|end|><|assistant|>\n"
        elif self.cfg.llm_config.architectures[0] == "InternLM2ForCausalLM":
            self._system = "You are an AI assistant whose name " "is InternLM (书生·浦语)."
            self.template["SYSTEM"] = "<|im_start|>system\n{system}<|im_end|>"
            self.template["INSTRUCTION"] = (
                "<|im_start|>user\n{input}" "<|im_end|><|im_start|>assistant\n"
            )
        else:
            raise NotImplementedError

        # Overrides without touching model config.json
        self.min_dynamic_patch = (
            min_dynamic_patch_override
            if min_dynamic_patch_override is not None
            else self.cfg.min_dynamic_patch
        )
        self.max_dynamic_patch = (
            max_dynamic_patch_override
            if max_dynamic_patch_override is not None
            else self.cfg.max_dynamic_patch
        )
        self.use_thumbnail = (
            use_thumbnail_override
            if use_thumbnail_override is not None
            else self.cfg.use_thumbnail
        )
        self.downsample_ratio = self.cfg.downsample_ratio
        self.image_size = self.cfg.force_image_size
        patch_size = self.cfg.vision_config.patch_size
        self.patch_token = int(
            (self.image_size // patch_size) ** 2 * (self.downsample_ratio**2)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self._add_special_tokens()

        self.transformer = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (self.image_size, self.image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )

        if loader_kwargs is None:
            loader_kwargs = {}

        print_log("Starting to load PKL data and calc length", logger="current")
        self.data = []
        self.image_folder = []
        self.group_length = []
        self.conv2length_text = {}

        self.pkl_loader = SingleFrameLoader(
            ann_file=pkl_path,
            dataset_root=dataset_root,
            split='val' if 'eval' in self.output_mode else 'train',
            **loader_kwargs,
            return_imgs=False,
        )
        self.pkl_path = pkl_path


        for sample in tqdm(self.pkl_loader, total=len(self.pkl_loader), desc="Loading anchor frames from PKL which is a subset of"):
            # images_dict = sample.get("images", {})
            # # Keep order deterministic
            # ordered_keys = ["front", "left", "back", "right"]
            # images = [images_dict[k] for k in ordered_keys if k in images_dict]
            # convs = self.build_conversations_from_sample(sample)

            # data_item = {
            #     "image": images,
            #     "conversations": convs,
            # }
            # extras = {k: v for k, v in sample.items() if k not in ("images", "depths", "instruction", "meta_action")}
            # data_item.update(copy.deepcopy(extras))
            # # print("[debug] ", data_item.keys())
            # self.data.append(data_item)
            # self.image_folder.append(None)  # not used in on-the-fly mode

            # conversations = "\n".join([temp["value"] for temp in convs])
            # str_length = len(conversations)
            # if str_length not in self.conv2length_text:
            #     token_length = self.tokenizer(
            #         conversations, return_tensors="pt", padding=False, truncation=False
            #     ).input_ids.size(1)
            #     self.conv2length_text[str_length] = token_length
            # else:
            #     token_length = self.conv2length_text[str_length]

            # image_token = len(images) * self.patch_token * (
            #     self.max_dynamic_patch + self.use_thumbnail
            # )
            # token_length = token_length + image_token
            # self.group_length.append(token_length)

            pano_path = sample["pano_path"]
            depth_path = sample["depth_paths"]
            base_yaw_deg = sample["agent_yaw_hab_rad"] * 180.0 / np.pi  # adjust if needed
            instr = sample.get("instruction", "")
            meta_action = sample.get("meta_action", "")
            # convs = self.build_conversations_from_sample(sample)
            if self.output_mode == "RoboPoint" or self.output_mode == "RoboPoint_eval":
                variant_specs = [{"variant": "RoboPoint"}]
            elif self.output_mode == "VQA" or self.output_mode == "VQA_eval":
                variant_specs = [{"variant": "RawChat"}]
            elif self.output_mode == "Affordance":
                variant_specs = [
                    {"variant": "SingleTokenEmbedding"},
                    # {"variant": "RawChat"}, # debug ablate
                ]
            elif self.output_mode == "Affordance_eval":
                variant_specs = [
                    {"variant": "SingleTokenEmbedding"},
                    # {"variant": "RawChat"}, # BUG : not sure why ste is severely wrong
                ]
            elif self.output_mode == "LSSAttn" or self.output_mode == "LSSAttn_eval":
                variant_specs = [
                    {"variant": "LSSAttn"},
                    # {"variant": "RawChat"},
                ]
            else:
                raise NotImplementedError
                
            for spec in variant_specs:
                convs = self.build_conversations_from_sample(
                    sample, variant=spec["variant"]
                )

                data_item = {
                    "pano_path": pano_path,
                    "depth_path": depth_path,
                    "base_yaw_deg": base_yaw_deg,
                    "instruction": instr,
                    "meta_action": meta_action,
                    "conversations": convs,
                    "variant": spec["variant"],
                    # keep any other metadata you need (scene, p_goal, etc.)
                }
                extras = {k: v for k, v in sample.items() if k not in ("images", "depths", "instruction", "meta_action")}
                data_item.update(copy.deepcopy(extras))
                self.data.append(data_item)
                self.image_folder.append(None)

                conversations = "\n".join([temp["value"] for temp in convs])
                str_length = len(conversations)
                if str_length not in self.conv2length_text:
                    token_length = self.tokenizer(
                        conversations, return_tensors="pt", padding=False, truncation=False
                    ).input_ids.size(1)
                    self.conv2length_text[str_length] = token_length
                else:
                    token_length = self.conv2length_text[str_length]

                # estimate image tokens: 4 views → 4 images
                image_token = 4 * self.patch_token * (
                    self.max_dynamic_patch + self.use_thumbnail
                )
                token_length = token_length + image_token
                self.group_length.append(token_length)

        if self.output_mode == "Affordance":    
            # for balanced sampling during training
            self.idx_affordance = [i for i, d in enumerate(self.data) if d["variant"] == "SingleTokenEmbedding"]
            self.idx_chat  = [i for i, d in enumerate(self.data) if d["variant"] == "RawChat"]

        # # compute region label class frequencies
        # self.region_class_freq = {}
        # print(len(self.data))
        # for sample in self.data:
        #     region_label = sample.get("region_label", [])
        #     self.region_class_freq[region_label] = self.region_class_freq.get(region_label, 0) + 1
        # total_regions = sum(self.region_class_freq.values())
        # # for lbl in self.region_class_freq:
        # #     self.region_class_freq[lbl] /= total_regions
        # # debug
        # region_label_to_names = {
        #     0: "FRONT_SMALL",
        #     1: "FRONT_BIG",
        #     2: "FRONTLEFT_SMALL",
        #     3: "FRONTLEFT_BIG",
        #     4: "LEFT_SMALL",
        #     5: "LEFT_BIG",
        #     6: "BACKLEFT_SMALL",
        #     7: "BACKLEFT_BIG",
        #     8: "BACK_SMALL",
        #     9: "BACK_BIG",
        #     10: "BACKRIGHT_SMALL",
        #     11: "BACKRIGHT_BIG",
        #     12: "RIGHT_SMALL",
        #     13: "RIGHT_BIG",
        #     14: "FRONTRIGHT_SMALL",
        #     15: "FRONTRIGHT_BIG",
        # }
        # print("Region class frequencies:")
        # for lbl, freq in self.region_class_freq.items():
        #     print(f"Region label {region_label_to_names.get(lbl, 'UNKNOWN')}: freq={freq:.4f}")
        # # Region label RIGHT_SMALL: freq=0.0562
        # # Region label FRONTRIGHT_SMALL: freq=0.1140
        # # Region label FRONT_BIG: freq=0.1282
        # # Region label BACKLEFT_BIG: freq=0.0252
        # # Region label FRONT_SMALL: freq=0.1455
        # # Region label LEFT_BIG: freq=0.0730
        # # Region label FRONTRIGHT_BIG: freq=0.1087
        # # Region label FRONTLEFT_SMALL: freq=0.1035
        # # Region label RIGHT_BIG: freq=0.0389
        # # Region label LEFT_SMALL: freq=0.0473
        # # Region label FRONTLEFT_BIG: freq=0.0972
        # # Region label BACK_SMALL: freq=0.0079
        # # Region label BACK_BIG: freq=0.0131
        # # Region label BACKLEFT_SMALL: freq=0.0158
        # # Region label BACKRIGHT_SMALL: freq=0.0084
        # # Region label BACKRIGHT_BIG: freq=0.0173
        # # so 18.4 times difference between most/least frequent regions
        # exit(0)

        # # box pipeline
        # self.box_pipeline = BoxPipeline()

        self.oracle_bev_generator = BEVMaskGenerator(
            # ann_file="<FALCON_DATA>/captures_v3_val_S9hNv5qa7GM/scanS9hNv5qa7GM_Landmark-RxR-dynamic.pkl",
            ann_file=f'{_BEACON_FALCON_DATA}/captures_v3_val_2azQ1b91cZZ/scan2azQ1b91cZZ_Landmark-RxR-dynamic.pkl',
            out_dir=f"{_BEACON_CACHE_ROOT}/traversable_masks",
            outer_half_xy=6.4,
            inner_half_xy=3.6,
            outer_pitch=0.2,
            inner_pitch=0.1,
            z_up=0.4,
            z_down=2.0,
            clearance_band=(-1.4, 0.0),
            support_band=(-1.6, -1.4),
            connectivity=4,
            corner_check=True,
            inner_vote_min=2,
        )

        assert len(self.group_length) == len(self.data)
        print_log("end loading PKL data and calc length", logger="current")
        print_log(f"=======total {len(self.data)} samples=======", logger="current")
        self._max_refetch = 1000
        with open(self.pkl_path, "rb") as f:
            _ann_raw = pickle.load(f)
        if "data_list" not in _ann_raw:
            raise KeyError(f"Missing `data_list` in PKL: {self.pkl_path}")
        self._raw_instruction_sub_lookup = defaultdict(lambda: defaultdict(list))
        for e in tqdm(_ann_raw["data_list"], desc="Building raw instruction-subindex lookup from PKL"):
            if "instruction_id" not in e or "sub_index" not in e:
                raise KeyError("Raw PKL entry missing `instruction_id` or `sub_index`.")
            scene = e.get("scene_token") or e.get("scene_name") or e.get("scene_idx")
            if not scene:
                raise KeyError("Raw PKL entry missing scene key (scene_token/scene_name/scene_idx).")
            token = str(e.get("token", ""))
            if token == "":
                raise KeyError("Raw PKL entry missing `token`.")
            if "habitat_base_yaw" not in e:
                raise KeyError("Raw PKL entry missing `habitat_base_yaw`.")
            if "ego2global" not in e:
                raise KeyError("Raw PKL entry missing `ego2global`.")
            vhash = token.split("_yaw")[0] if "_yaw" in token else (str(e.get("viewpoint_id", "")) or token)
            base_yaw_rad = self.pkl_loader.yaw_to_rad_auto(e["habitat_base_yaw"] + math.pi / 2)
            base_yaw_deg = self.pkl_loader.yaw_rad_to_deg_wrapped(base_yaw_rad)
            iid = e["instruction_id"]
            sub_idx = int(e["sub_index"])
            for cap in self.pkl_loader.capture_versions:
                self._raw_instruction_sub_lookup[iid][sub_idx].append(
                    {
                        "instruction_id": iid,
                        "sub_index": sub_idx,
                        "capture_version": str(cap).strip().lower(),
                        "scene": scene,
                        "token": token,
                        "viewpoint_hash": vhash,
                        "base_yaw_deg": base_yaw_deg,
                        "ego2global": np.asarray(e["ego2global"], dtype=np.float32),
                        "cam2imgs": {
                            "front": self.pkl_loader._cam2img(),
                            "left": self.pkl_loader._cam2img(),
                            "back": self.pkl_loader._cam2img(),
                            "right": self.pkl_loader._cam2img(),
                        },
                        "cam2egos": {
                            "front": self.pkl_loader._cam2ego(0.0),
                            "left": self.pkl_loader._cam2ego(90.0),
                            "back": self.pkl_loader._cam2ego(180.0),
                            "right": self.pkl_loader._cam2ego(-90.0),
                        },
                    }
                )
        self._instruction_sub_lookup = defaultdict(lambda: defaultdict(list))
        for idx, s in enumerate(self.data):
            iid = s.get("instruction_id", None)
            sub_idx = s.get("sub_index", None)
            if iid is None or sub_idx is None:
                continue
            try:
                sub_idx = int(sub_idx)
            except Exception:
                continue
            self._instruction_sub_lookup[iid][sub_idx].append(idx)

    def _add_special_tokens(self):
        assert hasattr(self, "tokenizer")
        target_tokens = ['[TGT]']
        region_tokens = [
            # '[FRONT_SMALL]',
            # '[FRONT_BIG]',
            # '[FRONTLEFT_SMALL]',
            # '[FRONTLEFT_BIG]',
            # '[LEFT_SMALL]',
            # '[LEFT_BIG]',
            # '[BACKLEFT_SMALL]',
            # '[BACKLEFT_BIG]',
            # '[BACK_SMALL]',
            # '[BACK_BIG]',
            # '[BACKRIGHT_SMALL]',
            # '[BACKRIGHT_BIG]',
            # '[RIGHT_SMALL]',
            # '[RIGHT_BIG]',
            # '[FRONTRIGHT_SMALL]',
            # '[FRONTRIGHT_BIG]',
        ]
        special_tokens = target_tokens + region_tokens
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        return

    @staticmethod
    def _stack_cam_mats(sample):
        cam2imgs = sample["cam2imgs"]
        cam2egos = sample["cam2egos"]
        if isinstance(cam2imgs, dict):
            order = ["front", "left", "back", "right"]
            cam2imgs = np.stack([cam2imgs[k] for k in order], axis=0)
            cam2egos = np.stack([cam2egos[k] for k in order], axis=0)
        cam2imgs = torch.as_tensor(cam2imgs, dtype=torch.float32)
        cam2egos = torch.as_tensor(cam2egos, dtype=torch.float32)
        intrinsics = cam2imgs[:, :3, :3]
        return cam2imgs, cam2egos, intrinsics

    def _load_depth_views(self, sample):
        depth_pano_patch = np.load(sample["depth_path"])
        depth_pano_range = self.pkl_loader.rectify_patchwork_pano_depth_to_range_ring(
            depth_pano_patch, max_depth=10.0, ring_vfov_deg=90.0, invalid_to=10.0
        )
        base_yaw_deg = sample["base_yaw_deg"]
        dirs = [0.0, 90.0, 180.0, -90.0]
        depths = []
        for off in dirs:
            depth_range = self.pkl_loader.pano_to_perspective(
                depth_pano_range,
                self.pkl_loader.fov_deg,
                self.pkl_loader.out_hw,
                base_yaw_deg + off,
                0.0,
                interp=cv2.INTER_NEAREST,
            )
            depths.append(self.pkl_loader.range_to_plane_depth_z(depth_range))
        return np.stack(depths).astype(np.float32)  # (4, H, W)

    @staticmethod
    def _warp_bev_mask_src_to_cur(mask_src, T_cur_from_src, x_min=-6.4, y_min=-6.4, res=0.1):
        # mask layout is [Dx, Dy] where axis-0 is x(forward), axis-1 is y(left)
        m = torch.as_tensor(mask_src, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 1,1,Dx,Dy
        _, _, Dx, Dy = m.shape
        device = m.device

        xs = x_min + (torch.arange(Dx, dtype=torch.float32, device=device) + 0.5) * res
        ys = y_min + (torch.arange(Dy, dtype=torch.float32, device=device) + 0.5) * res
        Xc, Yc = torch.meshgrid(xs, ys, indexing="ij")  # Dx,Dy

        ones = torch.ones_like(Xc)
        zeros = torch.zeros_like(Xc)
        p_cur = torch.stack([Xc, Yc, zeros, ones], dim=-1).view(-1, 4).T  # 4, Dx*Dy

        T_cur_from_src = torch.as_tensor(T_cur_from_src, dtype=torch.float32, device=device)
        T_src_from_cur = torch.linalg.inv(T_cur_from_src)
        p_src = T_src_from_cur @ p_cur
        Xs = p_src[0].view(Dx, Dy)
        Ys = p_src[1].view(Dx, Dy)

        i_src = (Xs - (x_min + 0.5 * res)) / res
        j_src = (Ys - (y_min + 0.5 * res)) / res
        gx = 2.0 * j_src / max(Dy - 1, 1) - 1.0  # width coord
        gy = 2.0 * i_src / max(Dx - 1, 1) - 1.0  # height coord
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # 1,Dx,Dy,2

        warped = torch.nn.functional.grid_sample(
            m, grid, mode="nearest", padding_mode="zeros", align_corners=True
        )
        return warped[0, 0] > 0.5

    def _neighbor_indices(self, sample, index):
        iid = sample.get("instruction_id", None)
        sub_idx = sample.get("sub_index", None)
        if iid is None or sub_idx is None:
            return [index]
        try:
            sub_idx = int(sub_idx)
        except Exception:
            return [index]

        picked = []
        for sidx in (sub_idx, sub_idx + 1):
            picked.extend(self._instruction_sub_lookup.get(iid, {}).get(sidx, []))
        if index not in picked:
            picked.insert(0, index)

        uniq = []
        seen = set()
        for k in picked:
            if k in seen:
                continue
            seen.add(k)
            uniq.append(k)
        return uniq if len(uniq) > 0 else [index]

    def _neighbor_raw_descs(self, sample, keep=(0, 1)):
        """
        keep:
        - tuple/list of relative sub-index offsets, e.g. (0, 1) for current + next
        - -1 means use all sub-indexes in the same instruction_id episode
        """
        iid = sample.get("instruction_id", None)
        sub_idx = sample.get("sub_index", None)
        if iid is None or sub_idx is None:
            raise KeyError("Current sample missing `instruction_id` or `sub_index`.")
        sub_idx = int(sub_idx)

        by_sub = self._raw_instruction_sub_lookup.get(iid, None)
        if by_sub is None or len(by_sub) == 0:
            raise RuntimeError(f"No raw episode pool for instruction_id={iid}")

        picked = []
        if keep == -1:
            for sidx in sorted(by_sub.keys()):
                picked.extend(by_sub[sidx])
        else:
            if not isinstance(keep, (tuple, list)):
                raise TypeError(f"`keep` must be -1 or tuple/list, got {type(keep)}")
            for off in keep:
                sidx = sub_idx + int(off)
                picked.extend(by_sub.get(sidx, []))

        if len(picked) == 0:
            raise RuntimeError(
                f"No raw neighbors for instruction_id={iid}, sub_index={sub_idx}, keep={keep}"
            )

        # dedup while preserving order
        uniq = []
        seen = set()
        for d in picked:
            k = (d["capture_version"], d["token"], int(d["sub_index"]))
            if k in seen:
                continue
            seen.add(k)
            uniq.append(d)

        if len(uniq) == 0:
            raise RuntimeError(
                f"Neighbors collapsed to empty after dedup for instruction_id={iid}, sub_index={sub_idx}, keep={keep}"
            )
        return uniq


    def _materialize_raw_desc(self, desc):
        depth_path = self.pkl_loader._find_depth(
            # desc["scene"], desc["token"], desc["viewpoint_hash"], capture_version=desc["capture_version"]
            desc["scene"], desc["token"], desc["viewpoint_hash"], capture_version="static" # this is for another bug that ped moving too slow in sim
        )
        if depth_path is None:
            # raise FileNotFoundError(
            #     f"Missing depth for raw neighbor scene={desc['scene']} token={desc['token']} "
            #     f"vhash={desc['viewpoint_hash']} capture={desc['capture_version']}"
            # )
            return None
        return {
            "depth_path": depth_path,
            "base_yaw_deg": desc["base_yaw_deg"],
            "cam2imgs": desc["cam2imgs"],
            "cam2egos": desc["cam2egos"],
            "ego2global": desc["ego2global"],
            "token": desc["token"],
            "capture_version": desc["capture_version"],
            "sub_index": desc["sub_index"],
        }
    
    def prepare_data(self, index):
    #     out = super().prepare_data(index)
    #     if out is None:
    #         return None
    #     sample = self.data[index]
    #     # attach everything except the fields the base already handled
    #     extras = {k: v for k, v in sample.items() if k not in ("images", "depths", "instruction", "meta_action")}
    #     out.update(copy.deepcopy(extras))  # or setdefault("meta", ...)
    #     # print("[debug] prepared data keys:", out.keys())
    #     return out

        sample = self.data[index]

        # load pano/depth
        pano = cv2.imread(sample["pano_path"], cv2.IMREAD_COLOR)
        # depth = np.load(sample["depth_path"]) if sample.get("depth_path") else None
        depth_pano_patch = np.load(sample["depth_path"]) if sample.get("depth_path") else None
        depth_pano_range = self.pkl_loader.rectify_patchwork_pano_depth_to_range_ring(
            depth_pano_patch, max_depth=10.0, ring_vfov_deg=90.0, invalid_to=10.0
        )
        base_yaw_deg = sample["base_yaw_deg"]

        # crop four views
        dirs = [("front", 0.0), ("left", 90.0), ("back", 180.0), ("right", -90.0)]
        images = []
        depths = []
        for _, off in dirs:
            images.append(
                self.pkl_loader.pano_to_perspective(
                    pano, self.pkl_loader.fov_deg, self.pkl_loader.out_hw, base_yaw_deg + off, 0.0, interp=cv2.INTER_LINEAR
                )
            )
            # images.append(
            #     self.pkl_loader.pano_to_linear_slice(
            #         pano, self.pkl_loader.fov_deg, self.pkl_loader.out_hw, base_yaw_deg + off, 0.0, interp=cv2.INTER_LINEAR
            #     )
            # )
            # depths.append(
            #     self.pkl_loader.pano_to_perspective(
            #         depth, self.pkl_loader.fov_deg, self.pkl_loader.out_hw, base_yaw_deg + off, 0.0, interp=cv2.INTER_NEAREST
            #     )
            # )
            # 1) crop range depth from corrected pano
            depth_range = self.pkl_loader.pano_to_perspective(
                depth_pano_range,
                self.pkl_loader.fov_deg,
                self.pkl_loader.out_hw,
                base_yaw_deg + off,
                0.0,
                interp=cv2.INTER_NEAREST,
            )

            # 2) convert range -> plane z-depth (so it's "normal depth" again)
            depths.append(self.pkl_loader.range_to_plane_depth_z(depth_range))

        
        # # debug
        # print(f"[debug] viewpoint: {sample['token']} at yaw {base_yaw_deg:.2f}°")
        # current_id = sample['viewpoint_hash']
        # cv2.imwrite(f'debug_outputs/debug_tile_left_{current_id}.jpg', images[1])
        # cv2.imwrite(f'debug_outputs/debug_tile_right_{current_id}.jpg', images[3])
        # cv2.imwrite(f'debug_outputs/debug_tile_front_{current_id}.jpg', images[0])
        # cv2.imwrite(f'debug_outputs/debug_tile_back_{current_id}.jpg', images[2])
        # print("111")
        # print("[debug] img shape:", images[0].shape, images[1].shape, images[2].shape, images[3].shape)
        # print("[debug] depth shape:", depths[0].shape, depths[1].shape, depths[2].shape, depths[3].shape)
        # print("[debug] img dtype min max:", images[0].dtype, images[0].min(), images[0].max())
        # print("[debug] instruction:", sample['instruction'])
        # print("[debug] p_goal shape:", sample['p_goal'].shape, sample['p_goal'])
        # exit(0)

        # # prepare the boxes as nodes
        # detected_boxes, selected_boxes = self.box_pipeline.run(
        #     images, depths, sample['conversations'], p_goal=sample['p_goal']
        # )
        # sample['detected_boxes'] = detected_boxes
        # sample['selected_boxes'] = selected_boxes

        pixel_value_chunks = []
        image_token_blocks = []
        for img_np in images:
            image = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
            tiles = dynamic_preprocess(
                image,
                self.min_dynamic_patch,
                self.max_dynamic_patch,
                self.image_size,
                self.use_thumbnail,
            )
            transformed = [self.transformer(t) for t in tiles]
            pixel_value_chunks.extend(transformed)
            num_tokens = len(transformed) * self.patch_token
            image_token_blocks.append(
                f"{self.IMG_START_TOKEN}"
                f"{self.IMG_CONTEXT_TOKEN * num_tokens}"
                f"{self.IMG_END_TOKEN}"
            )

        pixel_values = torch.stack(pixel_value_chunks)
        token_dict = self.get_inputid_labels(sample["conversations"], image_token_blocks)
        out = {"pixel_values": pixel_values}
        out["depth"] = np.stack(depths).astype(np.float32) # (4, H, W) float32 meters
        out.update(token_dict)
        out['raw_img'] = np.stack(images)  # (4, H, W, 3) uint8 BGR

        # ---- load annotations from npz ----
        npz_rel = sample.get("navigable_mask_path", "")
        if npz_rel:
            # assume npz_rel is relative to pkl directory 
            scan_id = sample.get('scene')
            npz_path = os.path.join(os.path.dirname(self.pkl_path), 'annotations', scan_id, npz_rel)
            if not os.path.isfile(npz_path):
                raise FileNotFoundError(f"NPZ not found: {npz_path}")

            npz = np.load(npz_path)
            A = npz["traversable_mask"].astype(np.uint8)   # (128,128)
            B = npz["visible_mask"].astype(np.uint8)       # (128,128)
            C = npz["affordance_mask"].astype(np.uint8)    # (128,128)

            # attach to output (per-sample; DataLoader will stack to Bx128x128)
            out["traversable_mask"] = torch.from_numpy(A)  # uint8
            out["visible_mask"] = torch.from_numpy(B)
            out["affordance_mask"] = torch.from_numpy(C)

            # visible_affordance = np.logical_and(B, C).astype(np.uint8)  # (128,128)
            # if visible_affordance.sum() == 0:
            #     print(f"[warning] Skipping zero visible affordance mask for sample idx {index} token {sample.get('token')}")
            #     return None
            # BUG: this will trigger rand another which will change the evaluation metrics from time to time, so we move it to SingleFrameLoader

        else:
            raise RuntimeError(f"Missing navigable_mask_path in sample idx {index} token {sample.get('token')}")
            # optional: keep keys for collate consistency
            out["traversable_mask"] = None
            out["visible_mask"] = None
            out["affordance_mask"] = None

        # Pseudo-label from nearby frames:
        compute_pseudo_label = True
        if compute_pseudo_label:
            # OR( free_ground_obs at sub_index and sub_index+1 in same instruction_id, warped to current ego )
            # AND current maskA.
            from xtuner.perception_modules.affordance_head import AffordanceHead

            # current frame geometry
            depth_cur = torch.from_numpy(out["depth"]).float().unsqueeze(0)  # [1,4,H,W]
            _, cam2egos_cur, intrinsics_cur = self._stack_cam_mats(sample)
            cam2egos_cur = cam2egos_cur.unsqueeze(0)      # [1,4,4,4]
            intrinsics_cur = intrinsics_cur.unsqueeze(0)  # [1,4,3,3]

            free_cur, blocker_cur, _, maskA_cur = AffordanceHead.prepare_free_obs_new(
                depth_cur,
                cam2egos_cur,
                intrinsics_cur,
                grid_config=dict(x=[-6.4, 6.4, 0.1], y=[-6.4, 6.4, 0.1]),
                Dx=128,
                Dy=128,
                ground_z_range=(-1.6, -1.4),
                blocker_z_range=(-1.0, -0.1),
                mid_z_range=(-0.1, 0.1),
                tall_z_range=(0.25, 0.35),
            )
            free_merged = (free_cur[0] > 0.5)  # [128,128], bool
            maskA_cur = (maskA_cur[0] > 0.5)   # [128,128], bool

            free_merged_debug_list = [free_merged]

            T_cur = torch.as_tensor(sample["ego2global"], dtype=torch.float32)
            raw_descs = self._neighbor_raw_descs(sample, keep=(0, 1, 2))
            for desc in raw_descs:
                nbr = self._materialize_raw_desc(desc)
                if nbr is None:
                    print(f"[WARNING] Due to not recordable in habitat, missing depth for raw neighbor scene={desc['scene']} token={desc['token']} "
                        f"vhash={desc['viewpoint_hash']} capture={desc['capture_version']}")
                    continue
                depth_nbr = torch.from_numpy(self._load_depth_views(nbr)).float().unsqueeze(0)  # [1,4,H,W]
                _, cam2egos_nbr, intrinsics_nbr = self._stack_cam_mats(nbr)
                cam2egos_nbr = cam2egos_nbr.unsqueeze(0)
                intrinsics_nbr = intrinsics_nbr.unsqueeze(0)
                free_nbr, _, _, _ = AffordanceHead.prepare_free_obs_new(
                    depth_nbr,
                    cam2egos_nbr,
                    intrinsics_nbr,
                    grid_config=dict(x=[-6.4, 6.4, 0.1], y=[-6.4, 6.4, 0.1]),
                    Dx=128,
                    Dy=128,
                    ground_z_range=(-1.6, -1.4),
                    blocker_z_range=(-1.0, -0.1),
                    mid_z_range=(-0.1, 0.1),
                    tall_z_range=(0.25, 0.35),
                )
                T_nbr = torch.as_tensor(nbr["ego2global"], dtype=torch.float32)
                T_cur_from_nbr = torch.linalg.inv(T_cur) @ T_nbr
                free_nbr_in_cur = self._warp_bev_mask_src_to_cur(
                    free_nbr[0] > 0.5, T_cur_from_nbr, x_min=-6.4, y_min=-6.4, res=0.1
                )
                free_merged = free_merged | free_nbr_in_cur
                free_merged_debug_list.append(free_nbr_in_cur)

            # pseudo_bound = (free_merged & maskA_cur).to(torch.float32)
            pseudo_bound = free_merged.to(torch.float32)
            out["pseudo_bound"] = pseudo_bound

        # # debug
        # debug_path = f'debug_outputs/0_pseudo_label.npz'
        # np.savez_compressed(
        #     debug_path,
        #     free_merged=free_merged.cpu().numpy(),
        #     maskA_cur=maskA_cur.cpu().numpy(),
        #     pseudo_bound=pseudo_bound.cpu().numpy(),
        #     free_merged_debug_list=[m.cpu().numpy() for m in free_merged_debug_list],
        #     traversable_mask=out['traversable_mask'].cpu().numpy(),
        #     visible_mask=out['visible_mask'].cpu().numpy(),
        #     affordance_mask=out['affordance_mask'].cpu().numpy(),
        # )
        # print(f"[pipeline] saved debug pseudo-label components to {debug_path}")
        # iid = sample["instruction_id"]
        # cur = int(sample["sub_index"])
        # all_sub = sorted(self._raw_instruction_sub_lookup[iid].keys())
        # is_last = (cur == all_sub[-1])
        # actual_n = len(free_merged_debug_list)
        # if actual_n == 1:
        #     assert is_last, (
        #         f"[pseudo-label] actual_n==1 but current is not last sub_index. "
        #         f"instruction_id={iid}, cur={cur}, all_sub={all_sub}, token={sample.get('token')}"
        #     )
        # # exit(0)
        # # end of debug


        if 'eval' not in self.output_mode:
            use_freespace_loss = False
            if use_freespace_loss:
                if 'traversable_mask' in out and out['traversable_mask'] is not None:
                    curr_token = sample.get("token")
                    # # debug
                    # save_path = f'debug_outputs/debug_sdf.npz'
                    # occ_field = self._sdf_from_mask(out['traversable_mask'], eight_neigh=True, d0_cells=1)  # (H, W) float32
                    # np.savez_compressed(save_path, occ_field=occ_field, raw_occ=out['traversable_mask'].numpy())
                    # print(f"[pipeline] saved debug sdf to {save_path}")
                    # exit(0)
                    # # end of debug
                    try:
                        occ_field = np.load(os.path.join(
                            f'{_BEACON_CACHE_ROOT}/temp_occ_field',
                            f"{curr_token}_occ_field.npy"
                        ))
                    except Exception as e:
                        print(f"[pipeline] preloaded occ field not found for {curr_token}, computing online...")
                        occ_field = self._sdf_from_mask(out['traversable_mask'], eight_neigh=True)  # (H, W) float32
                        save_path = f"{_BEACON_CACHE_ROOT}/temp_occ_field/{curr_token}_occ_field.npy"
                        
                        np.save(save_path, occ_field)
                        print(f"[pipeline] saved sdf to {save_path}")
                    out['occ_field'] = occ_field  # (H, W) float32
                    # exit(0)
                else:
                    occ_label_path = f'{_BEACON_FALCON_DATA}/captures_v3_train_S9hNv5qa7GM/multi_yaw_traversable_masks'
                    curr_base_yaw_rad = (sample['agent_yaw_hab_rad'] + 2*math.pi) % (2*math.pi) # wrap to [0, 2pi]
                    curr_base_yaw_deg = math.degrees(curr_base_yaw_rad)
                    viewpoint_hash = sample['viewpoint_hash']
                    try:
                        occ_with_yaw = np.load(os.path.join(occ_label_path, f"{viewpoint_hash}.npz"), allow_pickle=True)
                    except Exception as e:
                        print(f"[warning] failed to load precomputed occ for sample {sample['token']}: {e}, skipping...")
                        return None
                    traversable_mask = occ_with_yaw[f'traversable_mask']
                    combined_mask = np.all(traversable_mask, axis=1) # (256, H, W)
                    all_base_yaw_deg = occ_with_yaw[f'base_yaw'] # 256,
                    curr_base_yaw_bin = np.argmin(np.abs(all_base_yaw_deg - curr_base_yaw_deg))
                    occ_label = combined_mask[curr_base_yaw_bin].astype(np.uint8)  # (H, W)
                    out['occ_label'] = occ_label

                    # # debug
                    # import matplotlib.pyplot as plt
                    # plt.imshow(occ_label)
                    # plt.xlabel('+y (leftward)')
                    # plt.ylabel('+x (forward)')
                    # plt.colorbar()
                    # plt.savefig(f'debug_outputs/{current_id}_occ.png', bbox_inches='tight', dpi=150)
                    # plt.close()
                    # print("222")

                    # # debug
                    # from xtuner.perception_modules.oracle_bev import BEVMaskGenerator
                    # BEVMaskGenerator.save_vis(traversable_mask=occ_label, save_path='debug_outputs/debug_occ_combined.jpg')
                    # exit(0)
                    # occ_field = self._sdf_from_mask(occ_label, eight_neigh=True)  # (H, W) float32
                    # distance_field = self._distance_from_mask(
                    #     occ_label, sample['p_goal'][:2], x_min=-6.4, y_min=-6.4, res=0.2
                    # )  # (H, W) float32
                    # # geodesic_field = self._geodesic_distance_from_mask(
                    # #     occ_label, sample['p_goal'][:2], x_min=-6.4, y_min=-6.4, res=0.2
                    # # )  # (H, W) float32
                    # # out['geodesic_field'] = geodesic_field  # (H, W) float32
                    # out['distance_field'] = distance_field  # (H, W) float32
                    # out['occ_field'] = occ_field  # (H, W) float32
                    occ_field = None
                    distance_field = None
                    for attempt in range(3):
                        try:
                            occ_field = self._sdf_from_mask(occ_label, eight_neigh=True)  # (H, W) float32
                            distance_field = self._distance_from_mask(
                                occ_label, sample['p_goal'][:2], x_min=-6.4, y_min=-6.4, res=0.2
                            )  # (H, W) float32
                            break
                        except Exception as e:
                            if attempt == 2:
                                print(f"[warning] occ/distance failed for {sample.get('token')} idx {index}: {e}")
                                return None
                            print(f"[warning] sdf/distance computation error for {sample.get('token')} idx {index}: {e}, retrying...")
                            time.sleep(0.1)

                    out['distance_field'] = distance_field  # (H, W) float32
                    out['occ_field'] = occ_field  # (H, W) float32

        else:
            use_occ_label = False
            use_extra_label = False
            if use_occ_label:
                if 'traversable_mask' in out and out['traversable_mask'] is not None:
                    curr_token = sample.get("token")
                    try:
                        occ_field = np.load(os.path.join(
                            f'{_BEACON_CACHE_ROOT}/temp_occ_field',
                            f"{curr_token}_occ_field.npy"
                        ))
                    except Exception as e:
                        print(f"[pipeline] computing sdf for sample {curr_token}...")
                        occ_field = self._sdf_from_mask(out['traversable_mask'], eight_neigh=True)  # (H, W) float32
                        save_path = f"{_BEACON_CACHE_ROOT}/temp_occ_field/{curr_token}_occ_field.npy"
                        np.save(save_path, occ_field)
                        print(f"[pipeline] saved sdf to {save_path}")
                    out['occ_field'] = occ_field  # (H, W) float32
                else:
                    try:
                        occ_label_path = f'{_BEACON_FALCON_DATA}/captures_v3_train_S9hNv5qa7GM/multi_yaw_traversable_masks'
                        curr_base_yaw_rad = (sample['agent_yaw_hab_rad'] + 2*math.pi) % (2*math.pi) # wrap to [0, 2pi]
                        curr_base_yaw_deg = math.degrees(curr_base_yaw_rad)
                        viewpoint_hash = sample['viewpoint_hash']
                        try:
                            occ_with_yaw = np.load(os.path.join(occ_label_path, f"{viewpoint_hash}.npz"), allow_pickle=True)
                        except Exception as e:
                            raise RuntimeError
                        traversable_mask = occ_with_yaw[f'traversable_mask']
                        combined_mask = np.all(traversable_mask, axis=1) # (256, H, W)
                        all_base_yaw_deg = occ_with_yaw[f'base_yaw'] # 256,
                        curr_base_yaw_bin = np.argmin(np.abs(all_base_yaw_deg - curr_base_yaw_deg))
                        occ_label = combined_mask[curr_base_yaw_bin].astype(np.uint8)  # (H, W)
                        out['occ_label'] = occ_label

                        # # debug
                        # import matplotlib.pyplot as plt
                        # plt.imshow(occ_label)
                        # plt.xlabel('+y (leftward)')
                        # plt.ylabel('+x (forward)')
                        # plt.colorbar()
                        # plt.savefig(f'debug_outputs/{current_id}_occ.png', bbox_inches='tight', dpi=150)
                        # plt.close()
                        # print("222")

                        # from xtuner.perception_modules.oracle_bev import BEVMaskGenerator
                        # BEVMaskGenerator.save_vis(traversable_mask=occ_label, save_path=f'debug_outputs/debug_occ_combined_{current_id}.jpg')
                        # print("222")
                    except:
                        try:
                            traversable_mask, _, occluder_outer, _ = self.oracle_bev_generator.run_sample(
                                yaw_deg=math.degrees(sample['agent_yaw_hab_rad']),
                                target_token=sample['viewpoint_hash'],
                                save=False,
                                return_occluder=True,
                            )
                            limited_mask = self.oracle_bev_generator.limited_observation_mask_from_traversable(
                                traversable_mask.astype(bool),
                                occluder_mask=occluder_outer,
                                origin_radius_m=2.4,
                                unknown_val=255,
                                corner_check=True,
                                return_2ch=True
                            ) # 2, H, W
                            occ_label = np.logical_and(traversable_mask, limited_mask[1].astype(bool)) # both free and sure
                            out['occ_label'] = occ_label.astype(np.uint8)
                        except Exception as e:
                            print(f"[warning] oracle bev generation failed for sample {sample['token']}: {e}, skipping...")
                            return None
                    # # debug
                    # print(f"{combined.shape} at {math.degrees(sample['agent_yaw_hab_rad']):.2f}°")
                    # BEVMaskGenerator.save_vis(traversable_mask=combined.astype(np.uint8), save_path='debug_outputs/online_debug_occ_combined.jpg')
                    # exit(0)
                    # BUG: this may fail in multiworker, not sure here
                    if use_extra_label:
                        occ_field = self._sdf_from_mask(occ_label, eight_neigh=True)  # (H, W) float32
                        distance_field = self._distance_from_mask(
                            occ_label, sample['p_goal'][:2], x_min=-6.4, y_min=-6.4, res=0.2
                        )  # (H, W) float32
                        # geodesic_field = self._geodesic_distance_from_mask(
                        #     occ_label, sample['p_goal'][:2], x_min=-6.4, y_min=-6.4, res=0.2
                        # )  # (H, W) float32
                        # out['geodesic_field'] = geodesic_field  # (H, W) float32
                        out['distance_field'] = distance_field  # (H, W) float32
                        out['occ_field'] = occ_field  # (H, W) float32

        # # debug 
        # debug_costmap_path = 'debug_outputs/debug_costmap.npz'
        # np.savez_compressed(
        #     debug_costmap_path,
        #     sdf=occ_field,
        #     geodesic=geodesic_field,    
        #     distance=distance_field,
        #     goal_xy=sample['p_goal'][:2],
        #     occ_label=occ_label,
        # )
        # exit(0)

        # attach extras (ensure tensors for numeric fields)
        for k, v in sample.items():
            if k in ("pano_path", "depth_path", "base_yaw_deg"):
                continue
            if isinstance(v, np.ndarray):
                out[k] = torch.from_numpy(v)
            else:
                out[k] = v
        return out

    def _distance_from_mask(self, mask_free, goal_xy_m, x_min=-6.4, y_min=-6.4, res=0.2):
        H, W = np.asarray(mask_free).shape
        gx, gy = float(goal_xy_m[0]), float(goal_xy_m[1])

        # cell-center coordinates in meters
        xs = x_min + (np.arange(H, dtype=np.float32) + 0.5) * res  # (H,)
        ys = y_min + (np.arange(W, dtype=np.float32) + 0.5) * res  # (W,)

        # broadcast to (H,W)
        dx = xs[:, None] - gx
        dy = ys[None, :] - gy
        return np.sqrt(dx * dx + dy * dy).astype(np.float32)
    
    def _sdf_from_mask(self, mask_free, eight_neigh=True, d0_cells=1.0, res=0.1, use_l1=True, clamp_max=None, return_s=False):
        """
        Inputs:
        mask_free: (H,W) array-like, 1/True=free, 0/False=blocked
        eight_neigh: use 8-neighbour distances (1 and sqrt(2)) if True, else 4-neigh (1)
        d0_cells: margin d0 in *cell units* (for res=0.2m and margin=0.2m, d0_cells=1.0)
        clamp_max: optional float, clamp penalty to this max (helps huge unknown regions)
        return_s: if True, also return signed SDF s(i,j)

        Returns:
        occ_penalty: (H,W) float32, occ_penalty = relu(d0_cells - s)^2
        (optional) signed_s: (H,W) float32
        """
        free = (np.asarray(mask_free) > 0)
        H, W = free.shape

        # --- boundary cells (4-neigh defines interface) ---
        free_b = np.zeros((H, W), dtype=bool)  # free cells adjacent to blocked
        blk_b  = np.zeros((H, W), dtype=bool)  # blocked cells adjacent to free
        for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = np.zeros((H, W), dtype=bool)
            if di == -1: nb[1:,:]  = free[:-1,:]
            if di ==  1: nb[:-1,:] = free[1:,:]
            if dj == -1: nb[:,1:]  = free[:,:-1]
            if dj ==  1: nb[:,:-1] = free[:,1:]
            free_b |= free & (~nb)
            blk_b  |= (~free) & nb

        # --- multi-source dijkstra within a region mask ---
        moves = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0)]
        if eight_neigh:
            s2 = math.sqrt(2.0)
            moves += [(-1,-1,s2),(-1,1,s2),(1,-1,s2),(1,1,s2)]

        def ms_dijkstra(region_mask, sources_mask):
            dist = np.full((H, W), np.inf, dtype=np.float32)
            src = np.argwhere(sources_mask & region_mask)
            if src.size == 0:
                return dist
            
            # BUG: sometimes heapq.heappush is not callable??
            if not callable(heapq.heappush):
                import importlib
                importlib.reload(heapq)

            heap = []
            for i, j in src:
                dist[i, j] = 0.0
                heap.append((0.0, int(i), int(j)))
            heapq.heapify(heap)
            while heap:
                dcur, i, j = heapq.heappop(heap)
                # if dcur != dist[i, j]:
                if dcur > dist[i, j] + 1e-6:
                    continue
                for di, dj, w in moves:
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= H or nj < 0 or nj >= W:
                        continue
                    if not region_mask[ni, nj]:
                        continue
                    nd = dcur + w
                    if nd < dist[ni, nj]:
                        dist[ni, nj] = nd
                        heapq.heappush(heap, (nd, ni, nj))
            return dist

        # distance-to-boundary in cells, from boundary-cell centers; +0.5 approximates true interface line
        d_free = ms_dijkstra(free, free_b) + 0.5
        d_blk  = ms_dijkstra(~free, blk_b) + 0.5

        # degenerate cases (all-free or all-blocked => no boundary)
        big = float(max(H, W))
        d_free[~np.isfinite(d_free)] = big
        d_blk[~np.isfinite(d_blk)] = big

        signed_s = np.where(free, d_free, -d_blk).astype(np.float32)

        # # --- hinge penalty field ---
        # occ_penalty = np.maximum(0.0, d0_cells - signed_s) ** 2
        signed_m = signed_s * res
        margin_m = d0_cells * res
        if use_l1:
            occ_penalty = np.maximum(0.0, margin_m - signed_m)
        else:
            occ_penalty = np.maximum(0.0, margin_m - signed_m)**2
        if clamp_max is not None:
            occ_penalty = np.minimum(occ_penalty, float(clamp_max)).astype(np.float32)
        else:
            occ_penalty = occ_penalty.astype(np.float32)

        if return_s:
            return occ_penalty, signed_s
        return occ_penalty

    def _geodesic_distance_from_mask(self, mask_free, goal_xy_m, x_min=-6.4, y_min=-6.4, res=0.2):
        """
        mask_free: (H,W) 1/True=free, 0/False=blocked
        goal_xy_m: (2,) goal in meters (x forward, y left)
        returns: (H,W) geodesic distance in meters (8-neigh: 1 and sqrt(2))
                blocked + unreachable are set to Dmax (finite)
        """
        free = (np.asarray(mask_free) > 0)
        H, W = free.shape
        gx, gy = float(goal_xy_m[0]), float(goal_xy_m[1])
        gi = int(round((gx - x_min) / res))
        gj = int(round((gy - y_min) / res))
        gi = max(0, min(H - 1, gi))
        gj = max(0, min(W - 1, gj))

        Dmax = float(max(H, W) * math.sqrt(2) * res)
        dist = np.full((H, W), np.inf, dtype=np.float32)

        if not free[gi, gj]:
            return np.full((H, W), Dmax, dtype=np.float32)

        dist[gi, gj] = 0.0
        heap = [(0.0, gi, gj)]
        s2 = math.sqrt(2.0)
        moves = [(-1,0,1.0),(1,0,1.0),(0,-1,1.0),(0,1,1.0),
                (-1,-1,s2),(-1,1,s2),(1,-1,s2),(1,1,s2)]

        while heap:
            d, i, j = heapq.heappop(heap)
            # if d != dist[i, j]:
            if d > dist[i, j] + 1e-6:
                continue
            for di, dj, w in moves:
                ni, nj = i + di, j + dj
                if ni < 0 or ni >= H or nj < 0 or nj >= W:
                    continue
                if not free[ni, nj]:
                    continue
                nd = d + w * res
                if nd < dist[ni, nj]:
                    dist[ni, nj] = nd
                    heapq.heappush(heap, (nd, ni, nj))

        dist[~free] = Dmax
        dist[~np.isfinite(dist)] = Dmax
        return dist.astype(np.float32)


    def get_image(self, src):
        # Accept file paths (fallback to parent) or NumPy arrays (BGR->RGB handled)
        if isinstance(src, np.ndarray):
            img = src
            if img.ndim == 3 and img.shape[2] == 3:
                img = img[:, :, ::-1]  # BGR to RGB
            return Image.fromarray(img.astype(np.uint8))
        return super().get_image(src)
    
    def build_conversations_from_sample(self, sample: Dict, variant: str, use_end_orientation: bool = False) -> List[Dict]:
        instr = str(sample.get("instruction", "")).strip()
        meta_action = str(sample.get("meta_action", "")).strip()
        if use_end_orientation:
            output_template = "Move to the <MOVE_DIR> with a <RANGE_TAG> step, and end facing <END_DIR>."
            output_requirement_2 = "Here <MOVE_DIR> and <END_DIR> are discrete directions (e.g., Front, Front Right, Right, Back Right, Back, Back Left, Left, Front Left), and <RANGE_TAG> is either Small or Big."
        else:
            meta_action = meta_action.split(",")[0] + "."
            output_template = "Move to the <MOVE_DIR> with a <RANGE_TAG> step."
            output_requirement_2 = "Here <MOVE_DIR> is discrete directions (e.g., Front, Front Right, Right, Back Right, Back, Back Left, Left, Front Left), and <RANGE_TAG> is either Small or Big."
        
        output_requirement_1 = "Output a single navigation command in the exact format:"
        if variant == "RawChat":
            pass # keep as is
        elif variant == "RoboPoint":
            final_destination_xy = sample["p_goal"][:2]
            meta_action = f"{meta_action[:-1]} ({final_destination_xy[0]:.2f}, {final_destination_xy[1]:.2f})."
            output_template = f"{output_template[:-1]} (x.xx, y.yy)."
            output_requirement_2 += " The (x.xx, y.yy) indicates the final destination coordinates in local top-down frame (positive x is front, positive y is left)."
        elif variant == "SingleTokenEmbedding":

            # debug
            m = re.search(r"Move to the (.+?) with a (Small|Big) step", meta_action, flags=re.I)
            region_token = f"[{m.group(1).upper().replace(' ','').replace('-','')}_{m.group(2).upper()}]"
            region_token_to_label = {
                '[FRONT_SMALL]': 0,
                '[FRONT_BIG]': 1,
                '[FRONTLEFT_SMALL]': 2,
                '[FRONTLEFT_BIG]': 3,
                '[LEFT_SMALL]': 4,
                '[LEFT_BIG]': 5,
                '[BACKLEFT_SMALL]': 6,
                '[BACKLEFT_BIG]': 7,
                '[BACK_SMALL]': 8,
                '[BACK_BIG]': 9,
                '[BACKRIGHT_SMALL]': 10,
                '[BACKRIGHT_BIG]': 11,
                '[RIGHT_SMALL]': 12,
                '[RIGHT_BIG]': 13,
                '[FRONTRIGHT_SMALL]': 14,
                '[FRONTRIGHT_BIG]': 15,
            }
            sample["region_label"] = region_token_to_label[region_token] # explicit class label for LSSAttn
            # end of debug

            
            output_requirement_1 = "[TGT]"
            meta_action = ""
            output_template = ""
            output_requirement_2 = ""

        elif variant == "LSSAttn":
            m = re.search(r"Move to the (.+?) with a (Small|Big) step", meta_action, flags=re.I)
            region_token = f"[{m.group(1).upper().replace(' ','').replace('-','')}_{m.group(2).upper()}]"
            region_token_to_label = {
                '[FRONT_SMALL]': 0,
                '[FRONT_BIG]': 1,
                '[FRONTLEFT_SMALL]': 2,
                '[FRONTLEFT_BIG]': 3,
                '[LEFT_SMALL]': 4,
                '[LEFT_BIG]': 5,
                '[BACKLEFT_SMALL]': 6,
                '[BACKLEFT_BIG]': 7,
                '[BACK_SMALL]': 8,
                '[BACK_BIG]': 9,
                '[BACKRIGHT_SMALL]': 10,
                '[BACKRIGHT_BIG]': 11,
                '[RIGHT_SMALL]': 12,
                '[RIGHT_BIG]': 13,
                '[FRONTRIGHT_SMALL]': 14,
                '[FRONTRIGHT_BIG]': 15,
            }
            # meta_action = f"{meta_action[:-1]}" + region_token
            # output_template = f"{output_template[:-1]} [REGION TOKEN]."
            output_requirement_1 = "[TGT]"
            meta_action = ""
            output_template = ""
            output_requirement_2 = ""
            sample["region_label"] = region_token_to_label[region_token] # explicit class label for LSSAttn
        else:
            raise NotImplementedError
            # meta_action = f"<a> {meta_action} </a> [TGT]"
            # output_template = "<a> " + output_template + " </a> [TGT]"
        # prompt = (
        #     "You are an expert indoor navigation assistant for a mobile robot.\n"
        #     "You are given four synchronized camera views from the robot:\n"
        #     "Front view: <image>\n"
        #     "Left view: <image>\n"
        #     "Back view: <image>\n"
        #     "Right view: <image>\n\n"
        #     f"Instruction: {instr}\n\n"
        #     f"{output_requirement_1}\n"
        #     # "Move to the <MOVE_DIR> with a <RANGE_TAG> step.\n"
        #     f"{output_template}\n" if len(output_template) > 0 else ""
        #     # "Here <MOVE_DIR> and <RELATIVE_DIR> are discrete directions (e.g., Front, Front Right, Right, Back Right, Back, Back Left, Left, Front Left), and <RANGE_TAG> is either Small or Big.\n"
        #     f"{output_requirement_2}\n" if len(output_requirement_2) > 0 else ""
        # )
        prompt = (
            "You are an expert indoor navigation assistant for a mobile robot.\n"
            "You are given four synchronized camera views from the robot:\n"
            "Front view: <image>\n"
            "Left view: <image>\n"
            "Back view: <image>\n"
            "Right view: <image>\n\n"
            "Instruction: " + instr + "\n\n"
            + output_requirement_1 + "\n"
            + (output_template + "\n" if len(output_template) > 0 else "")
            + (output_requirement_2 + "\n" if len(output_requirement_2) > 0 else "")
        )

        return [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": meta_action},
        ]
