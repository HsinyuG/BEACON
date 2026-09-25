import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baseline_adaptors.sharded_npz_loader import ShardedNPZLoader
from robopoint.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from robopoint.conversation import conv_templates
from robopoint.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from robopoint.model.builder import load_pretrained_model
from robopoint.utils import disable_torch_init


def select_oracle_view_from_goal(p_goal: np.ndarray) -> Tuple[str, float]:
    x_fwd = float(p_goal[0])
    y_left = float(p_goal[1])
    yaw_deg = float(np.degrees(np.arctan2(y_left, x_fwd)))
    if -45.0 <= yaw_deg < 45.0:
        return "front", yaw_deg
    if 45.0 <= yaw_deg < 135.0:
        return "left", yaw_deg
    if yaw_deg >= 135.0 or yaw_deg < -135.0:
        return "backward", yaw_deg
    return "right", yaw_deg


def build_robopoint_question(instruction: str, view_key: str) -> str:
    view_text = {
        "front": "front-facing",
        "left": "left-facing",
        "backward": "backward-facing",
        "right": "right-facing",
    }[view_key]
    return (
        "<image>\n"
        f"You are given a single {view_text} view image from a 4-view panorama "
        f"(front, left, backward, right). This image is the selected view that is most likely "
        f"relevant to the instruction.\n"
        # "Locate a few points in the free space that best follow the instruction:\n"
        # "Locate a few points that best follow the instruction:\n" # NOTE: tuned, if including "free space", the model performs way worse even in invalid rate. And put this after the view description gives better performance than putting it before.
        "Locate a few points on the floor that best follow the instruction:\n" # best wording found so far
        f"{instruction}\n"
        "If the target is not clearly visible, return your best guess in this view.\n"
        "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], "
        "where each tuple contains the x and y coordinates of a point satisfying the conditions "
        "above. The coordinates should be between 0 and 1, indicating the normalized pixel "
        "locations of the points in the image."
    )


def ensure_pil_rgb(img: np.ndarray) -> Image.Image:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected image shape (..., 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # Shards store BGR; convert to RGB for PIL/model.
    arr = arr[..., ::-1]
    return Image.fromarray(arr, mode="RGB")


def parse_predicted_points(text: str) -> List[Tuple[float, float]]:
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points: List[Tuple[float, float]] = []
    for match in matches:
        vals = [float(v.strip()) for v in match.split(",")]
        if len(vals) == 2:
            points.append((vals[0], vals[1]))
        elif len(vals) == 4:
            x0, y0, x1, y1 = vals
            points.append(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
    return points


def xy_to_pixel(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        u = int(round(x * (w - 1)))
        v = int(round(y * (h - 1)))
    else:
        u = int(round(x))
        v = int(round(y))
    return int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))


def project_uv_to_ego(sample: Dict, view_key: str, u: int, v: int) -> Optional[np.ndarray]:
    depth = np.asarray(sample["depths"][view_key], dtype=np.float32)
    h, w = depth.shape
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))

    d = float(depth[v, u])
    if (not np.isfinite(d)) or d <= 1e-6:
        return None

    cam2ego = np.asarray(sample["cam2egos"][view_key], dtype=np.float32)
    cam2img = np.asarray(sample["cam2imgs"][view_key], dtype=np.float32)[:3, :3]
    try:
        inv_k = np.linalg.inv(cam2img)
    except np.linalg.LinAlgError:
        return None

    pixel = np.array([u, v, 1.0], dtype=np.float32) * d
    cam_xyz = inv_k @ pixel
    ego_h = cam2ego @ np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], 1.0], dtype=np.float32)
    if abs(float(ego_h[3])) < 1e-8:
        return None
    ego = ego_h[:3] / ego_h[3]
    if not np.all(np.isfinite(ego)):
        return None
    return ego.astype(np.float32)


class RoboPointRunner:
    def __init__(
        self,
        model_path: str,
        model_base: Optional[str],
        conv_mode: str,
        temperature: float,
        top_p: Optional[float],
        num_beams: int,
        max_new_tokens: int,
        load_4bit: bool = True,
    ):
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.temperature = float(temperature)
        self.top_p = top_p
        self.num_beams = int(num_beams)
        self.max_new_tokens = int(max_new_tokens)

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_path,
            self.model_base,
            model_name,
            load_4bit=load_4bit,
        )

    def infer_one(self, img_pil: Image.Image, question: str) -> Tuple[str, str]:
        qs = question
        if DEFAULT_IMAGE_TOKEN not in qs:
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).cuda()
        image_tensor = process_images([img_pil], self.image_processor, self.model.config)[0]

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[img_pil.size],
                do_sample=True if self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
                num_beams=self.num_beams,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return prompt, outputs


class BaselineRoboPoint:
    def __init__(
        self,
        data_root: str,
        output_dir: str,
        shard_pattern: str,
        model_path: str,
        model_base: Optional[str] = None,
        conv_mode: str = "llava_v1",
        max_samples: int = -1,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        max_new_tokens: int = 1024,
        overwrite: bool = False,
        verbose: bool = False,
    ):
        self.data_root = data_root
        self.output_dir = output_dir
        self.shard_pattern = shard_pattern
        self.max_samples = int(max_samples)
        self.overwrite = overwrite
        self.verbose = verbose

        self.loader = ShardedNPZLoader(data_root=self.data_root, pattern=self.shard_pattern, verbose=False)
        self.runner = RoboPointRunner(
            model_path=model_path,
            model_base=model_base,
            conv_mode=conv_mode,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            load_4bit=True,
        )

    def _candidate_output_path(self, shard_path: str) -> str:
        base = os.path.splitext(os.path.basename(shard_path))[0]
        return os.path.join(self.output_dir, f"{base}.robopoint_candidates.npz")

    @staticmethod
    def _fallback_xyz(cam_height: float) -> np.ndarray:
        return np.array([0.0, 0.0, -float(cam_height)], dtype=np.float32)

    def run(self):
        os.makedirs(self.output_dir, exist_ok=True)
        total_done = 0
        stop_all = False

        for shard_idx, shard_path in enumerate(self.loader.shards):
            out_path = self._candidate_output_path(shard_path)
            if os.path.exists(out_path) and not self.overwrite:
                print(f"[SKIP] exists: {out_path}")
                continue

            shard_data = ShardedNPZLoader._load_shard_into_ram(shard_path)
            n = int(shard_data["images"].shape[0])
            print(f"[INFO] processing shard {shard_idx+1}/{len(self.loader.shards)}: {os.path.basename(shard_path)} (N={n})")

            tokens: List[str] = []
            scenes: List[str] = []
            instructions: List[str] = []
            oracle_views: List[str] = []
            oracle_yaws_deg: List[float] = []
            cam_heights: List[float] = []
            p_goals: List[np.ndarray] = []
            ego2globals: List[np.ndarray] = []
            traversable_masks: List[np.ndarray] = []
            prompts: List[str] = []
            raw_outputs: List[str] = []
            statuses: List[str] = []
            parsed_points_norm: List[np.ndarray] = []
            candidate_uvs: List[np.ndarray] = []
            candidate_valids: List[np.ndarray] = []
            candidate_xyzs: List[np.ndarray] = []

            pbar = tqdm(range(n), desc=f"{os.path.basename(shard_path)}", unit="sample")
            for i in pbar:
                if self.max_samples > -1 and total_done >= self.max_samples:
                    stop_all = True
                    break

                sample = self.loader._make_sample(shard_data, i, total_done)
                tok = str(sample["token"])
                goal = np.asarray(sample["p_goal"], dtype=np.float32)
                cam_h = float(sample["cam_height"])
                view_key, yaw_deg = select_oracle_view_from_goal(goal)
                img_pil = ensure_pil_rgb(sample["images"][view_key])
                question = build_robopoint_question(str(sample["instruction"]), view_key)

                prompt_text = ""
                output_text = ""
                status = "ok"
                pts = []
                uvs: List[Tuple[int, int]] = []
                valids: List[bool] = []
                xyzs: List[np.ndarray] = []

                try:
                    prompt_text, output_text = self.runner.infer_one(img_pil, question)
                    pts = parse_predicted_points(output_text)
                    for x, y in pts:
                        u, v = xy_to_pixel(x, y, img_pil.width, img_pil.height)
                        uvs.append((u, v))
                        ego_xyz = project_uv_to_ego(sample, view_key, u, v)
                        if ego_xyz is None:
                            valids.append(False)
                            xyzs.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
                        else:
                            valids.append(True)
                            xyzs.append(ego_xyz.astype(np.float32))
                except Exception as e:
                    status = f"error:{type(e).__name__}"
                    if self.verbose:
                        print(f"[WARN] token={tok} inference error: {e}", flush=True)

                # # debug...
                # debug_dir = self.output_dir
                # os.makedirs(debug_dir, exist_ok=True)
                # debug_img_path = os.path.join(debug_dir, "debug_baseline_robopoint_input.jpg")
                # img_pil.save(debug_img_path)
                # print("\n[DEBUG] token:", tok)
                # print("[DEBUG] oracle view:", view_key, f"(yaw_deg={yaw_deg:.2f})")
                # print("[DEBUG] image saved:", debug_img_path)
                # print("[DEBUG] prompt sent to model:\n")
                # print(prompt_text if prompt_text else "<empty prompt>")
                # print("\n[DEBUG] model raw output:\n")
                # print(output_text if output_text else "<empty output>")
                # print("\n[DEBUG] parsed points:", pts)
                # print("[DEBUG] candidate_uvs:", uvs)
                # print("[DEBUG] candidate_valids:", valids)
                # print("[DEBUG] candidate_xyzs:", [x.tolist() for x in xyzs])
                # raise SystemExit(0)
                # # end of debug

                tokens.append(tok)
                scenes.append(str(sample.get("scene", "unknown_scene")))
                instructions.append(str(sample["instruction"]))
                oracle_views.append(view_key)
                oracle_yaws_deg.append(float(yaw_deg))
                cam_heights.append(cam_h)
                p_goals.append(goal.astype(np.float32))
                ego2globals.append(np.asarray(sample.get("ego2global", np.eye(4)), dtype=np.float32))
                traversable_masks.append(np.asarray(sample["traversable_mask"]).astype(bool))
                prompts.append(prompt_text)
                raw_outputs.append(output_text)
                statuses.append(status)
                parsed_points_norm.append(np.asarray(pts, dtype=np.float32).reshape(-1, 2))
                candidate_uvs.append(np.asarray(uvs, dtype=np.int32).reshape(-1, 2))
                candidate_valids.append(np.asarray(valids, dtype=bool))
                if len(xyzs) == 0:
                    candidate_xyzs.append(np.zeros((0, 3), dtype=np.float32))
                else:
                    candidate_xyzs.append(np.asarray(xyzs, dtype=np.float32).reshape(-1, 3))

                total_done += 1
                if self.verbose:
                    pbar.set_postfix({"view": view_key, "yaw": f"{yaw_deg:.1f}", "npts": len(pts), "status": status})

            n_saved = len(tokens)
            out = {
                "source_shard": np.asarray([os.path.basename(shard_path)] * n_saved, dtype=object),
                "token": np.asarray(tokens, dtype=object),
                "scene": np.asarray(scenes, dtype=object),
                "instruction": np.asarray(instructions, dtype=object),
                "oracle_view": np.asarray(oracle_views, dtype=object),
                "oracle_yaw_deg": np.asarray(oracle_yaws_deg, dtype=np.float32),
                "cam_height": np.asarray(cam_heights, dtype=np.float32),
                "p_goal": np.asarray(p_goals, dtype=np.float32) if n_saved else np.zeros((0, 3), dtype=np.float32),
                "ego2global_xfwd": np.asarray(ego2globals, dtype=np.float32) if n_saved else np.zeros((0, 4, 4), dtype=np.float32),
                "traversable_mask": np.asarray(traversable_masks, dtype=bool) if n_saved else np.zeros((0, 1, 1), dtype=bool),
                "prompt": np.asarray(prompts, dtype=object),
                "raw_output_text": np.asarray(raw_outputs, dtype=object),
                "status": np.asarray(statuses, dtype=object),
                "parsed_points_norm": np.asarray(parsed_points_norm, dtype=object),
                "candidate_uv": np.asarray(candidate_uvs, dtype=object),
                "candidate_valid": np.asarray(candidate_valids, dtype=object),
                "candidate_xyz": np.asarray(candidate_xyzs, dtype=object),
            }
            with open(out_path, "wb") as f:
                np.savez(f, **out)
            print(f"[SAVE] {out_path} (samples={n_saved})")

            del shard_data
            if stop_all:
                break


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--shard-pattern", type=str, default="shard_*.npz")
    parser.add_argument("--model-path", type=str, default="wentao-yuan/robopoint-v1-vicuna-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = BaselineRoboPoint(
        data_root=args.data_root,
        output_dir=args.output_dir,
        shard_pattern=args.shard_pattern,
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        max_samples=args.max_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    runner.run()
