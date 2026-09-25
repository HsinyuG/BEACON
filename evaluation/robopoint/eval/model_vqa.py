import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from robopoint.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from robopoint.conversation import conv_templates, SeparatorStyle
from robopoint.model.builder import load_pretrained_model
from robopoint.utils import disable_torch_init
from robopoint.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    if args.load_4bit and args.load_8bit:
        raise ValueError("Choose only one quantization mode: --load-4bit or --load-8bit.")

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name,
        load_8bit=args.load_8bit, load_4bit=args.load_4bit
    )

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answer_file = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")
    for line in tqdm(questions):
        idx = line["question_id"]
        image_file = line["image"]
        qs = line["text"]
        if DEFAULT_IMAGE_TOKEN not in qs:
            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
        else:
            cur_prompt = qs.split('\n', 1)[1]

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        image = Image.open(os.path.join(args.image_folder, image_file)).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                image_sizes=[image.size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                # no_repeat_ngram_size=3,
                max_new_tokens=1024,
                use_cache=True)

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        # debug...
        debug_dir = os.path.dirname(answer_file) if os.path.dirname(answer_file) else "."
        os.makedirs(debug_dir, exist_ok=True)
        raw_debug_path = os.path.join(debug_dir, "debug_raw_input_image.jpg")
        proc_debug_path = os.path.join(debug_dir, "debug_processed_input_image.jpg")
        image.save(raw_debug_path)

        proc = image_tensor
        if proc.ndim == 4:
            # anyres path may return a stack of patches; visualize the first one.
            proc = proc[0]
        proc = proc.detach().float().cpu()

        if hasattr(image_processor, "image_mean") and hasattr(image_processor, "image_std"):
            mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
            std = torch.tensor(image_processor.image_std).view(3, 1, 1)
            proc = proc * std + mean
        else:
            pmin, pmax = proc.min(), proc.max()
            if pmax > pmin:
                proc = (proc - pmin) / (pmax - pmin)
        proc = proc.clamp(0, 1)
        proc_np = (proc.permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(proc_np).save(proc_debug_path)

        print("\n[DEBUG] prompt sent to model:\n")
        print(prompt)
        print("\n[DEBUG] decoded model output:\n")
        print(outputs)
        print(f"\n[DEBUG] saved raw image: {raw_debug_path}")
        print(f"[DEBUG] saved processed image: {proc_debug_path}")
        raise SystemExit(0)
        # end debug

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "category": line["category"]}) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="robopoint-v1-vicuna-v1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="question.jsonl")
    parser.add_argument("--answer-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()

    eval_model(args)
