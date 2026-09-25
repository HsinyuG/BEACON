#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Standalone shard exporter for BEACON zero-shot baselines.

This script was extracted verbatim (in logic) from the ``__main__`` block of
``xtuner/dataset/internvl_dataset.py``. It iterates over an
``InternVL_V1_5_Dataset_Multiview_PKL`` in eval mode and dumps per-sample
multiview observations (images, depth, camera matrices, goal, BEV masks) into
compressed ``.npz`` shards plus a ``manifest.json``.

Paths default to the original HPC container layout (data bind-mounted at
``/data``, model weights at ``/model``). Override with env vars for a local
machine, e.g.::

    export BEACON_DATA_ROOT=/your/data ; export BEACON_MODEL_ROOT=/your/models

Every hardcoded setting is exposed as an argparse argument whose default
matches the original container value, so running with no arguments reproduces
the original behaviour exactly.
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from xtuner.dataset import InternVL_V1_5_Dataset_Multiview_PKL
from xtuner.utils import PROMPT_TEMPLATE

# Paths below default to the original HPC container layout (data bind-mounted at
# /data, model weights at /model). Override with env vars for a local machine.
BEACON_DATA_ROOT = os.environ.get("BEACON_DATA_ROOT", "/data")
BEACON_MODEL_ROOT = os.environ.get("BEACON_MODEL_ROOT", "/model")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export multiview eval samples into npz shards for zero-shot baselines.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for consistency (default: 0).")
    parser.add_argument("--output-mode", type=str, default="Affordance_eval",
                        help="Dataset output_mode; must include eval (default: Affordance_eval).")
    parser.add_argument("--samples-per-shard", type=int, default=200,
                        help="Number of samples per npz shard (default: 200).")
    parser.add_argument("--max-shards", type=int, default=-1,
                        help="Max shards to export; -1 => export all (default: -1).")
    parser.add_argument("--model-path", type=str,
                        default=f"{BEACON_MODEL_ROOT}/OpenGVLab/InternVL2-2B",
                        help="Path to the base VLM (default: $BEACON_MODEL_ROOT/OpenGVLab/InternVL2-2B).")
    parser.add_argument("--pkl-path", type=str,
                        default=f"{BEACON_DATA_ROOT}/val_unseen.pkl",
                        help="Path to the split pkl (default: $BEACON_DATA_ROOT/val_unseen.pkl).")
    parser.add_argument("--pkl-dataset-root", type=str,
                        default=f"{BEACON_DATA_ROOT}/captures",
                        help="Dataset capture root (default: $BEACON_DATA_ROOT/captures).")
    parser.add_argument("--max-length", type=int, default=1536,
                        help="Dataset max_length (default: 1536).")
    parser.add_argument("--expected-total-len", type=int, default=11656,
                        help="Expected dataset length; the exporter asserts against this "
                             "(default: 11656). Override for a different split.")
    parser.add_argument("--out-root", type=str,
                        default=f"{BEACON_DATA_ROOT}/processed_val_unseen_for_baselines",
                        help="Root directory the sharded output subfolder is created under "
                             "(default: $BEACON_DATA_ROOT/processed_val_unseen_for_baselines).")
    return parser.parse_args()


def main():
    args = parse_args()

    seed = args.seed
    output_mode = args.output_mode  # must include eval
    samples_per_shard = args.samples_per_shard
    max_shards = args.max_shards  # -1 => export all

    path = args.model_path
    pkl_path = args.pkl_path
    pkl_dataset_root = args.pkl_dataset_root
    pkl_loader_kwargs = {
        "fov_deg": 90.0,
        "out_hw": (448, 448),
        "use_version": "dynamic",
        "use_dynamic": True,  # backward compatibility
        "verbose": False,
        "max_distance_horizontal": 6.4,
        "max_distance_vertical": 0.5,
    }
    prompt_template = PROMPT_TEMPLATE.internlm2_chat
    max_length = args.max_length

    # =========================
    # Seed for consistency
    # =========================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # =========================
    # Build dataset (same as eval)
    # =========================
    ds = InternVL_V1_5_Dataset_Multiview_PKL(
        model_path=path,
        template=prompt_template,
        pkl_path=pkl_path,
        dataset_root=pkl_dataset_root,
        max_length=max_length,
        loader_kwargs=pkl_loader_kwargs,
        output_mode=output_mode,
    )

    total_len = len(ds)  # should match test loop denominator (e.g., 11656)
    assert total_len == args.expected_total_len, (
        f"Unexpected dataset length {total_len}, expected {args.expected_total_len}; "
        f"check if pkl_path and loader_kwargs are correct for the target split")

    # Determine number of samples to export
    if max_shards == -1:
        target_samples = total_len
        shard_tag = "full"
    else:
        target_samples = min(total_len, max_shards * samples_per_shard)
        shard_tag = f"maxshards{max_shards}"

    out_dir = os.path.join(
        args.out_root,
        f"N{target_samples}_size{samples_per_shard}_{shard_tag}",
    )
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] dataset loaded with {total_len} samples, preparing to export max {target_samples} samples in {max_shards} shards (if max_shards=-1, export all in one shard)")

    keys = [
        "id", "instruction",
        "images", "depth", "cam2imgs", "cam2egos", "p_goal", "ego2global", "cam_height",
        "traversable_mask", "visible_mask", "affordance_mask",
        "raw_img",
    ]
    buf = {k: [] for k in keys}

    def _to_ordered_cam_mats(x):
        # x can be dict {"front","left","back","right"} or already ndarray/tensor [4,4,4]
        if isinstance(x, dict):
            order = ["front", "left", "back", "right"]
            return np.stack([np.asarray(x[k], dtype=np.float32) for k in order], axis=0)
        return np.asarray(x, dtype=np.float32)

    def flush_shard(shard_idx):
        np.savez_compressed(
            os.path.join(out_dir, f"shard_{shard_idx:03d}.npz"),
            id=np.asarray(buf["id"], dtype=object),
            instruction=np.asarray(buf["instruction"], dtype=object),

            images=np.asarray(buf["images"], dtype=np.uint8),              # [N,4,448,448,3], BGR
            depth=np.asarray(buf["depth"], dtype=np.float32),              # [N,4,448,448]
            cam2imgs=np.asarray(buf["cam2imgs"], dtype=np.float32),        # [N,4,4,4]
            cam2egos=np.asarray(buf["cam2egos"], dtype=np.float32),        # [N,4,4,4]
            p_goal=np.asarray(buf["p_goal"], dtype=np.float32),            # [N,3]
            ego2global=np.asarray(buf["ego2global"], dtype=np.float32),    # [N,4,4]
            cam_height=np.asarray(buf["cam_height"], dtype=np.float32),    # [N]

            traversable_mask=np.asarray(buf["traversable_mask"], dtype=bool),  # [N,128,128]
            visible_mask=np.asarray(buf["visible_mask"], dtype=bool),          # [N,128,128]
            affordance_mask=np.asarray(buf["affordance_mask"], dtype=bool),    # [N,128,128]

            # raw_img=np.asarray(buf["raw_img"], dtype=np.uint8),            # optional alias to images
        )
        for k in keys:
            buf[k].clear()
        print(f"[INFO] flushed shard {shard_idx:03d}, saved_samples={saved}")

    # Resume-like behavior: skip contiguous shards that already exist.
    existing_shards = []
    saved = 0
    shard_idx = 0
    while True:
        shard_path = os.path.join(out_dir, f"shard_{shard_idx:03d}.npz")
        if not os.path.exists(shard_path):
            break
        with np.load(shard_path, allow_pickle=True) as z:
            n_in_shard = len(z["id"])
        existing_shards.append((shard_idx, n_in_shard))
        saved += n_in_shard
        shard_idx += 1
        if saved >= target_samples:
            break

    # Clamp in case pre-existing shards exceed current target setting.
    saved = min(saved, target_samples)
    if existing_shards:
        first_idx = existing_shards[0][0]
        last_idx = existing_shards[-1][0]
        print(
            f"[INFO] skipping existing shards {first_idx:03d}..{last_idx:03d} "
            f"(samples={saved}) in {out_dir}"
        )
    else:
        print(f"[INFO] no existing shards to skip in {out_dir}")
    print(f"[INFO] start export from sample index {saved}, shard index {shard_idx}")

    # Use ds[i] (as requested) so counting behavior matches real test pipeline behavior
    for i in tqdm(range(saved, target_samples), desc="Exporting shards"):
        if saved >= target_samples:
            break

        sample = ds[i]  # intentionally ds[i], not prepare_data(i)

        # id + instruction
        sample_id = sample.get("token", f"idx_{i:06d}")
        instruction = sample.get("instruction", "")

        # images / raw_img expected in this dataset
        images = np.asarray(sample["raw_img"], dtype=np.uint8)  # [4,448,448,3], BGR

        # depth
        depth = np.asarray(sample["depth"], dtype=np.float32)

        # cam mats
        cam2imgs = _to_ordered_cam_mats(sample["cam2imgs"])
        cam2egos = _to_ordered_cam_mats(sample["cam2egos"])

        # labels / meta
        p_goal = np.asarray(sample["p_goal"], dtype=np.float32)
        ego2global = np.asarray(sample["ego2global"], dtype=np.float32)
        cam_height = np.float32(sample["cam_height"])

        traversable_mask = np.asarray(sample["traversable_mask"], dtype=bool)
        visible_mask = np.asarray(sample["visible_mask"], dtype=bool)
        affordance_mask = np.asarray(sample["affordance_mask"], dtype=bool)

        # append
        buf["id"].append(sample_id)
        buf["instruction"].append(instruction)

        buf["images"].append(images)
        buf["depth"].append(depth)
        buf["cam2imgs"].append(cam2imgs)
        buf["cam2egos"].append(cam2egos)
        buf["p_goal"].append(p_goal)
        buf["ego2global"].append(ego2global)
        buf["cam_height"].append(cam_height)

        buf["traversable_mask"].append(traversable_mask)
        buf["visible_mask"].append(visible_mask)
        buf["affordance_mask"].append(affordance_mask)

        buf["raw_img"].append(images)

        saved += 1

        if len(buf["id"]) == samples_per_shard:
            flush_shard(shard_idx)
            shard_idx += 1
            if max_shards != -1 and shard_idx >= max_shards:
                break

    # final partial shard
    if len(buf["id"]) > 0 and (max_shards == -1 or shard_idx < max_shards):
        flush_shard(shard_idx)
        shard_idx += 1

    manifest = {
        "seed": seed,
        "output_mode": output_mode,
        "dataset_len": total_len,
        "saved_samples": saved,
        "samples_per_shard": samples_per_shard,
        "saved_shards": shard_idx,
        "max_shards": max_shards,
        "pkl_path": pkl_path,
        "dataset_root": pkl_dataset_root,
        "note": "Load with np.load(..., allow_pickle=True) for object arrays id/instruction.",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[DONE] saved_samples={saved}, saved_shards={shard_idx}, out_dir={out_dir}")


if __name__ == "__main__":
    main()
