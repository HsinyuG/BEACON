import argparse
import heapq
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

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
from robopoint.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from robopoint.model.builder import load_pretrained_model
from robopoint.utils import disable_torch_init


def select_oracle_view_from_goal(p_goal: np.ndarray) -> Tuple[str, float]:
    """Pick one of front/left/backward/right from goal direction in ego frame."""
    x_fwd = float(p_goal[0])
    y_left = float(p_goal[1])
    yaw_deg = math.degrees(math.atan2(y_left, x_fwd))
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
        f"<image>\nYou are given a single {view_text} view image from a 4-view panorama "
        f"(front, left, backward, right). This image is the selected view that is most likely "
        f"relevant to the instruction.\n"
        f"Instruction: {instruction}\n\n"
        "Predict several candidate target points in this image that best follow the instruction. "
        "If the target is not clearly visible, return your best guess in this view.\n"
        "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], "
        "where each tuple contains x and y coordinates. The coordinates should be normalized to "
        "the range [0, 1]."
    )


def ensure_pil_rgb(img: np.ndarray) -> Image.Image:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected image shape (..., 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # Dataset shards store color channels in BGR order; convert to RGB for PIL/model.
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
    # If values are normalized, map to pixel coordinates.
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        u = int(round(x * (w - 1)))
        v = int(round(y * (h - 1)))
    else:
        u = int(round(x))
        v = int(round(y))
    u = int(np.clip(u, 0, w - 1))
    v = int(np.clip(v, 0, h - 1))
    return u, v


def project_uv_to_ego(
    sample: Dict, view_key: str, u: int, v: int
) -> Optional[np.ndarray]:
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


def bounded_geodesic_dist_cells(
    traversable: np.ndarray, si: int, sj: int, rmax_cells: float
) -> np.ndarray:
    h, w = traversable.shape
    inf = 1e9
    dist = np.full((h, w), inf, dtype=np.float32)
    visited = np.zeros((h, w), dtype=bool)

    dist[si, sj] = 0.0
    pq: List[Tuple[float, int, int]] = [(0.0, si, sj)]
    sq2 = math.sqrt(2.0)
    moves = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, sq2),
        (-1, 1, sq2),
        (1, -1, sq2),
        (1, 1, sq2),
    ]

    while pq:
        d, y, x = heapq.heappop(pq)
        if visited[y, x]:
            continue
        visited[y, x] = True
        if d > rmax_cells:
            break

        for dy, dx, step in moves:
            ny = y + dy
            nx = x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if not traversable[ny, nx]:
                continue
            if dy != 0 and dx != 0:
                if not (traversable[y, nx] and traversable[ny, x]):
                    continue
            nd = d + step
            if nd <= rmax_cells and nd < float(dist[ny, nx]):
                dist[ny, nx] = nd
                heapq.heappush(pq, (nd, ny, nx))
    return dist


def build_goal_dist_map(
    traversable: np.ndarray,
    p_goal: np.ndarray,
    res: float = 0.1,
    half_xy: float = 6.4,
    rmax_m: float = 1.5,
) -> np.ndarray:
    h, w = traversable.shape
    inf = 1e9
    if not traversable.any():
        return np.full((h, w), inf, dtype=np.float32)

    seed_i = int(round((float(p_goal[0]) + half_xy - 0.5 * res) / res))
    seed_j = int(round((float(p_goal[1]) + half_xy - 0.5 * res) / res))
    seed_i = max(0, min(h - 1, seed_i))
    seed_j = max(0, min(w - 1, seed_j))

    if not traversable[seed_i, seed_j]:
        ys, xs = np.where(traversable)
        d2 = (ys.astype(np.float32) - float(seed_i)) ** 2 + (xs.astype(np.float32) - float(seed_j)) ** 2
        arg = int(np.argmin(d2))
        seed_i = int(ys[arg])
        seed_j = int(xs[arg])

    rmax_cells = float(rmax_m / res)
    return bounded_geodesic_dist_cells(traversable, seed_i, seed_j, rmax_cells)


def geohit_score_for_xy(
    pred_xy: np.ndarray,
    traversable: np.ndarray,
    dist_map: np.ndarray,
    res: float = 0.1,
    half_xy: float = 6.4,
) -> Tuple[float, int, int, int, float, bool]:
    h, w = traversable.shape
    gx = int(round((float(pred_xy[0]) + half_xy - 0.5 * res) / res))
    gy = int(round((float(pred_xy[1]) + half_xy - 0.5 * res) / res))
    out_of_bounds = gx < 0 or gx >= h or gy < 0 or gy >= w
    if out_of_bounds:
        return 0.0, 0, 0, 0, float("inf"), True
    if not bool(traversable[gx, gy]):
        return 0.0, 0, 0, 0, float("inf"), True

    d = float(dist_map[gx, gy])
    h05 = int(d <= (0.5 / res))
    h10 = int(d <= (1.0 / res))
    h15 = int(d <= (1.5 / res))
    mean_hit = (h05 + h10 + h15) / 3.0
    return mean_hit, h05, h10, h15, d, False


def fallback_prediction(cam_height: float) -> np.ndarray:
    return np.array([0.0, 0.0, -float(cam_height)], dtype=np.float32)


class BaselineRoboPointOracleBestGeoHit:
    def __init__(
        self,
        data_root: str,
        output_path: str,
        model_path: str,
        model_base: Optional[str] = None,
        conv_mode: str = "llava_v1",
        max_samples: int = -1,
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        num_beams: int = 1,
        max_new_tokens: int = 1024,
        verbose: bool = False,
        res: float = 0.1,
        half_xy: float = 6.4,
    ):
        self.loader = ShardedNPZLoader(data_root=data_root, pattern="shard_*.npz", verbose=False)
        self.output_path = output_path
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.max_samples = int(max_samples)
        self.temperature = float(temperature)
        self.top_p = top_p
        self.num_beams = int(num_beams)
        self.max_new_tokens = int(max_new_tokens)
        self.verbose = verbose
        self.res = float(res)
        self.half_xy = float(half_xy)

        disable_torch_init()
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            self.model_path,
            self.model_base,
            model_name,
            load_4bit=True,
        )

    def infer_on_single_image(self, img_pil: Image.Image, question: str) -> Tuple[str, str]:
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
        output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return prompt, output_text

    def run(self):
        p_pred: List[np.ndarray] = []
        p_goal: List[np.ndarray] = []
        ego2global_xfwd: List[np.ndarray] = []
        cam_height: List[float] = []
        scene_token: List[str] = []
        token: List[str] = []
        oracle_view: List[str] = []

        pbar = tqdm(self.loader, desc="RoboPoint OracleGeoHit", unit="sample")
        for sample in pbar:
            if self.max_samples > -1 and len(p_pred) >= self.max_samples:
                break

            tok = str(sample.get("token", f"sample_{len(p_pred):06d}"))
            goal = np.asarray(sample["p_goal"], dtype=np.float32)
            view_key, yaw_deg = select_oracle_view_from_goal(goal)
            oracle_view.append(view_key)

            cam_h = float(sample.get("cam_height", 1.5))
            pred = fallback_prediction(cam_h)
            status = "fallback"
            dbg_prompt = ""
            dbg_output_text = ""
            dbg_points: List[Tuple[float, float]] = []
            dbg_rank_rows: List[Dict] = []
            dbg_selected = None

            try:
                img_pil = ensure_pil_rgb(sample["images"][view_key])
                question = build_robopoint_question(str(sample["instruction"]), view_key)
                prompt_text, output_text = self.infer_on_single_image(img_pil, question)
                dbg_prompt = prompt_text
                dbg_output_text = output_text
                points = parse_predicted_points(output_text)
                dbg_points = points

                projected_candidates: List[np.ndarray] = []
                for x, y in points:
                    u, v = xy_to_pixel(x, y, img_pil.width, img_pil.height)
                    ego_xyz = project_uv_to_ego(sample, view_key, u, v)
                    if ego_xyz is not None:
                        projected_candidates.append(ego_xyz)

                if projected_candidates:
                    traversable_mask = np.asarray(sample["traversable_mask"]).astype(bool)
                    dist_map = build_goal_dist_map(
                        traversable_mask,
                        goal,
                        res=self.res,
                        half_xy=self.half_xy,
                        rmax_m=1.5,
                    )

                    best_key = None
                    best_pred = None
                    for cand in projected_candidates:
                        mean_hit, h05, h10, h15, d_cells, invalid = geohit_score_for_xy(
                            cand[:2], traversable_mask, dist_map, res=self.res, half_xy=self.half_xy
                        )
                        euclid = float(np.linalg.norm(cand[:2] - goal[:2]))
                        # Maximize GeoHit mean first, then tie-break by tighter hits and shorter dists.
                        rank_key = (
                            mean_hit,
                            h05,
                            h10,
                            h15,
                            -d_cells if np.isfinite(d_cells) else -1e9,
                            -euclid,
                            0 if not invalid else -1,
                        )
                        dbg_rank_rows.append(
                            {
                                "cand_xy": [float(cand[0]), float(cand[1])],
                                "mean_hit": float(mean_hit),
                                "hit05": int(h05),
                                "hit10": int(h10),
                                "hit15": int(h15),
                                "geo_cells": float(d_cells),
                                "euclid_m": float(euclid),
                                "invalid": bool(invalid),
                                "rank_key": rank_key,
                            }
                        )
                        if best_key is None or rank_key > best_key:
                            best_key = rank_key
                            best_pred = cand
                            dbg_selected = dbg_rank_rows[-1]

                    if best_pred is not None:
                        pred = best_pred.astype(np.float32)
                        status = "ok"
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] token={tok} fallback due to error: {e}", flush=True)

            # # debug...
            # debug_dir = os.path.dirname(self.output_path) or "."
            # os.makedirs(debug_dir, exist_ok=True)
            # debug_img_path = os.path.join(debug_dir, "debug_robopoint_selected_view.jpg")
            # ensure_pil_rgb(sample["images"][view_key]).save(debug_img_path)

            # print("\n[DEBUG] token:", tok)
            # print("[DEBUG] instruction:", sample["instruction"])
            # print("[DEBUG] oracle view:", view_key, f"(yaw_deg={yaw_deg:.2f})")
            # print("[DEBUG] image saved:", debug_img_path)
            # print("[DEBUG] prompt sent to model:\n")
            # print(dbg_prompt if dbg_prompt else "<empty prompt>")
            # print("\n[DEBUG] model raw output:\n")
            # print(dbg_output_text if dbg_output_text else "<empty output>")
            # print("\n[DEBUG] parsed points:", dbg_points)
            # print("[DEBUG] candidate geohit ranking rows:")
            # for irow, row in enumerate(dbg_rank_rows):
            #     print(f"  - cand#{irow}: {row}")
            # print("[DEBUG] selected candidate row:", dbg_selected)
            # print("[DEBUG] final pred xyz:", pred.tolist(), "| status:", status)
            # raise SystemExit(0)
            # # end of debug

            p_pred.append(pred.astype(np.float32))
            p_goal.append(goal.astype(np.float32))
            ego2global_xfwd.append(np.asarray(sample.get("ego2global", np.eye(4)), dtype=np.float32))
            cam_height.append(cam_h)
            scene_token.append(str(sample.get("scene", "unknown_scene")))
            token.append(tok)

            if self.verbose:
                pbar.set_postfix({"status": status, "view": view_key, "yaw_deg": f"{yaw_deg:.1f}"})

        out = {
            "p_pred": np.asarray(p_pred, dtype=np.float32),
            "p_goal": np.asarray(p_goal, dtype=np.float32),
            "ego2global_xfwd": np.asarray(ego2global_xfwd, dtype=np.float32),
            "cam_height": np.asarray(cam_height, dtype=np.float32),
            "scene_token": np.asarray(scene_token),
            "token": np.asarray(token),
            "oracle_view": np.asarray(oracle_view),
        }
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "wb") as f:
            np.savez(f, **out)
        print(f"Saved {len(p_pred)} samples to {self.output_path}")


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument(
        "--model-path",
        type=str,
        default="wentao-yuan/robopoint-v1-vicuna-v1.5-13b",
    )
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--res", type=float, default=0.1)
    parser.add_argument("--half-xy", type=float, default=6.4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = build_args()
    runner = BaselineRoboPointOracleBestGeoHit(
        data_root=args.data_root,
        output_path=args.output_path,
        model_path=args.model_path,
        model_base=args.model_base,
        conv_mode=args.conv_mode,
        max_samples=args.max_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose,
        res=args.res,
        half_xy=args.half_xy,
    )
    runner.run()
