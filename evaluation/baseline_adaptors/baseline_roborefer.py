import argparse
import base64
import io
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baseline_adaptors.sharded_npz_loader import ShardedNPZLoader


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


def ensure_pil_rgb(img: np.ndarray) -> Image.Image:
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[-1] != 3:
        raise ValueError(f"Expected image shape (..., 3), got {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    # Shards store BGR; convert to RGB for PIL/RoboRefer visual input.
    arr = arr[..., ::-1]
    return Image.fromarray(arr, mode="RGB")


def pil_to_base64(img: Image.Image, format_: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format_)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def parse_predicted_points(text: str) -> List[Tuple[float, float]]:
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points: List[Tuple[float, float]] = []
    for match in matches:
        try:
            vals = [float(v.strip()) for v in match.split(",")]
        except Exception:
            continue
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


def build_roborefer_prompt(instruction: str, oracle_view: str) -> str:
    suffix = (
        "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], "
        "where each tuple contains the x and y coordinates of a point satisfying the conditions "
        "above. The coordinates should be between 0 and 1, indicating the normalized pixel "
        "locations of the points in the image."
    )

    view_prefix = (
        f"You are given a single {oracle_view}-facing view image from a 4-view panorama "
        "(front, left, backward, right). This image is the selected view that is most likely "
        "relevant to the instruction."
    )

    # Candidate A (default / most confident): RoboPoint-style wording + explicit floor-only bias.
    # Avoids adding the phrase "free space" in the instruction wrapper (see baseline_robopoint.py note).
    # prompt = (
    #     f"{view_prefix}\n"
    #     "For mobile-base navigation, interpret the target as a reachable point on the floor "
    #     "(not on tables, shelves, or other elevated surfaces).\n"
    #     "Locate a few points on the floor that best follow the instruction:\n"
    #     f"{instruction}\n"
    #     "If the target is not clearly visible, return your best guess in this view.\n"
    #     "Return only the tuple list and no extra explanation.\n"
    #     f"{suffix}"
    # )

    # Candidate B (closer to RoboRefer benchmark prompt style, still floor-biased):
    prompt = (
        f"{view_prefix}\n"
        "Please point out one or more valid points on the floor that best follow the instruction:\n"
        f"{instruction}\n"
        "If the target is not clearly visible, return your best guess in this view.\n"
        # "Do not choose points on furniture surfaces; choose floor locations only.\n" # Adding this differs from others and also harms the performance
        # "Return only the tuple list and no extra explanation.\n" # no need to add this, it already does so.
        f"{suffix}"
    )

    # Candidate C (more concise / stronger output-format control):
    # prompt = (
    #     f"{view_prefix}\n"
    #     "Task: select a few candidate floor points for mobile navigation that best follow the instruction below.\n"
    #     f"{instruction}\n"
    #     "Floor points only. If uncertain, provide your best guess in this view.\n"
    #     f"{suffix}\n"
    #     "Output only a Python-style list of tuples."
    # )

    return prompt


class RoboReferServerRunner:
    def __init__(
        self,
        server_url: str = "http://127.0.0.1:25547",
        enable_depth: bool = True,
        retry: int = 3,
        request_timeout: float = 120.0,
        image_format: str = "PNG",
        verbose: bool = False,
    ):
        self.server_url = server_url.rstrip("/")
        self.enable_depth = bool(enable_depth)
        self.retry = int(retry)
        self.request_timeout = float(request_timeout)
        self.image_format = image_format
        self.verbose = verbose

    def infer(self, images: List[Image.Image], prompt: str) -> str:
        payload = {
            "image_url": [pil_to_base64(img, format_=self.image_format) for img in images],
            "depth_url": [],
            "enable_depth": int(self.enable_depth),
            "text": prompt,
        }

        for attempt in range(1, self.retry + 1):
            try:
                response = requests.post(
                    self.server_url + "/query",
                    json=payload,
                    timeout=self.request_timeout,
                )
                if response.status_code != 200:
                    print(
                        f"[WARN] RoboRefer server returned status {response.status_code} "
                        f"on attempt {attempt}/{self.retry}"
                    )
                    time.sleep(1)
                    continue
                data = response.json()
                if self.verbose:
                    print(data)
                if not isinstance(data, dict) or "answer" not in data:
                    raise ValueError(f"Unexpected response JSON: {data}")
                return str(data.get("answer", ""))
            except Exception as e:
                print(f"[WARN] RoboRefer query failed attempt {attempt}/{self.retry}: {e}")
                time.sleep(1)
        return "Failed: Query Error"


class BaselineRoboRefer:
    def __init__(
        self,
        data_root: str,
        output_dir: str,
        shard_pattern: str = "shard_*.npz",
        server_url: str = "http://127.0.0.1:25547",
        enable_depth: bool = True,
        max_samples: int = -1,
        retry: int = 3,
        request_timeout: float = 120.0,
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
        self.runner = RoboReferServerRunner(
            server_url=server_url,
            enable_depth=enable_depth,
            retry=retry,
            request_timeout=request_timeout,
            verbose=verbose,
        )

    def _candidate_output_path(self, shard_path: str) -> str:
        base = os.path.splitext(os.path.basename(shard_path))[0]
        return os.path.join(self.output_dir, f"{base}.roborefer_candidates.npz")

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
            selected_views_for_projection: List[str] = []
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
                oracle_view, yaw_deg = select_oracle_view_from_goal(goal)

                prompt = build_roborefer_prompt(str(sample["instruction"]), oracle_view=oracle_view)
                image_order_keys = [oracle_view]
                images_pil = [ensure_pil_rgb(sample["images"][k]) for k in image_order_keys]

                raw_output = ""
                status = "ok"
                pts: List[Tuple[float, float]] = []
                project_view = oracle_view
                uvs: List[Tuple[int, int]] = []
                valids: List[bool] = []
                xyzs: List[np.ndarray] = []

                try:
                    raw_output = self.runner.infer(images_pil, prompt)
                    pts = parse_predicted_points(raw_output)

                    img_for_proj = ensure_pil_rgb(sample["images"][project_view])
                    for x, y in pts:
                        u, v = xy_to_pixel(x, y, img_for_proj.width, img_for_proj.height)
                        uvs.append((u, v))
                        ego_xyz = project_uv_to_ego(sample, project_view, u, v)
                        if ego_xyz is None:
                            valids.append(False)
                            xyzs.append(np.array([np.nan, np.nan, np.nan], dtype=np.float32))
                        else:
                            valids.append(True)
                            xyzs.append(ego_xyz.astype(np.float32))
                except Exception as e:
                    status = f"error:{type(e).__name__}"
                    if self.verbose:
                        print(f"[WARN] token={tok} RoboRefer error: {e}", flush=True)

                # # debug...
                # debug_dir = self.output_dir
                # os.makedirs(debug_dir, exist_ok=True)
                # for di, (k, imgp) in enumerate(zip(image_order_keys, images_pil)):
                #     imgp.save(os.path.join(debug_dir, f"debug_baseline_roborefer_input_{di:02d}_{k}.jpg"))
                # print("\n[DEBUG] token:", tok)
                # print("[DEBUG] oracle_view:", oracle_view, f"(yaw_deg={yaw_deg:.2f})")
                # print("[DEBUG] prompt sent to RoboRefer:\n")
                # print(prompt)
                # print("\n[DEBUG] raw RoboRefer output:\n")
                # print(raw_output)
                # print("[DEBUG] parsed_points:", pts)
                # print("[DEBUG] candidate_uvs:", uvs)
                # print("[DEBUG] candidate_valids:", valids)
                # print("[DEBUG] candidate_xyzs:", [x.tolist() for x in xyzs])
                # print(f"[DEBUG] saved img to {debug_dir}/debug_baseline_roborefer_input_XX_{{view}}.jpg")
                # raise SystemExit(0)
                # # end of debug

                tokens.append(tok)
                scenes.append(str(sample.get("scene", "unknown_scene")))
                instructions.append(str(sample["instruction"]))
                oracle_views.append(oracle_view)
                selected_views_for_projection.append(project_view)
                oracle_yaws_deg.append(float(yaw_deg))
                cam_heights.append(cam_h)
                p_goals.append(goal.astype(np.float32))
                ego2globals.append(np.asarray(sample.get("ego2global", np.eye(4)), dtype=np.float32))
                traversable_masks.append(np.asarray(sample["traversable_mask"]).astype(bool))
                prompts.append(prompt)
                raw_outputs.append(raw_output)
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
                    pbar.set_postfix(
                        {
                            "oracle": oracle_view,
                            "npts": len(pts),
                            "status": status,
                        }
                    )

            n_saved = len(tokens)
            out = {
                "source_shard": np.asarray([os.path.basename(shard_path)] * n_saved, dtype=object),
                "token": np.asarray(tokens, dtype=object),
                "scene": np.asarray(scenes, dtype=object),
                "instruction": np.asarray(instructions, dtype=object),
                "oracle_view": np.asarray(oracle_views, dtype=object),
                "selected_view_for_projection": np.asarray(selected_views_for_projection, dtype=object),
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
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:25547")
    parser.add_argument("--enable-depth", type=int, choices=[0, 1], default=1)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = BaselineRoboRefer(
        data_root=args.data_root,
        output_dir=args.output_dir,
        shard_pattern=args.shard_pattern,
        server_url=args.server_url,
        enable_depth=bool(args.enable_depth),
        max_samples=args.max_samples,
        retry=args.retry,
        request_timeout=args.request_timeout,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    runner.run()
