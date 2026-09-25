import argparse
import base64
import io
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baseline_adaptors.sharded_npz_loader import ShardedNPZLoader


def _load_openai_client(api_key_path: Optional[str] = None):
    from openai import OpenAI

    # Provide the key via the OPENAI_API_KEY env var, or --api-key-path pointing to a
    # text file that contains the key. No key is bundled with this repository.
    default_key_path = None
    key_path = api_key_path or default_key_path
    base_url = os.getenv("OPENAI_BASE_URL") or None
    api_key = os.getenv("OPENAI_API_KEY")
    if (not api_key) and key_path and os.path.exists(key_path):
        with open(key_path, "r") as f:
            api_key = f.read().strip()
    if not api_key:
        raise RuntimeError("OpenAI API key not found. Set OPENAI_API_KEY or provide --api-key-path.")
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


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
    # Shards store BGR; convert to RGB for PIL/OpenAI visual input.
    arr = arr[..., ::-1]
    return Image.fromarray(arr, mode="RGB")


def pil_to_data_url(img: Image.Image, format_: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format_)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/jpeg" if format_.upper() == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def parse_predicted_points(text: str) -> List[Tuple[float, float]]:
    # Robust tuple parser (adapted from summarize_acc + local parsers).
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
            # If bbox appears, convert to center point to preserve pipeline compatibility.
            x0, y0, x1, y1 = vals
            points.append(((x0 + x1) * 0.5, (y0 + y1) * 0.5))
    return points


def parse_view_from_text(text: str) -> Optional[str]:
    t = text.lower()
    # Prefer explicit key-value patterns first.
    m = re.search(r'"view"\s*:\s*"([^"]+)"', t)
    if m:
        raw = m.group(1).strip()
    else:
        m = re.search(r"\bview\s*[:=]\s*(front|left|right|backward|back)\b", t)
        raw = m.group(1).strip() if m else None
    if raw is None:
        # Fallback: find first standalone direction token.
        for pat in [r"\bbackward\b", r"\bfront\b", r"\bleft\b", r"\bright\b", r"\bback\b"]:
            m2 = re.search(pat, t)
            if m2:
                raw = m2.group(0)
                break
    if raw is None:
        return None
    raw = raw.lower()
    if raw == "back":
        raw = "backward"
    return raw if raw in {"front", "left", "backward", "right"} else None


def xy_to_pixel(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    # If normalized, map to pixels; otherwise treat as absolute.
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


def build_prompt(
    instruction: str,
    view_mode: str,
    oracle_view: Optional[str] = None,
) -> str:
    # Prompt variants (swap manually if you want to ablate later):
    # Variant A (RoboPoint-like adapter, current default):
    #   - explicit free-space point request
    #   - exact tuple/normalized output suffix
    # Variant B (RoboRefer/GPT style):
    #   - "Please point out ..."
    # Variant C (view-selection explicit JSON):
    #   - ask for {"view": "...", "points": [[x,y], ...]}
    #
    # We keep current default minimal and tuple-based for parser compatibility.

    suffix = (
        "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], "
        "where each tuple contains the x and y coordinates of a point satisfying the conditions "
        "above. The coordinates should be between 0 and 1, indicating the normalized pixel "
        "locations of the points in the image."
    )

    if view_mode == "oracle_single_view":
        view_text = oracle_view or "front"
        return (
            f"You are given a single {view_text}-facing view image from a 4-view panorama "
            "(front, left, backward, right). This image is the selected view that is most likely "
            "relevant to the instruction.\n"
            "Locate a few points on the floor that best follow the instruction:\n"
            f"{instruction}\n"
            "If the target is not clearly visible, return your best guess in this view.\n"
            f"{suffix}"
        )

    if view_mode == "model_select_view":
        return (
            "You are given 4 images from a panorama in this order: front, left, backward, right.\n"
            "First determine which single view is most relevant to the instruction. Then locate a few "
            "points on the floor that best follow the instruction in that selected view.\n"
            f"Instruction:\n{instruction}\n"
            "IMPORTANT: Start your answer with the selected view using one of these exact words: "
            "front, left, backward, right. Then provide the points for that selected view.\n"
            "Example: view: front; [(0.45, 0.62), (0.47, 0.63)]\n"
            "If unsure, still choose one view and provide your best estimate.\n"
            f"{suffix}"
        )

    raise ValueError(f"Unknown view_mode: {view_mode}")


class GPT4ORunner:
    def __init__(
        self,
        model_name: str = "gpt-4o-2024-05-13",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        image_detail: str = "high",
        retry: int = 100,
        api_key_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.client = _load_openai_client(api_key_path=api_key_path)
        self.model_name = model_name
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.image_detail = image_detail
        self.retry = int(retry)
        self.verbose = verbose

    def infer(self, images: List[Image.Image], prompt: str) -> str:
        content = [{"type": "text", "text": prompt}]
        for img in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": pil_to_data_url(img, format_="JPEG"), "detail": self.image_detail},
                }
            )
        for attempt in range(1, self.retry + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=self.max_tokens,
                    n=1,
                    temperature=self.temperature,
                )
                u = response.usage
                print(
                    f"[TOKENS] prompt={u.prompt_tokens}, completion={u.completion_tokens}, total={u.total_tokens}"
                )
                if self.verbose:
                    print(response)
                return response.choices[0].message.content or ""
            except Exception as e:
                print(f"[WARN] OpenAI query failed attempt {attempt}/{self.retry}: {e}")
                time.sleep(1)
        return "Failed: Query Error"


class BaselineChatGPT:
    def __init__(
        self,
        data_root: str,
        output_dir: str,
        shard_pattern: str = "shard_*.npz",
        view_mode: str = "oracle_single_view",
        model_name: str = "gpt-4o-2024-05-13",
        max_samples: int = -1,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        image_detail: str = "high",
        retry: int = 20,
        overwrite: bool = False,
        api_key_path: Optional[str] = None,
        verbose: bool = False,
    ):
        self.data_root = data_root
        self.output_dir = output_dir
        self.shard_pattern = shard_pattern
        self.view_mode = view_mode
        self.max_samples = int(max_samples)
        self.overwrite = overwrite
        self.verbose = verbose

        self.loader = ShardedNPZLoader(data_root=self.data_root, pattern=self.shard_pattern, verbose=False)
        self.runner = GPT4ORunner(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            image_detail=image_detail,
            retry=retry,
            api_key_path=api_key_path,
            verbose=verbose,
        )

    def _candidate_output_path(self, shard_path: str) -> str:
        base = os.path.splitext(os.path.basename(shard_path))[0]
        return os.path.join(self.output_dir, f"{base}.chatgpt_candidates.npz")

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

                prompt = build_prompt(
                    str(sample["instruction"]),
                    view_mode=self.view_mode,
                    oracle_view=oracle_view,
                )

                if self.view_mode == "oracle_single_view":
                    image_order_keys = [oracle_view]
                elif self.view_mode == "model_select_view":
                    image_order_keys = ["front", "left", "backward", "right"]
                else:
                    raise ValueError(f"Unknown view_mode: {self.view_mode}")

                images_pil = [ensure_pil_rgb(sample["images"][k]) for k in image_order_keys]

                raw_output = ""
                status = "ok"
                pts: List[Tuple[float, float]] = []
                project_view = oracle_view if self.view_mode == "oracle_single_view" else None
                uvs: List[Tuple[int, int]] = []
                valids: List[bool] = []
                xyzs: List[np.ndarray] = []

                try:
                    raw_output = self.runner.infer(images_pil, prompt)
                    if self.view_mode == "model_select_view":
                        parsed_view = parse_view_from_text(raw_output)
                        project_view = parsed_view
                    pts = parse_predicted_points(raw_output)

                    if project_view is None:
                        status = "no_view_parsed"
                    elif project_view not in {"front", "left", "backward", "right"}:
                        status = f"bad_view:{project_view}"
                    else:
                        # In model_select_view, point coords are interpreted in the selected view's image.
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
                        print(f"[WARN] token={tok} GPT error: {e}", flush=True)

                # # debug...
                # debug_dir = self.output_dir
                # os.makedirs(debug_dir, exist_ok=True)
                # for di, (k, imgp) in enumerate(zip(image_order_keys, images_pil)):
                #     imgp.save(os.path.join(debug_dir, f"debug_baseline_chatgpt_input_{di:02d}_{k}.jpg"))
                # print("\n[DEBUG] token:", tok)
                # print("[DEBUG] view_mode:", self.view_mode)
                # print("[DEBUG] oracle_view:", oracle_view, f"(yaw_deg={yaw_deg:.2f})")
                # print("[DEBUG] project_view:", project_view)
                # print("[DEBUG] prompt sent to GPT:\n")
                # print(prompt)
                # print("\n[DEBUG] raw GPT output:\n")
                # print(raw_output)
                # print("[DEBUG] parsed_points:", pts)
                # print("[DEBUG] candidate_uvs:", uvs)
                # print("[DEBUG] candidate_valids:", valids)
                # print("[DEBUG] candidate_xyzs:", [x.tolist() for x in xyzs])
                # raise SystemExit(0)
                # # end of debug

                tokens.append(tok)
                scenes.append(str(sample.get("scene", "unknown_scene")))
                instructions.append(str(sample["instruction"]))
                oracle_views.append(oracle_view)
                selected_views_for_projection.append(project_view or "")
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
                            "sel": (project_view or "-"),
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
    parser.add_argument("--view-mode", type=str, choices=["oracle_single_view", "model_select_view"], default="oracle_single_view")
    parser.add_argument("--model-name", type=str, default="gpt-4o-2024-05-13")
    parser.add_argument("--api-key-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--image-detail", type=str, default="high", choices=["low", "high", "auto"])
    parser.add_argument("--retry", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = BaselineChatGPT(
        data_root=args.data_root,
        output_dir=args.output_dir,
        shard_pattern=args.shard_pattern,
        view_mode=args.view_mode,
        model_name=args.model_name,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        image_detail=args.image_detail,
        retry=args.retry,
        overwrite=args.overwrite,
        api_key_path=args.api_key_path,
        verbose=args.verbose,
    )
    runner.run()
