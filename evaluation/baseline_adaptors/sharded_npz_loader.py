import glob
import os
from typing import Dict, Iterator, Optional

import numpy as np
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


class ShardedNPZLoader:
    """
    Fast shard loader: loads each shard fully into RAM once, then yields samples.

    Required fields (N-first):
      - instruction: object [N]
      - images: uint8 [N,4,H,W,3] order: front,left,backward,right
      - depth: float32 [N,4,H,W] same order
      - cam2imgs: float32 [N,4,4,4]
      - cam2egos: float32 [N,4,4,4]
      - p_goal: float32 [N,3]
    Optional:
      - ego2global: float32 [N,4,4]
      - cam_height: float32 [N]
      - scene: object [N]
      - token: object [N]
      - ep_idx: int [N]
      - traversable_mask / visible_mask / affordance_mask: [N,...]
      - raw_img: uint8 [N,4,H,W,3] (if present, overrides raw_img view)
    """

    _DIR_KEYS = ("front", "left", "backward", "right")
    _DIR_TO_IDX = {"front": 0, "left": 1, "backward": 2, "right": 3}

    def __init__(self, data_root: str, pattern: str = "shard_*.npz", verbose: bool = False):
        self.data_root = data_root
        self.pattern = pattern
        self.verbose = verbose
        self.shards = sorted(glob.glob(os.path.join(self.data_root, self.pattern)))
        if not self.shards:
            raise FileNotFoundError(f"No shard files found: {os.path.join(self.data_root, self.pattern)}")

    def __iter__(self) -> Iterator[Dict]:
        global_idx = 0
        shard_iter = self.shards
        if self.verbose and tqdm is not None:
            shard_iter = tqdm(self.shards, desc="Shards", unit="shard")

        for si, shard_path in enumerate(shard_iter):
            if self.verbose:
                print(f"[ShardedNPZLoader] loading shard {si+1}/{len(self.shards)}: {shard_path}", flush=True)

            shard_data = self._load_shard_into_ram(shard_path)
            n = int(shard_data["images"].shape[0])

            if self.verbose:
                print(f"[ShardedNPZLoader] shard={os.path.basename(shard_path)} N={n}", flush=True)

            for i in range(n):
                yield self._make_sample(shard_data, i, global_idx)
                global_idx += 1

            # drop shard RAM explicitly (optional; helps peak memory behavior)
            del shard_data

    @staticmethod
    def _load_shard_into_ram(shard_path: str) -> Dict[str, Optional[np.ndarray]]:
        import time
        import os

        required = ("images", "depth", "cam2imgs", "cam2egos", "instruction", "p_goal")
        optional = (
            "ego2global",
            "cam_height",
            "scene",
            "token",
            "ep_idx",
            "traversable_mask",
            "visible_mask",
            "affordance_mask",
            "raw_img",
        )
        out: Dict[str, Optional[np.ndarray]] = {}

        t0 = time.perf_counter()
        with np.load(shard_path, allow_pickle=True) as d:
            t_open = time.perf_counter()

            # copy() forces full decompression -> RAM (this is usually the expensive part)
            for k in required:
                out[k] = d[k].copy()
            for k in optional:
                out[k] = d[k].copy() if k in d else None

        t1 = time.perf_counter()

        fname = os.path.basename(shard_path)
        n = int(out["images"].shape[0]) if out.get("images", None) is not None else -1
        print(
            f"[ShardLoad] {fname} N={n} | open={t_open - t0:.3f}s "
            f"| copy+decompress={t1 - t_open:.3f}s | total={t1 - t0:.3f}s",
            flush=True,
        )

        return out
    
    def _make_sample(self, d: Dict[str, Optional[np.ndarray]], i: int, global_idx: int) -> Dict:
        # Pull 4-view blocks once (reduces Python indexing overhead)
        img4 = d["images"][i]      # (4,H,W,3)
        dep4 = d["depth"][i]       # (4,H,W)
        c2i4 = d["cam2imgs"][i]    # (4,4,4)
        c2e4 = d["cam2egos"][i]    # (4,4,4)

        # raw_img should be (4,H,W,3) in [front,left,backward,right] order
        raw_img = img4  # view, no copy
        if d.get("raw_img", None) is not None:
            raw_img = d["raw_img"][i]  # optional override if explicitly stored

        images = {k: raw_img[self._DIR_TO_IDX[k]] for k in self._DIR_KEYS}
        depths = {k: dep4[self._DIR_TO_IDX[k]] for k in self._DIR_KEYS}
        cam2imgs = {k: c2i4[self._DIR_TO_IDX[k]] for k in self._DIR_KEYS}
        cam2egos = {k: c2e4[self._DIR_TO_IDX[k]] for k in self._DIR_KEYS}

        instruction = str(d["instruction"][i])
        p_goal = np.asarray(d["p_goal"][i], dtype=np.float32)

        ego2global = self._optional_array(d, "ego2global", i, np.eye(4, dtype=np.float32))
        cam_height = float(self._optional_scalar(d, "cam_height", i, 1.5))
        scene = str(self._optional_scalar(d, "scene", i, "unknown_scene"))
        token = str(self._optional_scalar(d, "token", i, f"sample_{global_idx:06d}"))
        ep_idx = int(d["ep_idx"][i]) if d.get("ep_idx", None) is not None else global_idx

        traversable_mask = self._optional_array(d, "traversable_mask", i, None)
        visible_mask = self._optional_array(d, "visible_mask", i, None)
        affordance_mask = self._optional_array(d, "affordance_mask", i, None)

        return {
            "instruction": instruction,
            "images": images,
            "scene": scene,
            "token": token,
            "viewpoint_hash": token.split("_yaw")[0] if "_yaw" in token else token,
            "agent_pos_hab": np.zeros(3, dtype=np.float32),
            "agent_yaw_hab_rad": 0.0,
            "cam_height": cam_height,
            "waypoints_hab": np.zeros((0, 3), dtype=np.float32),
            "pano_path": "",
            "depths": depths,
            "depth": depths,  # alias expected downstream
            "depth_paths": "",
            "cam2imgs": cam2imgs,
            "cam2egos": cam2egos,
            "ego2global": np.asarray(ego2global, dtype=np.float32),
            "p_goal": p_goal,
            "traversable_mask": traversable_mask,
            "visible_mask": visible_mask,
            "affordance_mask": affordance_mask,
            "raw_img": raw_img,
            "ep_idx": ep_idx,
        }

    @staticmethod
    def _optional_array(d: Dict[str, Optional[np.ndarray]], key: str, i: int, default):
        arr = d.get(key, None)
        return arr[i] if arr is not None else default

    @staticmethod
    def _optional_scalar(d: Dict[str, Optional[np.ndarray]], key: str, i: int, default):
        arr = d.get(key, None)
        return arr[i] if arr is not None else default


if __name__ == "__main__":
    loader = ShardedNPZLoader(
        data_root="/path/to/processed_val_unseen_for_baselines/N400_size200_maxshards2",
        pattern="shard_*.npz",
        verbose=True,
    )

    it = iter(loader)
    sample = next(it)
    print("Sanity sample keys:", sorted(sample.keys()))
    print("instruction:", sample["instruction"])
    for k in ("front", "left", "backward", "right"):
        print(
            f"{k:9s} image={sample['images'][k].shape} {sample['images'][k].dtype}, "
            f"depth={sample['depths'][k].shape} {sample['depths'][k].dtype}"
        )
    print("p_goal:", sample["p_goal"], sample["p_goal"].dtype)
    print("cam2img(front):", sample["cam2imgs"]["front"].shape, sample["cam2imgs"]["front"].dtype)
    print("cam2ego(front):", sample["cam2egos"]["front"].shape, sample["cam2egos"]["front"].dtype)
    for key in ("traversable_mask", "visible_mask", "affordance_mask", "raw_img"):
        val = sample[key]
        if val is None:
            print(f"{key}: MISSING")
        else:
            print(f"{key}: {val.shape} {val.dtype}")