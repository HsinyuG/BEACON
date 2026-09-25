import argparse
import glob
import heapq
import math
import os
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


INF_GEO = 1e9


def fallback_xyz(cam_height: float) -> np.ndarray:
    return np.array([0.0, 0.0, -float(cam_height)], dtype=np.float32)


def bounded_geodesic_dist_cells(
    traversable: np.ndarray, si: int, sj: int, rmax_cells: float
) -> np.ndarray:
    h, w = traversable.shape
    dist = np.full((h, w), INF_GEO, dtype=np.float32)
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
    if not traversable.any():
        return np.full((h, w), INF_GEO, dtype=np.float32)

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

    return bounded_geodesic_dist_cells(traversable, seed_i, seed_j, rmax_m / res)


def geohit_rank_key_for_xy(
    pred_xy: np.ndarray,
    traversable: np.ndarray,
    dist_map: np.ndarray,
    p_goal: np.ndarray,
    res: float = 0.1,
    half_xy: float = 6.4,
) -> Tuple:
    h, w = traversable.shape
    gx = int(round((float(pred_xy[0]) + half_xy - 0.5 * res) / res))
    gy = int(round((float(pred_xy[1]) + half_xy - 0.5 * res) / res))
    out_of_bounds = gx < 0 or gx >= h or gy < 0 or gy >= w
    if out_of_bounds or (not bool(traversable[gx, gy])):
        d_cells = INF_GEO
        h05 = h10 = h15 = 0
        invalid = True
    else:
        d_cells = float(dist_map[gx, gy])
        h05 = int(d_cells <= (0.5 / res))
        h10 = int(d_cells <= (1.0 / res))
        h15 = int(d_cells <= (1.5 / res))
        invalid = False
    mean_hit = (h05 + h10 + h15) / 3.0
    euclid = float(np.linalg.norm(np.asarray(pred_xy, dtype=np.float32) - np.asarray(p_goal[:2], dtype=np.float32)))
    return (
        mean_hit,
        h05,
        h10,
        h15,
        -d_cells if np.isfinite(d_cells) else -INF_GEO,
        -euclid,
        0 if not invalid else -1,
    )


def select_first(candidate_xyz: np.ndarray, candidate_valid: np.ndarray, cam_height: float) -> np.ndarray:
    if candidate_xyz.shape[0] == 0:
        return fallback_xyz(cam_height)
    if candidate_valid.shape[0] == 0:
        return fallback_xyz(cam_height)
    if bool(candidate_valid[0]):
        xyz = np.asarray(candidate_xyz[0], dtype=np.float32)
        if np.all(np.isfinite(xyz)):
            return xyz
    return fallback_xyz(cam_height)


def select_best(
    candidate_xyz: np.ndarray,
    candidate_valid: np.ndarray,
    traversable_mask: np.ndarray,
    p_goal: np.ndarray,
    cam_height: float,
    res: float,
    half_xy: float,
) -> np.ndarray:
    if candidate_xyz.shape[0] == 0:
        return fallback_xyz(cam_height)
    valid_idx = [i for i in range(candidate_xyz.shape[0]) if i < candidate_valid.shape[0] and bool(candidate_valid[i])]
    if not valid_idx:
        return fallback_xyz(cam_height)

    traversable = np.asarray(traversable_mask).astype(bool)
    dist_map = build_goal_dist_map(traversable, np.asarray(p_goal, dtype=np.float32), res=res, half_xy=half_xy, rmax_m=1.5)

    best_key = None
    best_xyz = None
    for i in valid_idx:
        xyz = np.asarray(candidate_xyz[i], dtype=np.float32)
        if not np.all(np.isfinite(xyz)):
            continue
        key = geohit_rank_key_for_xy(xyz[:2], traversable, dist_map, p_goal, res=res, half_xy=half_xy)
        if best_key is None or key > best_key:
            best_key = key
            best_xyz = xyz
    if best_xyz is None:
        return fallback_xyz(cam_height)
    return best_xyz.astype(np.float32)


class ShardedResultSelector:
    def __init__(
        self,
        candidate_dir: str,
        pattern: str,
        output_path: str,
        mode: str,
        res: float = 0.1,
        half_xy: float = 6.4,
        verbose: bool = False,
    ):
        self.candidate_dir = candidate_dir
        self.pattern = pattern
        self.output_path = output_path
        self.mode = mode
        self.res = float(res)
        self.half_xy = float(half_xy)
        self.verbose = verbose

        self.files = sorted(glob.glob(os.path.join(self.candidate_dir, self.pattern)))
        if not self.files:
            raise FileNotFoundError(f"No candidate shard files matched: {os.path.join(self.candidate_dir, self.pattern)}")

    def _select_one(self, rec: Dict, idx: int) -> np.ndarray:
        cam_h = float(rec["cam_height"][idx])
        cand_xyz = np.asarray(rec["candidate_xyz"][idx], dtype=np.float32).reshape(-1, 3)
        cand_valid = np.asarray(rec["candidate_valid"][idx], dtype=bool).reshape(-1)
        if self.mode == "first":
            return select_first(cand_xyz, cand_valid, cam_h)
        if self.mode == "best":
            traversable = np.asarray(rec["traversable_mask"][idx]).astype(bool)
            p_goal = np.asarray(rec["p_goal"][idx], dtype=np.float32)
            return select_best(cand_xyz, cand_valid, traversable, p_goal, cam_h, self.res, self.half_xy)
        raise ValueError(f"Unknown mode: {self.mode}")

    def run(self):
        tokens: List[str] = []
        scenes: List[str] = []
        oracle_views: List[str] = []
        p_pred: List[np.ndarray] = []
        p_goal: List[np.ndarray] = []
        ego2global_xfwd: List[np.ndarray] = []
        cam_height: List[float] = []

        pbar = tqdm(self.files, desc=f"Selecting mode={self.mode}", unit="shard")
        for fpath in pbar:
            with np.load(fpath, allow_pickle=True) as d:
                n = int(len(d["token"]))
                for i in range(n):
                    pred = self._select_one(d, i)

                    # debug...
                    # print("\n[DEBUG] file:", fpath)
                    # print("[DEBUG] token:", str(d['token'][i]))
                    # print("[DEBUG] mode:", self.mode)
                    # print("[DEBUG] candidate_valid:", np.asarray(d['candidate_valid'][i]).tolist())
                    # print("[DEBUG] candidate_xyz:", np.asarray(d['candidate_xyz'][i], dtype=np.float32).tolist())
                    # print("[DEBUG] selected pred:", pred.tolist())
                    # raise SystemExit(0)
                    # end of debug

                    tokens.append(str(d["token"][i]))
                    scenes.append(str(d["scene"][i]))
                    oracle_views.append(str(d["oracle_view"][i]))
                    p_pred.append(np.asarray(pred, dtype=np.float32))
                    p_goal.append(np.asarray(d["p_goal"][i], dtype=np.float32))
                    ego2global_xfwd.append(np.asarray(d["ego2global_xfwd"][i], dtype=np.float32))
                    cam_height.append(float(d["cam_height"][i]))

        out = {
            "p_pred": np.asarray(p_pred, dtype=np.float32),
            "p_goal": np.asarray(p_goal, dtype=np.float32),
            "ego2global_xfwd": np.asarray(ego2global_xfwd, dtype=np.float32),
            "cam_height": np.asarray(cam_height, dtype=np.float32),
            "scene_token": np.asarray(scenes, dtype=object),
            "token": np.asarray(tokens, dtype=object),
            "oracle_view": np.asarray(oracle_views, dtype=object),
            "selection_mode": np.asarray([self.mode] * len(tokens), dtype=object),
        }
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "wb") as f:
            np.savez(f, **out)
        print(f"[SAVE] merged selection NPZ: {self.output_path} (samples={len(tokens)}, mode={self.mode})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*.robopoint_candidates.npz")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["first", "best"], required=True)
    parser.add_argument("--res", type=float, default=0.1)
    parser.add_argument("--half-xy", type=float, default=6.4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    selector = ShardedResultSelector(
        candidate_dir=args.candidate_dir,
        pattern=args.pattern,
        output_path=args.output_path,
        mode=args.mode,
        res=args.res,
        half_xy=args.half_xy,
        verbose=args.verbose,
    )
    selector.run()
