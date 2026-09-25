import argparse
import glob
import os
import sys
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from baseline_adaptors.sharded_npz_loader import ShardedNPZLoader
from baseline_adaptors.sharded_result_selector import select_best, select_first
from baseline_adaptors.single_frame_evaluator import ExternalMetric


class ShardedNPZEvaluator:
    def __init__(
        self,
        candidate_dir: str,
        dataset_root: str,
        mode: str,
        pattern: str = "*.robopoint_candidates.npz",
        match_mode: str = "token",
        res: float = 0.1,
        half_xy: float = 6.4,
        verbose: bool = False,
    ):
        self.candidate_dir = candidate_dir
        self.dataset_root = dataset_root
        self.mode = mode
        self.pattern = pattern
        self.match_mode = match_mode
        self.res = float(res)
        self.half_xy = float(half_xy)
        self.verbose = verbose

        self.metric_evaluator = ExternalMetric(collect_device="cpu", res=self.res, half_xy=self.half_xy)
        self.candidate_files = sorted(glob.glob(os.path.join(self.candidate_dir, self.pattern)))
        if not self.candidate_files:
            raise FileNotFoundError(f"No candidate shard files matched: {os.path.join(self.candidate_dir, self.pattern)}")

    @staticmethod
    def _as_bool_tensor(x: np.ndarray) -> torch.Tensor:
        if x is None:
            raise ValueError("[ERR] got None where mask was expected")
        return torch.from_numpy(np.asarray(x)).bool()

    @staticmethod
    def _as_f32_tensor(x: np.ndarray) -> torch.Tensor:
        if x is None:
            raise ValueError("[ERR] got None where float tensor was expected")
        return torch.from_numpy(np.asarray(x)).float()

    @staticmethod
    def _fallback_xyz(cam_height: float) -> np.ndarray:
        return np.array([0.0, 0.0, -float(cam_height)], dtype=np.float32)

    def _load_candidate_map(self):
        cand_by_token: Dict[str, Dict] = {}
        dup = 0
        total_rows = 0
        for fpath in tqdm(self.candidate_files, desc="Loading candidate shards", unit="shard"):
            with np.load(fpath, allow_pickle=True) as d:
                n = int(len(d["token"]))
                total_rows += n
                for i in range(n):
                    tok = str(d["token"][i])
                    if tok in cand_by_token:
                        dup += 1
                        continue
                    cand_by_token[tok] = {
                        "token": tok,
                        "scene": str(d["scene"][i]) if "scene" in d else "",
                        "oracle_view": str(d["oracle_view"][i]) if "oracle_view" in d else "",
                        "cam_height": float(d["cam_height"][i]),
                        "p_goal": np.asarray(d["p_goal"][i], dtype=np.float32),
                        "ego2global_xfwd": np.asarray(d["ego2global_xfwd"][i], dtype=np.float32),
                        "traversable_mask": np.asarray(d["traversable_mask"][i]).astype(bool),
                        "candidate_xyz": np.asarray(d["candidate_xyz"][i], dtype=np.float32).reshape(-1, 3),
                        "candidate_valid": np.asarray(d["candidate_valid"][i], dtype=bool).reshape(-1),
                    }
        if dup > 0:
            print(f"[WARN] duplicate tokens across candidate shards: {dup} (kept first occurrence)")
        print(f"[INFO] loaded candidate rows={total_rows}, unique tokens={len(cand_by_token)}")
        return cand_by_token

    def _load_candidate_rows_in_order(self):
        rows: List[Dict] = []
        total_rows = 0
        for fpath in tqdm(self.candidate_files, desc="Loading candidate shards (order)", unit="shard"):
            with np.load(fpath, allow_pickle=True) as d:
                n = int(len(d["token"]))
                total_rows += n
                for i in range(n):
                    rows.append({
                        "token": str(d["token"][i]),
                        "scene": str(d["scene"][i]) if "scene" in d else "",
                        "oracle_view": str(d["oracle_view"][i]) if "oracle_view" in d else "",
                        "cam_height": float(d["cam_height"][i]),
                        "p_goal": np.asarray(d["p_goal"][i], dtype=np.float32),
                        "ego2global_xfwd": np.asarray(d["ego2global_xfwd"][i], dtype=np.float32),
                        "traversable_mask": np.asarray(d["traversable_mask"][i]).astype(bool),
                        "candidate_xyz": np.asarray(d["candidate_xyz"][i], dtype=np.float32).reshape(-1, 3),
                        "candidate_valid": np.asarray(d["candidate_valid"][i], dtype=bool).reshape(-1),
                    })
        print(f"[INFO] loaded candidate rows (order mode)={total_rows}")
        return rows

    def _build_data_batch(self, sample: Dict) -> Dict:
        view_order = ["front", "left", "backward", "right"]
        A = self._as_bool_tensor(sample["traversable_mask"]).unsqueeze(0)
        B = self._as_bool_tensor(sample["visible_mask"]).unsqueeze(0)
        C = self._as_bool_tensor(sample["affordance_mask"]).unsqueeze(0)
        p_goal = self._as_f32_tensor(sample["p_goal"]).view(1, 3)

        depths = sample["depths"]
        cam2imgs = sample["cam2imgs"]
        cam2egos = sample["cam2egos"]
        if not isinstance(depths, dict) or not isinstance(cam2imgs, dict) or not isinstance(cam2egos, dict):
            raise TypeError(f"[ERR] expected dict depths/cam2imgs/cam2egos in sample token={sample.get('token','')}")

        depth = torch.stack([self._as_f32_tensor(depths[k]) for k in view_order], dim=0).unsqueeze(0)
        cam2img = torch.stack([self._as_f32_tensor(cam2imgs[k]) for k in view_order], dim=0).unsqueeze(0)
        cam2ego = torch.stack([self._as_f32_tensor(cam2egos[k]) for k in view_order], dim=0).unsqueeze(0)

        return {
            "data": {
                "traversable_mask": A,
                "visible_mask": B,
                "affordance_mask": C,
                "p_goal": p_goal,
                "depth": depth,
                "cam2imgs": cam2img,
                "cam2egos": cam2ego,
            }
        }

    @staticmethod
    def _make_data_sample_from_xyz(pred_xyz: np.ndarray, A_1hw: torch.Tensor) -> List[Dict]:
        pred_xy = torch.tensor(np.asarray(pred_xyz[:2], dtype=np.float32), dtype=torch.float32)
        pred_aff = torch.zeros_like(A_1hw[0], dtype=torch.bool)
        return [{"pred_xy": pred_xy, "pred_aff": pred_aff}]

    def evaluate(self):
        loader = ShardedNPZLoader(data_root=self.dataset_root, pattern="shard_*.npz", verbose=False)
        cand_by_token = None
        cand_rows = None
        pred_token_set = set()
        remaining = set()
        used_pred_tokens = set()
        cand_idx = 0
        if self.match_mode == "token":
            cand_by_token = self._load_candidate_map()
            pred_token_set = set(cand_by_token.keys())
            remaining = set(pred_token_set)
        elif self.match_mode == "order":
            cand_rows = self._load_candidate_rows_in_order()
        else:
            raise ValueError(f"Unknown match_mode: {self.match_mode}")

        total = 0
        matched = 0
        missing_pred = 0
        total_candidate_points = 0
        fallback_count = 0
        samples_with_valid_candidates = 0

        pbar = tqdm(loader, desc=f"Evaluating mode={self.mode}", unit="sample")
        for sample in pbar:
            total += 1
            tok = str(sample.get("token", ""))
            if self.match_mode == "token":
                if tok == "":
                    raise ValueError("[ERR] sample has empty token")
                if tok not in cand_by_token:
                    missing_pred += 1
                    continue
                rec = cand_by_token[tok]
                matched += 1
                used_pred_tokens.add(tok)
                remaining.discard(tok)
            else:
                if cand_idx >= len(cand_rows):
                    missing_pred += 1
                    continue
                rec = cand_rows[cand_idx]
                cand_idx += 1
                matched += 1

            data_batch = self._build_data_batch(sample)
            A = data_batch["data"]["traversable_mask"]  # (1,H,W), reused for pred_aff shape

            cand_xyz = rec["candidate_xyz"]
            cand_valid = rec["candidate_valid"]
            cam_h = rec["cam_height"]

            preds_to_eval: List[np.ndarray] = []
            if self.mode == "first":
                preds_to_eval = [select_first(cand_xyz, cand_valid, cam_h)]
            elif self.mode == "best":
                preds_to_eval = [select_best(
                    cand_xyz,
                    cand_valid,
                    rec["traversable_mask"],
                    rec["p_goal"],
                    cam_h,
                    self.res,
                    self.half_xy,
                )]
            elif self.mode == "all":
                valid_preds = []
                for i in range(cand_xyz.shape[0]):
                    is_valid = i < cand_valid.shape[0] and bool(cand_valid[i])
                    xyz = np.asarray(cand_xyz[i], dtype=np.float32)
                    if is_valid and np.all(np.isfinite(xyz)):
                        valid_preds.append(xyz.astype(np.float32))
                if valid_preds:
                    preds_to_eval = valid_preds
                    samples_with_valid_candidates += 1
                else:
                    preds_to_eval = [self._fallback_xyz(cam_h)]
                    fallback_count += 1
            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            if self.mode in ("first", "best"):
                chosen = np.asarray(preds_to_eval[0], dtype=np.float32)
                if np.allclose(chosen[:2], np.array([0.0, 0.0], dtype=np.float32)):
                    # fallback convention in current pipeline
                    fallback_count += 1
                if np.any(cand_valid):
                    samples_with_valid_candidates += 1

            total_candidate_points += len(preds_to_eval)
            for pred_xyz in preds_to_eval:
                data_samples = self._make_data_sample_from_xyz(pred_xyz, A)
                self.metric_evaluator.process(data_batch, data_samples)

            # debug...
            # print("\n[DEBUG] token:", tok)
            # print("[DEBUG] mode:", self.mode)
            # print("[DEBUG] match_mode:", self.match_mode)
            # print("[DEBUG] candidate_valid:", cand_valid.tolist())
            # print("[DEBUG] candidate_xyz:", cand_xyz.tolist())
            # print("[DEBUG] preds_to_eval:", [np.asarray(p).tolist() for p in preds_to_eval])
            # raise SystemExit(0)
            # end of debug

            if self.match_mode == "token" and self.mode != "all" and len(remaining) == 0:
                break

        print("\n========== Eval coverage ==========")
        print(f"Mode:                    {self.mode}")
        print(f"Match mode:              {self.match_mode}")
        print(f"Dataset samples scanned:  {total}")
        print(f"Matched (evaluated):      {matched}")
        print(f"Missing preds (skipped):  {missing_pred}")
        print(f"Total eval points:        {total_candidate_points}")
        print(f"Samples w/ valid cand:    {samples_with_valid_candidates}")
        print(f"Fallback count:           {fallback_count}")
        print("===================================\n")

        metrics = self.metric_evaluator.compute_metrics(self.metric_evaluator.results)
        return metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["first", "best", "all"], required=True)
    parser.add_argument("--pattern", type=str, default="*.robopoint_candidates.npz")
    parser.add_argument("--match-mode", type=str, choices=["token", "order"], default="token")
    parser.add_argument("--res", type=float, default=0.1)
    parser.add_argument("--half-xy", type=float, default=6.4)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluator = ShardedNPZEvaluator(
        candidate_dir=args.candidate_dir,
        dataset_root=args.dataset_root,
        mode=args.mode,
        pattern=args.pattern,
        match_mode=args.match_mode,
        res=args.res,
        half_xy=args.half_xy,
        verbose=args.verbose,
    )
    metrics = evaluator.evaluate()
    print(metrics)
