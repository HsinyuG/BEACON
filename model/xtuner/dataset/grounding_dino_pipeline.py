import os
import sys, torch
from typing import Tuple, List
from PIL import Image, ImageDraw  # for rendering results
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import spacy
import re

# Grounded-Segment-Anything (GroundingDINO) is an optional third-party dependency
# used only by BoxPipeline (not instantiated in the released train/eval paths).
# Point GSA_ROOT at your local clone to enable it; the paths below are added to
# sys.path only when that directory actually exists.
_GSA_ROOT = os.environ.get("GSA_ROOT", "")
for _p in (
    os.path.join(_GSA_ROOT, "segment_anything"),
    _GSA_ROOT,
    os.environ.get("BEACON_AO_PLANNER_ROOT", ""),
):
    if _p and os.path.isdir(_p):
        sys.path.insert(0, _p)

class GroundingDINOBoxer:
    def __init__(
        self,
        device: str = "cuda",
        config_path: str = os.environ.get(
            "GROUNDINGDINO_CONFIG",
            "/path/to/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        ),
        ckpt_path: str = os.environ.get(
            "GROUNDINGDINO_CKPT",
            "/path/to/Grounded-Segment-Anything/groundingdino_swint_ogc.pth",
        ),
    ):


        import GroundingDINO.groundingdino.datasets.transforms as T
        from GroundingDINO.groundingdino.models import build_model
        from GroundingDINO.groundingdino.util.slconfig import SLConfig
        from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
        from PIL import Image

        self.T = T
        self.Image = Image
        self.get_phrases_from_posmap = get_phrases_from_posmap

        args = SLConfig.fromfile(config_path)
        args.device = device
        model = build_model(args)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
        self.model = model.eval().to(device)
        self.device = device

        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _load_image_from_path(self, image_path: str):
        image_pil = self.Image.open(image_path).convert("RGB")
        image_t, _ = self.transform(image_pil, None)
        return image_pil, image_t
    
    def _load_image(self, image_in):
        # image_in can be: path(str/Path), PIL.Image, or np.ndarray (H,W,3) from cv2 (BGR)
        if isinstance(image_in, (str, Path)):
            image_pil = self.Image.open(str(image_in)).convert("RGB")
        elif isinstance(image_in, self.Image.Image):
            image_pil = image_in.convert("RGB")
        elif isinstance(image_in, np.ndarray):
            if image_in.ndim != 3 or image_in.shape[2] != 3:
                raise ValueError(f"Expected HxWx3 uint8 image, got {image_in.shape}")
            # assume OpenCV BGR -> RGB
            rgb = image_in[..., ::-1].astype(np.uint8, copy=False)
            image_pil = self.Image.fromarray(rgb, mode="RGB")
        elif torch.is_tensor(image_in):
            t = image_in
            # accept (H,W,3) or (3,H,W)
            if t.ndim == 3 and t.shape[-1] == 3:
                arr = t.detach().cpu().numpy()
            elif t.ndim == 3 and t.shape[0] == 3:
                arr = t.permute(1, 2, 0).detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported torch image shape: {tuple(t.shape)}")

            # ensure uint8 0..255
            if arr.dtype != np.uint8:
                mx = float(arr.max()) if arr.size else 0.0
                if mx <= 1.0:
                    arr = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)

            # assume BGR -> RGB
            rgb = arr[..., ::-1]
            image_pil = self.Image.fromarray(rgb, mode="RGB")
        else:
            raise TypeError(f"Unsupported image type: {type(image_in)}")

        image_t, _ = self.transform(image_pil, None)
        return image_pil, image_t

    def predict_boxes(
        self,
        image_in,
        text_prompt: str = "ground",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> Tuple[List[List[float]], List[str]]:
        image_pil, image_t = self._load_image(image_in)
        caption = text_prompt.strip().lower()
        if not caption.endswith("."):
            caption += "."

        with torch.no_grad():
            outputs = self.model(image_t[None].to(self.device), captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]   # (nq, 256)
        boxes = outputs["pred_boxes"][0]                # (nq, 4) cx,cy,w,h normalized

        keep = logits.max(dim=1)[0] > box_threshold
        logits_f, boxes_f = logits[keep], boxes[keep]

        tokenized = self.model.tokenizer(caption)
        phrases = []
        for logit, box in zip(logits_f, boxes_f):
            phrase = self.get_phrases_from_posmap(logit > text_threshold, tokenized, self.model.tokenizer)
            phrases.append(f"{phrase}({logit.max().item():.3f})")

        # convert to pixel x0,y0,x1,y1
        W, H = image_pil.size
        boxes_xyxy = []
        for b in boxes_f:
            b = b * torch.tensor([W, H, W, H], device=boxes_f.device)
            b[:2] -= b[2:] / 2
            b[2:] += b[:2]
            boxes_xyxy.append(b.tolist())

        return boxes_xyxy, phrases
    
    def predict_boxes_batch(
        self,
        images_in,
        captions,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ):
        """
        images_in: list length N; each can be np(H,W,3), torch(H,W,3), PIL, or path
        captions:  list length N (string prompt per image)
        returns: (all_boxes_xyxy, all_phrases)
        - all_boxes_xyxy: list of N lists of [x0,y0,x1,y1]
        - all_phrases:    list of N lists of phrase strings like 'chair(0.512)'
        """
        assert len(images_in) == len(captions), "images_in and captions must be same length"
        N = len(images_in)

        # preprocess all images
        image_ts = []
        sizes = []
        caps = []
        for img, cap in zip(images_in, captions):
            image_pil, image_t = self._load_image(img)
            sizes.append(image_pil.size)  # (W,H) of original
            image_ts.append(image_t)
            c = cap.strip().lower()
            if not c.endswith("."):
                c += "."
            caps.append(c)

        batch = torch.stack(image_ts, dim=0).to(self.device)  # (N,3,H',W')

        with torch.no_grad():
            outputs = self.model(batch, captions=caps)

        pred_logits = outputs["pred_logits"].sigmoid()  # (N,nq,256)
        pred_boxes  = outputs["pred_boxes"]             # (N,nq,4) cxcywh normalized

        all_boxes_xyxy = []
        all_phrases = []

        for i in range(N):
            logits = pred_logits[i]
            boxes  = pred_boxes[i]

            keep = logits.max(dim=1)[0] > box_threshold
            logits_f = logits[keep]
            boxes_f  = boxes[keep]

            tokenized = self.model.tokenizer(caps[i])

            phrases = []
            for logit in logits_f:
                phrase = self.get_phrases_from_posmap(
                    logit > text_threshold, tokenized, self.model.tokenizer
                )
                phrases.append(f"{phrase}({logit.max().item():.3f})")

            W, H = sizes[i]
            if boxes_f.numel() == 0:
                all_boxes_xyxy.append([])
                all_phrases.append([])
                continue

            # cxcywh -> xyxy in pixels
            scale = torch.tensor([W, H, W, H], device=boxes_f.device)
            b = boxes_f * scale
            cxcy = b[:, :2]
            wh   = b[:, 2:]
            x0y0 = cxcy - wh / 2
            x1y1 = cxcy + wh / 2
            bxyxy = torch.cat([x0y0, x1y1], dim=1)

            all_boxes_xyxy.append(bxyxy.detach().cpu().tolist())
            all_phrases.append(phrases)

        return all_boxes_xyxy, all_phrases

import math
class PointConverter():
    def __init__(self):
        self.out_hw = (448, 448)
        self.fov_deg = 90.0  # horizontal fov
        
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
    
    def project_points_to_ego(self, points_pixel, yaw_deg):
        """
        points_pixel: (N,3) array of (u, v, depth)
        yaw_deg: camera yaw in degrees (front=0, left=90, back=180, right=270)
        returns: (N,3) array of (x, y, z) in ego frame
        """
        pts = np.asarray(points_pixel, dtype=np.float32)
        if pts.shape[0] == 0:
            return pts.reshape(0, 3)

        K = self._cam2img()[:3, :3]
        K_inv = np.linalg.inv(K)
        T = self._cam2ego(yaw_deg)

        uv1 = np.concatenate([pts[:, :2], np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)  # (N,3)
        rays = (K_inv @ uv1.T).T  # (N,3), normalized directions
        cam_xyz = rays * pts[:, 2:3]  # scale by depth
        cam_h = np.concatenate([cam_xyz, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)  # (N,4)

        ego = (T @ cam_h.T).T  # (N,4)
        return ego[:, :3]



class BoxPipeline:
    def __init__(self, 
                 iou_thresh=0.5, 
                 size_ratio_thresh=0.5,
                 dist_thr=3.0, 
                 max_within=4, 
                 min_total=2,
                 max_repeat_within=2,
                 ):
        self.boxer = GroundingDINOBoxer()
        self.converter = PointConverter()
        self.iou_thresh = iou_thresh
        self.size_ratio_thresh = size_ratio_thresh
        self.yaw_map = {"front": 0.0, "left": 90.0, "back": 180.0, "right": 270.0}
        self.nlp = spacy.load("en_core_web_sm")
        self.dist_thr = dist_thr
        self.max_within = max_within
        self.max_repeat_within = max_repeat_within
        self.min_total = min_total
    
    def _extract_instruction_text(self, raw_instruction):
        # Case 1: already a string
        if isinstance(raw_instruction, str):
            s = raw_instruction
        # Case 2: list[dict] chat format
        elif isinstance(raw_instruction, list):
            s = ""
            for m in raw_instruction:
                if isinstance(m, dict) and m.get("from") == "human":
                    s = m.get("value", "")
                    break
        # Fallback
        else:
            s = str(raw_instruction)

        # Pull only the "Instruction: ..." part if present
        m = re.search(r"Instruction:\s*(.*?)(?:\n\s*\[TGT\]|\Z)", s, flags=re.S)
        if m:
            return m.group(1).strip()
        # If pattern not found, just return the whole human text
        return s.strip()

    def _noun_phrases(self, text: str):
        doc = self.nlp(text)
        out, seen = [], set()
        for np in doc.noun_chunks:
            p = " ".join(t.lemma_.lower() for t in np
                        if t.pos_ in {"NOUN","PROPN","ADJ"} and not t.is_stop)
            if p and p not in seen:
                seen.add(p); out.append(p)
        print(f"[BoxPipeline] Extracted noun phrases: {out}")
        return out

    def _parse_phrase(self, phrase):
        if "(" in phrase and phrase.endswith(")"):
            conf = float(phrase.split("(")[-1][:-1])
            label = phrase.rsplit("(", 1)[0]
        else:
            label, conf = phrase, 0.0
        return label, conf

    def _iou(self, A, B):
        ax0, ay0, ax1, ay1 = A
        bx0, by0, bx1, by1 = B
        inter = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
        a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
        b = max(0, bx1 - bx0) * max(0, by1 - by0)
        union = a + b - inter
        return inter / union if union > 0 else 0.0

    def _depth_stats(self, box, depth):
        H, W = depth.shape
        x0, y0, x1, y1 = [int(v) for v in box]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = depth[y0:y1, x0:x1]
        valid = crop[np.isfinite(crop)]
        if valid.size == 0:
            return None
        d = float(valid.mean())
        u = 0.5 * (x0 + x1)
        v = 0.5 * (y0 + y1)
        area = int((y1 - y0) * (x1 - x0))
        return u, v, d, area, W, H

    def _nms_per_label(self, boxes, phrases, W, H):
        kept_boxes, kept_phrases = [], []
        by_label = {}
        for box, phr in zip(boxes, phrases):
            label, conf = self._parse_phrase(phr)
            by_label.setdefault(label, []).append((conf, box, phr))
        for _, items in by_label.items():
            items.sort(key=lambda x: x[0], reverse=True)  # confidence sort
            selected = []
            for conf, box, phr in items:
                area = max(0, box[2]-box[0]) * max(0, box[3]-box[1])
                if area / float(W * H) > self.size_ratio_thresh:
                    continue
                if any(self._iou(box, sb) > self.iou_thresh for _, sb, _ in selected):
                    continue
                selected.append((conf, box, phr))
            for conf, box, phr in selected:
                kept_boxes.append(box)
                kept_phrases.append(phr)
        return kept_boxes, kept_phrases

    def _select_near_targets(
        self,
        points,
        target_xy,
    ):
        if not points:
            raise RuntimeError("No detection points (empty pts). Likely dataloading / depth / detection bug.")
        
        dist_thr = self.dist_thr
        max_within = self.max_within
        min_total = self.min_total
        max_rep = self.max_repeat_within

        tx, ty = float(target_xy[0]), float(target_xy[1])

        # distances for detections (ego excluded here)
        det = []
        for p in points:
            d = math.hypot(float(p["xyz"][0]) - tx, float(p["xyz"][1]) - ty)
            det.append((d, p))
        det.sort(key=lambda x: x[0])

        within = [(d, p) for (d, p) in det if d <= dist_thr]

        # Case A: enough detections within threshold => return up to max_within, exclude ego
        if len(within) >= min_total:
            out = []
            label_cnt = {}
            for d, p in within:  # already sorted by distance
                lab = p.get("label", "")
                if max_rep > 0 and label_cnt.get(lab, 0) >= max_rep:
                    continue

                label_cnt[lab] = label_cnt.get(lab, 0) + 1

                item = dict(p)
                item.update({"is_ego": False, "dist_xy": float(d), "rank": len(out)})
                out.append(item)

                if len(out) >= max_within:
                    break
            return out


        # Case B: not enough within threshold => ignore threshold, allow ego, return top min_total globally
        ego_dist = math.hypot(0.0 - tx, 0.0 - ty)
        global_cands = [(ego_dist, None)] + det  # None denotes ego
        global_cands.sort(key=lambda x: x[0])

        out = []
        for d, p in global_cands:
            if p is None:
                out.append({"is_ego": True, "dist_xy": float(d), "rank": len(out)})
            else:
                item = dict(p)
                item.update({"is_ego": False, "dist_xy": float(d), "rank": len(out)})
                out.append(item)
            if len(out) >= min_total:
                break

        return out
    
    def _select_near_targets_idx(self, points, target_xy):
        """
        points: list of detection dicts for ONE sample
        returns: list of dicts with keys {idx, rank, dist_xy, is_ego}
        idx refers to index in `points`. ego uses idx=-1.
        """
        if not points:
            raise RuntimeError("No detection points (empty pts).")

        dist_thr = float(self.dist_thr)
        max_within = int(self.max_within)
        min_total = int(self.min_total)
        max_rep = int(self.max_repeat_within)

        tx, ty = float(target_xy[0]), float(target_xy[1])

        # compute distances, keep index
        det = []
        for i, p in enumerate(points):
            d = math.hypot(float(p["xyz"][0]) - tx, float(p["xyz"][1]) - ty)
            det.append((d, i, p.get("label", "")))
        det.sort(key=lambda x: x[0])

        within = [(d, i, lab) for (d, i, lab) in det if d <= dist_thr]

        # Case A: enough within threshold
        if len(within) >= min_total:
            out = []
            label_cnt = {}
            for d, i, lab in within:
                if max_rep > 0 and label_cnt.get(lab, 0) >= max_rep:
                    continue
                label_cnt[lab] = label_cnt.get(lab, 0) + 1

                out.append({"idx": int(i), "rank": len(out), "dist_xy": float(d), "is_ego": False})
                if len(out) >= max_within:
                    break
            return out

        # Case B: fallback include ego, ignore threshold
        ego_dist = math.hypot(0.0 - tx, 0.0 - ty)
        global_cands = [(ego_dist, -1, "ego")] + det
        global_cands.sort(key=lambda x: x[0])

        out = []
        label_cnt = {}
        for d, i, lab in global_cands:
            if i == -1:
                out.append({"idx": -1, "rank": len(out), "dist_xy": float(d), "is_ego": True})
            else:
                if max_rep > 0 and label_cnt.get(lab, 0) >= max_rep:
                    continue
                label_cnt[lab] = label_cnt.get(lab, 0) + 1
                out.append({"idx": int(i), "rank": len(out), "dist_xy": float(d), "is_ego": False})

            if len(out) >= min_total:
                break

        return out


    def _points_from_view(self, name, img_np, depth_np, raw_instruction):
        depth = depth_np
        # GroundingDINO tends to like period-separated objects
        instr_text = self._extract_instruction_text(raw_instruction)
        # print(f"[BoxPipeline] Detecting boxes in view '{name}' with prompt: {instr_text}")
        text_prompt = ". ".join(self._noun_phrases(instr_text))
        boxes, phrases = self.boxer.predict_boxes(
            image_in=img_np,
            text_prompt=text_prompt,
            box_threshold=0.3,
            text_threshold=0.25,
        )
        H, W = img_np.shape[:2]

        boxes, phrases = self._nms_per_label(boxes, phrases, W, H)
        pts_pixel = []
        labels = []
        for box, phr in zip(boxes, phrases):
            stats = self._depth_stats(box, depth)
            if stats is None:
                continue
            u, v, d, area, _, _ = stats
            pts_pixel.append((u, v, d, area, phr, box))
        if not pts_pixel:
            return []
        uvds = np.array([[p[0], p[1], p[2]] for p in pts_pixel], dtype=np.float32)
        ego_xyz = self.converter.project_points_to_ego(uvds, self.yaw_map[name])
        out = []
        for (u, v, d, area, phr, box), xyz in zip(pts_pixel, ego_xyz):
            label, conf = self._parse_phrase(phr)
            x0, y0, x1, y1 = [float(vv) for vv in box]
            out.append({
                "label": label,
                "conf": conf,
                "phrase": phr,

                "view": name,
                "view_idx": {"front":0, "left":1, "back":2, "right":3}[name],
                "img_hw": [int(H), int(W)],

                # box defines the covered (u,v) rectangle
                "box_xyxy": [x0, y0, x1, y1],
                "uv_center": [float(u), float(v)],
                "depth_mean": float(d),

                "xyz": xyz,
                "area": int(area),
            })
        return out


    def run(self, images, depths, raw_instruction, p_goal=None):
        """
        views: list of (name, rgb_path, depth_path), name in {"front","left","back","right"}
        raw_instruction: string prompt to feed the detector
        returns: list of dicts {label, xyz, area}
        """
        pts = []
        names = ["front", "left", "back", "right"]
        for name, img_np, depth_np in zip(names, images, depths):
            pts.extend(self._points_from_view(name, img_np, depth_np, raw_instruction))

        if not pts:
            raise RuntimeError("No detection points after processing all 4 views (pts empty).")

        selected = None
        if p_goal is not None:
            target_xy = np.asarray(p_goal, dtype=np.float32).reshape(-1)[:2]
            selected = self._select_near_targets(pts, target_xy)

        return pts, selected

    def run_batched(self, raw_img, depth, conversations, p_goal=None):
        """
        raw_img: torch.Tensor (B,4,H,W,3) or numpy
        depth:   torch.Tensor (B,4,H,W)   or numpy
        conversations: list length B; each item is list[dict] (chat)
        p_goal:  torch.Tensor (B,2 or 3) or numpy; we use [:2]
        returns:
        detected_boxes: list length B; each is list[dict] detections
        selected_boxes: list length B; each is list[dict] selection (or None if p_goal None)
        """
        # normalize conversations shape (sometimes B=1 may come unbatched)
        if isinstance(conversations, list) and len(conversations) > 0 and isinstance(conversations[0], dict):
            conversations = [conversations]

        # move tensors to CPU once (avoid per-image gpu->cpu copies)
        if torch.is_tensor(raw_img):
            raw_img_cpu = raw_img.detach().cpu()
            B, V, H, W, C = raw_img_cpu.shape
        else:
            raw_img_cpu = np.asarray(raw_img)
            B, V, H, W, C = raw_img_cpu.shape

        if torch.is_tensor(depth):
            depth_cpu = depth.detach().cpu().numpy()
        else:
            depth_cpu = np.asarray(depth)

        assert V == 4 and C == 3, f"Expected raw_img (B,4,H,W,3), got {tuple(raw_img_cpu.shape)}"
        assert depth_cpu.shape[:2] == (B, 4), f"Expected depth (B,4,H,W), got {tuple(depth_cpu.shape)}"

        # build per-sample caption (noun prompt)
        caps_b = []
        for b in range(B):
            instr = self._extract_instruction_text(conversations[b])
            np_list = self._noun_phrases(instr)
            np_list += ['door'] if 'door' not in np_list else []
            cap = ". ".join(np_list) if len(np_list) > 0 else "object"
            caps_b.append(cap)

        # flatten Bx4 -> N = B*4
        view_names = ["front", "left", "back", "right"]
        images_flat = []
        caps_flat = []
        for b in range(B):
            for v in range(4):
                images_flat.append(raw_img_cpu[b, v])   # torch(H,W,3) or np(H,W,3)
                caps_flat.append(caps_b[b])

        # SINGLE GDINO CALL
        all_boxes, all_phrases = self.boxer.predict_boxes_batch(
            images_flat,
            caps_flat,
            box_threshold=0.3,
            text_threshold=0.25,
        )

        detected = [[] for _ in range(B)]

        # decode per image -> points
        for idx in range(B * 4):
            b = idx // 4
            v = idx % 4
            name = view_names[v]

            img = raw_img_cpu[b, v]
            H_img = int(img.shape[0])
            W_img = int(img.shape[1])

            boxes = all_boxes[idx]
            phrases = all_phrases[idx]
            if len(boxes) == 0:
                continue

            boxes, phrases = self._nms_per_label(boxes, phrases, W_img, H_img)

            depth_view = depth_cpu[b, v]  # (H,W)
            pts_pixel = []
            for box, phr in zip(boxes, phrases):
                stats = self._depth_stats(box, depth_view)
                if stats is None:
                    continue
                u, vv, d, area, _, _ = stats
                pts_pixel.append((u, vv, d, area, phr, box))

            if not pts_pixel:
                continue

            uvds = np.array([[p[0], p[1], p[2]] for p in pts_pixel], dtype=np.float32)
            ego_xyz = self.converter.project_points_to_ego(uvds, self.yaw_map[name])

            for (u, vv, d, area, phr, box), xyz in zip(pts_pixel, ego_xyz):
                label, conf = self._parse_phrase(phr)
                x0, y0, x1, y1 = [float(vv2) for vv2 in box]
                detected[b].append({
                    "label": label,
                    "conf": conf,
                    "phrase": phr,

                    "view": name,
                    "view_idx": v,
                    "img_hw": [H_img, W_img],

                    "box_xyxy": [x0, y0, x1, y1],
                    "uv_center": [float(u), float(vv)],
                    "depth_mean": float(d),

                    "xyz": xyz.astype(np.float32),
                    "area": int(area),
                })

        # enforce "must not be missing" (with fallback prompt if empty)
        empty_bs = [b for b in range(B) if len(detected[b]) == 0]
        if len(empty_bs) > 0:
            print(f"[BoxPipeline] Warning: {len(empty_bs)}/{B} batch items have NO detections; applying fallback prompt.")
            fallback_cap = "door. window. object"

            # build a smaller batch for only the missing samples (4 views each)
            images_fb = []
            caps_fb = []
            bvidx = []  # list of (b, v)
            for b in empty_bs:
                for v in range(4):
                    images_fb.append(raw_img_cpu[b, v])
                    caps_fb.append(fallback_cap)
                    bvidx.append((b, v))

            fb_boxes, fb_phrases = self.boxer.predict_boxes_batch(
                images_fb,
                caps_fb,
                box_threshold=0.3,
                text_threshold=0.25,
            )

            view_names = ["front", "left", "back", "right"]

            # decode fallback detections into detected[b]
            for j in range(len(images_fb)):
                b, v = bvidx[j]
                name = view_names[v]

                img = raw_img_cpu[b, v]
                H_img = int(img.shape[0])
                W_img = int(img.shape[1])

                boxes = fb_boxes[j]
                phrases = fb_phrases[j]
                if len(boxes) == 0:
                    continue

                boxes, phrases = self._nms_per_label(boxes, phrases, W_img, H_img)

                depth_view = depth_cpu[b, v]
                pts_pixel = []
                for box, phr in zip(boxes, phrases):
                    stats = self._depth_stats(box, depth_view)
                    if stats is None:
                        continue
                    u, vv, d, area, _, _ = stats
                    pts_pixel.append((u, vv, d, area, phr, box))

                if not pts_pixel:
                    continue

                uvds = np.array([[p[0], p[1], p[2]] for p in pts_pixel], dtype=np.float32)
                ego_xyz = self.converter.project_points_to_ego(uvds, self.yaw_map[name])

                for (u, vv, d, area, phr, box), xyz in zip(pts_pixel, ego_xyz):
                    label, conf = self._parse_phrase(phr)
                    x0, y0, x1, y1 = [float(vv2) for vv2 in box]
                    detected[b].append({
                        "label": label,
                        "conf": conf,
                        "phrase": phr,

                        "view": name,
                        "view_idx": v,
                        "img_hw": [H_img, W_img],

                        "box_xyxy": [x0, y0, x1, y1],
                        "uv_center": [float(u), float(vv)],
                        "depth_mean": float(d),

                        "xyz": xyz.astype(np.float32),
                        "area": int(area),
                    })

        # if still empty after fallback => raise
        for b in range(B):
            if len(detected[b]) == 0:
                raise RuntimeError(f"No detection points for batch item {b} (even after fallback prompt).")


        selected = [None for _ in range(B)]
        if p_goal is not None:
            if torch.is_tensor(p_goal):
                p_goal_np = p_goal.detach().cpu().numpy()
            else:
                p_goal_np = np.asarray(p_goal)

            for b in range(B):
                target_xy = np.asarray(p_goal_np[b], dtype=np.float32).reshape(-1)[:2]
                # selected[b] = self._select_near_targets(detected[b], target_xy)
                selected[b] = self._select_near_targets_idx(detected[b], target_xy)

        return detected, selected




    @staticmethod
    def plot(points, outfile, target_xy=None):
        fig = go.Figure()
        by_label = {}
        ego_is_here = False
        for p in points:
            if p.get("is_ego", False):
                ego_is_here = True
                continue
            by_label.setdefault(p["label"], []).append(p)

        # regular per-label traces
        for label, grp in by_label.items():
            fig.add_trace(go.Scatter3d(
                x=[g["xyz"][0] for g in grp],
                y=[g["xyz"][1] for g in grp],
                z=[g["xyz"][2] for g in grp],
                mode="markers+text",
                text=[label]*len(grp),
                marker=dict(size=[max(4, (g["area"]**0.5/5.0)) for g in grp]),
                name=label,
            ))
        # ego base marker
        ego_color = "red" if ego_is_here else "black"
        ego_label = "ego (selected)" if ego_is_here else "ego (not selected)"
        fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[0],
                                mode="markers", marker=dict(color=ego_color, size=8),
                                name=ego_label))
        # target + closest highlights
        if target_xy is not None:
            tx, ty = float(target_xy[0]), float(target_xy[1])
            # target marker at z=0
            fig.add_trace(go.Scatter3d(x=[tx], y=[ty], z=[0],
                                    mode="markers+text",
                                    text=["target"],
                                    marker=dict(color="blue", symbol="x", size=10),
                                    name="target"))
            # gather candidates (points + ego) for nearest search
            cands = [{"name": p["label"], "xyz": p["xyz"]} for p in points]
            cands.append({"name": "ego", "xyz": np.array([0.0, 0.0, 0.0], dtype=np.float32)})
            for c in cands:
                c["dist"] = math.hypot(c["xyz"][0] - tx, c["xyz"][1] - ty)
            cands.sort(key=lambda c: c["dist"])
            closest_two = cands[:2] if len(cands) >= 2 else cands
            for i, c in enumerate(closest_two):
                fig.add_trace(go.Scatter3d(
                    x=[c["xyz"][0]], y=[c["xyz"][1]], z=[c["xyz"][2]],
                    mode="markers+text",
                    text=[f"{c['name']} ({'closest' if i == 0 else 'second'})"],
                    marker=dict(
                        color="magenta" if i == 0 else "orange",
                        size=12 if i == 0 else 10,
                        symbol="diamond" if i == 0 else "square"
                    ),
                    name=f"{'closest' if i == 0 else 'second'}"
                ))
        fig.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                        scene_aspectmode="data")
        fig.write_html(outfile)
        print(f"Saved plot: {outfile}")


# if __name__ == "__main__":

    # # sample 1

    # left_img = '/path/to/Datasets/MP3D/tools/result_episode_S9hNv5qa7GM_ep0390/data/S9hNv5qa7GM/cam/scanS9hNv5qa7GM_ep0390_g0014_sub05_i00_left_rgb.png'
    # front_img = '/path/to/Datasets/MP3D/tools/result_episode_S9hNv5qa7GM_ep0390/data/S9hNv5qa7GM/cam/scanS9hNv5qa7GM_ep0390_g0014_sub05_i00_front_rgb.png'
    # right_img = '/path/to/Datasets/MP3D/tools/result_episode_S9hNv5qa7GM_ep0390/data/S9hNv5qa7GM/cam/scanS9hNv5qa7GM_ep0390_g0014_sub05_i00_right_rgb.png'
    # back_img = '/path/to/Datasets/MP3D/tools/result_episode_S9hNv5qa7GM_ep0390/data/S9hNv5qa7GM/cam/scanS9hNv5qa7GM_ep0390_g0014_sub05_i00_back_rgb.png'

    # sample_instruction = 'If we turn slightly right, we are going to see a couch and a window behind the couch.\
    #     We are going to walk between the window and the couch. We are going to stop right in the corner, \
    #     looking at the window on the left side and looking at a shelf with sculptures, has a rabbit, paintings, a pear. That would be our destination.'
    # target_xy = np.array([1.5743311558822684, -2.8971120353902036])

    # # sample 2

    # left_img = '/path/to/Datasets/MP3D/tools/result_episode_2azQ1b91cZZ_ep0306/data/2azQ1b91cZZ/cam/scan2azQ1b91cZZ_ep0306_g0008_sub03_i00_left_rgb.png'
    # front_img = '/path/to/Datasets/MP3D/tools/result_episode_2azQ1b91cZZ_ep0306/data/2azQ1b91cZZ/cam/scan2azQ1b91cZZ_ep0306_g0008_sub03_i00_front_rgb.png'
    # right_img = '/path/to/Datasets/MP3D/tools/result_episode_2azQ1b91cZZ_ep0306/data/2azQ1b91cZZ/cam/scan2azQ1b91cZZ_ep0306_g0008_sub03_i00_right_rgb.png'
    # back_img = '/path/to/Datasets/MP3D/tools/result_episode_2azQ1b91cZZ_ep0306/data/2azQ1b91cZZ/cam/scan2azQ1b91cZZ_ep0306_g0008_sub03_i00_back_rgb.png'

    # sample_instruction = 'turn to your right, so you see the 4th vase, go down the hall way.'
    # target_xy = np.array([1.680629218613101, -1.485264298405087])

    # # sample 3

    # left_img = '/path/to/Datasets/MP3D/tools/result_episode_2azQ1b91cZZ_ep0306/data/2azQ1b91cZZ/cam/scan2azQ1b91cZZ_ep0306_g0012_sub06_i00_left_rgb.png'
    # front_img = '/path/to/Datasets/MP3D/tools/result_episode_2azQ1b91cZZ_ep0306/data/2azQ1b91cZZ/cam/scan2azQ1b91cZZ_ep0306_g0012_sub06_i00_front_rgb.png'
    # right_img = '/path/to/Datasets/MP3D/tools/result_episode_2azQ1b91cZZ_ep0306/data/2azQ1b91cZZ/cam/scan2azQ1b91cZZ_ep0306_g0012_sub06_i00_right_rgb.png'
    # back_img = '/path/to/Datasets/MP3D/tools/result_episode_2azQ1b91cZZ_ep0306/data/2azQ1b91cZZ/cam/scan2azQ1b91cZZ_ep0306_g0012_sub06_i00_back_rgb.png'

    # sample_instruction = 'walk in between the two green chairs to the left of the bed.'
    # target_xy = np.array([1.587, 0.742])

    # # sample 4
    # left_img = '/path/to/Datasets/MP3D/tools/result_episode_2azQ1b91cZZ_ep0306/data/2azQ1b91cZZ/cam/scan2azQ1b91cZZ_ep0306_g0009_sub04_i00_left_rgb.png'
    # front_img = '/path/to/Datasets/MP3D/tools/result_episode_2azQ1b91cZZ_ep0306/data/2azQ1b91cZZ/cam/scan2azQ1b91cZZ_ep0306_g0009_sub04_i00_front_rgb.png'
    # right_img = '/path/to/Datasets/MP3D/tools/result_episode_2azQ1b91cZZ_ep0306/data/2azQ1b91cZZ/cam/scan2azQ1b91cZZ_ep0306_g0009_sub04_i00_right_rgb.png'
    # back_img = '/path/to/Datasets/MP3D/tools/result_episode_2azQ1b91cZZ_ep0306/data/2azQ1b91cZZ/cam/scan2azQ1b91cZZ_ep0306_g0009_sub04_i00_back_rgb.png'

    # sample_instruction = 'continue down the hallway into the bedroom.'

    # left_depth = left_img.replace('_rgb.png', '_depth.npy').replace('/cam/', '/depth/')
    # front_depth = front_img.replace('_rgb.png', '_depth.npy').replace('/cam/', '/depth/')
    # right_depth = right_img.replace('_rgb.png', '_depth.npy').replace('/cam/', '/depth/')
    # back_depth = back_img.replace('_rgb.png', '_depth.npy').replace('/cam/', '/depth/')





    # nouns = noun_phrases(sample_instruction)
    # print(nouns)

    # def save_boxed_image(image_path, boxes, phrases):
    #     img = Image.open(image_path).convert("RGB")
    #     draw = ImageDraw.Draw(img)
    #     for box, phrase in zip(boxes, phrases):
    #         x0, y0, x1, y1 = [int(v) for v in box]
    #         draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
    #         draw.text((x0, max(0, y0 - 12)), phrase, fill="red")
    #     out_path = Path(__file__).with_name(f"{Path(image_path).stem}_boxed.jpg")
    #     img.save(out_path)
    #     print(f"Saved: {out_path}")

    # boxer = GroundingDINOBoxer()

    # for img_path in [left_img, front_img, right_img, back_img]:
    #     boxes, phrases = boxer.predict_boxes(
    #         image_path=img_path,
    #         text_prompt=". ".join(nouns),
    #         box_threshold=0.3,
    #         text_threshold=0.25,
    #     )
    #     save_boxed_image(img_path, boxes, phrases)
    #     print(f"Image: {img_path}")
    #     for box, phrase in zip(boxes, phrases):
    #         print(f" Box: {box}, Phrase: {phrase}")
    #     print()

    # converter = PointConverter()
    # pipeline = BoxPipeline(boxer, converter)
    # views = [("left", left_img, left_depth),
    #          ("front", front_img, front_depth),
    #          ("right", right_img, right_depth),
    #          ("back", back_img, back_depth)]
    # #points = pipeline.run(views, ". ".join(nouns))
    # points, selected = pipeline.run(views, sample_instruction, target_xy=target_xy)
    # # selected is ordered; use selected[0], selected[1] if you need the first two
    # for p in points:
    #     print(f"Point: label={p['label']}, xyz={p['xyz']}, area={p['area']}")
    # pipeline.plot(points, Path(__file__).with_name("boxed_points.html"), target_xy=target_xy)