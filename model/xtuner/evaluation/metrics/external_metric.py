import os, numpy as np, torch
from mmengine.evaluator import BaseMetric
import math
import heapq

# the one used for vqa and robopoint
class ExternalMetric_QA(BaseMetric):
    def __init__(self, save_path=None, flat_output=True, **kwargs):
        super().__init__(**kwargs)
        self.save_path = save_path
        self.flat_output = flat_output

    def process(self, data_batch, data_samples):
        data = data_batch["data"]
        preds = data_samples
        if isinstance(preds, dict):
            preds = [preds]  # val loop returns dict per batch

        for i, pred in enumerate(preds):
            pred_xy = pred["p_pred"].detach().cpu().float()
            loss_ce = pred.get("loss_ce", torch.tensor(0.0)).detach().cpu().float()
            loss_reg = pred.get("loss_reg", torch.tensor(0.0)).detach().cpu().float()
            print(f"Eval sample {i} CE loss: {loss_ce.item():.4f}")
            print(f"Eval sample {i} Reg loss: {loss_reg.item():.4f}")
            cam_h = data["cam_height"][i].detach().cpu().float()
            # z comes from cam height when flat_output=True
            z = -cam_h if self.flat_output else torch.tensor(0.0)
            p_pred = torch.cat([pred_xy, z.view(1)]).numpy()

            self.results.append({
                "p_pred": p_pred,
                "p_goal": data["p_goal"][i].detach().cpu().numpy(),           # (3,)
                "ego2global_xfwd": data["ego2global"][i].detach().cpu().numpy(),  # (4,4)
                "cam_height": cam_h.item(),
                "scene_token": data["scene"][i],
                "token": data["token"][i],
                "ep_idx": int(data["ep_idx"][i]),
                "succ1_dir": pred.get("succ1_dir", None),
                'succ1_dir_and_size': pred.get('succ1_dir_and_size', None),
                "succ0_dir": pred.get("succ0_dir", None),
                'succ0_dir_and_size': pred.get('succ0_dir_and_size', None),
                "succ2_dir": pred.get("succ2_dir", None),
                "path_pred": pred.get("path_pred", None),
                "path_gt": data['waypoints'][i].detach().cpu().numpy(), # 6 x 3
            })

    def compute_metrics(self, results):
        path = self.save_path or os.path.join(self.work_dir, "val_preds.npz")
        np.savez(
            path,
            p_pred=np.stack([r["p_pred"] for r in results]),
            p_goal=np.stack([r["p_goal"] for r in results]),
            ego2global_xfwd=np.stack([r["ego2global_xfwd"] for r in results]),
            cam_height=np.array([r["cam_height"] for r in results], dtype=np.float32),
            scene_token=np.array([r["scene_token"] for r in results], dtype=object),
            token=np.array([r["token"] for r in results], dtype=object),
            ep_idx=np.array([r["ep_idx"] for r in results], dtype=np.int64),
            # waypoints_pred=np.stack([r["path_pred"] for r in results if r.get("path_pred", None) is not None]),
        )
        print(f"Saved predictions to {path}")
        
        # evaluate succ_1_dir if available
        succ1_dir_list = [r['succ1_dir'] for r in results if r['succ1_dir'] is not None]
        succ1_dir_and_size_list = [r['succ1_dir_and_size'] for r in results if r['succ1_dir_and_size'] is not None]
        if len(succ1_dir_list) > 0:
            succ1_dir_array = np.array(succ1_dir_list, dtype=np.int64)
            succ1_dir_acc = np.mean(succ1_dir_array)
            print(f"Succ_1_dir accuracy: {succ1_dir_acc:.4f}")
            succ1_dir_and_size_array = np.array(succ1_dir_and_size_list, dtype=np.int64)
            succ1_dir_and_size_acc = np.mean(succ1_dir_and_size_array)
            print(f"Succ_1_dir_and_size accuracy: {succ1_dir_and_size_acc:.4f}")
        
        succ0_dir_list = [r['succ0_dir'] for r in results if r['succ0_dir'] is not None]
        succ0_dir_and_size_list = [r['succ0_dir_and_size'] for r in results if r['succ0_dir_and_size'] is not None]
        if len(succ0_dir_list) > 0:
            succ0_dir_array = np.array(succ0_dir_list, dtype=np.int64)
            succ0_dir_acc = np.mean(succ0_dir_array)
            print(f"Succ_0_dir accuracy: {succ0_dir_acc:.4f}")
            succ0_dir_and_size_array = np.array(succ0_dir_and_size_list, dtype=np.int64)
            succ0_dir_and_size_acc = np.mean(succ0_dir_and_size_array)
            print(f"Succ_0_dir_and_size accuracy: {succ0_dir_and_size_acc:.4f}")

        succ2_dir_list = [r['succ2_dir'] for r in results if r.get('succ2_dir', None) is not None]
        if len(succ2_dir_list) > 0:
            succ2_dir_array = np.array(succ2_dir_list, dtype=np.int64)
            succ2_dir_acc = np.mean(succ2_dir_array)
            print(f"Succ_2_dir accuracy: {succ2_dir_acc:.4f}")

        path_pred = [r['path_pred'] for r in results if r.get('path_pred', None) is not None]
        path_gt = [r['path_gt'][:, :2] for r in results if r.get('path_pred', None) is not None]
        if len(path_pred) > 0:
            # compute l2 error
            l2_errors = []
            for pp, pg in zip(path_pred, path_gt):
                l2_error = np.linalg.norm(pp - pg, axis=1)  # per waypoint
                l2_errors.append(l2_error)
            l2_errors = np.stack(l2_errors, axis=0)  # N x num_waypoints
            mean_l2_error = np.mean(l2_errors, axis=0)
            print(f"Mean L2 error per waypoint: {mean_l2_error}")
            print(f"Overall ADE: {np.mean(mean_l2_error):.4f}")

        return {"dump_path": path}

# hit rate and collision
class ExternalMetric_v1(BaseMetric):
    # TODO: distance somehow
    def __init__(self, eps=1e-6, collect_device='cpu', prefix=None, **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.eps = eps

    def process(self, data_batch, data_samples):
        data = data_batch["data"]
        A = data['traversable_mask'].to(self.collect_device).bool()  # B,H,W
        B = data['visible_mask'].to(self.collect_device).bool()
        C = data['affordance_mask'].to(self.collect_device).bool()

        preds = data_samples
        if isinstance(preds, dict):
            preds = [preds]

        for i, pred in enumerate(preds):
            Ai, Bi, Ci = A[i], B[i], C[i]  # H,W

            pred_aff = pred['pred_aff'].to(self.collect_device).bool()  # H,W (or 1,H,W)
            if pred_aff.dim() == 3:
                pred_aff = pred_aff[0]

            pred_xy = pred['pred_xy'].to(self.collect_device).float()  # 2 or N x 2
            if pred_xy.dim() == 1:
                pred_xy = pred_xy.view(1, 2)

            gt = (Ai & Ci) & Bi
            p = pred_aff & Bi
            tp = (p & gt).sum()
            fp = (p & (~gt) & Bi).sum()
            fn = ((~p) & gt).sum()

            x = torch.round((pred_xy[:, 0] + 6.4 - 0.05) / 0.1).long()
            y = torch.round((pred_xy[:, 1] + 6.4 - 0.05) / 0.1).long()
            H, W = Ai.shape
            inb = (x >= 0) & (x < H) & (y >= 0) & (y < W)
            x = x[inb]; y = y[inb]

            # known = Bi[x, y]
            known = torch.ones_like(Bi[x, y], dtype=torch.bool) # all known for now
            hit = (known & Ai[x, y] & Ci[x, y]).sum()
            coll = (known & (~Ai[x, y])).sum()
            known_num = known.sum()

            self.results.append(dict(
                tp=tp.cpu(), fp=fp.cpu(), fn=fn.cpu(),
                hit_num=hit.cpu(), coll_num=coll.cpu(), known_num=known_num.cpu(),
            ))

    def compute_metrics(self, results):
        tp = sum(r['tp'] for r in results)
        fp = sum(r['fp'] for r in results)
        fn = sum(r['fn'] for r in results)
        hit_num = sum(r['hit_num'] for r in results)
        coll_num = sum(r['coll_num'] for r in results)
        known_num = sum(r['known_num'] for r in results)

        iou = (tp.float() / (tp + fp + fn).float().clamp_min(self.eps)).item()
        hit_known = (hit_num.float() / known_num.float().clamp_min(self.eps)).item()
        collision_rate = (coll_num.float() / known_num.float().clamp_min(self.eps)).item()
        print(f"total hit: {hit_num}, total known: {known_num}, hit_known: {hit_known:.4f}")
        return dict(iou=iou, hit_known=hit_known, collision_rate=collision_rate)

# hit rate and collision with geodesic distance computed online
class ExternalMetric_v2(BaseMetric):
    # TODO: distance somehow
    def __init__(self, 
                 
                 eps=1e-6, 
                 collect_device='cpu', 
                 prefix=None, 
                 hit_thresholds_m=[0.5, 1.0, 1.5],
                 res=0.1,
                 half_xy=6.4,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.eps = eps
        self.hit_thresholds_m = hit_thresholds_m
        self.res = res
        self.half_xy = half_xy

    @staticmethod
    def _bounded_geodesic_dist_cells(Ai_cpu: torch.Tensor, si: int, sj: int, rmax_cells: float) -> torch.Tensor:
        """
        Ai_cpu: (H,W) bool on CPU. True = traversable.
        si,sj: seed indices (row=i, col=j)
        rmax_cells: max radius in cell-units (cardinal=1, diagonal=sqrt(2))
        Returns dist (H,W) float32 on CPU, INF for unreachable/outside rmax.
        """
        H, W = Ai_cpu.shape
        INF = 1e9
        dist = torch.full((H, W), INF, dtype=torch.float32)
        visited = torch.zeros((H, W), dtype=torch.bool)

        dist[si, sj] = 0.0
        pq = [(0.0, si, sj)]

        SQ2 = math.sqrt(2.0)
        moves = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2),
        ]

        while pq:
            d, y, x = heapq.heappop(pq)
            if visited[y, x]:
                continue
            visited[y, x] = True
            if d > rmax_cells:
                break

            for dy, dx, w in moves:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                if not Ai_cpu[ny, nx]:
                    continue

                # strict diagonal gating
                if dy != 0 and dx != 0:
                    if not (Ai_cpu[y, nx] and Ai_cpu[ny, x]):
                        continue

                nd = d + w
                if nd <= rmax_cells and nd < float(dist[ny, nx]):
                    dist[ny, nx] = float(nd)
                    heapq.heappush(pq, (float(nd), ny, nx))

        return dist


    def process(self, data_batch, data_samples):
        data = data_batch["data"]
        A = data['traversable_mask'].to(self.collect_device).bool()  # B,H,W
        B = data['visible_mask'].to(self.collect_device).bool()
        C = data['affordance_mask'].to(self.collect_device).bool()
        p_goal = data["p_goal"].to(self.collect_device).float()  # B,3 (x_fwd, y_left, z_up)

        preds = data_samples
        if isinstance(preds, dict):
            preds = [preds]

        for i, pred in enumerate(preds):
            Ai, Bi, Ci = A[i], B[i], C[i]  # H,W

            pred_aff = pred['pred_aff'].to(self.collect_device).bool()  # H,W (or 1,H,W)
            if pred_aff.dim() == 3:
                pred_aff = pred_aff[0]

            pred_xy = pred['pred_xy'].to(self.collect_device).float()  # 2 or N x 2
            if pred_xy.dim() == 1:
                pred_xy = pred_xy.view(1, 2)

            gt = (Ai & Ci) & Bi
            p = pred_aff & Bi
            tp = (p & gt).sum()
            fp = (p & (~gt) & Bi).sum()
            fn = ((~p) & gt).sum()

            x = torch.round((pred_xy[:, 0] + 6.4 - 0.05) / 0.1).long()
            y = torch.round((pred_xy[:, 1] + 6.4 - 0.05) / 0.1).long()
            H, W = Ai.shape
            inb = (x >= 0) & (x < H) & (y >= 0) & (y < W)
            x = x[inb]; y = y[inb]

            # ---- NEW: bounded geodesic dist from p_goal (compute once per sample) ----
            thr_m = self.hit_thresholds_m
            rmax_m = max(thr_m)
            rmax_cells = float(rmax_m / self.res)

            # seed indices from p_goal (offset is half of resolution => 0.5*self.res)
            seed_i = int(torch.round((p_goal[i, 0] + self.half_xy - 0.5 * self.res) / self.res).item())  # row (H), forward
            seed_j = int(torch.round((p_goal[i, 1] + self.half_xy - 0.5 * self.res) / self.res).item())  # col (W), left

            # compute dist on CPU (small neighborhood), then bring back to collect_device
            Ai_cpu = Ai.detach().to("cpu")
            dist_cpu = self._bounded_geodesic_dist_cells(Ai_cpu, seed_i, seed_j, rmax_cells)  # cell units
            dist = dist_cpu.to(self.collect_device)
            # ------------------------------------------------------------------------

            # # debug
            # debug_save_path = 'debug_outputs/1_metric_disc.npz'
            # np.savez(
            #     debug_save_path,
            #     traversable = Ai_cpu.numpy(),
            #     visible = Bi.detach().to("cpu").numpy(),
            #     affordance = Ci.detach().to("cpu").numpy(),
            #     disc_05 = (dist_cpu.numpy() <= (0.5 / self.res)).astype(np.uint8),
            #     disc_10 = (dist_cpu.numpy() <= (1.0 / self.res)).astype(np.uint8),
            #     disc_15 = (dist_cpu.numpy() <= (1.5 / self.res)).astype(np.uint8),
            # )
            # exit(0)
            # # end of debug

            # known = Bi[x, y]
            known = torch.ones_like(Bi[x, y], dtype=torch.bool) # all known for now
            # hit = (known & Ai[x, y] & Ci[x, y]).sum()
            hit_05 = (known & Ai[x, y] & (dist[x, y] <= (0.5 / self.res))).sum()
            hit_10 = (known & Ai[x, y] & (dist[x, y] <= (1.0 / self.res))).sum()
            hit_15 = (known & Ai[x, y] & (dist[x, y] <= (1.5 / self.res))).sum()
            coll = (known & (~Ai[x, y])).sum()
            known_num = known.sum()

            self.results.append(dict(
                tp=tp.cpu(), fp=fp.cpu(), fn=fn.cpu(),
                # hit_num=hit.cpu(), 
                hit_num_05=hit_05.cpu(),
                hit_num_10=hit_10.cpu(),
                hit_num_15=hit_15.cpu(),
                coll_num=coll.cpu(), 
                known_num=known_num.cpu(),
            ))

    def compute_metrics(self, results):
        tp = sum(r['tp'] for r in results)
        fp = sum(r['fp'] for r in results)
        fn = sum(r['fn'] for r in results)
        # hit_num = sum(r['hit_num'] for r in results)
        hit_05 = sum(r["hit_num_05"] for r in results)
        hit_10 = sum(r["hit_num_10"] for r in results)
        hit_15 = sum(r["hit_num_15"] for r in results)
        coll_num = sum(r['coll_num'] for r in results)
        known_num = sum(r['known_num'] for r in results)

        iou = (tp.float() / (tp + fp + fn).float().clamp_min(self.eps)).item()
        # hit_known = (hit_num.float() / known_num.float().clamp_min(self.eps)).item()
        collision_rate = (coll_num.float() / known_num.float().clamp_min(self.eps)).item()
        # print(f"total hit: {hit_num}, total known: {known_num}, hit_known: {hit_known:.4f}")
        # return dict(iou=iou, hit_known=hit_known, collision_rate=collision_rate)
        hit_at_05 = (hit_05.float() / known_num.float().clamp_min(self.eps)).item()
        hit_at_10 = (hit_10.float() / known_num.float().clamp_min(self.eps)).item()
        hit_at_15 = (hit_15.float() / known_num.float().clamp_min(self.eps)).item()
        hit_mean = (hit_at_05 + hit_at_10 + hit_at_15) / 3.0

        print(
            f"hit@0.5={hit_at_05:.4f}, hit@1.0={hit_at_10:.4f}, hit@1.5={hit_at_15:.4f}, "
            f"hit_mean={hit_mean:.4f} | total known: {known_num}"
        )

        return dict(
            iou=iou,
            hit_at_0_5=hit_at_05,
            hit_at_1_0=hit_at_10,
            hit_at_1_5=hit_at_15,
            hit_mean=hit_mean,
            collision_rate=collision_rate,
        )

# hit rate and collision with geodesic distance computed online, with occlusion group reported separately
class ExternalMetric_v3(BaseMetric):
    # TODO: distance somehow
    def __init__(self, 
                 eps=1e-6, 
                 collect_device='cpu', 
                 prefix=None, 
                 hit_thresholds_m=[0.5, 1.0, 1.5],
                 res=0.1,
                 half_xy=6.4,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.eps = eps
        self.hit_thresholds_m = hit_thresholds_m
        self.res = res
        self.half_xy = half_xy

    @staticmethod
    def _bounded_geodesic_dist_cells(Ai_cpu: torch.Tensor, si: int, sj: int, rmax_cells: float) -> torch.Tensor:
        """
        Ai_cpu: (H,W) bool on CPU. True = traversable.
        si,sj: seed indices (row=i, col=j)
        rmax_cells: max radius in cell-units (cardinal=1, diagonal=sqrt(2))
        Returns dist (H,W) float32 on CPU, INF for unreachable/outside rmax.
        """
        H, W = Ai_cpu.shape
        INF = 1e9
        dist = torch.full((H, W), INF, dtype=torch.float32)
        visited = torch.zeros((H, W), dtype=torch.bool)

        dist[si, sj] = 0.0
        pq = [(0.0, si, sj)]

        SQ2 = math.sqrt(2.0)
        moves = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2),
        ]

        while pq:
            d, y, x = heapq.heappop(pq)
            if visited[y, x]:
                continue
            visited[y, x] = True
            if d > rmax_cells:
                break

            for dy, dx, w in moves:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                if not Ai_cpu[ny, nx]:
                    continue

                # strict diagonal gating
                if dy != 0 and dx != 0:
                    if not (Ai_cpu[y, nx] and Ai_cpu[ny, x]):
                        continue

                nd = d + w
                if nd <= rmax_cells and nd < float(dist[ny, nx]):
                    dist[ny, nx] = float(nd)
                    heapq.heappush(pq, (float(nd), ny, nx))

        return dist


    def process(self, data_batch, data_samples):
        data = data_batch["data"]
        A = data['traversable_mask'].to(self.collect_device).bool()  # B,H,W
        B = data['visible_mask'].to(self.collect_device).bool()
        C = data['affordance_mask'].to(self.collect_device).bool()
        p_goal = data["p_goal"].to(self.collect_device).float()  # B,3 (x_fwd, y_left, z_up)

        # ---- NEW: goal occlusion inputs (depth + intrinsics + cam2egos) ----
        depth = data["depth"]  # (B,4,H,W) or dict
        cam2imgs = data["cam2imgs"]
        cam2egos = data["cam2egos"]

        # handle dict case (same order as your pipeline)
        if isinstance(cam2imgs, dict):
            order = ["front", "left", "back", "right"]
            cam2imgs = torch.stack([cam2imgs[k] for k in order], dim=1)   # (B,4,4,4)
            cam2egos = torch.stack([cam2egos[k] for k in order], dim=1)   # (B,4,4,4)
            depth = torch.stack([depth[k] for k in order], dim=1) if isinstance(depth, dict) else depth

        depth = depth.to(self.collect_device).float()        # (B,4,448,448), z-depth in meters
        cam2imgs = cam2imgs.to(self.collect_device).float()
        cam2egos = cam2egos.to(self.collect_device).float()
        intrinsics = cam2imgs[:, :, :3, :3]                 # (B,4,3,3)
        # ---------------------------------------------------

        preds = data_samples
        if isinstance(preds, dict):
            preds = [preds]

        for i, pred in enumerate(preds):
            Ai, Bi, Ci = A[i], B[i], C[i]  # H,W

            pred_aff = pred['pred_aff'].to(self.collect_device).bool()  # H,W (or 1,H,W)
            if pred_aff.dim() == 3:
                pred_aff = pred_aff[0]

            pred_xy = pred['pred_xy'].to(self.collect_device).float()  # 2 or N x 2
            if pred_xy.dim() == 1:
                pred_xy = pred_xy.view(1, 2)

            gt = (Ai & Ci) & Bi
            p = pred_aff & Bi
            tp = (p & gt).sum()
            fp = (p & (~gt) & Bi).sum()
            fn = ((~p) & gt).sum()

            x = torch.round((pred_xy[:, 0] + 6.4 - 0.05) / 0.1).long()
            y = torch.round((pred_xy[:, 1] + 6.4 - 0.05) / 0.1).long()
            H, W = Ai.shape
            inb = (x >= 0) & (x < H) & (y >= 0) & (y < W)
            x = x[inb]; y = y[inb]

            # ---- NEW: bounded geodesic dist from p_goal (compute once per sample) ----
            thr_m = self.hit_thresholds_m
            rmax_m = max(thr_m)
            rmax_cells = float(rmax_m / self.res)

            # seed indices from p_goal (offset is half of resolution => 0.5*self.res)
            seed_i = int(torch.round((p_goal[i, 0] + self.half_xy - 0.5 * self.res) / self.res).item())  # row (H), forward
            seed_j = int(torch.round((p_goal[i, 1] + self.half_xy - 0.5 * self.res) / self.res).item())  # col (W), left

            # compute dist on CPU (small neighborhood), then bring back to collect_device
            Ai_cpu = Ai.detach().to("cpu")
            dist_cpu = self._bounded_geodesic_dist_cells(Ai_cpu, seed_i, seed_j, rmax_cells)  # cell units
            dist = dist_cpu.to(self.collect_device)
            # ------------------------------------------------------------------------

            # ---- NEW: occlusion flag for p_goal (per sample) ----
            tol = 0.10  # fixed tolerance in meters
            pg = p_goal[i]  # (3,) ego
            pg_h = torch.tensor([pg[0], pg[1], pg[2], 1.0], device=self.collect_device, dtype=torch.float32)
            # ego <- cam given, so cam <- ego is inverse
            T_ego_cam = cam2egos[i]                       # (4,4,4)
            T_cam_ego = torch.linalg.inv(T_ego_cam)       # (4,4,4)
            # project into each cam
            pg_cam = (T_cam_ego @ pg_h.view(4, 1)).squeeze(-1)   # (4,4) -> per cam (x,y,z,1)
            X = pg_cam[:, 0]
            Y = pg_cam[:, 1]
            Z = pg_cam[:, 2]
            # valid if in front
            front = Z > 1e-6
            # intrinsics for this sample
            K = intrinsics[i]  # (4,3,3)
            fx = K[:, 0, 0]; fy = K[:, 1, 1]
            cx = K[:, 0, 2]; cy = K[:, 1, 2]
            u = torch.round(fx * (X / Z) + cx).long()
            v = torch.round(fy * (Y / Z) + cy).long()
            Himg, Wimg = depth.shape[-2], depth.shape[-1]
            inb_img = (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg)
            valid_cam = front & inb_img
            n_valid = int(valid_cam.sum().item())
            assert n_valid <= 1, f"[ERR] p_goal visible in >1 cams (n={n_valid}) -> conversion wrong"
            is_weird = False
            if n_valid == 0:
                # If p_goal is outside all camera frustums, we only accept it as "too close"
                # when its XY is within 2m in ego frame; otherwise treat as a conversion bug.
                assert float(torch.abs(pg[0])) <= 2.0 and float(torch.abs(pg[1])) <= 2.0, \
                    f"[ERR] n_valid==0 but p_goal XY not within 2m: p_goal={pg.tolist()}"

                occluded_goal = False  # treat as not-occluded (too close / outside FOV)

            else:
                cam_idx = int(torch.nonzero(valid_cam, as_tuple=False)[0].item())
                uu = int(u[cam_idx].item())
                vv = int(v[cam_idx].item())
                zq = float(Z[cam_idx].item())

                d = depth[i, cam_idx, vv, uu]
                d_ok = torch.isfinite(d) & (d > 1e-6)

                if not bool(d_ok.item()):
                    # 3x3 neighbor fallback
                    v0 = max(0, vv - 1); v1 = min(Himg, vv + 2)
                    u0 = max(0, uu - 1); u1 = min(Wimg, uu + 2)
                    patch = depth[i, cam_idx, v0:v1, u0:u1]
                    m = torch.isfinite(patch) & (patch > 1e-6)
                    if m.any():
                        d = patch[m].mean()
                        d_ok = torch.isfinite(d) & (d > 1e-6)
                    else:
                        d_ok = torch.tensor(False, device=self.collect_device)

                if not bool(d_ok.item()):
                    occluded_goal = True
                else:
                    dq = float(d.item())
                    diff = zq - dq

                    # only abs within tol is "not occluded"
                    if abs(diff) <= tol:
                        occluded_goal = False
                    elif diff > tol:
                        # z is behind the observed surface => occluded
                        occluded_goal = True
                    else:
                        occluded_goal = False
                        print(f"[WARN] Z < depth - tol: cam={cam_idx} (u,v)=({uu},{vv}) Z={zq:.3f} d={dq:.3f} diff={diff:.3f} p_goal={pg.tolist()}")
                        is_weird = True # likely p_goal is above the mesh ground somehow
            # ------------------------------------------


            # # debug for geodesic dist
            # debug_save_path = 'debug_outputs/1_metric_disc.npz'
            # np.savez(
            #     debug_save_path,
            #     traversable = Ai_cpu.numpy(),
            #     visible = Bi.detach().to("cpu").numpy(),
            #     affordance = Ci.detach().to("cpu").numpy(),
            #     disc_05 = (dist_cpu.numpy() <= (0.5 / self.res)).astype(np.uint8),
            #     disc_10 = (dist_cpu.numpy() <= (1.0 / self.res)).astype(np.uint8),
            #     disc_15 = (dist_cpu.numpy() <= (1.5 / self.res)).astype(np.uint8),
            # )
            # exit(0)
            # # end of debug

            # # debug for occlusion
            # debug_save_path = 'debug_outputs/1_metric_occlusion.npz'
            # if n_valid == 1:
            #     cam_idx_save = int(cam_idx)
            #     uv_save = np.array([int(uu), int(vv)], dtype=np.int32)  # (u,v)
            # else:
            #     cam_idx_save = -1
            #     uv_save = np.array([-1, -1], dtype=np.int32)
            # if cam_idx_save >= 2 and is_weird:
            #     np.savez(
            #         debug_save_path,
            #         depth=depth[i].detach().to("cpu").numpy(),
            #         cam_idx=np.int32(cam_idx_save),
            #         uv=uv_save,
            #         instr=data["instruction"][i],
            #         p_goal=p_goal[i].detach().to("cpu").numpy(),
            #     )
            #     exit(0)
            # # end of debug

            # known = Bi[x, y]
            known = torch.ones_like(Bi[x, y], dtype=torch.bool) # all known for now
            # hit = (known & Ai[x, y] & Ci[x, y]).sum()
            hit_05 = (known & Ai[x, y] & (dist[x, y] <= (0.5 / self.res))).sum()
            hit_10 = (known & Ai[x, y] & (dist[x, y] <= (1.0 / self.res))).sum()
            hit_15 = (known & Ai[x, y] & (dist[x, y] <= (1.5 / self.res))).sum()
            coll = (known & (~Ai[x, y])).sum()
            known_num = known.sum()

            # ---- DEBUG DUMP: collect occluded-goal samples for visualization ----
            # (comment out this whole block to disable)
            if occluded_goal:
                # lazy init to avoid touching __init__
                if not hasattr(self, "_occ_dump"):
                    self._occ_dump = []

                # fetch raw_img + extra obs masks from data_batch
                raw_img_i = data["raw_img"][i].detach().to("cpu").numpy()  # (4,448,448,3) uint8 (BGR)
                free_ground_i = pred["free_ground_obs"].detach().to("cpu").numpy().astype(np.uint8)  # (128,128)
                free_cam_i = pred["free_cam_obs"].detach().to("cpu").numpy().astype(np.uint8)        # (128,128)

                # core masks (128,128) -> uint8 for easier viewing
                trav_i = Ai.detach().to("cpu").numpy().astype(np.uint8)
                vis_i = Bi.detach().to("cpu").numpy().astype(np.uint8)
                aff_i = Ci.detach().to("cpu").numpy().astype(np.uint8)

                # pred_xy is K=1 always; keep (2,)
                pred_xy_i = pred_xy[0].detach().to("cpu").numpy().astype(np.float32)  # (2,)
                p_goal_i = p_goal[i].detach().to("cpu").numpy().astype(np.float32)    # (3,)
                pred_aff_i = pred_aff.detach().to("cpu").numpy().astype(np.uint8)          # (128,128)

                instr_i = data["instruction"][i]  # keep as python string

                self._occ_dump.append(dict(
                    raw_img=raw_img_i,
                    traversable_mask=trav_i,
                    visible_mask=vis_i,
                    affordance_mask=aff_i,
                    free_ground_obs=free_ground_i,
                    free_cam_obs=free_cam_i,
                    pred_xy=pred_xy_i,
                    p_goal=p_goal_i,
                    instruction=instr_i,
                    pred_affordance=pred_aff_i,
                ))
            # -------------------------------------------------------------------


            # ---- NEW: subgroup sums for occluded-goal only (keep global sums unchanged) ----
            if occluded_goal:
                hit_05_occ, hit_10_occ, hit_15_occ = hit_05, hit_10, hit_15
                coll_occ, known_occ = coll, known_num
                occ_goal_num = torch.tensor(1, device=self.collect_device, dtype=torch.long)
            else:
                hit_05_occ = torch.zeros_like(hit_05)
                hit_10_occ = torch.zeros_like(hit_10)
                hit_15_occ = torch.zeros_like(hit_15)
                coll_occ = torch.zeros_like(coll)
                known_occ = torch.zeros_like(known_num)
                occ_goal_num = torch.tensor(0, device=self.collect_device, dtype=torch.long)
            # -------------------------------------------------------------------------------


            self.results.append(dict(
                tp=tp.cpu(), fp=fp.cpu(), fn=fn.cpu(),
                # hit_num=hit.cpu(), 
                hit_num_05=hit_05.cpu(),
                hit_num_10=hit_10.cpu(),
                hit_num_15=hit_15.cpu(),
                coll_num=coll.cpu(), 
                known_num=known_num.cpu(),
                occ_goal_num=occ_goal_num.cpu(),
                hit_num_05_occ=hit_05_occ.cpu(),
                hit_num_10_occ=hit_10_occ.cpu(),
                hit_num_15_occ=hit_15_occ.cpu(),
                coll_num_occ=coll_occ.cpu(),
                known_num_occ=known_occ.cpu(),
                weird_num=torch.tensor(1 if is_weird else 0, device=self.collect_device, dtype=torch.long).cpu(),
            ))

    def compute_metrics(self, results):
        tp = sum(r['tp'] for r in results)
        fp = sum(r['fp'] for r in results)
        fn = sum(r['fn'] for r in results)
        # hit_num = sum(r['hit_num'] for r in results)
        hit_05 = sum(r["hit_num_05"] for r in results)
        hit_10 = sum(r["hit_num_10"] for r in results)
        hit_15 = sum(r["hit_num_15"] for r in results)
        coll_num = sum(r['coll_num'] for r in results)
        known_num = sum(r['known_num'] for r in results)

        iou = (tp.float() / (tp + fp + fn).float().clamp_min(self.eps)).item()
        # hit_known = (hit_num.float() / known_num.float().clamp_min(self.eps)).item()
        collision_rate = (coll_num.float() / known_num.float().clamp_min(self.eps)).item()
        # print(f"total hit: {hit_num}, total known: {known_num}, hit_known: {hit_known:.4f}")
        # return dict(iou=iou, hit_known=hit_known, collision_rate=collision_rate)
        hit_at_05 = (hit_05.float() / known_num.float().clamp_min(self.eps)).item()
        hit_at_10 = (hit_10.float() / known_num.float().clamp_min(self.eps)).item()
        hit_at_15 = (hit_15.float() / known_num.float().clamp_min(self.eps)).item()
        hit_mean = (hit_at_05 + hit_at_10 + hit_at_15) / 3.0

        print(
            f"hit@0.5={hit_at_05:.4f}, hit@1.0={hit_at_10:.4f}, hit@1.5={hit_at_15:.4f}, "
            f"hit_mean={hit_mean:.4f} | total known: {known_num}"
        )

        # ---- NEW: occluded-goal subgroup metrics ----
        occ_goal_num = sum(r["occ_goal_num"] for r in results)
        total_samples = len(results)
        weird_num = sum(r.get("weird_num", 0) for r in results)
        if weird_num > 0:
            print(f"[GOAL somehow ABOVE GROUND] weird_num={int(weird_num)}/{len(results)} ({float(weird_num)/max(1,len(results)):.4f})")

        occ_ratio = float(occ_goal_num) / max(1, total_samples)

        hit_05_occ = sum(r["hit_num_05_occ"] for r in results)
        hit_10_occ = sum(r["hit_num_10_occ"] for r in results)
        hit_15_occ = sum(r["hit_num_15_occ"] for r in results)
        coll_occ = sum(r["coll_num_occ"] for r in results)
        known_occ = sum(r["known_num_occ"] for r in results)

        hit_at_05_occ = (hit_05_occ.float() / known_occ.float().clamp_min(self.eps)).item()
        hit_at_10_occ = (hit_10_occ.float() / known_occ.float().clamp_min(self.eps)).item()
        hit_at_15_occ = (hit_15_occ.float() / known_occ.float().clamp_min(self.eps)).item()
        hit_mean_occ = (hit_at_05_occ + hit_at_10_occ + hit_at_15_occ) / 3.0
        collision_rate_occ = (coll_occ.float() / known_occ.float().clamp_min(self.eps)).item()

        print(
            f"[GOAL OCC] occ_ratio={occ_ratio:.4f} (occ={int(occ_goal_num)}/{total_samples}) | "
            f"hit@0.5_occ={hit_at_05_occ:.4f}, hit@1.0_occ={hit_at_10_occ:.4f}, hit@1.5_occ={hit_at_15_occ:.4f}, "
            f"hit_mean_occ={hit_mean_occ:.4f} | coll_occ={collision_rate_occ:.4f} | known_occ={int(known_occ)}"
        )
        # -------------------------------------------

        # ---- DEBUG DUMP: write all occluded-goal samples in one NPZ ----
        # (comment out this whole block to disable)
        if hasattr(self, "_occ_dump") and len(self._occ_dump) > 0:
            out_path = os.environ.get(
                "BEACON_OCC_DUMP", "debug_outputs/2_miss_when_occ_debug.npz"
            )
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

            raw_imgs = np.stack([d["raw_img"] for d in self._occ_dump], axis=0)  # (N,4,448,448,3)
            trav = np.stack([d["traversable_mask"] for d in self._occ_dump], axis=0)  # (N,128,128)
            vis = np.stack([d["visible_mask"] for d in self._occ_dump], axis=0)       # (N,128,128)
            aff = np.stack([d["affordance_mask"] for d in self._occ_dump], axis=0)    # (N,128,128)
            free_ground = np.stack([d["free_ground_obs"] for d in self._occ_dump], axis=0)  # (N,128,128)
            free_cam = np.stack([d["free_cam_obs"] for d in self._occ_dump], axis=0)        # (N,128,128)

            pred_xy = np.stack([d["pred_xy"] for d in self._occ_dump], axis=0)  # (N,2)
            p_goal_np = np.stack([d["p_goal"] for d in self._occ_dump], axis=0) # (N,3)
            instr = np.array([d["instruction"] for d in self._occ_dump], dtype=object)  # (N,)
            pred_aff = np.stack([d["pred_affordance"] for d in self._occ_dump], axis=0)  # (N,128,128)

            np.savez_compressed(
                out_path,
                raw_img=raw_imgs,
                traversable_mask=trav,
                visible_mask=vis,
                affordance_mask=aff,
                free_ground_obs=free_ground,
                free_cam_obs=free_cam,
                pred_xy=pred_xy,
                p_goal=p_goal_np,
                instruction=instr,
                pred_affordance=pred_aff,
            )
            print(f"[DEBUG] saved debug at: {out_path} (N={len(self._occ_dump)})")
        # -----------------------------------------------------------------


        return dict(
            iou=iou,
            hit_at_0_5=hit_at_05,
            hit_at_1_0=hit_at_10,
            hit_at_1_5=hit_at_15,
            hit_mean=hit_mean,
            collision_rate=collision_rate,
            occ_goal_ratio=occ_ratio,
            hit_at_0_5_occ=hit_at_05_occ,
            hit_at_1_0_occ=hit_at_10_occ,
            hit_at_1_5_occ=hit_at_15_occ,
            hit_mean_occ=hit_mean_occ,
            collision_rate_occ=collision_rate_occ,
        )

# sem_hit, geo_hit, invalid rate, snapgeo_hit & meansnapdist
class ExternalMetric(BaseMetric):
    # TODO: distance somehow
    def __init__(self, 
                 eps=1e-6, 
                 collect_device='cpu', 
                 prefix=None, 
                 hit_thresholds_m=[0.5, 1.0, 1.5],
                 res=0.1,
                 half_xy=6.4,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.eps = eps
        self.hit_thresholds_m = hit_thresholds_m
        self.res = res
        self.half_xy = half_xy

    @staticmethod
    def _bounded_geodesic_dist_cells(Ai_cpu: torch.Tensor, si: int, sj: int, rmax_cells: float) -> torch.Tensor:
        """
        Ai_cpu: (H,W) bool on CPU. True = traversable.
        si,sj: seed indices (row=i, col=j)
        rmax_cells: max radius in cell-units (cardinal=1, diagonal=sqrt(2))
        Returns dist (H,W) float32 on CPU, INF for unreachable/outside rmax.
        """
        H, W = Ai_cpu.shape
        INF = 1e9
        dist = torch.full((H, W), INF, dtype=torch.float32)
        visited = torch.zeros((H, W), dtype=torch.bool)

        dist[si, sj] = 0.0
        pq = [(0.0, si, sj)]

        SQ2 = math.sqrt(2.0)
        moves = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2),
        ]

        while pq:
            d, y, x = heapq.heappop(pq)
            if visited[y, x]:
                continue
            visited[y, x] = True
            if d > rmax_cells:
                break

            for dy, dx, w in moves:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                if not Ai_cpu[ny, nx]:
                    continue

                # strict diagonal gating
                if dy != 0 and dx != 0:
                    if not (Ai_cpu[y, nx] and Ai_cpu[ny, x]):
                        continue

                nd = d + w
                if nd <= rmax_cells and nd < float(dist[ny, nx]):
                    dist[ny, nx] = float(nd)
                    heapq.heappush(pq, (float(nd), ny, nx))

        return dist


    def process(self, data_batch, data_samples):
        data = data_batch["data"]
        A = data['traversable_mask'].to(self.collect_device).bool()  # B,H,W
        B = data['visible_mask'].to(self.collect_device).bool()
        C = data['affordance_mask'].to(self.collect_device).bool()
        p_goal = data["p_goal"].to(self.collect_device).float()  # B,3 (x_fwd, y_left, z_up)

        # ---- NEW: goal occlusion inputs (depth + intrinsics + cam2egos) ----
        depth = data["depth"]  # (B,4,H,W) or dict
        cam2imgs = data["cam2imgs"]
        cam2egos = data["cam2egos"]

        # handle dict case (same order as your pipeline)
        if isinstance(cam2imgs, dict):
            order = ["front", "left", "back", "right"]
            cam2imgs = torch.stack([cam2imgs[k] for k in order], dim=1)   # (B,4,4,4)
            cam2egos = torch.stack([cam2egos[k] for k in order], dim=1)   # (B,4,4,4)
            depth = torch.stack([depth[k] for k in order], dim=1) if isinstance(depth, dict) else depth

        depth = depth.to(self.collect_device).float()        # (B,4,448,448), z-depth in meters
        cam2imgs = cam2imgs.to(self.collect_device).float()
        cam2egos = cam2egos.to(self.collect_device).float()
        intrinsics = cam2imgs[:, :, :3, :3]                 # (B,4,3,3)
        # ---------------------------------------------------

        preds = data_samples
        if isinstance(preds, dict):
            preds = [preds]

        for i, pred in enumerate(preds):
            Ai, Bi, Ci = A[i], B[i], C[i]  # H,W

            pred_aff = pred['pred_aff'].to(self.collect_device).bool()  # H,W (or 1,H,W)
            if pred_aff.dim() == 3:
                pred_aff = pred_aff[0]

            pred_xy = pred['pred_xy'].to(self.collect_device).float()  # 2 or N x 2
            if pred_xy.dim() == 1:
                pred_xy = pred_xy.view(1, 2)
            
            # ---- NEW: SemHit (Euclidean in meters, independent of traversability) ----
            gt_xy_m = p_goal[i, 0:2]  # (2,) in meters (x_fwd, y_left)
            diff_xy = pred_xy - gt_xy_m.view(1, 2)  # (K,2)
            euc_dist_m = torch.sqrt((diff_xy ** 2).sum(dim=1) + self.eps)  # (K,)
            sem_hit_05 = (euc_dist_m <= 0.5).sum()
            sem_hit_10 = (euc_dist_m <= 1.0).sum()
            sem_hit_15 = (euc_dist_m <= 1.5).sum()
            # ------------------------------------------------------------------------


            gt = (Ai & Ci) & Bi
            p = pred_aff & Bi
            tp = (p & gt).sum()
            fp = (p & (~gt) & Bi).sum()
            fn = ((~p) & gt).sum()

            x = torch.round((pred_xy[:, 0] + 6.4 - 0.05) / 0.1).long()
            y = torch.round((pred_xy[:, 1] + 6.4 - 0.05) / 0.1).long()
            H, W = Ai.shape
            inb = (x >= 0) & (x < H) & (y >= 0) & (y < W)

            # clamp for safe indexing; keep inb to mark OOB as invalid later
            x_clamp = x.clamp(0, H - 1)
            y_clamp = y.clamp(0, W - 1)


            # ---- NEW: bounded geodesic dist from p_goal (compute once per sample) ----
            thr_m = self.hit_thresholds_m
            rmax_m = max(thr_m)
            rmax_cells = float(rmax_m / self.res)

            # seed indices from p_goal (offset is half of resolution => 0.5*self.res)
            seed_i = int(torch.round((p_goal[i, 0] + self.half_xy - 0.5 * self.res) / self.res).item())  # row (H), forward
            seed_j = int(torch.round((p_goal[i, 1] + self.half_xy - 0.5 * self.res) / self.res).item())  # col (W), left

            # ---- NEW: clamp seed to valid grid bounds ----
            seed_i = max(0, min(H - 1, seed_i))
            seed_j = max(0, min(W - 1, seed_j))
            # --------------------------------------------


            # compute dist on CPU (small neighborhood), then bring back to collect_device
            Ai_cpu = Ai.detach().to("cpu")
            # ---- NEW: if seed is not traversable, snap seed to nearest traversable cell ----
            if not bool(Ai_cpu[seed_i, seed_j].item()):
                trav_idx_seed = torch.nonzero(Ai_cpu, as_tuple=False)  # (M,2)
                if trav_idx_seed.numel() > 0:
                    di = trav_idx_seed[:, 0].float() - float(seed_i)
                    dj = trav_idx_seed[:, 1].float() - float(seed_j)
                    d2 = di * di + dj * dj
                    arg = int(torch.argmin(d2).item())
                    seed_i = int(trav_idx_seed[arg, 0].item())
                    seed_j = int(trav_idx_seed[arg, 1].item())
            # ------------------------------------------------------------------------------
            dist_cpu = self._bounded_geodesic_dist_cells(Ai_cpu, seed_i, seed_j, rmax_cells)  # cell units
            dist = dist_cpu.to(self.collect_device)
            # ------------------------------------------------------------------------

            # ---- NEW: occlusion flag for p_goal (per sample) ----
            tol = 0.10  # fixed tolerance in meters
            pg = p_goal[i]  # (3,) ego
            pg_h = torch.tensor([pg[0], pg[1], pg[2], 1.0], device=self.collect_device, dtype=torch.float32)
            # ego <- cam given, so cam <- ego is inverse
            T_ego_cam = cam2egos[i]                       # (4,4,4)
            T_cam_ego = torch.linalg.inv(T_ego_cam)       # (4,4,4)
            # project into each cam
            pg_cam = (T_cam_ego @ pg_h.view(4, 1)).squeeze(-1)   # (4,4) -> per cam (x,y,z,1)
            X = pg_cam[:, 0]
            Y = pg_cam[:, 1]
            Z = pg_cam[:, 2]
            # valid if in front
            front = Z > 1e-6
            # intrinsics for this sample
            K = intrinsics[i]  # (4,3,3)
            fx = K[:, 0, 0]; fy = K[:, 1, 1]
            cx = K[:, 0, 2]; cy = K[:, 1, 2]
            u = torch.round(fx * (X / Z) + cx).long()
            v = torch.round(fy * (Y / Z) + cy).long()
            Himg, Wimg = depth.shape[-2], depth.shape[-1]
            inb_img = (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg)
            valid_cam = front & inb_img
            n_valid = int(valid_cam.sum().item())
            assert n_valid <= 1, f"[ERR] p_goal visible in >1 cams (n={n_valid}) -> conversion wrong"
            is_weird = False
            if n_valid == 0:
                # If p_goal is outside all camera frustums, we only accept it as "too close"
                # when its XY is within 2m in ego frame; otherwise treat as a conversion bug.
                assert float(torch.abs(pg[0])) <= 2.0 and float(torch.abs(pg[1])) <= 2.0, \
                    f"[ERR] n_valid==0 but p_goal XY not within 2m: p_goal={pg.tolist()}"

                occluded_goal = False  # treat as not-occluded (too close / outside FOV)

            else:
                cam_idx = int(torch.nonzero(valid_cam, as_tuple=False)[0].item())
                uu = int(u[cam_idx].item())
                vv = int(v[cam_idx].item())
                zq = float(Z[cam_idx].item())

                d = depth[i, cam_idx, vv, uu]
                d_ok = torch.isfinite(d) & (d > 1e-6)

                if not bool(d_ok.item()):
                    # 3x3 neighbor fallback
                    v0 = max(0, vv - 1); v1 = min(Himg, vv + 2)
                    u0 = max(0, uu - 1); u1 = min(Wimg, uu + 2)
                    patch = depth[i, cam_idx, v0:v1, u0:u1]
                    m = torch.isfinite(patch) & (patch > 1e-6)
                    if m.any():
                        d = patch[m].mean()
                        d_ok = torch.isfinite(d) & (d > 1e-6)
                    else:
                        d_ok = torch.tensor(False, device=self.collect_device)

                if not bool(d_ok.item()):
                    occluded_goal = True
                else:
                    dq = float(d.item())
                    diff = zq - dq

                    # only abs within tol is "not occluded"
                    if abs(diff) <= tol:
                        occluded_goal = False
                    elif diff > tol:
                        # z is behind the observed surface => occluded
                        occluded_goal = True
                    else:
                        occluded_goal = False
                        print(f"[WARN] Z < depth - tol: cam={cam_idx} (u,v)=({uu},{vv}) Z={zq:.3f} d={dq:.3f} diff={diff:.3f} p_goal={pg.tolist()}")
                        is_weird = True # likely p_goal is above the mesh ground somehow
            # ------------------------------------------


            # # debug for geodesic dist
            # debug_save_path = 'debug_outputs/1_metric_disc.npz'
            # np.savez(
            #     debug_save_path,
            #     traversable = Ai_cpu.numpy(),
            #     visible = Bi.detach().to("cpu").numpy(),
            #     affordance = Ci.detach().to("cpu").numpy(),
            #     disc_05 = (dist_cpu.numpy() <= (0.5 / self.res)).astype(np.uint8),
            #     disc_10 = (dist_cpu.numpy() <= (1.0 / self.res)).astype(np.uint8),
            #     disc_15 = (dist_cpu.numpy() <= (1.5 / self.res)).astype(np.uint8),
            # )
            # exit(0)
            # # end of debug

            # # debug for occlusion
            # debug_save_path = 'debug_outputs/1_metric_occlusion.npz'
            # if n_valid == 1:
            #     cam_idx_save = int(cam_idx)
            #     uv_save = np.array([int(uu), int(vv)], dtype=np.int32)  # (u,v)
            # else:
            #     cam_idx_save = -1
            #     uv_save = np.array([-1, -1], dtype=np.int32)
            # if cam_idx_save >= 2 and is_weird:
            #     np.savez(
            #         debug_save_path,
            #         depth=depth[i].detach().to("cpu").numpy(),
            #         cam_idx=np.int32(cam_idx_save),
            #         uv=uv_save,
            #         instr=data["instruction"][i],
            #         p_goal=p_goal[i].detach().to("cpu").numpy(),
            #     )
            #     exit(0)
            # # end of debug

            # invalid = OOB OR non-traversable
            invalid = (~inb) | (~Ai[x_clamp, y_clamp])

            # GeoHit@r: only counts if in-bounds + traversable + within geodesic threshold
            hit_05 = ((~invalid) & (dist[x_clamp, y_clamp] <= (0.5 / self.res))).sum()
            hit_10 = ((~invalid) & (dist[x_clamp, y_clamp] <= (1.0 / self.res))).sum()
            hit_15 = ((~invalid) & (dist[x_clamp, y_clamp] <= (1.5 / self.res))).sum()

            # InvalidRate numerator (was collision_rate)
            coll = invalid.sum()

            # denominator: number of predicted points (K)
            known_num = torch.tensor(pred_xy.shape[0], device=self.collect_device, dtype=torch.long)

            # ---- NEW: invalid count for SnapDist invalid-only mean ----
            invalid_count = coll.clone()
            # ----------------------------------------------------------


            # ---- NEW: Snap to nearest traversable cell (Ai==1), then SnapGeoHit + SnapDist ----
            Ai_cpu = Ai.detach().to("cpu")
            x0_cpu = x_clamp.detach().to("cpu").long()
            y0_cpu = y_clamp.detach().to("cpu").long()

            # precompute all traversable indices once per sample (CPU)
            trav_idx = torch.nonzero(Ai_cpu, as_tuple=False)  # (M,2) with (i,j)
            has_trav = trav_idx.numel() > 0

            xs_list = []
            ys_list = []

            if has_trav:
                # For each prediction: pick nearest traversable cell in Euclidean grid distance
                for kk in range(x0_cpu.shape[0]):
                    i0 = int(x0_cpu[kk].item())
                    j0 = int(y0_cpu[kk].item())
                    di = trav_idx[:, 0].float() - float(i0)
                    dj = trav_idx[:, 1].float() - float(j0)
                    d2 = di * di + dj * dj
                    arg = int(torch.argmin(d2).item())
                    xs_list.append(int(trav_idx[arg, 0].item()))
                    ys_list.append(int(trav_idx[arg, 1].item()))
            else:
                # degenerate: no traversable cells, keep clamped as "snapped"
                for kk in range(x0_cpu.shape[0]):
                    xs_list.append(int(x0_cpu[kk].item()))
                    ys_list.append(int(y0_cpu[kk].item()))

            xs = torch.tensor(xs_list, device=self.collect_device, dtype=torch.long)
            ys = torch.tensor(ys_list, device=self.collect_device, dtype=torch.long)

            # SnapGeoHit@r (geodesic evaluated at snapped cell)
            snap_hit_05 = (dist[xs, ys] <= (0.5 / self.res)).sum()
            snap_hit_10 = (dist[xs, ys] <= (1.0 / self.res)).sum()
            snap_hit_15 = (dist[xs, ys] <= (1.5 / self.res)).sum()

            # SnapDist: straight Euclidean in meters from original pred_xy to snapped cell center
            # cell center inverse mapping consistent with your meters->cell quantization
            xs_m = (xs.float() * self.res) - self.half_xy + 0.5 * self.res
            ys_m = (ys.float() * self.res) - self.half_xy + 0.5 * self.res
            snap_dx = pred_xy[:, 0] - xs_m
            snap_dy = pred_xy[:, 1] - ys_m
            snap_dist = torch.sqrt(snap_dx * snap_dx + snap_dy * snap_dy + self.eps)  # (K,)

            # ---- NEW: only invalid/OOB contribute to snap distance; valid -> 0 ----
            snap_dist = snap_dist * invalid.to(snap_dist.dtype)
            # ---------------------------------------------------------------

            snap_dist_sum = snap_dist.sum()

            # total predicted points (K) — keep if you still want "expected correction per query"
            snap_count = torch.tensor(pred_xy.shape[0], device=self.collect_device, dtype=torch.long)

            # invalid-only count for invalid-only mean
            snap_invalid_count = invalid_count.clone()
            # -------------------------------------------------------------------------------


            # ---- DEBUG DUMP: collect occluded-goal samples for visualization ----
            # (comment out this whole block to disable)
            if occluded_goal:
                # lazy init to avoid touching __init__
                if not hasattr(self, "_occ_dump"):
                    self._occ_dump = []

                # fetch raw_img + extra obs masks from data_batch
                raw_img_i = data["raw_img"][i].detach().to("cpu").numpy()  # (4,448,448,3) uint8 (BGR)
                free_ground_i = pred["free_ground_obs"].detach().to("cpu").numpy().astype(np.uint8)  # (128,128)
                free_cam_i = pred["free_cam_obs"].detach().to("cpu").numpy().astype(np.uint8)        # (128,128)

                # core masks (128,128) -> uint8 for easier viewing
                trav_i = Ai.detach().to("cpu").numpy().astype(np.uint8)
                vis_i = Bi.detach().to("cpu").numpy().astype(np.uint8)
                aff_i = Ci.detach().to("cpu").numpy().astype(np.uint8)

                # pred_xy is K=1 always; keep (2,)
                pred_xy_i = pred_xy[0].detach().to("cpu").numpy().astype(np.float32)  # (2,)
                p_goal_i = p_goal[i].detach().to("cpu").numpy().astype(np.float32)    # (3,)
                pred_aff_i = pred_aff.detach().to("cpu").numpy().astype(np.uint8)          # (128,128)

                instr_i = data["instruction"][i]  # keep as python string

                self._occ_dump.append(dict(
                    raw_img=raw_img_i,
                    traversable_mask=trav_i,
                    visible_mask=vis_i,
                    affordance_mask=aff_i,
                    free_ground_obs=free_ground_i,
                    free_cam_obs=free_cam_i,
                    pred_xy=pred_xy_i,
                    p_goal=p_goal_i,
                    instruction=instr_i,
                    pred_affordance=pred_aff_i,
                ))
            # -------------------------------------------------------------------


            # ---- NEW: subgroup sums for occluded-goal only (keep global sums unchanged) ----
            if occluded_goal:
                hit_05_occ, hit_10_occ, hit_15_occ = hit_05, hit_10, hit_15
                coll_occ, known_occ = coll, known_num

                sem_hit_05_occ, sem_hit_10_occ, sem_hit_15_occ = sem_hit_05, sem_hit_10, sem_hit_15
                snap_hit_05_occ, snap_hit_10_occ, snap_hit_15_occ = snap_hit_05, snap_hit_10, snap_hit_15
                snap_dist_sum_occ, snap_count_occ = snap_dist_sum, snap_count

                # ---- NEW: invalid-only count for SnapDist invalid-only mean (occ subset) ----
                snap_invalid_count_occ = snap_invalid_count
                # ---------------------------------------------------------------------------

                occ_goal_num = torch.tensor(1, device=self.collect_device, dtype=torch.long)
            else:
                hit_05_occ = torch.zeros_like(hit_05)
                hit_10_occ = torch.zeros_like(hit_10)
                hit_15_occ = torch.zeros_like(hit_15)
                coll_occ = torch.zeros_like(coll)
                known_occ = torch.zeros_like(known_num)

                sem_hit_05_occ = torch.zeros_like(sem_hit_05)
                sem_hit_10_occ = torch.zeros_like(sem_hit_10)
                sem_hit_15_occ = torch.zeros_like(sem_hit_15)
                snap_hit_05_occ = torch.zeros_like(snap_hit_05)
                snap_hit_10_occ = torch.zeros_like(snap_hit_10)
                snap_hit_15_occ = torch.zeros_like(snap_hit_15)
                snap_dist_sum_occ = torch.zeros_like(snap_dist_sum)
                snap_count_occ = torch.zeros_like(snap_count)

                # ---- NEW: invalid-only count for SnapDist invalid-only mean (occ subset) ----
                snap_invalid_count_occ = torch.zeros_like(snap_invalid_count)
                # ---------------------------------------------------------------------------

                occ_goal_num = torch.tensor(0, device=self.collect_device, dtype=torch.long)

            # -------------------------------------------------------------------------------


            self.results.append(dict(
                tp=tp.cpu(), fp=fp.cpu(), fn=fn.cpu(),
                # hit_num=hit.cpu(), 
                hit_num_05=hit_05.cpu(),
                hit_num_10=hit_10.cpu(),
                hit_num_15=hit_15.cpu(),
                coll_num=coll.cpu(), 
                known_num=known_num.cpu(),
                occ_goal_num=occ_goal_num.cpu(),
                hit_num_05_occ=hit_05_occ.cpu(),
                hit_num_10_occ=hit_10_occ.cpu(),
                hit_num_15_occ=hit_15_occ.cpu(),
                coll_num_occ=coll_occ.cpu(),
                known_num_occ=known_occ.cpu(),
                weird_num=torch.tensor(1 if is_weird else 0, device=self.collect_device, dtype=torch.long).cpu(),
                
                # NEW
                sem_hit_num_05=sem_hit_05.cpu(),
                sem_hit_num_10=sem_hit_10.cpu(),
                sem_hit_num_15=sem_hit_15.cpu(),

                snap_hit_num_05=snap_hit_05.cpu(),
                snap_hit_num_10=snap_hit_10.cpu(),
                snap_hit_num_15=snap_hit_15.cpu(),
                snap_dist_sum=snap_dist_sum.cpu(),
                snap_count=snap_count.cpu(),
                snap_invalid_count=snap_invalid_count.cpu(),

                sem_hit_num_05_occ=sem_hit_05_occ.cpu(),
                sem_hit_num_10_occ=sem_hit_10_occ.cpu(),
                sem_hit_num_15_occ=sem_hit_15_occ.cpu(),

                snap_hit_num_05_occ=snap_hit_05_occ.cpu(),
                snap_hit_num_10_occ=snap_hit_10_occ.cpu(),
                snap_hit_num_15_occ=snap_hit_15_occ.cpu(),
                snap_dist_sum_occ=snap_dist_sum_occ.cpu(),
                snap_count_occ=snap_count_occ.cpu(),
                snap_invalid_count_occ=snap_invalid_count_occ.cpu(),
            ))

    def compute_metrics(self, results):
        tp = sum(r['tp'] for r in results)
        fp = sum(r['fp'] for r in results)
        fn = sum(r['fn'] for r in results)
        # hit_num = sum(r['hit_num'] for r in results)
        hit_05 = sum(r["hit_num_05"] for r in results)
        hit_10 = sum(r["hit_num_10"] for r in results)
        hit_15 = sum(r["hit_num_15"] for r in results)

        sem_05 = sum(r["sem_hit_num_05"] for r in results)
        sem_10 = sum(r["sem_hit_num_10"] for r in results)
        sem_15 = sum(r["sem_hit_num_15"] for r in results)

        snap_05 = sum(r["snap_hit_num_05"] for r in results)
        snap_10 = sum(r["snap_hit_num_10"] for r in results)
        snap_15 = sum(r["snap_hit_num_15"] for r in results)

        snap_dist_sum = sum(r["snap_dist_sum"] for r in results)
        snap_count = sum(r["snap_count"] for r in results)
        snap_invalid_count = sum(r["snap_invalid_count"] for r in results)

        coll_num = sum(r['coll_num'] for r in results)
        known_num = sum(r['known_num'] for r in results)

        iou = (tp.float() / (tp + fp + fn).float().clamp_min(self.eps)).item()
        # hit_known = (hit_num.float() / known_num.float().clamp_min(self.eps)).item()
        collision_rate = (coll_num.float() / known_num.float().clamp_min(self.eps)).item()
        # print(f"total hit: {hit_num}, total known: {known_num}, hit_known: {hit_known:.4f}")
        # return dict(iou=iou, hit_known=hit_known, collision_rate=collision_rate)
        hit_at_05 = (hit_05.float() / known_num.float().clamp_min(self.eps)).item()
        hit_at_10 = (hit_10.float() / known_num.float().clamp_min(self.eps)).item()
        hit_at_15 = (hit_15.float() / known_num.float().clamp_min(self.eps)).item()
        hit_mean = (hit_at_05 + hit_at_10 + hit_at_15) / 3.0

        sem_at_05 = (sem_05.float() / known_num.float().clamp_min(self.eps)).item()
        sem_at_10 = (sem_10.float() / known_num.float().clamp_min(self.eps)).item()
        sem_at_15 = (sem_15.float() / known_num.float().clamp_min(self.eps)).item()
        sem_mean = (sem_at_05 + sem_at_10 + sem_at_15) / 3.0

        snap_at_05 = (snap_05.float() / known_num.float().clamp_min(self.eps)).item()
        snap_at_10 = (snap_10.float() / known_num.float().clamp_min(self.eps)).item()
        snap_at_15 = (snap_15.float() / known_num.float().clamp_min(self.eps)).item()
        snap_mean = (snap_at_05 + snap_at_10 + snap_at_15) / 3.0

        # expected correction per prediction (denom = all K)
        snap_dist_mean = (snap_dist_sum.float() / snap_count.float().clamp_min(self.eps)).item()

        # invalid-only correction severity (denom = invalid count)
        snap_dist_mean_invalid = (snap_dist_sum.float() / snap_invalid_count.float().clamp_min(self.eps)).item()


        print("\n========== ExternalMetric (GLOBAL) ==========")
        print(f"[Seg/Mask]  IoU={iou:.4f}")
        print(
            f"[GeoHit]    @0.5={hit_at_05:.4f}  @1.0={hit_at_10:.4f}  @1.5={hit_at_15:.4f}  mean={hit_mean:.4f}"
        )
        print(f"[Invalid]   rate={collision_rate:.4f}   (invalid/total = {int(coll_num)}/{int(known_num)})")
        print(
            f"[SemHit]    @0.5={sem_at_05:.4f}  @1.0={sem_at_10:.4f}  @1.5={sem_at_15:.4f}  mean={sem_mean:.4f}"
        )
        print(
            f"[SnapGeo]   @0.5={snap_at_05:.4f}  @1.0={snap_at_10:.4f}  @1.5={snap_at_15:.4f}  mean={snap_mean:.4f}"
        )
        print(f"[SnapDist]  invalid_mean={snap_dist_mean_invalid:.4f} m   (expected_all={snap_dist_mean:.4f} m)")
        print("============================================\n")


        # ---- NEW: occluded-goal subgroup metrics ----
        occ_goal_num = sum(r["occ_goal_num"] for r in results)
        total_samples = len(results)
        weird_num = sum(r.get("weird_num", 0) for r in results)
        if weird_num > 0:
            print(f"[GOAL somehow ABOVE GROUND] weird_num={int(weird_num)}/{len(results)} ({float(weird_num)/max(1,len(results)):.4f})")

        occ_ratio = float(occ_goal_num) / max(1, total_samples)

        hit_05_occ = sum(r["hit_num_05_occ"] for r in results)
        hit_10_occ = sum(r["hit_num_10_occ"] for r in results)
        hit_15_occ = sum(r["hit_num_15_occ"] for r in results)
        coll_occ = sum(r["coll_num_occ"] for r in results)
        known_occ = sum(r["known_num_occ"] for r in results)

        sem_05_occ = sum(r["sem_hit_num_05_occ"] for r in results)
        sem_10_occ = sum(r["sem_hit_num_10_occ"] for r in results)
        sem_15_occ = sum(r["sem_hit_num_15_occ"] for r in results)

        snap_05_occ = sum(r["snap_hit_num_05_occ"] for r in results)
        snap_10_occ = sum(r["snap_hit_num_10_occ"] for r in results)
        snap_15_occ = sum(r["snap_hit_num_15_occ"] for r in results)

        snap_dist_sum_occ = sum(r["snap_dist_sum_occ"] for r in results)
        snap_count_occ = sum(r["snap_count_occ"] for r in results)
        snap_invalid_count_occ = sum(r["snap_invalid_count_occ"] for r in results)

        hit_at_05_occ = (hit_05_occ.float() / known_occ.float().clamp_min(self.eps)).item()
        hit_at_10_occ = (hit_10_occ.float() / known_occ.float().clamp_min(self.eps)).item()
        hit_at_15_occ = (hit_15_occ.float() / known_occ.float().clamp_min(self.eps)).item()
        hit_mean_occ = (hit_at_05_occ + hit_at_10_occ + hit_at_15_occ) / 3.0
        collision_rate_occ = (coll_occ.float() / known_occ.float().clamp_min(self.eps)).item()

        sem_at_05_occ = (sem_05_occ.float() / known_occ.float().clamp_min(self.eps)).item()
        sem_at_10_occ = (sem_10_occ.float() / known_occ.float().clamp_min(self.eps)).item()
        sem_at_15_occ = (sem_15_occ.float() / known_occ.float().clamp_min(self.eps)).item()
        sem_mean_occ = (sem_at_05_occ + sem_at_10_occ + sem_at_15_occ) / 3.0

        snap_at_05_occ = (snap_05_occ.float() / known_occ.float().clamp_min(self.eps)).item()
        snap_at_10_occ = (snap_10_occ.float() / known_occ.float().clamp_min(self.eps)).item()
        snap_at_15_occ = (snap_15_occ.float() / known_occ.float().clamp_min(self.eps)).item()
        snap_mean_occ = (snap_at_05_occ + snap_at_10_occ + snap_at_15_occ) / 3.0

        # expected correction per prediction (occ subset)
        snap_dist_mean_occ = (snap_dist_sum_occ.float() / snap_count_occ.float().clamp_min(self.eps)).item()

        # invalid-only correction severity (occ subset)
        snap_dist_mean_invalid_occ = (snap_dist_sum_occ.float() / snap_invalid_count_occ.float().clamp_min(self.eps)).item()



        print("========== ExternalMetric (GOAL OCCLUDED SUBSET) ==========")
        print(f"[OccRatio]  {occ_ratio:.4f}   (occ={int(occ_goal_num)}/{total_samples})")
        print(
            f"[GeoHit]    @0.5={hit_at_05_occ:.4f}  @1.0={hit_at_10_occ:.4f}  @1.5={hit_at_15_occ:.4f}  mean={hit_mean_occ:.4f}"
        )
        print(f"[Invalid]   rate={collision_rate_occ:.4f}   (invalid/total = {int(coll_occ)}/{int(known_occ)})")
        print(
            f"[SemHit]    @0.5={sem_at_05_occ:.4f}  @1.0={sem_at_10_occ:.4f}  @1.5={sem_at_15_occ:.4f}  mean={sem_mean_occ:.4f}"
        )
        print(
            f"[SnapGeo]   @0.5={snap_at_05_occ:.4f}  @1.0={snap_at_10_occ:.4f}  @1.5={snap_at_15_occ:.4f}  mean={snap_mean_occ:.4f}"
        )
        print(f"[SnapDist]  invalid_mean={snap_dist_mean_invalid_occ:.4f} m   (expected_all={snap_dist_mean_occ:.4f} m)")
        print("==========================================================\n")
        # -------------------------------------------

        # ---- DEBUG DUMP: write all occluded-goal samples in one NPZ ----
        # (comment out this whole block to disable)
        if hasattr(self, "_occ_dump") and len(self._occ_dump) > 0:
            # out_path = "debug_outputs/2_miss_when_occ_debug.npz"
            out_path = "/workspace/debug_outputs/2_miss_when_occ_debug.npz"

            raw_imgs = np.stack([d["raw_img"] for d in self._occ_dump], axis=0)  # (N,4,448,448,3)
            trav = np.stack([d["traversable_mask"] for d in self._occ_dump], axis=0)  # (N,128,128)
            vis = np.stack([d["visible_mask"] for d in self._occ_dump], axis=0)       # (N,128,128)
            aff = np.stack([d["affordance_mask"] for d in self._occ_dump], axis=0)    # (N,128,128)
            free_ground = np.stack([d["free_ground_obs"] for d in self._occ_dump], axis=0)  # (N,128,128)
            free_cam = np.stack([d["free_cam_obs"] for d in self._occ_dump], axis=0)        # (N,128,128)

            pred_xy = np.stack([d["pred_xy"] for d in self._occ_dump], axis=0)  # (N,2)
            p_goal_np = np.stack([d["p_goal"] for d in self._occ_dump], axis=0) # (N,3)
            instr = np.array([d["instruction"] for d in self._occ_dump], dtype=object)  # (N,)
            pred_aff = np.stack([d["pred_affordance"] for d in self._occ_dump], axis=0)  # (N,128,128)

            np.savez_compressed(
                out_path,
                raw_img=raw_imgs,
                traversable_mask=trav,
                visible_mask=vis,
                affordance_mask=aff,
                free_ground_obs=free_ground,
                free_cam_obs=free_cam,
                pred_xy=pred_xy,
                p_goal=p_goal_np,
                instruction=instr,
                pred_affordance=pred_aff,
            )
            print(f"[DEBUG] saved debug at: {out_path} (N={len(self._occ_dump)})")
        # -----------------------------------------------------------------


        return dict(
            iou=iou,
            hit_at_0_5=hit_at_05,
            hit_at_1_0=hit_at_10,
            hit_at_1_5=hit_at_15,
            hit_mean=hit_mean,
            collision_rate=collision_rate,
            occ_goal_ratio=occ_ratio,
            hit_at_0_5_occ=hit_at_05_occ,
            hit_at_1_0_occ=hit_at_10_occ,
            hit_at_1_5_occ=hit_at_15_occ,
            hit_mean_occ=hit_mean_occ,
            collision_rate_occ=collision_rate_occ,

            sem_hit_at_0_5=sem_at_05,
            sem_hit_at_1_0=sem_at_10,
            sem_hit_at_1_5=sem_at_15,
            sem_hit_mean=sem_mean,

            snap_hit_at_0_5=snap_at_05,
            snap_hit_at_1_0=snap_at_10,
            snap_hit_at_1_5=snap_at_15,
            snap_hit_mean=snap_mean,
            snap_dist_mean=snap_dist_mean,
            snap_dist_mean_invalid=snap_dist_mean_invalid,

            sem_hit_at_0_5_occ=sem_at_05_occ,
            sem_hit_at_1_0_occ=sem_at_10_occ,
            sem_hit_at_1_5_occ=sem_at_15_occ,
            sem_hit_mean_occ=sem_mean_occ,

            snap_hit_at_0_5_occ=snap_at_05_occ,
            snap_hit_at_1_0_occ=snap_at_10_occ,
            snap_hit_at_1_5_occ=snap_at_15_occ,
            snap_hit_mean_occ=snap_mean_occ,
            snap_dist_mean_occ=snap_dist_mean_occ,
            snap_dist_mean_invalid_occ=snap_dist_mean_invalid_occ,

        )

# separately evaluated
class ExternalMetric_v5(BaseMetric):
    # TODO: distance somehow
    def __init__(self, 
                 eps=1e-6, 
                 collect_device='cpu', 
                 prefix=None, 
                 hit_thresholds_m=[0.5, 1.0, 1.5],
                 res=0.1,
                 half_xy=6.4,
                 **kwargs):
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.eps = eps
        self.hit_thresholds_m = hit_thresholds_m
        self.res = res
        self.half_xy = half_xy

    @staticmethod
    def _bounded_geodesic_dist_cells(Ai_cpu: torch.Tensor, si: int, sj: int, rmax_cells: float) -> torch.Tensor:
        """
        Ai_cpu: (H,W) bool on CPU. True = traversable.
        si,sj: seed indices (row=i, col=j)
        rmax_cells: max radius in cell-units (cardinal=1, diagonal=sqrt(2))
        Returns dist (H,W) float32 on CPU, INF for unreachable/outside rmax.
        """
        H, W = Ai_cpu.shape
        INF = 1e9
        dist = torch.full((H, W), INF, dtype=torch.float32)
        visited = torch.zeros((H, W), dtype=torch.bool)

        dist[si, sj] = 0.0
        pq = [(0.0, si, sj)]

        SQ2 = math.sqrt(2.0)
        moves = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, SQ2), (-1, 1, SQ2), (1, -1, SQ2), (1, 1, SQ2),
        ]

        while pq:
            d, y, x = heapq.heappop(pq)
            if visited[y, x]:
                continue
            visited[y, x] = True
            if d > rmax_cells:
                break

            for dy, dx, w in moves:
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= H or nx < 0 or nx >= W:
                    continue
                if not Ai_cpu[ny, nx]:
                    continue

                # strict diagonal gating
                if dy != 0 and dx != 0:
                    if not (Ai_cpu[y, nx] and Ai_cpu[ny, x]):
                        continue

                nd = d + w
                if nd <= rmax_cells and nd < float(dist[ny, nx]):
                    dist[ny, nx] = float(nd)
                    heapq.heappush(pq, (float(nd), ny, nx))

        return dist


    def process(self, data_batch, data_samples):
        data = data_batch["data"]
        A = data['traversable_mask'].to(self.collect_device).bool()  # B,H,W
        B = data['visible_mask'].to(self.collect_device).bool()
        C = data['affordance_mask'].to(self.collect_device).bool()
        p_goal = data["p_goal"].to(self.collect_device).float()  # B,3 (x_fwd, y_left, z_up)
        capture_versions = data.get("capture_version", None)
        if capture_versions is None:
            raise KeyError(
                "ExternalMetric expects `capture_version` in data_batch['data'] "
                "(e.g., 'static'/'dynamic') to report per-version metrics."
            )
        gt_waypoints = data.get("waypoints", None)

        # ---- NEW: goal occlusion inputs (depth + intrinsics + cam2egos) ----
        depth = data["depth"]  # (B,4,H,W) or dict
        cam2imgs = data["cam2imgs"]
        cam2egos = data["cam2egos"]

        # handle dict case (same order as your pipeline)
        if isinstance(cam2imgs, dict):
            order = ["front", "left", "back", "right"]
            cam2imgs = torch.stack([cam2imgs[k] for k in order], dim=1)   # (B,4,4,4)
            cam2egos = torch.stack([cam2egos[k] for k in order], dim=1)   # (B,4,4,4)
            depth = torch.stack([depth[k] for k in order], dim=1) if isinstance(depth, dict) else depth

        depth = depth.to(self.collect_device).float()        # (B,4,448,448), z-depth in meters
        cam2imgs = cam2imgs.to(self.collect_device).float()
        cam2egos = cam2egos.to(self.collect_device).float()
        intrinsics = cam2imgs[:, :, :3, :3]                 # (B,4,3,3)
        # ---------------------------------------------------

        preds = data_samples
        if isinstance(preds, dict):
            preds = [preds]

        for i, pred in enumerate(preds):
            Ai, Bi, Ci = A[i], B[i], C[i]  # H,W
            if isinstance(capture_versions, (list, tuple)):
                cap_ver = capture_versions[i]
            else:
                cap_ver = capture_versions
            cap_ver = str(cap_ver).strip().lower()

            pred_aff = pred['pred_aff'].to(self.collect_device).bool()  # H,W (or 1,H,W)
            if pred_aff.dim() == 3:
                pred_aff = pred_aff[0]

            pred_xy = pred['pred_xy'].to(self.collect_device).float()  # 2 or N x 2
            if pred_xy.dim() == 1:
                pred_xy = pred_xy.view(1, 2)
            
            # ---- NEW: SemHit (Euclidean in meters, independent of traversability) ----
            gt_xy_m = p_goal[i, 0:2]  # (2,) in meters (x_fwd, y_left)
            diff_xy = pred_xy - gt_xy_m.view(1, 2)  # (K,2)
            euc_dist_m = torch.sqrt((diff_xy ** 2).sum(dim=1) + self.eps)  # (K,)
            sem_hit_05 = (euc_dist_m <= 0.5).sum()
            sem_hit_10 = (euc_dist_m <= 1.0).sum()
            sem_hit_15 = (euc_dist_m <= 1.5).sum()
            # ------------------------------------------------------------------------


            gt = (Ai & Ci) & Bi
            p = pred_aff & Bi
            tp = (p & gt).sum()
            fp = (p & (~gt) & Bi).sum()
            fn = ((~p) & gt).sum()

            x = torch.round((pred_xy[:, 0] + 6.4 - 0.05) / 0.1).long()
            y = torch.round((pred_xy[:, 1] + 6.4 - 0.05) / 0.1).long()
            H, W = Ai.shape
            inb = (x >= 0) & (x < H) & (y >= 0) & (y < W)

            # clamp for safe indexing; keep inb to mark OOB as invalid later
            x_clamp = x.clamp(0, H - 1)
            y_clamp = y.clamp(0, W - 1)


            # ---- NEW: bounded geodesic dist from p_goal (compute once per sample) ----
            thr_m = self.hit_thresholds_m
            rmax_m = max(thr_m)
            rmax_cells = float(rmax_m / self.res)

            # seed indices from p_goal (offset is half of resolution => 0.5*self.res)
            seed_i = int(torch.round((p_goal[i, 0] + self.half_xy - 0.5 * self.res) / self.res).item())  # row (H), forward
            seed_j = int(torch.round((p_goal[i, 1] + self.half_xy - 0.5 * self.res) / self.res).item())  # col (W), left

            # ---- NEW: clamp seed to valid grid bounds ----
            seed_i = max(0, min(H - 1, seed_i))
            seed_j = max(0, min(W - 1, seed_j))
            # --------------------------------------------


            # compute dist on CPU (small neighborhood), then bring back to collect_device
            Ai_cpu = Ai.detach().to("cpu")
            # ---- NEW: if seed is not traversable, snap seed to nearest traversable cell ----
            if not bool(Ai_cpu[seed_i, seed_j].item()):
                trav_idx_seed = torch.nonzero(Ai_cpu, as_tuple=False)  # (M,2)
                if trav_idx_seed.numel() > 0:
                    di = trav_idx_seed[:, 0].float() - float(seed_i)
                    dj = trav_idx_seed[:, 1].float() - float(seed_j)
                    d2 = di * di + dj * dj
                    arg = int(torch.argmin(d2).item())
                    seed_i = int(trav_idx_seed[arg, 0].item())
                    seed_j = int(trav_idx_seed[arg, 1].item())
            # ------------------------------------------------------------------------------
            dist_cpu = self._bounded_geodesic_dist_cells(Ai_cpu, seed_i, seed_j, rmax_cells)  # cell units
            dist = dist_cpu.to(self.collect_device)
            # ------------------------------------------------------------------------

            # ---- NEW: occlusion flag for p_goal (per sample) ----
            tol = 0.10  # fixed tolerance in meters
            pg = p_goal[i]  # (3,) ego
            pg_h = torch.tensor([pg[0], pg[1], pg[2], 1.0], device=self.collect_device, dtype=torch.float32)
            # ego <- cam given, so cam <- ego is inverse
            T_ego_cam = cam2egos[i]                       # (4,4,4)
            T_cam_ego = torch.linalg.inv(T_ego_cam)       # (4,4,4)
            # project into each cam
            pg_cam = (T_cam_ego @ pg_h.view(4, 1)).squeeze(-1)   # (4,4) -> per cam (x,y,z,1)
            X = pg_cam[:, 0]
            Y = pg_cam[:, 1]
            Z = pg_cam[:, 2]
            # valid if in front
            front = Z > 1e-6
            # intrinsics for this sample
            K = intrinsics[i]  # (4,3,3)
            fx = K[:, 0, 0]; fy = K[:, 1, 1]
            cx = K[:, 0, 2]; cy = K[:, 1, 2]
            u = torch.round(fx * (X / Z) + cx).long()
            v = torch.round(fy * (Y / Z) + cy).long()
            Himg, Wimg = depth.shape[-2], depth.shape[-1]
            inb_img = (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg)
            valid_cam = front & inb_img
            n_valid = int(valid_cam.sum().item())
            assert n_valid <= 1, f"[ERR] p_goal visible in >1 cams (n={n_valid}) -> conversion wrong"
            is_weird = False
            if n_valid == 0:
                # If p_goal is outside all camera frustums, we only accept it as "too close"
                # when its XY is within 2m in ego frame; otherwise treat as a conversion bug.
                assert float(torch.abs(pg[0])) <= 2.0 and float(torch.abs(pg[1])) <= 2.0, \
                    f"[ERR] n_valid==0 but p_goal XY not within 2m: p_goal={pg.tolist()}"

                occluded_goal = False  # treat as not-occluded (too close / outside FOV)

            else:
                cam_idx = int(torch.nonzero(valid_cam, as_tuple=False)[0].item())
                uu = int(u[cam_idx].item())
                vv = int(v[cam_idx].item())
                zq = float(Z[cam_idx].item())

                d = depth[i, cam_idx, vv, uu]
                d_ok = torch.isfinite(d) & (d > 1e-6)

                if not bool(d_ok.item()):
                    # 3x3 neighbor fallback
                    v0 = max(0, vv - 1); v1 = min(Himg, vv + 2)
                    u0 = max(0, uu - 1); u1 = min(Wimg, uu + 2)
                    patch = depth[i, cam_idx, v0:v1, u0:u1]
                    m = torch.isfinite(patch) & (patch > 1e-6)
                    if m.any():
                        d = patch[m].mean()
                        d_ok = torch.isfinite(d) & (d > 1e-6)
                    else:
                        d_ok = torch.tensor(False, device=self.collect_device)

                if not bool(d_ok.item()):
                    occluded_goal = True
                else:
                    dq = float(d.item())
                    diff = zq - dq

                    # only abs within tol is "not occluded"
                    if abs(diff) <= tol:
                        occluded_goal = False
                    elif diff > tol:
                        # z is behind the observed surface => occluded
                        occluded_goal = True
                    else:
                        occluded_goal = False
                        print(f"[WARN] Z < depth - tol: cam={cam_idx} (u,v)=({uu},{vv}) Z={zq:.3f} d={dq:.3f} diff={diff:.3f} p_goal={pg.tolist()}")
                        is_weird = True # likely p_goal is above the mesh ground somehow
            # ------------------------------------------

            # ---- NEW: waypoint ADE (per-point) + per-sample collision score ----
            wp_ade_sum = torch.tensor(0.0, device=self.collect_device, dtype=torch.float32)
            wp_ade_cnt = torch.tensor(0, device=self.collect_device, dtype=torch.long)
            wp_coll_score_sum = torch.tensor(0.0, device=self.collect_device, dtype=torch.float32)
            wp_coll_score_cnt = torch.tensor(0, device=self.collect_device, dtype=torch.long)

            if gt_waypoints is not None and isinstance(pred, dict) and ("pred_waypoints" in pred):
                # pred waypoints: [N,2] (or [1,N,2])
                pred_wps = pred["pred_waypoints"]
                if not isinstance(pred_wps, torch.Tensor):
                    pred_wps = torch.as_tensor(pred_wps)
                pred_wps = pred_wps.to(self.collect_device).float()
                if pred_wps.dim() == 3:
                    pred_wps = pred_wps[0]

                gt_wps_i = gt_waypoints[i]
                if not isinstance(gt_wps_i, torch.Tensor):
                    gt_wps_i = torch.as_tensor(gt_wps_i)
                gt_wps_i = gt_wps_i.to(self.collect_device).float()
                if gt_wps_i.dim() == 3:
                    gt_wps_i = gt_wps_i[0]
                gt_wps_xy = gt_wps_i[:, :2]

                N = int(min(pred_wps.shape[0], gt_wps_xy.shape[0]))
                if N > 0:
                    pred_xy_wp = pred_wps[:N, :2]
                    gt_xy_wp = gt_wps_xy[:N, :2]

                    # ADE: sum L2 over points, aggregate as sum/cnt
                    per_point_l2 = torch.sqrt(((pred_xy_wp - gt_xy_wp) ** 2).sum(dim=-1) + self.eps)  # (N,)
                    wp_ade_sum = per_point_l2.sum()
                    wp_ade_cnt = torch.tensor(N, device=self.collect_device, dtype=torch.long)

                    # collision count on predicted waypoints
                    Ht, Wt = Ai.shape
                    ix = torch.round((pred_xy_wp[:, 0] + self.half_xy - 0.5 * self.res) / self.res).long()
                    iy = torch.round((pred_xy_wp[:, 1] + self.half_xy - 0.5 * self.res) / self.res).long()
                    inb_wp = (ix >= 0) & (ix < Ht) & (iy >= 0) & (iy < Wt)
                    ix_c = ix.clamp(0, Ht - 1)
                    iy_c = iy.clamp(0, Wt - 1)
                    collide_wp = (~inb_wp) | (~Ai[ix_c, iy_c])
                    c = int(collide_wp.long().sum().item())

                    # per-sample collision score: 0, 0.5, or 1
                    if c == 0:
                        score = 0.0
                    elif c == 1:
                        score = 0.5
                    else:
                        score = 1.0
                    wp_coll_score_sum = torch.tensor(score, device=self.collect_device, dtype=torch.float32)
                    wp_coll_score_cnt = torch.tensor(1, device=self.collect_device, dtype=torch.long)

            if occluded_goal:
                wp_ade_sum_occ, wp_ade_cnt_occ = wp_ade_sum, wp_ade_cnt
                wp_coll_score_sum_occ, wp_coll_score_cnt_occ = wp_coll_score_sum, wp_coll_score_cnt
            else:
                wp_ade_sum_occ = torch.zeros_like(wp_ade_sum)
                wp_ade_cnt_occ = torch.zeros_like(wp_ade_cnt)
                wp_coll_score_sum_occ = torch.zeros_like(wp_coll_score_sum)
                wp_coll_score_cnt_occ = torch.zeros_like(wp_coll_score_cnt)
            # ---------------------------------------------------------------

            # # debug for geodesic dist
            # debug_save_path = 'debug_outputs/1_metric_disc.npz'
            # np.savez(
            #     debug_save_path,
            #     traversable = Ai_cpu.numpy(),
            #     visible = Bi.detach().to("cpu").numpy(),
            #     affordance = Ci.detach().to("cpu").numpy(),
            #     disc_05 = (dist_cpu.numpy() <= (0.5 / self.res)).astype(np.uint8),
            #     disc_10 = (dist_cpu.numpy() <= (1.0 / self.res)).astype(np.uint8),
            #     disc_15 = (dist_cpu.numpy() <= (1.5 / self.res)).astype(np.uint8),
            # )
            # exit(0)
            # # end of debug

            # # debug for occlusion
            # debug_save_path = 'debug_outputs/1_metric_occlusion.npz'
            # if n_valid == 1:
            #     cam_idx_save = int(cam_idx)
            #     uv_save = np.array([int(uu), int(vv)], dtype=np.int32)  # (u,v)
            # else:
            #     cam_idx_save = -1
            #     uv_save = np.array([-1, -1], dtype=np.int32)
            # if cam_idx_save >= 2 and is_weird:
            #     np.savez(
            #         debug_save_path,
            #         depth=depth[i].detach().to("cpu").numpy(),
            #         cam_idx=np.int32(cam_idx_save),
            #         uv=uv_save,
            #         instr=data["instruction"][i],
            #         p_goal=p_goal[i].detach().to("cpu").numpy(),
            #     )
            #     exit(0)
            # # end of debug

            # invalid = OOB OR non-traversable
            invalid = (~inb) | (~Ai[x_clamp, y_clamp])

            # GeoHit@r: only counts if in-bounds + traversable + within geodesic threshold
            hit_05 = ((~invalid) & (dist[x_clamp, y_clamp] <= (0.5 / self.res))).sum()
            hit_10 = ((~invalid) & (dist[x_clamp, y_clamp] <= (1.0 / self.res))).sum()
            hit_15 = ((~invalid) & (dist[x_clamp, y_clamp] <= (1.5 / self.res))).sum()

            # InvalidRate numerator (was collision_rate)
            coll = invalid.sum()

            # denominator: number of predicted points (K)
            known_num = torch.tensor(pred_xy.shape[0], device=self.collect_device, dtype=torch.long)

            # ---- NEW: invalid count for SnapDist invalid-only mean ----
            invalid_count = coll.clone()
            # ----------------------------------------------------------


            # ---- NEW: Snap to nearest traversable cell (Ai==1), then SnapGeoHit + SnapDist ----
            Ai_cpu = Ai.detach().to("cpu")
            x0_cpu = x_clamp.detach().to("cpu").long()
            y0_cpu = y_clamp.detach().to("cpu").long()

            # precompute all traversable indices once per sample (CPU)
            trav_idx = torch.nonzero(Ai_cpu, as_tuple=False)  # (M,2) with (i,j)
            has_trav = trav_idx.numel() > 0

            xs_list = []
            ys_list = []

            if has_trav:
                # For each prediction: pick nearest traversable cell in Euclidean grid distance
                for kk in range(x0_cpu.shape[0]):
                    i0 = int(x0_cpu[kk].item())
                    j0 = int(y0_cpu[kk].item())
                    di = trav_idx[:, 0].float() - float(i0)
                    dj = trav_idx[:, 1].float() - float(j0)
                    d2 = di * di + dj * dj
                    arg = int(torch.argmin(d2).item())
                    xs_list.append(int(trav_idx[arg, 0].item()))
                    ys_list.append(int(trav_idx[arg, 1].item()))
            else:
                # degenerate: no traversable cells, keep clamped as "snapped"
                for kk in range(x0_cpu.shape[0]):
                    xs_list.append(int(x0_cpu[kk].item()))
                    ys_list.append(int(y0_cpu[kk].item()))

            xs = torch.tensor(xs_list, device=self.collect_device, dtype=torch.long)
            ys = torch.tensor(ys_list, device=self.collect_device, dtype=torch.long)

            # SnapGeoHit@r (geodesic evaluated at snapped cell)
            snap_hit_05 = (dist[xs, ys] <= (0.5 / self.res)).sum()
            snap_hit_10 = (dist[xs, ys] <= (1.0 / self.res)).sum()
            snap_hit_15 = (dist[xs, ys] <= (1.5 / self.res)).sum()

            # SnapDist: straight Euclidean in meters from original pred_xy to snapped cell center
            # cell center inverse mapping consistent with your meters->cell quantization
            xs_m = (xs.float() * self.res) - self.half_xy + 0.5 * self.res
            ys_m = (ys.float() * self.res) - self.half_xy + 0.5 * self.res
            snap_dx = pred_xy[:, 0] - xs_m
            snap_dy = pred_xy[:, 1] - ys_m
            snap_dist = torch.sqrt(snap_dx * snap_dx + snap_dy * snap_dy + self.eps)  # (K,)

            # ---- NEW: only invalid/OOB contribute to snap distance; valid -> 0 ----
            snap_dist = snap_dist * invalid.to(snap_dist.dtype)
            # ---------------------------------------------------------------

            snap_dist_sum = snap_dist.sum()

            # total predicted points (K) — keep if you still want "expected correction per query"
            snap_count = torch.tensor(pred_xy.shape[0], device=self.collect_device, dtype=torch.long)

            # invalid-only count for invalid-only mean
            snap_invalid_count = invalid_count.clone()
            # -------------------------------------------------------------------------------


            # ---- DEBUG DUMP: collect occluded-goal samples for visualization ----
            # (comment out this whole block to disable)
            if occluded_goal:
                # lazy init to avoid touching __init__
                if not hasattr(self, "_occ_dump"):
                    self._occ_dump = []

                # fetch raw_img + extra obs masks from data_batch
                raw_img_i = data["raw_img"][i].detach().to("cpu").numpy()  # (4,448,448,3) uint8 (BGR)
                free_ground_i = pred["free_ground_obs"].detach().to("cpu").numpy().astype(np.uint8)  # (128,128)
                free_cam_i = pred["free_cam_obs"].detach().to("cpu").numpy().astype(np.uint8)        # (128,128)

                # core masks (128,128) -> uint8 for easier viewing
                trav_i = Ai.detach().to("cpu").numpy().astype(np.uint8)
                vis_i = Bi.detach().to("cpu").numpy().astype(np.uint8)
                aff_i = Ci.detach().to("cpu").numpy().astype(np.uint8)

                # pred_xy is K=1 always; keep (2,)
                pred_xy_i = pred_xy[0].detach().to("cpu").numpy().astype(np.float32)  # (2,)
                p_goal_i = p_goal[i].detach().to("cpu").numpy().astype(np.float32)    # (3,)
                pred_aff_i = pred_aff.detach().to("cpu").numpy().astype(np.uint8)          # (128,128)

                instr_i = data["instruction"][i]  # keep as python string

                self._occ_dump.append(dict(
                    raw_img=raw_img_i,
                    traversable_mask=trav_i,
                    visible_mask=vis_i,
                    affordance_mask=aff_i,
                    free_ground_obs=free_ground_i,
                    free_cam_obs=free_cam_i,
                    pred_xy=pred_xy_i,
                    p_goal=p_goal_i,
                    instruction=instr_i,
                    pred_affordance=pred_aff_i,
                ))
            # -------------------------------------------------------------------


            # ---- NEW: subgroup sums for occluded-goal only (keep global sums unchanged) ----
            if occluded_goal:
                hit_05_occ, hit_10_occ, hit_15_occ = hit_05, hit_10, hit_15
                coll_occ, known_occ = coll, known_num

                sem_hit_05_occ, sem_hit_10_occ, sem_hit_15_occ = sem_hit_05, sem_hit_10, sem_hit_15
                snap_hit_05_occ, snap_hit_10_occ, snap_hit_15_occ = snap_hit_05, snap_hit_10, snap_hit_15
                snap_dist_sum_occ, snap_count_occ = snap_dist_sum, snap_count

                # ---- NEW: invalid-only count for SnapDist invalid-only mean (occ subset) ----
                snap_invalid_count_occ = snap_invalid_count
                # ---------------------------------------------------------------------------

                occ_goal_num = torch.tensor(1, device=self.collect_device, dtype=torch.long)
            else:
                hit_05_occ = torch.zeros_like(hit_05)
                hit_10_occ = torch.zeros_like(hit_10)
                hit_15_occ = torch.zeros_like(hit_15)
                coll_occ = torch.zeros_like(coll)
                known_occ = torch.zeros_like(known_num)

                sem_hit_05_occ = torch.zeros_like(sem_hit_05)
                sem_hit_10_occ = torch.zeros_like(sem_hit_10)
                sem_hit_15_occ = torch.zeros_like(sem_hit_15)
                snap_hit_05_occ = torch.zeros_like(snap_hit_05)
                snap_hit_10_occ = torch.zeros_like(snap_hit_10)
                snap_hit_15_occ = torch.zeros_like(snap_hit_15)
                snap_dist_sum_occ = torch.zeros_like(snap_dist_sum)
                snap_count_occ = torch.zeros_like(snap_count)

                # ---- NEW: invalid-only count for SnapDist invalid-only mean (occ subset) ----
                snap_invalid_count_occ = torch.zeros_like(snap_invalid_count)
                # ---------------------------------------------------------------------------

                occ_goal_num = torch.tensor(0, device=self.collect_device, dtype=torch.long)

            # -------------------------------------------------------------------------------


            self.results.append(dict(
                capture_version=cap_ver,
                tp=tp.cpu(), fp=fp.cpu(), fn=fn.cpu(),
                # hit_num=hit.cpu(), 
                hit_num_05=hit_05.cpu(),
                hit_num_10=hit_10.cpu(),
                hit_num_15=hit_15.cpu(),
                coll_num=coll.cpu(), 
                known_num=known_num.cpu(),
                occ_goal_num=occ_goal_num.cpu(),
                hit_num_05_occ=hit_05_occ.cpu(),
                hit_num_10_occ=hit_10_occ.cpu(),
                hit_num_15_occ=hit_15_occ.cpu(),
                coll_num_occ=coll_occ.cpu(),
                known_num_occ=known_occ.cpu(),
                weird_num=torch.tensor(1 if is_weird else 0, device=self.collect_device, dtype=torch.long).cpu(),
                
                # NEW
                sem_hit_num_05=sem_hit_05.cpu(),
                sem_hit_num_10=sem_hit_10.cpu(),
                sem_hit_num_15=sem_hit_15.cpu(),

                snap_hit_num_05=snap_hit_05.cpu(),
                snap_hit_num_10=snap_hit_10.cpu(),
                snap_hit_num_15=snap_hit_15.cpu(),
                snap_dist_sum=snap_dist_sum.cpu(),
                snap_count=snap_count.cpu(),
                snap_invalid_count=snap_invalid_count.cpu(),

                sem_hit_num_05_occ=sem_hit_05_occ.cpu(),
                sem_hit_num_10_occ=sem_hit_10_occ.cpu(),
                sem_hit_num_15_occ=sem_hit_15_occ.cpu(),

                snap_hit_num_05_occ=snap_hit_05_occ.cpu(),
                snap_hit_num_10_occ=snap_hit_10_occ.cpu(),
                snap_hit_num_15_occ=snap_hit_15_occ.cpu(),
                snap_dist_sum_occ=snap_dist_sum_occ.cpu(),
                snap_count_occ=snap_count_occ.cpu(),
                snap_invalid_count_occ=snap_invalid_count_occ.cpu(),

                # NEW: waypoint metrics
                wp_ade_sum=wp_ade_sum.cpu(),
                wp_ade_cnt=wp_ade_cnt.cpu(),
                wp_coll_score_sum=wp_coll_score_sum.cpu(),
                wp_coll_score_cnt=wp_coll_score_cnt.cpu(),
                wp_ade_sum_occ=wp_ade_sum_occ.cpu(),
                wp_ade_cnt_occ=wp_ade_cnt_occ.cpu(),
                wp_coll_score_sum_occ=wp_coll_score_sum_occ.cpu(),
                wp_coll_score_cnt_occ=wp_coll_score_cnt_occ.cpu(),
            ))

    def compute_metrics(self, results):
        def _compute_and_print(subset_results, *, title: str, print_report: bool = True):
            tp = sum(r['tp'] for r in subset_results)
            fp = sum(r['fp'] for r in subset_results)
            fn = sum(r['fn'] for r in subset_results)
            hit_05 = sum(r["hit_num_05"] for r in subset_results)
            hit_10 = sum(r["hit_num_10"] for r in subset_results)
            hit_15 = sum(r["hit_num_15"] for r in subset_results)

            sem_05 = sum(r["sem_hit_num_05"] for r in subset_results)
            sem_10 = sum(r["sem_hit_num_10"] for r in subset_results)
            sem_15 = sum(r["sem_hit_num_15"] for r in subset_results)

            snap_05 = sum(r["snap_hit_num_05"] for r in subset_results)
            snap_10 = sum(r["snap_hit_num_10"] for r in subset_results)
            snap_15 = sum(r["snap_hit_num_15"] for r in subset_results)

            snap_dist_sum = sum(r["snap_dist_sum"] for r in subset_results)
            snap_count = sum(r["snap_count"] for r in subset_results)
            snap_invalid_count = sum(r["snap_invalid_count"] for r in subset_results)

            coll_num = sum(r['coll_num'] for r in subset_results)
            known_num = sum(r['known_num'] for r in subset_results)

            iou = (tp.float() / (tp + fp + fn).float().clamp_min(self.eps)).item()
            collision_rate = (coll_num.float() / known_num.float().clamp_min(self.eps)).item()
            hit_at_05 = (hit_05.float() / known_num.float().clamp_min(self.eps)).item()
            hit_at_10 = (hit_10.float() / known_num.float().clamp_min(self.eps)).item()
            hit_at_15 = (hit_15.float() / known_num.float().clamp_min(self.eps)).item()
            hit_mean = (hit_at_05 + hit_at_10 + hit_at_15) / 3.0

            sem_at_05 = (sem_05.float() / known_num.float().clamp_min(self.eps)).item()
            sem_at_10 = (sem_10.float() / known_num.float().clamp_min(self.eps)).item()
            sem_at_15 = (sem_15.float() / known_num.float().clamp_min(self.eps)).item()
            sem_mean = (sem_at_05 + sem_at_10 + sem_at_15) / 3.0

            snap_at_05 = (snap_05.float() / known_num.float().clamp_min(self.eps)).item()
            snap_at_10 = (snap_10.float() / known_num.float().clamp_min(self.eps)).item()
            snap_at_15 = (snap_15.float() / known_num.float().clamp_min(self.eps)).item()
            snap_mean = (snap_at_05 + snap_at_10 + snap_at_15) / 3.0

            snap_dist_mean = (snap_dist_sum.float() / snap_count.float().clamp_min(self.eps)).item()
            snap_dist_mean_invalid = (snap_dist_sum.float() / snap_invalid_count.float().clamp_min(self.eps)).item()

            # waypoint metrics
            wp_ade_sum = sum(r.get("wp_ade_sum", 0.0) for r in subset_results)
            wp_ade_cnt = sum(r.get("wp_ade_cnt", 0) for r in subset_results)
            wp_ade = (wp_ade_sum.float() / wp_ade_cnt.float().clamp_min(self.eps)).item() if wp_ade_cnt > 0 else float("nan")

            wp_coll_score_sum = sum(r.get("wp_coll_score_sum", 0.0) for r in subset_results)
            wp_coll_score_cnt = sum(r.get("wp_coll_score_cnt", 0) for r in subset_results)
            wp_coll_score = (wp_coll_score_sum.float() / wp_coll_score_cnt.float().clamp_min(self.eps)).item() if wp_coll_score_cnt > 0 else float("nan")

            wp_ade_sum_occ = sum(r.get("wp_ade_sum_occ", 0.0) for r in subset_results)
            wp_ade_cnt_occ = sum(r.get("wp_ade_cnt_occ", 0) for r in subset_results)
            wp_ade_occ = (wp_ade_sum_occ.float() / wp_ade_cnt_occ.float().clamp_min(self.eps)).item() if wp_ade_cnt_occ > 0 else float("nan")

            wp_coll_score_sum_occ = sum(r.get("wp_coll_score_sum_occ", 0.0) for r in subset_results)
            wp_coll_score_cnt_occ = sum(r.get("wp_coll_score_cnt_occ", 0) for r in subset_results)
            wp_coll_score_occ = (wp_coll_score_sum_occ.float() / wp_coll_score_cnt_occ.float().clamp_min(self.eps)).item() if wp_coll_score_cnt_occ > 0 else float("nan")

            if print_report:
                print(f"\n========== ExternalMetric ({title}) ==========")
                print(f"[Seg/Mask]  IoU={iou:.4f}")
                print(
                    f"[GeoHit]    @0.5={hit_at_05:.4f}  @1.0={hit_at_10:.4f}  @1.5={hit_at_15:.4f}  mean={hit_mean:.4f}"
                )
                print(f"[Invalid]   rate={collision_rate:.4f}   (invalid/total = {int(coll_num)}/{int(known_num)})")
                print(
                    f"[SemHit]    @0.5={sem_at_05:.4f}  @1.0={sem_at_10:.4f}  @1.5={sem_at_15:.4f}  mean={sem_mean:.4f}"
                )
                print(
                    f"[SnapGeo]   @0.5={snap_at_05:.4f}  @1.0={snap_at_10:.4f}  @1.5={snap_at_15:.4f}  mean={snap_mean:.4f}"
                )
                print(f"[SnapDist]  invalid_mean={snap_dist_mean_invalid:.4f} m   (expected_all={snap_dist_mean:.4f} m)")
                if wp_ade_cnt > 0:
                    print(f"[WP_ADE]    {wp_ade:.4f} m")
                    print(f"[WP_Coll]   {wp_coll_score:.4f}   (mean per-sample score: 0/0.5/1)")
                print("============================================\n")

            # occluded-goal subgroup metrics
            occ_goal_num = sum(r["occ_goal_num"] for r in subset_results)
            total_samples = len(subset_results)
            weird_num = sum(r.get("weird_num", 0) for r in subset_results)
            if print_report and weird_num > 0:
                print(
                    f"[GOAL somehow ABOVE GROUND] weird_num={int(weird_num)}/{len(subset_results)} "
                    f"({float(weird_num)/max(1,len(subset_results)):.4f})"
                )

            occ_ratio = float(occ_goal_num) / max(1, total_samples)

            hit_05_occ = sum(r["hit_num_05_occ"] for r in subset_results)
            hit_10_occ = sum(r["hit_num_10_occ"] for r in subset_results)
            hit_15_occ = sum(r["hit_num_15_occ"] for r in subset_results)
            coll_occ = sum(r["coll_num_occ"] for r in subset_results)
            known_occ = sum(r["known_num_occ"] for r in subset_results)

            sem_05_occ = sum(r["sem_hit_num_05_occ"] for r in subset_results)
            sem_10_occ = sum(r["sem_hit_num_10_occ"] for r in subset_results)
            sem_15_occ = sum(r["sem_hit_num_15_occ"] for r in subset_results)

            snap_05_occ = sum(r["snap_hit_num_05_occ"] for r in subset_results)
            snap_10_occ = sum(r["snap_hit_num_10_occ"] for r in subset_results)
            snap_15_occ = sum(r["snap_hit_num_15_occ"] for r in subset_results)

            snap_dist_sum_occ = sum(r["snap_dist_sum_occ"] for r in subset_results)
            snap_count_occ = sum(r["snap_count_occ"] for r in subset_results)
            snap_invalid_count_occ = sum(r["snap_invalid_count_occ"] for r in subset_results)

            hit_at_05_occ = (hit_05_occ.float() / known_occ.float().clamp_min(self.eps)).item()
            hit_at_10_occ = (hit_10_occ.float() / known_occ.float().clamp_min(self.eps)).item()
            hit_at_15_occ = (hit_15_occ.float() / known_occ.float().clamp_min(self.eps)).item()
            hit_mean_occ = (hit_at_05_occ + hit_at_10_occ + hit_at_15_occ) / 3.0
            collision_rate_occ = (coll_occ.float() / known_occ.float().clamp_min(self.eps)).item()

            sem_at_05_occ = (sem_05_occ.float() / known_occ.float().clamp_min(self.eps)).item()
            sem_at_10_occ = (sem_10_occ.float() / known_occ.float().clamp_min(self.eps)).item()
            sem_at_15_occ = (sem_15_occ.float() / known_occ.float().clamp_min(self.eps)).item()
            sem_mean_occ = (sem_at_05_occ + sem_at_10_occ + sem_at_15_occ) / 3.0

            snap_at_05_occ = (snap_05_occ.float() / known_occ.float().clamp_min(self.eps)).item()
            snap_at_10_occ = (snap_10_occ.float() / known_occ.float().clamp_min(self.eps)).item()
            snap_at_15_occ = (snap_15_occ.float() / known_occ.float().clamp_min(self.eps)).item()
            snap_mean_occ = (snap_at_05_occ + snap_at_10_occ + snap_at_15_occ) / 3.0

            snap_dist_mean_occ = (snap_dist_sum_occ.float() / snap_count_occ.float().clamp_min(self.eps)).item()
            snap_dist_mean_invalid_occ = (snap_dist_sum_occ.float() / snap_invalid_count_occ.float().clamp_min(self.eps)).item()

            if print_report:
                print(f"========== ExternalMetric ({title} | GOAL OCCLUDED SUBSET) ==========")
                print(f"[OccRatio]  {occ_ratio:.4f}   (occ={int(occ_goal_num)}/{total_samples})")
                print(
                    f"[GeoHit]    @0.5={hit_at_05_occ:.4f}  @1.0={hit_at_10_occ:.4f}  @1.5={hit_at_15_occ:.4f}  mean={hit_mean_occ:.4f}"
                )
                print(f"[Invalid]   rate={collision_rate_occ:.4f}   (invalid/total = {int(coll_occ)}/{int(known_occ)})")
                print(
                    f"[SemHit]    @0.5={sem_at_05_occ:.4f}  @1.0={sem_at_10_occ:.4f}  @1.5={sem_at_15_occ:.4f}  mean={sem_mean_occ:.4f}"
                )
                print(
                    f"[SnapGeo]   @0.5={snap_at_05_occ:.4f}  @1.0={snap_at_10_occ:.4f}  @1.5={snap_at_15_occ:.4f}  mean={snap_mean_occ:.4f}"
                )
                print(f"[SnapDist]  invalid_mean={snap_dist_mean_invalid_occ:.4f} m   (expected_all={snap_dist_mean_occ:.4f} m)")
                if wp_ade_cnt_occ > 0:
                    print(f"[WP_ADE]    {wp_ade_occ:.4f} m")
                    print(f"[WP_Coll]   {wp_coll_score_occ:.4f}   (mean per-sample score: 0/0.5/1)")
                print("==========================================================\n")

            return dict(
                iou=iou,
                hit_at_0_5=hit_at_05,
                hit_at_1_0=hit_at_10,
                hit_at_1_5=hit_at_15,
                hit_mean=hit_mean,
                collision_rate=collision_rate,
                occ_goal_ratio=occ_ratio,
                hit_at_0_5_occ=hit_at_05_occ,
                hit_at_1_0_occ=hit_at_10_occ,
                hit_at_1_5_occ=hit_at_15_occ,
                hit_mean_occ=hit_mean_occ,
                collision_rate_occ=collision_rate_occ,
                sem_hit_at_0_5=sem_at_05,
                sem_hit_at_1_0=sem_at_10,
                sem_hit_at_1_5=sem_at_15,
                sem_hit_mean=sem_mean,
                snap_hit_at_0_5=snap_at_05,
                snap_hit_at_1_0=snap_at_10,
                snap_hit_at_1_5=snap_at_15,
                snap_hit_mean=snap_mean,
                snap_dist_mean=snap_dist_mean,
                snap_dist_mean_invalid=snap_dist_mean_invalid,
                sem_hit_at_0_5_occ=sem_at_05_occ,
                sem_hit_at_1_0_occ=sem_at_10_occ,
                sem_hit_at_1_5_occ=sem_at_15_occ,
                sem_hit_mean_occ=sem_mean_occ,
                snap_hit_at_0_5_occ=snap_at_05_occ,
                snap_hit_at_1_0_occ=snap_at_10_occ,
                snap_hit_at_1_5_occ=snap_at_15_occ,
                snap_hit_mean_occ=snap_mean_occ,
                snap_dist_mean_occ=snap_dist_mean_occ,
                snap_dist_mean_invalid_occ=snap_dist_mean_invalid_occ,

                wp_ade=wp_ade,
                wp_collision_score=wp_coll_score,
                wp_ade_occ=wp_ade_occ,
                wp_collision_score_occ=wp_coll_score_occ,
            )

        static_results = [r for r in results if r.get("capture_version", "") == "static"]
        dynamic_results = [r for r in results if r.get("capture_version", "") == "dynamic"]

        # Keep backward-compatible global metrics (returned), but print only per-version blocks.
        global_metrics = _compute_and_print(results, title="GLOBAL", print_report=False) if len(results) > 0 else {}
        static_metrics = _compute_and_print(static_results, title="STATIC") if len(static_results) > 0 else {}
        dynamic_metrics = _compute_and_print(dynamic_results, title="DYNAMIC") if len(dynamic_results) > 0 else {}
        if len(static_results) == 0:
            print("[ExternalMetric] No STATIC samples in results; skipping STATIC report.")
        if len(dynamic_results) == 0:
            print("[ExternalMetric] No DYNAMIC samples in results; skipping DYNAMIC report.")

        # ---- DEBUG DUMP: write all occluded-goal samples in one NPZ ----
        # (comment out this whole block to disable)
        if hasattr(self, "_occ_dump") and len(self._occ_dump) > 0:
            out_path = os.environ.get(
                "BEACON_OCC_DUMP", "debug_outputs/2_miss_when_occ_debug.npz"
            )
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

            raw_imgs = np.stack([d["raw_img"] for d in self._occ_dump], axis=0)  # (N,4,448,448,3)
            trav = np.stack([d["traversable_mask"] for d in self._occ_dump], axis=0)  # (N,128,128)
            vis = np.stack([d["visible_mask"] for d in self._occ_dump], axis=0)       # (N,128,128)
            aff = np.stack([d["affordance_mask"] for d in self._occ_dump], axis=0)    # (N,128,128)
            free_ground = np.stack([d["free_ground_obs"] for d in self._occ_dump], axis=0)  # (N,128,128)
            free_cam = np.stack([d["free_cam_obs"] for d in self._occ_dump], axis=0)        # (N,128,128)

            pred_xy = np.stack([d["pred_xy"] for d in self._occ_dump], axis=0)  # (N,2)
            p_goal_np = np.stack([d["p_goal"] for d in self._occ_dump], axis=0) # (N,3)
            instr = np.array([d["instruction"] for d in self._occ_dump], dtype=object)  # (N,)
            pred_aff = np.stack([d["pred_affordance"] for d in self._occ_dump], axis=0)  # (N,128,128)

            np.savez_compressed(
                out_path,
                raw_img=raw_imgs,
                traversable_mask=trav,
                visible_mask=vis,
                affordance_mask=aff,
                free_ground_obs=free_ground,
                free_cam_obs=free_cam,
                pred_xy=pred_xy,
                p_goal=p_goal_np,
                instruction=instr,
                pred_affordance=pred_aff,
            )
            print(f"[DEBUG] saved debug at: {out_path} (N={len(self._occ_dump)})")
        # -----------------------------------------------------------------


        # Return global + per-version metrics for logging.
        out = dict(global_metrics)
        for k, v in static_metrics.items():
            out[f"{k}_static"] = v
        for k, v in dynamic_metrics.items():
            out[f"{k}_dynamic"] = v
        return out

      
