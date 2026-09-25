import os
import math
import pickle
from collections import deque

import numpy as np
import trimesh
from time import time


class BEVMaskGenerator:
    def __init__(
        self,
        ann_file,
        out_dir,
        outer_half_xy,
        inner_half_xy,
        outer_pitch,
        inner_pitch,
        z_up,
        z_down,
        clearance_band,
        support_band,
        connectivity,
        corner_check,
        inner_vote_min,
    ):
        self.ann_file = ann_file
        self.out_dir = out_dir
        self.outer_half_xy = outer_half_xy
        self.inner_half_xy = inner_half_xy
        self.outer_pitch = outer_pitch
        self.inner_pitch = inner_pitch
        self.z_up = z_up
        self.z_down = z_down
        self.clearance_band = clearance_band
        self.support_band = support_band
        self.connectivity = connectivity
        self.corner_check = corner_check
        self.inner_vote_min = inner_vote_min

        self.mesh_cache = {}
        self._data_list = None

    def sanitize_token(self, token):
        return str(token).replace("/", "_").replace("\\", "_")

    def _get_data_list(self):
        if self._data_list is None:
            with open(self.ann_file, "rb") as f:
                info = pickle.load(f)
            self._data_list = info["data_list"] if isinstance(info, dict) and "data_list" in info else info
        return self._data_list

    def load_mesh_cached(self, path):
        if path in self.mesh_cache:
            return self.mesh_cache[path]
        mesh = trimesh.load(path, process=False)
        if isinstance(mesh, trimesh.Scene):
            try:
                geoms = mesh.to_geometry()
                if isinstance(geoms, dict):
                    mesh = trimesh.util.concatenate(tuple(geoms.values()))
                else:
                    mesh = trimesh.util.concatenate(tuple(geoms))
            except Exception:
                mesh = mesh.dump(concatenate=True)
        self.mesh_cache[path] = mesh
        return mesh

    def yaw_from_T_mp(self, T):
        return math.atan2(T[0, 2], T[2, 2])

    def crop_occ_from_mesh(self, mesh, center_xyz, yaw_anchor_rad, half_xy, z_up, z_down, pitch):
        if mesh is None or mesh.vertices.size == 0:
            print("Warning: empty mesh provided; returning empty occupancy grid.")
            return np.zeros((1, 1, 1), np.uint8)
        V = mesh.vertices.view(np.ndarray)
        F = mesh.faces.view(np.ndarray)

        cz, sz = math.cos(yaw_anchor_rad), math.sin(yaw_anchor_rad)
        R_wl = np.array([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]], np.float64)
        V_local = (V - center_xyz) @ R_wl.T

        inside = (
            (np.abs(V_local[:, 0]) <= half_xy)
            & (np.abs(V_local[:, 1]) <= half_xy)
            & (V_local[:, 2] <= z_up)
            & (V_local[:, 2] >= -z_down)
        )
        keep_faces = inside[F].any(axis=1)
        F_keep = F[keep_faces]

        xy_dim = int(round((2 * half_xy) / pitch))
        z_dim = int(round((z_up + z_down) / pitch))
        occ = np.zeros((xy_dim, xy_dim, z_dim), np.uint8)
        if F_keep.size == 0:
            print("Warning: no faces remain after cropping; returning empty occupancy grid.")
            return occ

        uidx = np.unique(F_keep)
        remap = -np.ones(V_local.shape[0], int)
        remap[uidx] = np.arange(uidx.shape[0])
        F_local = remap[F_keep]
        V_local_crop = V_local[uidx]

        mesh_local = trimesh.Trimesh(vertices=V_local_crop, faces=F_local, process=False)
        vg = mesh_local.voxelized(pitch=pitch)
        pts = None if vg is None else vg.points
        if pts is None or len(pts) == 0:
            print("Warning: voxelization produced no points; returning empty occupancy grid.")
            return occ

        x_min = -half_xy
        y_min = -half_xy
        z_min = -z_down
        for px, py, pz in pts:
            ix = int(np.round((px - x_min) / pitch))
            iy = int(np.round((py - y_min) / pitch))
            iz = int(np.round((pz - z_min) / pitch))
            if 0 <= ix < xy_dim and 0 <= iy < xy_dim and 0 <= iz < z_dim:
                occ[ix, iy, iz] = 1
        return occ

    def band_to_idx(self, z0, z1, z_min, pitch, nz):
        z0, z1 = sorted((z0, z1))
        g0 = int(np.floor((z0 - z_min) / pitch))
        g1 = int(np.floor((z1 - z_min) / pitch))
        g0 = max(0, min(nz - 1, g0))
        g1 = max(0, min(nz - 1, g1))
        if g1 < g0:
            g1 = g0
        return g0, g1

    def band_occupied(self, occ, band, pitch, z_min):
        g0, g1 = self.band_to_idx(band[0], band[1], z_min, pitch, occ.shape[2])
        return (occ[:, :, g0:g1 + 1] > 0).any(axis=2)

    def build_free_xy(self, occ, clearance_band, support_band, pitch, z_min):
        blocked = self.band_occupied(occ, clearance_band, pitch, z_min)
        support = self.band_occupied(occ, support_band, pitch, z_min)
        return (~blocked) & support

    def downsample_free_mask(self, free_inner, inner_pitch, outer_pitch, min_free_votes):
        ratio = int(round(outer_pitch / inner_pitch))
        if abs(ratio * inner_pitch - outer_pitch) > 1e-6:
            raise ValueError("inner_pitch must evenly divide outer_pitch.")
        nx, ny = free_inner.shape
        if nx % ratio != 0 or ny % ratio != 0:
            raise ValueError("inner free mask size not divisible by ratio.")
        out_x = nx // ratio
        out_y = ny // ratio
        counts = free_inner.reshape(out_x, ratio, out_y, ratio).sum(axis=(1, 3))
        return counts >= min_free_votes

    def overlay_inner_on_outer(self, free_outer, free_inner_ds, inner_half_xy, outer_half_xy, outer_pitch):
        outer_size = free_outer.shape[0]
        inner_size = free_inner_ds.shape[0]
        expected_inner_size = int(round((2 * inner_half_xy) / outer_pitch))
        if inner_size != expected_inner_size:
            raise ValueError("inner downsample size does not match expected size.")
        start = (outer_size - inner_size) // 2
        end = start + inner_size
        if start < 0 or end > outer_size:
            raise ValueError("inner overlay region out of bounds.")
        out = free_outer.copy()
        out[start:end, start:end] = free_inner_ds
        return out

    def reachable_mask(self, free_xy, start, connectivity=4, corner_check=True):
        sx, sy = start
        visited = np.zeros_like(free_xy, dtype=np.uint8)
        if not (0 <= sx < free_xy.shape[0] and 0 <= sy < free_xy.shape[1]):
            return visited
        if not free_xy[sx, sy]:
            return visited

        if connectivity == 8:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                         (-1, -1), (-1, 1), (1, -1), (1, 1)]
        else:
            neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        q = deque([(sx, sy)])
        visited[sx, sy] = 1
        while q:
            x, y = q.popleft()
            for dx, dy in neighbors:
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= free_xy.shape[0] or ny >= free_xy.shape[1]:
                    continue
                if not free_xy[nx, ny] or visited[nx, ny]:
                    continue
                if connectivity == 8 and corner_check and dx != 0 and dy != 0:
                    if not (free_xy[x + dx, y] or free_xy[x, y + dy]):  # less strict corner check
                        continue
                visited[nx, ny] = 1
                q.append((nx, ny))
        return visited

    def run_sample(self, yaw_deg, token_key="waypoint_token", save=False, target_token=None, sample=None, return_occluder: bool = False):
        yaw_input = math.radians(yaw_deg)
        if save and self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

        assert target_token is not None or sample is not None, "Either target_token or sample must be provided."
        if sample is None:
            data_list = self._get_data_list()
            for s in data_list:
                if s.get(token_key) == target_token:
                    sample = s
                    break
            if sample is None:
                raise ValueError(f"Token {target_token} not found using key {token_key}")

        wp_token = sample.get("waypoint_token") or target_token

        occ_rel = sample.get("occ_path", "")
        if not occ_rel:
            raise ValueError(f"Missing occ_path for waypoint {wp_token}")
        occ_path = occ_rel  # TODO: make it rel
        if not os.path.exists(occ_path):
            raise FileNotFoundError(f"Occupancy mesh not found: {occ_path}")

        ego2global = np.asarray(sample["ego2global"], dtype=np.float32)
        ego_T = ego2global[0] if ego2global.ndim == 3 else ego2global

        mesh = self.load_mesh_cached(occ_path)

        occ_outer = self.crop_occ_from_mesh(
            mesh, ego_T[:3, 3], self.yaw_from_T_mp(ego_T) + yaw_input,
            half_xy=self.outer_half_xy, z_up=self.z_up, z_down=self.z_down, pitch=self.outer_pitch
        )
        occluder_outer, max_z_outer = self.occluder_mask_from_occ(
            occ_outer,
            pitch=self.outer_pitch,
            z_min=-self.z_down,
            cam_z=0.0,
            eps=1e-3,
        )

        free_outer = self.build_free_xy(
            occ_outer, self.clearance_band, self.support_band, self.outer_pitch, -self.z_down
        )

        occ_inner = self.crop_occ_from_mesh(
            mesh, ego_T[:3, 3], self.yaw_from_T_mp(ego_T) + yaw_input,
            half_xy=self.inner_half_xy, z_up=self.z_up, z_down=self.z_down, pitch=self.inner_pitch
        )
        free_inner = self.build_free_xy(
            occ_inner, self.clearance_band, self.support_band, self.inner_pitch, -self.z_down
        )

        free_inner_ds = self.downsample_free_mask(
            free_inner, self.inner_pitch, self.outer_pitch, self.inner_vote_min
        )
        free_final = self.overlay_inner_on_outer(
            free_outer, free_inner_ds, self.inner_half_xy, self.outer_half_xy, self.outer_pitch
        )

        center_idx = int(round(self.outer_half_xy / self.outer_pitch))
        center_idx = max(0, min(free_final.shape[0] - 1, center_idx))
        center_x = center_idx
        center_y = center_idx

        # corner case handling: if center is not free, find closest free cell
        if not free_final[center_x, center_y]:
            print(f"Is the center free? {free_final[center_x, center_y]}")
            center_idx_candidates = np.argwhere(free_final)
            if center_idx_candidates.size == 0:
                print("Warning: no free cells in final mask.")
            else: # find the closest free cell to center
                # dists = np.linalg.norm(center_idx_candidates - np.array([center_x, center_y]), axis=1)
                # best_idx = np.argmin(dists)
                # best_cell = center_idx_candidates[best_idx]
                # center_x = best_cell[0]
                # center_y = best_cell[1]
                # print(f"Closest free cell to center is at {best_cell} with distance {dists[best_idx]:.3f}, distance should be at most 1")
                free_final[center_x, center_y] = True  # force it to be free
                print(f"Is center free now? {free_final[center_x, center_y]}")
        reachable = self.reachable_mask(
            free_final, (center_x, center_y),
            connectivity=self.connectivity, corner_check=self.corner_check
        )

        traversable_mask = reachable.astype(np.uint8)
        out_path = None
        if save:
            out_path = os.path.join(self.out_dir, f"{self.sanitize_token(wp_token)}.npz")
            np.savez_compressed(out_path, occ=occ_outer, traversable_mask=traversable_mask)
        if return_occluder:
            return traversable_mask, out_path, occluder_outer, max_z_outer
        return traversable_mask, out_path
    
    
    def oracle_free_point(self, yaw_deg, target_token, direction, step_size, token_key="waypoint_token"):
        # 1) get traversable mask in memory
        traversable_mask, _, occluder_outer, _ = self.run_sample(
            yaw_deg=yaw_deg,
            target_token=target_token,
            token_key=token_key,
            save=False,
            return_occluder=True,
        )
        trav_bool = traversable_mask.astype(bool)
        # # trial 1
        # limited_mask = self.limited_observation_mask_from_traversable(
        #     trav_bool,
        #     origin_radius_m=2.4,
        #     unknown_val=255,
        #     corner_check=True,
        #     return_2ch=True
        # ) # 2, H, W
        # trav_bool = limited_mask[0].astype(bool)  # use the limited traversable mask
        # # end of trial 1

        # trial 2
        limited_mask = self.limited_observation_mask_from_traversable(
            trav_bool,
            occluder_mask=occluder_outer,
            origin_radius_m=2.4,
            unknown_val=255,
            corner_check=True,
            return_2ch=True
        ) # 2, H, W
        combined = np.logical_and(trav_bool, limited_mask[1].astype(bool)) # both free and sure
        trav_bool = combined  # use the limited traversable mask
        # end of trial 2
        
        nx, ny = trav_bool.shape
        x_min = -self.outer_half_xy
        y_min = -self.outer_half_xy

        xs = x_min + (np.arange(nx) + 0.5) * self.outer_pitch
        ys = y_min + (np.arange(ny) + 0.5) * self.outer_pitch
        X, Y = np.meshgrid(xs, ys, indexing="ij")

        r = np.hypot(X, Y)
        angle_deg = (np.degrees(np.arctan2(Y, X)) + 360.0) % 360.0

        # 2) direction fan (CCW from +x)
        dir_key = direction.upper().replace(" ", "")
        dir_centers = {
            "FRONT": 0.0,
            "FRONTLEFT": 45.0,
            "LEFT": 90.0,
            "BACKLEFT": 135.0,
            "BACK": 180.0,
            "BACKRIGHT": 225.0,
            "RIGHT": 270.0,
            "FRONTRIGHT": 315.0,
        }
        if dir_key not in dir_centers:
            raise ValueError(f"Unknown direction: {direction}")

        center = dir_centers[dir_key]
        half_width = 22.5
        start = (center - half_width) % 360.0
        end = (center + half_width) % 360.0
        if start < end:
            angle_mask = (angle_deg >= start) & (angle_deg < end)
        else:
            angle_mask = (angle_deg >= start) | (angle_deg < end)

        # 3) ring mask
        step_key = step_size.upper().replace(" ", "")
        if step_key == "SMALL":
            ring_mask = r < 2.0
            ring_center = 1.0
        elif step_key == "BIG":
            ring_mask = (r >= 2.0) & (r <= self.outer_half_xy)
            ring_center = 0.5 * (2.0 + self.outer_half_xy)
        else:
            raise ValueError(f"Unknown step_size: {step_size}")

        fan_mask = angle_mask & ring_mask
        candidates = fan_mask & trav_bool

        if not candidates.any():
            empty_fan = np.zeros_like(traversable_mask, dtype=bool)
            return traversable_mask, empty_fan, (-1, -1), np.array([0.0, 0.0], dtype=np.float32)

        # 4) distance-to-obstacle + tie-breaks
        from scipy.ndimage import distance_transform_edt

        padded = np.pad(trav_bool, 1, mode="constant", constant_values=False)
        dist = distance_transform_edt(padded)[1:-1, 1:-1]

        cand_idx = np.argwhere(candidates)
        dt_vals = dist[cand_idx[:, 0], cand_idx[:, 1]]

        ang_vals = angle_deg[cand_idx[:, 0], cand_idx[:, 1]]
        ang_diff = np.abs(((ang_vals - center + 180.0) % 360.0) - 180.0)

        r_vals = r[cand_idx[:, 0], cand_idx[:, 1]]
        r_diff = np.abs(r_vals - ring_center)

        order = np.lexsort((r_diff, ang_diff, -dt_vals))
        best_ix, best_iy = cand_idx[order[0]]

        best_x = x_min + (best_ix + 0.5) * self.outer_pitch
        best_y = y_min + (best_iy + 0.5) * self.outer_pitch
        best_xy = np.array([best_x, best_y], dtype=np.float32)

        return traversable_mask, fan_mask, (int(best_ix), int(best_iy)), best_xy
    
    # --- begin of copied from the internvl_dataaset.py ---
    # small math helpers
    # --------------------------
    def wrap_to_pi(self, a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def yaw_to_rad_auto(self, yaw_val: float) -> float:
        """If someone stored degrees by accident, convert; else treat as radians."""
        y = float(yaw_val)
        if abs(y) > (2.0 * math.pi + 1e-3):  # likely degrees
            return math.radians(y)
        return y

    def yaw_rad_to_deg_wrapped(self, yaw_rad: float) -> float:
        return math.degrees(self.wrap_to_pi(float(yaw_rad)))
    
    # --------------------------
    # end of copied from the internvl_dataset.py ---
    
    def oracle_bev_from_dataloader(self, sample, return_limited_mask: bool = False, origin_radius_m: float = 2.4):
        t_start = time()
        base_yaw_rad = self.yaw_to_rad_auto(sample["habitat_base_yaw"])  # magic fix is gone LOL
        base_yaw_deg = self.yaw_rad_to_deg_wrapped(base_yaw_rad)
        traversable_mask, _, occluder_outer, _ = self.run_sample(
            yaw_deg=base_yaw_deg,
            save=False,
            sample=sample,
            return_occluder=True,
        )
        trav_bool = traversable_mask.astype(bool)

        if not return_limited_mask:
            return trav_bool  # H,W bool (your original behavior)

        # limited-observation map: uint8 {1 free, 0 not free, 255 unknown}    
        limited_mask = self.limited_observation_mask_from_traversable(
            trav_bool,
            occluder_mask=occluder_outer,
            origin_radius_m=origin_radius_m,
            unknown_val=255,
            corner_check=True,
            return_2ch=True
        ) # 2, H, W
        limited_mask[0] = trav_bool.astype(np.uint8) #  we still return the raw traversable mask as the first channel
        # # DEBUG
        # unknown = (limited_mask[1] == 0).astype(np.uint8) # sure channel: 1 sure, 0 unknown
        # combined = np.logical_and(trav_bool, limited_mask[1].astype(bool))

        # self.save_vis(traversable_mask=trav_bool.astype(np.uint8),
        #     unknown_mask=unknown,
        #     save_path='debug_outputs/traversable_with_unknown.png',
        # )
        # self.save_vis(
        #     traversable_mask=trav_bool,
        #     save_path='debug_outputs/traversable_raw.png',
        # )
        # self.save_vis(
        #     traversable_mask=combined,
        #     save_path='debug_outputs/traversable_limited.png',
        # )
        # exit(0)
        # # end of DEBUG
        t_end = time()
        print(f"[Oracle BEV] time taken: {t_end - t_start:.3f} s")
        return limited_mask

    
    # --------------------------
    # Limited-observation mask on top of traversable_mask
    # --------------------------
    def _local_reachable_within_steps_4n(self, trav: np.ndarray, start, max_steps: int) -> np.ndarray:
        """
        4-neighbor BFS on traversable (bool) with a hard step budget.
        Returns bool mask of cells reachable from start within <= max_steps moves.
        """
        nx, ny = trav.shape
        sx, sy = start
        out = np.zeros_like(trav, dtype=bool)
        if not (0 <= sx < nx and 0 <= sy < ny) or not trav[sx, sy]:
            return out

        dist = -np.ones((nx, ny), dtype=np.int16)
        q = deque()
        dist[sx, sy] = 0
        out[sx, sy] = True
        q.append((sx, sy))

        nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while q:
            x, y = q.popleft()
            d = dist[x, y]
            if d >= max_steps:
                continue
            nd = d + 1
            for dx, dy in nbrs:
                xx, yy = x + dx, y + dy
                if xx < 0 or yy < 0 or xx >= nx or yy >= ny:
                    continue
                if dist[xx, yy] != -1:
                    continue
                if not trav[xx, yy]:
                    continue
                dist[xx, yy] = nd
                out[xx, yy] = True
                q.append((xx, yy))
        return out

    def _boundary_cells(self, nx: int, ny: int):
        """All boundary cell indices (unique), as a list of (x,y)."""
        out = []
        # top/bottom rows
        for x in range(nx):
            out.append((x, 0))
            if ny > 1:
                out.append((x, ny - 1))
        # left/right cols (excluding corners to avoid duplicates)
        for y in range(1, ny - 1):
            out.append((0, y))
            if nx > 1:
                out.append((nx - 1, y))
        return out

    # ===================== newly added =====================
    def max_z_map_from_occ(self, occ: np.ndarray, pitch: float, z_min: float) -> np.ndarray:
        """
        occ: (H,W,Z) uint8 {0,1}
        Returns max_z_map (H,W) float32, -inf where empty.
        z at voxel center: z_min + (iz + 0.5)*pitch
        """
        nx, ny, nz = occ.shape
        max_z = np.full((nx, ny), -np.inf, dtype=np.float32)
        idx = np.argwhere(occ > 0)
        if idx.size == 0:
            return max_z
        ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
        z = z_min + (iz.astype(np.float32) + 0.5) * pitch
        np.maximum.at(max_z, (ix, iy), z)
        return max_z

    def occluder_mask_from_occ(
        self,
        occ: np.ndarray,
        pitch: float,
        z_min: float,
        cam_z: float = 0.0,
        eps: float = 1e-3,
    ):
        """
        Occluder if max_z >= cam_z - eps.
        Returns: occluder(bool H,W), max_z(float32 H,W)
        """
        max_z = self.max_z_map_from_occ(occ, pitch=pitch, z_min=z_min)
        occluder = max_z >= (cam_z - eps)
        return occluder, max_z

    def _cast_ray_sure_to_target(
        self,
        stop_mask: np.ndarray,   # bool H,W : True means occluder (stop here)
        sure: np.ndarray,        # bool H,W : will be updated
        start_xy,
        target_xy,
        corner_check: bool = True,
        eps: float = 1e-12,
    ):
        """
        DDA traversal from start cell center -> target cell center.
        - Marks every visited cell as sure=True
        - Stops when entering stop_mask cell (occluder), but that cell is sure too.
        Corner anti-squeeze: if crossing exact corner and BOTH side cells are stop_mask, stop.
        """
        nx, ny = stop_mask.shape
        sx, sy = start_xy
        tx, ty = target_xy
        if not (0 <= sx < nx and 0 <= sy < ny):
            return
        if (sx, sy) == (tx, ty):
            sure[sx, sy] = True
            return

        x0, y0 = sx + 0.5, sy + 0.5
        x1, y1 = tx + 0.5, ty + 0.5
        dx, dy = x1 - x0, y1 - y0

        step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
        step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

        inv_dx = 1.0 / dx if dx != 0 else float("inf")
        inv_dy = 1.0 / dy if dy != 0 else float("inf")
        t_delta_x = abs(inv_dx) if dx != 0 else float("inf")
        t_delta_y = abs(inv_dy) if dy != 0 else float("inf")

        if step_x > 0:
            t_max_x = ((sx + 1) - x0) * abs(inv_dx) if dx != 0 else float("inf")
        elif step_x < 0:
            t_max_x = (x0 - sx) * abs(inv_dx) if dx != 0 else float("inf")
        else:
            t_max_x = float("inf")

        if step_y > 0:
            t_max_y = ((sy + 1) - y0) * abs(inv_dy) if dy != 0 else float("inf")
        elif step_y < 0:
            t_max_y = (y0 - sy) * abs(inv_dy) if dy != 0 else float("inf")
        else:
            t_max_y = float("inf")

        x, y = sx, sy
        sure[x, y] = True
        if stop_mask[x, y]:
            return

        max_iter = (nx + ny) * 3 + 10
        for _ in range(max_iter):
            if (x, y) == (tx, ty):
                return

            if abs(t_max_x - t_max_y) <= eps:
                # corner crossing
                if step_x == 0 or step_y == 0:
                    # degenerate
                    if t_max_x < t_max_y:
                        t_max_x = float("inf")
                    else:
                        t_max_y = float("inf")
                    continue

                x_side1, y_side1 = x + step_x, y
                x_side2, y_side2 = x, y + step_y
                b1 = (0 <= x_side1 < nx and 0 <= y_side1 < ny and stop_mask[x_side1, y_side1])
                b2 = (0 <= x_side2 < nx and 0 <= y_side2 < ny and stop_mask[x_side2, y_side2])

                if corner_check and b1 and b2:
                    sure[x_side1, y_side1] = True
                    sure[x_side2, y_side2] = True
                    return

                x += step_x
                y += step_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
            elif t_max_x < t_max_y:
                x += step_x
                t_max_x += t_delta_x
            else:
                y += step_y
                t_max_y += t_delta_y

            if x < 0 or y < 0 or x >= nx or y >= ny:
                return

            sure[x, y] = True
            if stop_mask[x, y]:
                return
    # ===================== newly added end =====================

    # def _cast_ray_to_target(
    #     self,
    #     trav: np.ndarray,
    #     observed: np.ndarray,
    #     start_xy,
    #     target_xy,
    #     corner_check: bool = True,
    #     eps: float = 1e-12,
    # ):
    #     """
    #     Grid traversal from start cell center to target cell center.
    #     - Marks traversable cells as 1.
    #     - Stops at first non-traversable cell, marks it 0.
    #     - Leaves everything else as 255 (unknown).
    #     Corner rule (anti-squeeze):
    #       if crossing an exact grid corner and BOTH side-adjacent cells are non-traversable,
    #       stop (and mark those side cells 0 if in-bounds).
    #     """
    #     nx, ny = trav.shape
    #     sx, sy = start_xy
    #     tx, ty = target_xy

    #     # start already should be marked by caller; still guard
    #     if not (0 <= sx < nx and 0 <= sy < ny):
    #         return
    #     if (sx, sy) == (tx, ty):
    #         return

    #     # Continuous coordinates in "cell units": boundaries at integers, centers at i+0.5
    #     x0, y0 = sx + 0.5, sy + 0.5
    #     x1, y1 = tx + 0.5, ty + 0.5
    #     dx, dy = x1 - x0, y1 - y0

    #     # Direction steps
    #     step_x = 1 if dx > 0 else (-1 if dx < 0 else 0)
    #     step_y = 1 if dy > 0 else (-1 if dy < 0 else 0)

    #     # Parametric traversal increments
    #     inv_dx = 1.0 / dx if dx != 0 else float("inf")
    #     inv_dy = 1.0 / dy if dy != 0 else float("inf")
    #     t_delta_x = abs(inv_dx) if dx != 0 else float("inf")
    #     t_delta_y = abs(inv_dy) if dy != 0 else float("inf")

    #     # Distance (in t) to first vertical/horizontal grid boundary
    #     if step_x > 0:
    #         next_xb = (sx + 1)
    #         t_max_x = (next_xb - x0) * abs(inv_dx) if dx != 0 else float("inf")
    #     elif step_x < 0:
    #         next_xb = sx
    #         t_max_x = (x0 - next_xb) * abs(inv_dx) if dx != 0 else float("inf")
    #     else:
    #         t_max_x = float("inf")

    #     if step_y > 0:
    #         next_yb = (sy + 1)
    #         t_max_y = (next_yb - y0) * abs(inv_dy) if dy != 0 else float("inf")
    #     elif step_y < 0:
    #         next_yb = sy
    #         t_max_y = (y0 - next_yb) * abs(inv_dy) if dy != 0 else float("inf")
    #     else:
    #         t_max_y = float("inf")

    #     x, y = sx, sy
    #     # Safety cap: worst-case steps ~ nx+ny; a bit more for ties
    #     max_iter = (nx + ny) * 3 + 10

    #     for _ in range(max_iter):
    #         if (x, y) == (tx, ty):
    #             break

    #         # Corner (tie): cross a grid corner
    #         if abs(t_max_x - t_max_y) <= eps:
    #             if step_x == 0 or step_y == 0:
    #                 # Degenerate, treat as axis move
    #                 if t_max_x < t_max_y:
    #                     t_max_x = float("inf")
    #                 else:
    #                     t_max_y = float("inf")
    #                 continue

    #             # Anti-squeeze: if both side-adjacent cells are blocked, stop
    #             x_side1, y_side1 = x + step_x, y
    #             x_side2, y_side2 = x, y + step_y
    #             b1 = (0 <= x_side1 < nx and 0 <= y_side1 < ny and not trav[x_side1, y_side1])
    #             b2 = (0 <= x_side2 < nx and 0 <= y_side2 < ny and not trav[x_side2, y_side2])

    #             if corner_check and b1 and b2:
    #                 observed[x_side1, y_side1] = 0
    #                 observed[x_side2, y_side2] = 0
    #                 return

    #             # Otherwise move diagonally (do NOT “touch” side cells)
    #             x += step_x
    #             y += step_y
    #             t_max_x += t_delta_x
    #             t_max_y += t_delta_y
    #         elif t_max_x < t_max_y:
    #             x += step_x
    #             t_max_x += t_delta_x
    #         else:
    #             y += step_y
    #             t_max_y += t_delta_y

    #         if x < 0 or y < 0 or x >= nx or y >= ny:
    #             return

    #         if trav[x, y]:
    #             observed[x, y] = 1
    #         else:
    #             observed[x, y] = 0
    #             return

    def limited_observation_mask_from_traversable(
        self,
        traversable_mask: np.ndarray,
        occluder_mask: np.ndarray = None,
        origin_radius_m: float = 2.4,
        unknown_val: int = 255,
        corner_check: bool = True,
        return_2ch=True,
    ) -> np.ndarray:
        """
        Build limited-observation mask on top of traversable_mask (bool or uint8).
        Output uint8 with {1 free, 0 not free, 255 unknown}.
        """
        trav = traversable_mask.astype(bool)
        nx, ny = trav.shape

        center_idx = int(round(self.outer_half_xy / self.outer_pitch))
        center_idx = max(0, min(nx - 1, center_idx))
        start = (center_idx, center_idx)

        max_steps = int(math.floor(origin_radius_m / self.outer_pitch + 1e-6))
        origin_mask = self._local_reachable_within_steps_4n(trav, start, max_steps=max_steps)

        # stopping condition for visibility:
        # - if occluder_mask provided: stop at both non-traversable and with tall occluders (height >= cam), 
        # - - cuz occluders are generated from low resolution so should yield to traversable mask with inner range from high-res
        # - else fallback: stop at non-traversable (old behavior)
        if occluder_mask is not None:
            stop_mask = (~trav) & occluder_mask.astype(bool)
        else:
            stop_mask = ~trav


        targets = self._boundary_cells(nx, ny)
        origins = np.argwhere(origin_mask)

        sure = np.zeros((nx, ny), dtype=bool)
        sure[origin_mask] = True

        for ox, oy in origins:
            s = (int(ox), int(oy))
            for t in targets:
                self._cast_ray_sure_to_target(stop_mask, sure, s, t, corner_check=corner_check)

        if return_2ch:
            trav_ch = trav.astype(np.uint8)      # you overwrite later anyway
            sure_ch = sure.astype(np.uint8)      # 1 known, 0 unknown
            return np.stack([trav_ch, sure_ch], axis=0)
        else:
            return sure.astype(np.uint8)

    
    @staticmethod
    def save_vis(
        traversable_mask: np.ndarray,
        unknown_mask: np.ndarray = None,   # <-- NEW: bool or 0/1 mask, True=unknown
        fan_mask: np.ndarray = None,
        best_xy=None,
        best_idx=(-1, -1),
        save_path: str = "bev_debug.png",
        title: str = "BEV Debug",
        dpi: int = 200,
        unknown_alpha: float = 0.35,
        half_xy: float = 6.4,
        pitch: float = 0.2,
    ):
        import matplotlib.pyplot as plt
        import numpy as np

        nx, ny = traversable_mask.shape
        x_min, y_min = -half_xy, -half_xy
        x_max, y_max = x_min + nx * pitch, y_min + ny * pitch

        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

        # Base traversable
        ax.imshow(
            traversable_mask.T,
            origin="lower",
            cmap="gray",
            extent=[x_min, x_max, y_min, y_max],
            interpolation="nearest",
            aspect="equal",
        )

        # Unknown fog overlay (uniform gray wherever unknown_mask==True)
        if unknown_mask is not None:
            if unknown_mask.shape != (nx, ny):
                raise ValueError(f"unknown_mask shape {unknown_mask.shape} must match traversable_mask {(nx, ny)}")
            unk = unknown_mask.astype(bool)

            gray_field = np.full((ny, nx), 0.5, dtype=np.float32)  # note transpose shape
            fog = np.ma.masked_where(~unk.T, gray_field)
            ax.imshow(
                fog,
                origin="lower",
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
                extent=[x_min, x_max, y_min, y_max],
                interpolation="nearest",
                alpha=unknown_alpha,
            )

        # Fan overlay
        if fan_mask is not None:
            fan_overlay = np.ma.masked_where(~fan_mask.T, fan_mask.T)
            ax.imshow(
                fan_overlay,
                origin="lower",
                cmap="gray",
                extent=[x_min, x_max, y_min, y_max],
                interpolation="nearest",
                alpha=0.35,
            )

        # Best point
        if best_xy is not None:
            label = "selected" if best_idx != (-1, -1) else "default origin"
            ax.scatter(
                [best_xy[0]], [best_xy[1]],
                s=60, c="cyan", edgecolors="black", linewidths=0.8,
                label=label,
            )
            ax.legend(loc="upper right")

        ax.set_xlabel("Forward X (m)")
        ax.set_ylabel("Left Y (m)")
        ax.set_title(title)

        im = ax.images[0]
        fig.colorbar(im, ax=ax, label="Traversable", fraction=0.046, pad=0.04)

        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def oracle_free_xy_from_dir48(
        self,
        yaw_deg,
        target_token,
        dir_id: int,                      # 0..47, bin [k*7.5, (k+1)*7.5)
        token_key: str = "waypoint_token",
        max_range_m: float = 6.4,
        r_min_m: float = 0.5,             # set >0 (e.g. 0.3) if you want to ignore near-origin cells
    ):
        """
        Returns:
        best_xy: np.ndarray (2,) float32
            - uses your combined mask (free & sure)
            - casts 2 rays inside the 7.5° bin: +1.875° and +5.625°
            - chooses the ray with MAX number of free segments (conservative vs corner-slip)
            - then chooses the FURTHEST free segment along that ray
            - returns the midpoint of that furthest segment in meters (x forward, y left)
        """

        # ---------- helpers (nested, so "single function") ----------
        def bresenham(i0, j0, i1, j1):
            """List grid cells (i,j) on a straight line between endpoints."""
            i0, j0, i1, j1 = int(i0), int(j0), int(i1), int(j1)
            di = abs(i1 - i0)
            dj = abs(j1 - j0)
            si = 1 if i0 < i1 else -1
            sj = 1 if j0 < j1 else -1
            err = di - dj
            ii, jj = [], []
            i, j = i0, j0
            while True:
                ii.append(i); jj.append(j)
                if i == i1 and j == j1:
                    break
                e2 = 2 * err
                if e2 > -dj:
                    err -= dj
                    i += si
                if e2 < di:
                    err += di
                    j += sj
            return np.asarray(ii, np.int32), np.asarray(jj, np.int32)

        def ij_to_xy(i, j):
            x = x_min + (i + 0.5) * pitch
            y = y_min + (j + 0.5) * pitch
            return x, y

        def xy_to_ij(x, y):
            i = int(np.round((x - x_min) / pitch - 0.5))
            j = int(np.round((y - y_min) / pitch - 0.5))
            i = int(np.clip(i, 0, nx - 1))
            j = int(np.clip(j, 0, ny - 1))
            return i, j

        def ray_endpoint_ij(theta_rad):
            """
            Origin at (0,0). Clamp by (a) map boundary (square) and (b) max_range_m.
            """
            c = float(np.cos(theta_rad))
            s = float(np.sin(theta_rad))
            eps = 1e-9

            # map is [-outer_half_xy, outer_half_xy] in both x and y
            half = float(self.outer_half_xy)

            # t to hit x boundary
            if abs(c) < eps:
                t_x = float("inf")
            else:
                t_x = (half / c) if c > 0 else (-half / c)  # positive

            # t to hit y boundary
            if abs(s) < eps:
                t_y = float("inf")
            else:
                t_y = (half / s) if s > 0 else (-half / s)  # positive

            t_bound = min(t_x, t_y)
            t = min(max_range_m, t_bound)

            x_end = t * c
            y_end = t * s
            return xy_to_ij(x_end, y_end)

        def analyze_ray(ii, jj):
            """
            Returns a tuple describing the chosen point on this ray:
            (n_sections, furthest_end_k, seg_len, mid_i, mid_j)
            or None if no free segments.
            """
            if ii.size == 0:
                return None

            # optional: skip near-origin steps
            if r_min_m > 0:
                k0 = int(np.ceil(r_min_m / pitch))
                if k0 >= ii.size:
                    return None
                ii2, jj2 = ii[k0:], jj[k0:]
            else:
                k0 = 0
                ii2, jj2 = ii, jj

            ray = free[ii2, jj2]  # bool 1D
            L = ray.shape[0]
            if L == 0:
                return None

            # find all True segments
            segments = []  # list of (start_k, end_k) in the *ii2/jj2* index space
            k = 0
            while k < L:
                if not ray[k]:
                    k += 1
                    continue
                s0 = k
                while k < L and ray[k]:
                    k += 1
                e0 = k - 1
                segments.append((s0, e0))

            n_sections = len(segments)
            if n_sections == 0:
                return None

            # choose the FURTHEST segment (largest end index)
            s_best, e_best = max(segments, key=lambda se: se[1])
            seg_len = e_best - s_best + 1
            mid_k = (s_best + e_best) // 2

            mi = int(ii2[mid_k])
            mj = int(jj2[mid_k])

            # convert back to "original" k along full ray (for consistent tie-break)
            furthest_end_k = e_best + k0

            return (n_sections, furthest_end_k, seg_len, mi, mj)

        # ---------- 1) get combined (free & sure) mask ----------
        traversable_mask, _, occluder_outer, _ = self.run_sample(
            yaw_deg=yaw_deg,
            target_token=target_token,
            token_key=token_key,
            save=False,
            return_occluder=True,
        )
        trav_bool = traversable_mask.astype(bool)

        limited_mask = self.limited_observation_mask_from_traversable(
            trav_bool,
            occluder_mask=occluder_outer,
            origin_radius_m=2.4,
            unknown_val=255,
            corner_check=True,
            return_2ch=True
        )  # (2, H, W)

        free = np.logical_and(trav_bool, limited_mask[1].astype(bool))  # free & sure

        nx, ny = free.shape
        x_min = -float(self.outer_half_xy)
        y_min = -float(self.outer_half_xy)
        pitch = float(self.outer_pitch)

        # origin cell index (closest to (0,0))
        i0, j0 = xy_to_ij(0.0, 0.0)

        # ---------- 2) two rays inside the bin ----------
        bw = 360.0 / 48.0  # 7.5°
        yaw0 = (int(dir_id) % 48) * bw
        yaws = [yaw0 + bw * 0.25, yaw0 + bw * 0.75]  # 1.875°, 5.625° inside the bin

        candidates = []
        for y in yaws:
            theta = np.deg2rad(y % 360.0)
            i1, j1 = ray_endpoint_ij(theta)
            ii, jj = bresenham(i0, j0, i1, j1)
            cand = analyze_ray(ii, jj)
            if cand is not None:
                candidates.append(cand)

        if len(candidates) == 0:
            return np.array([0.0, 0.0], dtype=np.float32)

        # ---------- 3) choose ray result conservatively ----------
        # primary: MAX n_sections
        # tie1:    MAX furthest_end_k (more "behind")
        # tie2:    MAX seg_len
        best = max(candidates, key=lambda t: (t[0], t[1], t[2]))
        _, _, _, bi, bj = best

        best_x, best_y = ij_to_xy(bi, bj)
        return np.array([best_x, best_y], dtype=np.float32)

    # def oracle_free_point_from_nodes(
    #     self,
    #     yaw_deg,
    #     target_token,
    #     node_xy_list,
    #     token_key="waypoint_token",
    #     origin_radius_m=2.4,
    #     debug=False,
    #     debug_path=None,
    # ):
    #     """
    #     Given ego-frame nodes [p0, p1, ..., pN] (each (x,y) in meters), find the last free
    #     cell along the first segment (in priority order) that contains any free cell.
    #     Priority: p1->p0, p2->p0, ..., pN->p0, then p2->p1, p3->p1, ..., pN->p1, etc.
    #     If all fail, fallback to origin (0,0)->p0. Returns XY (meters) of the chosen cell.
    #     Free means traversable AND sure (combined limited-observation mask).
    #     """
    #     if node_xy_list is None or len(node_xy_list) == 0:
    #         raise ValueError("node_xy_list must contain at least p0.")
    #     nodes = [tuple(map(float, p)) for p in node_xy_list]

    #     # 1) build combined free mask
    #     traversable_mask, _, occluder_outer, _ = self.run_sample(
    #         yaw_deg=yaw_deg,
    #         target_token=target_token,
    #         token_key=token_key,
    #         save=False,
    #         return_occluder=True,
    #     )
    #     trav_bool = traversable_mask.astype(bool)
    #     limited_mask = self.limited_observation_mask_from_traversable(
    #         trav_bool,
    #         occluder_mask=occluder_outer,
    #         origin_radius_m=origin_radius_m,
    #         unknown_val=255,
    #         corner_check=True,
    #         return_2ch=True,
    #     )
    #     sure_mask = limited_mask[1].astype(bool)
    #     free_mask = np.logical_and(trav_bool, sure_mask)

    #     nx, ny = free_mask.shape
    #     pitch = self.outer_pitch
    #     x_min = -self.outer_half_xy
    #     y_min = -self.outer_half_xy

    #     def xy_to_idx(x, y):
    #         ix = int(np.round((x - x_min) / pitch - 0.5))
    #         iy = int(np.round((y - y_min) / pitch - 0.5))
    #         if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
    #             return None
    #         return ix, iy

    #     def idx_to_xy(ix, iy):
    #         x = x_min + (ix + 0.5) * pitch
    #         y = y_min + (iy + 0.5) * pitch
    #         return np.array([x, y], dtype=np.float32)

    #     def grid_line(idx0, idx1):
    #         x0, y0 = idx0
    #         x1, y1 = idx1
    #         dx = abs(x1 - x0)
    #         dy = abs(y1 - y0)
    #         sx = 1 if x0 < x1 else -1
    #         sy = 1 if y0 < y1 else -1
    #         err = dx - dy
    #         x, y = x0, y0
    #         while True:
    #             yield x, y
    #             if (x, y) == (x1, y1):
    #                 break
    #             e2 = 2 * err
    #             if e2 > -dy:
    #                 err -= dy
    #                 x += sx
    #             if e2 < dx:
    #                 err += dx
    #                 y += sy

    #     best_idx = None
    #     used_segment = None

    #     # segment priority: pj->pi with j>i, sweeping i=0..N-1
    #     num_nodes = len(nodes)
    #     for end_idx in range(num_nodes - 1):
    #         for start_idx in range(end_idx + 1, num_nodes):
    #             start_grid = xy_to_idx(*nodes[start_idx])
    #             end_grid = xy_to_idx(*nodes[end_idx])
    #             if start_grid is None or end_grid is None:
    #                 continue
    #             last_free = None
    #             for ix, iy in grid_line(start_grid, end_grid):
    #                 if 0 <= ix < nx and 0 <= iy < ny and free_mask[ix, iy]:
    #                     last_free = (ix, iy)
    #             if last_free is not None:
    #                 best_idx = last_free
    #                 used_segment = (start_idx, end_idx)
    #                 break
    #         if best_idx is not None:
    #             break

    #     # fallback: origin -> p0
    #     if best_idx is None:
    #         origin_grid = xy_to_idx(0.0, 0.0)
    #         end_grid = xy_to_idx(*nodes[0])
    #         if origin_grid is None or end_grid is None:
    #             raise ValueError("Origin or p0 is out of BEV bounds.")
    #         last_free = None
    #         for ix, iy in grid_line(origin_grid, end_grid):
    #             if 0 <= ix < nx and 0 <= iy < ny and free_mask[ix, iy]:
    #                 last_free = (ix, iy)
    #         if last_free is None:
    #             raise RuntimeError("Fallback (0,0)->p0 found no free cell, unexpected.")
    #         best_idx = last_free
    #         used_segment = ("origin", 0)

    #     pred_xy = idx_to_xy(*best_idx)

    #     if debug and debug_path is not None:
    #         import matplotlib.pyplot as plt

    #         plt.figure(figsize=(6, 6))
    #         plt.imshow(free_mask.T, origin="lower", cmap="gray")
    #         # mark nodes with numbers
    #         for i, (x, y) in enumerate(nodes):
    #             idx = xy_to_idx(x, y)
    #             if idx is None:
    #                 continue
    #             px, py = idx
    #             plt.scatter(px, py, s=30, c="lime", edgecolors="black", linewidths=0.5)
    #             plt.text(px + 0.2, py + 0.2, str(i), color="yellow", fontsize=8)
    #         # mark prediction
    #         plt.scatter(best_idx[0], best_idx[1], s=50, c="red", marker="x", linewidths=1.5)
    #         plt.title(f"oracle_free_point_from_nodes | seg={used_segment}")
    #         plt.tight_layout()
    #         plt.savefig(debug_path, dpi=200)
    #         plt.close()

    #     return pred_xy # np.ndarray (2,) float32

    def oracle_free_point_from_nodes(
        self,
        yaw_deg,
        target_token,
        node_xy_list,
        token_key="waypoint_token",
        origin_radius_m=2.4,
        debug=False,
        debug_path=None,
        selected_node_xy=None,
        goal_xy=None,
        instruction=None,
    ):
        """
        Given ego-frame nodes [p0, p1, ..., pN] (each (x,y) meters), find the second-to-last
        free-and-sure cell along the first valid segment (priority below) that has >=2 free cells.
        Priority: p1->p0, p2->p0, ..., pN->p0, then p2->p1, p3->p1, ..., pN->p1, etc.
        If all fail, fallback to origin (0,0)->p0 (also requires >=2 free cells).
        Returns XY (meters) of the chosen cell.
        instruction: optional string to append to the plot title.
        """
        if node_xy_list is None or len(node_xy_list) == 0:
            raise ValueError("node_xy_list must contain at least p0.")
        if debug and selected_node_xy is None:
            raise ValueError("selected_node_xy must be provided when debug=True.")
        nodes = [tuple(map(float, p[:2])) for p in node_xy_list]

        traversable_mask, _, occluder_outer, _ = self.run_sample(
            yaw_deg=yaw_deg,
            target_token=target_token,
            token_key=token_key,
            save=False,
            return_occluder=True,
        )
        trav_bool = traversable_mask.astype(bool)
        limited_mask = self.limited_observation_mask_from_traversable(
            trav_bool,
            occluder_mask=occluder_outer,
            origin_radius_m=origin_radius_m,
            unknown_val=255,
            corner_check=True,
            return_2ch=True,
        )
        sure_mask = limited_mask[1].astype(bool)
        free_mask = np.logical_and(trav_bool, sure_mask)

        nx, ny = free_mask.shape
        pitch = self.outer_pitch
        x_min = -self.outer_half_xy
        y_min = -self.outer_half_xy

        def xy_to_idx(x, y):
            ix = int(np.round((x - x_min) / pitch - 0.5))
            iy = int(np.round((y - y_min) / pitch - 0.5))
            if ix < 0 or iy < 0 or ix >= nx or iy >= ny:
                return None
            return ix, iy

        def idx_to_xy(ix, iy):
            x = x_min + (ix + 0.5) * pitch
            y = y_min + (iy + 0.5) * pitch
            return np.array([x, y], dtype=np.float32)

        def grid_line(idx0, idx1):
            x0, y0 = idx0
            x1, y1 = idx1
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy
            x, y = x0, y0
            while True:
                yield x, y
                if (x, y) == (x1, y1):
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

        best_idx = None
        used_segment = None
        seg_start_idx = None
        seg_end_idx = None

        num_nodes = len(nodes)
        for end_idx in range(num_nodes - 1):
            for start_idx in range(end_idx + 1, num_nodes):
                start_grid = xy_to_idx(*nodes[start_idx])
                end_grid = xy_to_idx(*nodes[end_idx])
                if start_grid is None or end_grid is None:
                    continue
                free_cells = []
                for ix, iy in grid_line(start_grid, end_grid):
                    if 0 <= ix < nx and 0 <= iy < ny and free_mask[ix, iy]:
                        free_cells.append((ix, iy))
                if len(free_cells) >= 2:
                    best_idx = free_cells[-2]  # second-to-last free cell
                    used_segment = (start_idx, end_idx)
                    seg_start_idx, seg_end_idx = start_idx, end_idx
                    break
            if best_idx is not None:
                break

        if best_idx is None:
            origin_grid = xy_to_idx(0.0, 0.0)
            end_grid = xy_to_idx(*nodes[0])
            if origin_grid is None or end_grid is None:
                raise ValueError("Origin or p0 is out of BEV bounds.")
            free_cells = []
            for ix, iy in grid_line(origin_grid, end_grid):
                if 0 <= ix < nx and 0 <= iy < ny and free_mask[ix, iy]:
                    free_cells.append((ix, iy))
            if len(free_cells) < 2:
                raise RuntimeError("Fallback (0,0)->p0 found fewer than 2 free cells, unexpected.")
            best_idx = free_cells[-2]
            used_segment = ("origin", 0)
            seg_start_idx, seg_end_idx = None, 0

        pred_xy = idx_to_xy(*best_idx)

        if debug and debug_path is not None:
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D

            extent = [x_min, x_min + pitch * nx, y_min, y_min + pitch * ny]
            plt.figure(figsize=(6, 6))
            plt.imshow(free_mask.T, origin="lower", cmap="gray", extent=extent)

            handles = []

            ego_sc = plt.scatter(0.0, 0.0, s=50, c="red", marker="o",
                                 edgecolors="black", linewidths=0.6, alpha=0.9)
            handles.append(Line2D([0], [0], marker="o", color="w", label="ego",
                                  markerfacecolor="red", markeredgecolor="black", markersize=7, alpha=0.9))

            node_sc = None
            for i, (x, y) in enumerate(nodes):
                node_sc = plt.scatter(x, y, s=30, c="lime", edgecolors="black",
                                      linewidths=0.5, alpha=0.7)
                plt.text(x + 0.05, y + 0.05, str(i), color="yellow", fontsize=8, alpha=0.8)
            if node_sc is not None:
                handles.append(Line2D([0], [0], marker="o", color="w", label="nodes",
                                      markerfacecolor="lime", markeredgecolor="black", markersize=6, alpha=0.7))

            if seg_start_idx is not None and seg_end_idx is not None:
                sx, sy = nodes[seg_start_idx]
                ex, ey = nodes[seg_end_idx]
                plt.scatter(sx, sy, s=70, c="cyan", marker="^", edgecolors="black",
                            linewidths=0.8, alpha=0.6)
                plt.scatter(ex, ey, s=70, c="magenta", marker="D", edgecolors="black",
                            linewidths=0.8, alpha=0.6)
            elif seg_end_idx is not None:
                ex, ey = nodes[seg_end_idx]
                plt.scatter(0.0, 0.0, s=70, c="cyan", marker="^", edgecolors="black",
                            linewidths=0.8, alpha=0.6)
                plt.scatter(ex, ey, s=70, c="magenta", marker="D", edgecolors="black",
                            linewidths=0.8, alpha=0.6)
            if seg_end_idx is not None:
                handles.append(Line2D([0], [0], marker="^", color="w", label="segment start",
                                      markerfacecolor="cyan", markeredgecolor="black", markersize=7, alpha=0.6))
                handles.append(Line2D([0], [0], marker="D", color="w", label="segment end",
                                      markerfacecolor="magenta", markeredgecolor="black", markersize=7, alpha=0.6))

            if selected_node_xy:
                alphas = np.linspace(1.0, 0.2, num=len(selected_node_xy))
                for (pt, a) in zip(selected_node_xy, alphas):
                    plt.scatter(pt[0], pt[1], s=45, c="orange", alpha=float(a),
                                edgecolors="black", linewidths=0.5)
                handles.append(Line2D([0], [0], marker="o", color="w",
                                      label="selected_node_xy (ordered)",
                                      markerfacecolor="orange", markeredgecolor="black",
                                      markersize=7, alpha=0.7))

            if goal_xy is not None:
                plt.scatter(goal_xy[0], goal_xy[1], s=60, c="green", marker="*",
                            linewidths=1.2, alpha=0.8)
                handles.append(Line2D([0], [0], marker="*", color="w", label="goal",
                                      markerfacecolor="green", markeredgecolor="green",
                                      markersize=10, alpha=0.8))

            plt.scatter(pred_xy[0], pred_xy[1], s=70, c="blue", marker="x",
                        linewidths=1.5, alpha=0.9)
            handles.append(Line2D([0], [0], marker="x", color="blue", label="prediction",
                                  markeredgewidth=1.5, markersize=8, alpha=0.9))

            plt.xlabel(f"x (m) [{x_min}, {x_min + pitch * nx}]")
            plt.ylabel(f"y (m) [{y_min}, {y_min + pitch * ny}]")
            title = f"oracle_free_point_from_nodes | seg={used_segment}"
            if instruction is not None:
                title = f"{title}\n{instruction}"
            plt.title(title)
            plt.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7)
            plt.tight_layout()
            plt.savefig(debug_path, dpi=200)
            plt.close()

        return pred_xy

    def debug_plot_pred_and_goal(
        self,
        yaw_deg,
        target_token,
        pred_xy,
        goal_xy=None,
        debug_path=None,
        instruction=None,
        token_key="waypoint_token",
        origin_radius_m=2.4,
    ):
        """
        Draw pred_xy and (optional) goal_xy on top of the BEV free+sure mask produced by run_sample().
        Saves to debug_path if provided.

        Inputs you care about:
        yaw_deg, target_token, pred_xy, goal_xy, debug_path, instruction
        """

        if debug_path is None:
            raise ValueError("debug_path must be provided to save the figure.")
        if pred_xy is None:
            raise ValueError("pred_xy must be provided.")

        # --- Rebuild the same visualization background as oracle_free_point_from_nodes ---
        traversable_mask, _, occluder_outer, _ = self.run_sample(
            yaw_deg=yaw_deg,
            target_token=target_token,
            token_key=token_key,
            save=False,
            return_occluder=True,
        )
        trav_bool = traversable_mask.astype(bool)
        limited_mask = self.limited_observation_mask_from_traversable(
            trav_bool,
            occluder_mask=occluder_outer,
            origin_radius_m=origin_radius_m,
            unknown_val=255,
            corner_check=True,
            return_2ch=True,
        )
        sure_mask = limited_mask[1].astype(bool)
        free_mask = np.logical_and(trav_bool, sure_mask)

        nx, ny = free_mask.shape
        pitch = self.outer_pitch
        x_min = -self.outer_half_xy
        y_min = -self.outer_half_xy
        extent = [x_min, x_min + pitch * nx, y_min, y_min + pitch * ny]

        # --- Plot ---
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        plt.figure(figsize=(6, 6))
        plt.imshow(free_mask.T, origin="lower", cmap="gray", extent=extent)

        handles = []

        # Ego
        plt.scatter(0.0, 0.0, s=50, c="red", marker="o",
                    edgecolors="black", linewidths=0.6, alpha=0.9)
        handles.append(Line2D([0], [0], marker="o", color="w", label="ego",
                            markerfacecolor="red", markeredgecolor="black",
                            markersize=7, alpha=0.9))

        # Goal
        if goal_xy is not None:
            plt.scatter(goal_xy[0], goal_xy[1], s=70, c="green", marker="*",
                        linewidths=1.2, alpha=0.9)
            handles.append(Line2D([0], [0], marker="*", color="w", label="goal",
                                markerfacecolor="green", markeredgecolor="green",
                                markersize=10, alpha=0.9))

        # Prediction
        plt.scatter(pred_xy[0], pred_xy[1], s=80, c="blue", marker="x",
                    linewidths=1.7, alpha=0.95)
        handles.append(Line2D([0], [0], marker="x", color="blue", label="prediction",
                            markeredgewidth=1.7, markersize=8, alpha=0.95))

        plt.xlabel(f"x (m) [{x_min}, {x_min + pitch * nx}]")
        plt.ylabel(f"y (m) [{y_min}, {y_min + pitch * ny}]")

        title = "debug_plot_pred_and_goal"
        if instruction is not None:
            title = f"{title}\n{instruction}"
        plt.title(title)

        plt.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.7)
        plt.tight_layout()
        plt.savefig(debug_path, dpi=200)
        plt.close()


# # ---- usage ----
# import matplotlib.pyplot as plt
# if __name__ == "__main__":
#     gen = BEVMaskGenerator(
#         ann_file="/path/to/Datasets/MP3D/Falcon/data/captures_v3_val_S9hNv5qa7GM/scanS9hNv5qa7GM_Landmark-RxR-dynamic.pkl",
#         out_dir="/path/to/Datasets/MP3D/GaussTR/generate_bev_masks/traversable_masks",
#         outer_half_xy=6.4,
#         inner_half_xy=3.6,
#         outer_pitch=0.2,
#         inner_pitch=0.1,
#         z_up=0.4,
#         z_down=2.0,
#         clearance_band=(-1.4, 0.0),
#         support_band=(-1.6, -1.4),
#         connectivity=4,
#         corner_check=True,
#         inner_vote_min=2,
#     )

#     # traversable_mask, _ = gen.run_sample(yaw_deg=-45, target_token="a3aef3290dd544e7bc7d60c9015b031a", token_key="waypoint_token")
#     traversable_mask, fan_mask, best_idx, best_xy = gen.oracle_free_point(
#         yaw_deg=-45,
#         target_token="a3aef3290dd544e7bc7d60c9015b031a",
#         direction="backright",
#         step_size="small",
#         token_key="waypoint_token",
#     )

#     # for your plotting cell
#     HALF_XY = gen.outer_half_xy
#     PITCH = gen.outer_pitch
#     OUT_DIR = gen.out_dir

#     nx, ny = traversable_mask.shape
#     x_min = -HALF_XY
#     y_min = -HALF_XY
#     x_max = x_min + nx * PITCH
#     y_max = y_min + ny * PITCH

#     # ---- 2D traversable mask (forward x, left y) ----
#     plt.figure(figsize=(6, 6))
#     plt.imshow(
#         traversable_mask.T,
#         origin="lower",
#         cmap="gray",
#         extent=[x_min, x_max, y_min, y_max],
#         interpolation="nearest",
#         aspect="equal",
#     )

#     # overlay fan sector
#     fan_overlay = np.ma.masked_where(~fan_mask.T, fan_mask.T)
#     plt.imshow(
#         fan_overlay,
#         origin="lower",
#         cmap="gray",
#         extent=[x_min, x_max, y_min, y_max],
#         interpolation="nearest",
#         alpha=0.35,
#     )

#     # mark best point
#     print(f"Using best point" if best_idx != (-1, -1) else "Using default origin")
#     plt.scatter(
#         [best_xy[0]], [best_xy[1]],
#         s=60, c="cyan", edgecolors="black", linewidths=0.8,
#         label="selected"
#     )
#     plt.legend(loc="upper right")

#     plt.xlabel("Forward X (m)")
#     plt.ylabel("Left Y (m)")
#     plt.title("Traversable Mask + Fan + Selected Point")
#     plt.colorbar(label="Traversable")
#     plt.show()