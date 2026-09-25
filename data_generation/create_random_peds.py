# =====================================================================
# create_random_peds.py  (BEACON data generation)
# =====================================================================
# Faithful .py conversion of cell 1 ("# 1. load the original ped data
# per scene, downsample and save to json.gz") of `create_random_peds_v2.ipynb`
# from the author's private Falcon fork. It randomly samples pedestrian
# starting positions on the navmesh and rolls out coverage-optimized,
# collision-avoiding trajectories for one MP3D scan, then saves a single
# merged episode to `crowd-mp3d/{split}/content/{scan_id}.json.gz`
# (consumed by `auto_save_pano.py`).
#
# The notebook's other code cells (3, 5, 7) were matplotlib-display /
# gzip-inspection debug helpers and are intentionally NOT included here;
# this cell is self-contained and does not depend on them.
#
# The ONLY changes vs. the notebook are:
#   1) hardcoded absolute paths replaced by env-var-derived roots
#      (MP3D_ROOT / FALCON_DATA_ROOT / BEACON_VIS_DIR, see block below);
#   2) `os.makedirs(VIS_DIR, exist_ok=True)` added before saving the
#      debug floor-plan PNGs (the original wrote into a directory that
#      already existed on the author's machine).
# No algorithm logic was changed.
#
# KNOWN QUIRKS OF THE ORIGINAL CODE (preserved as-is, do not "fix" silently):
#   - In the AUTO_DETECT_SPLIT_AND_SEED block, when the scan is missing from
#     BOTH train and val (the `MISSING_SCAN_ID` fallback case), `scan_exist_in`
#     is set to None inside the branch but then unconditionally overwritten to
#     "val" by the following line `scan_exist_in = "train" if train_exists else "val"`.
#   - The navmesh-Y fallback path (scan missing from social-mp3d) computes
#     `floor_heights`, but the episode-saving section at the bottom still uses
#     `data["episodes"][0]` from the social-mp3d gz, which is undefined in that
#     fallback case -> the script would raise NameError there. In practice the
#     fallback was only used to inspect floor heights for `MISSING_SCAN_ID`.
#   - `plt.cm.get_cmap` is deprecated in newer matplotlib; kept for fidelity.
#
# Run from the ROOT of your Falcon clone (habitat-lab must be importable):
#   PYTHONPATH=.:habitat-baselines:habitat-lab python create_random_peds.py
# =====================================================================

import gzip, json, os, random
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import habitat_sim
from habitat.utils.visualizations import maps
from tqdm.auto import tqdm

# ---------------------------------------------------------------------
# Machine-specific dataset roots
# ---------------------------------------------------------------------
# Configure these for your machine (originally the author's local MP3D/Falcon
# layout). Set the env vars or edit defaults.
MP3D_ROOT = os.environ.get("MP3D_ROOT", "/path/to/mp3d")            # .../mp3d_habitat/mp3d
FALCON_DATA_ROOT = os.environ.get("FALCON_DATA_ROOT", "/path/to/Falcon/data")
VIS_DIR = os.environ.get("BEACON_VIS_DIR", "vis")  # debug floor-plan PNG output dir

# -----------------------------
# Deterministic randomness
# -----------------------------
# RANDOM_SEED = 42 # 42 is train, 0 is val
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)

AUTO_DETECT_SPLIT_AND_SEED = True
DATA_ROOT = os.path.join(FALCON_DATA_ROOT, "datasets", "pointnav", "social-mp3d") # source
scan_id = '8WUmhLawc2A' #'82sE5b5pLXE' #'7y3sRwLe3Va' #'759xd9YjKW5' #'5q7pvUzZiYa' # '5LpN3gDmAk7' #'2n8kARJN3HM' # '29hnd4uzFmX' # '1pXnuDYAj8r' # '1LXtFkjw3qL' # "HxpKQynjfin" # '2azQ1b91cZZ' # 'JmbYfDe2QKZ' # 'pa4otMbVnkk' # "SN83YJsR3w2" # "S9hNv5qa7GM" # 17DRP5sb8fy
MISSING_SCAN_ID = "HxpKQynjfin"

HAS_SOCIAL_FILE = True
if AUTO_DETECT_SPLIT_AND_SEED:
    train_path = os.path.join(DATA_ROOT, "train", "content", f"{scan_id}.json.gz")
    val_path   = os.path.join(DATA_ROOT, "val",   "content", f"{scan_id}.json.gz")

    train_exists = os.path.exists(train_path)
    val_exists   = os.path.exists(val_path)

    if train_exists and val_exists:
        raise RuntimeError(f"Leakage: {scan_id} exists in BOTH train and val")
    if not train_exists and not val_exists:
        HAS_SOCIAL_FILE = False
        scan_exist_in = None
        RANDOM_SEED = 42  # arbitrary; only matters for your synthetic generation randomness
        print(f"[WARN] {scan_id} not found in social-mp3d train/val -> will use navmesh-Y fallback")
        assert scan_id == MISSING_SCAN_ID, f"Unexpected missing scan: {scan_id}"

    scan_exist_in = "train" if train_exists else "val"
    RANDOM_SEED = 42 if scan_exist_in == "train" else 0
    print(f"Auto-detected {scan_id} in split '{scan_exist_in}' -> using RANDOM_SEED={RANDOM_SEED}")
else:
    scan_exist_in = "train"   # old manual behavior
    RANDOM_SEED = 42        # old manual behavior
    print(f"Auto-detection disabled: using scan_exist_in='{scan_exist_in}' and RANDOM_SEED={RANDOM_SEED}")

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# episode_gz = os.path.join(DATA_ROOT, scan_exist_in, "content", f"{scan_id}.json.gz")
episode_gz = (
    os.path.join(DATA_ROOT, scan_exist_in, "content", f"{scan_id}.json.gz")
    if HAS_SOCIAL_FILE
    else None
)

# -----------------------------
# Trajectory generation tunables
# -----------------------------
GEN_STEP_M = 0.5                 # (D) grid resolution and step length for generation
WAYPOINTS_PER_HUMAN = 500
MIN_SEPARATION_M = 0.2           # discourage closer than this
HUMAN_DENSITY_PER_M2 = 1 # 0.33 # 0.19       # (A) increase density so human_count per floor is higher
MAX_HUMANS_PER_FLOOR = 1000 # 30
HUMAN_RADIUS = 0.25               # (B) inflate navmesh by this - 0.1

COVERAGE_WEIGHT = 1.0
COLLISION_WEIGHT = 5.0
BACKTRACK_WEIGHT = 0.2

MAX_HUMANS_TOTAL = 1000 # 30

# (E) mandatory navmesh neighbor validation
USE_GEODESIC_NEIGHBOR_CHECK = True
GEODESIC_RATIO_MAX = 1.2         # geodesic <= 1.2 * euclid

# Optional: prevent A<->B swaps in one timestep (cheap, can disable)
AVOID_SWAP_COLLISIONS = True

# -----------------------------
# Floor detection / plotting
# -----------------------------
FLOOR_GAP = 2.0
FLOOR_TOL = 0.5                      # when snapping nodes, max Y difference to intended floor
PLOT_METERS_PER_PIXEL = 0.05     # fine-res map for area calc and plotting
MIN_FLOOR_AREA_M2 = None  # None = OFF. Set e.g. 20.0 to enable.

NAVMESH_Y_SAMPLES = 50000   # start with 50k; increase if floors are missed
NAVMESH_Y_STEP = 50         # subsampling for clustering

MAX_SNAP_XZ_M = 0.2  # start 0.5; try 0.3 if still bad nodes



# -----------------------------
# Paths
# -----------------------------
scene_root = f"{MP3D_ROOT}/{scan_id}"
scene_id = f"{scene_root}/{scan_id}.glb"
navmesh_path = f"{scene_root}/{scan_id}.navmesh"
# # episode_gz = f"{FALCON_DATA_ROOT}/datasets/pointnav/social-mp3d/train/content/{scan_id}.json.gz"
# episode_gz = f"{FALCON_DATA_ROOT}/datasets/pointnav/social-mp3d/{scan_exist_in}/content/{scan_id}.json.gz"


# -----------------------------
# Helpers
# -----------------------------
def extract_humans(info):
    humans = defaultdict(dict)
    for k, v in info.items():
        if k.startswith("human_") and "_waypoint_" in k:
            parts = k.split("_")
            h = int(parts[1])
            wp = int(parts[3])
            field = "pos" if k.endswith("_position") else "rot"
            humans[h].setdefault(wp, {})[field] = v
    merged = {}
    for h, by_wp in humans.items():
        merged[h] = [by_wp[i] for i in sorted(by_wp.keys())]
    return merged

def floor_heights_from_y(all_y, gap=FLOOR_GAP, step=50):
    # simple 1D clustering by gap
    if step > 1:
        all_y = all_y[::step]
    all_y = np.sort(all_y)
    bins = []
    for y in tqdm(all_y, desc="Floor clustering"):
        if not bins or abs(y - np.mean(bins[-1])) > gap:
            bins.append([y])
        else:
            bins[-1].append(y)
    return [float(np.mean(b)) for b in bins]

def farthest_point_sampling(points, k):
    k = min(k, len(points))
    if k <= 0:
        return []
    chosen = [random.randrange(len(points))]
    dists = np.full(len(points), np.inf, dtype=np.float32)
    for _ in range(1, k):
        last = points[chosen[-1]]
        d = np.linalg.norm(points - last, axis=1)
        dists = np.minimum(dists, d)
        chosen.append(int(np.argmax(dists)))
    return chosen

def compute_floor_area_m2(pathfinder, fh, meters_per_pixel=PLOT_METERS_PER_PIXEL):
    td_plot = maps.get_topdown_map(
        pathfinder, height=fh, meters_per_pixel=meters_per_pixel, draw_border=True
    )
    area_m2 = float((td_plot == maps.MAP_VALID_POINT).sum() * (meters_per_pixel ** 2))
    return td_plot, area_m2

def build_grid_and_graph(pathfinder, fh, step_size=GEN_STEP_M,
                         use_geodesic_check=True, geodesic_ratio_max=1.2,
                         floor_tol=FLOOR_TOL):
    """
    Build a node set from topdown valid cells at (fh, step_size),
    then connect 8-neighborhood edges.
    (E) Edge is kept only if a shortest path exists and geodesic <= ratio * euclid.
    """
    td = maps.get_topdown_map(
        pathfinder, height=fh, meters_per_pixel=step_size, draw_border=False
    )
    H, W = td.shape
    valid = np.argwhere(td == maps.MAP_VALID_POINT)
    if len(valid) == 0:
        return None, None, None, None

    grid_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(valid)}

    # Convert valid grid cells to world points and snap them onto navmesh for stability.
    world_pts = np.zeros((len(valid), 3), dtype=np.float32)
    good_mask = np.ones(len(valid), dtype=bool)

    for i, (r, c) in enumerate(valid):
        wz, wx = maps.from_grid(int(r), int(c), (H, W), pathfinder=pathfinder)
        p = np.array([wx, fh, wz], dtype=np.float32)
        # try:
        #     ps = pathfinder.snap_point(p)
        # except Exception:
        #     ps = p
        # # Ensure we didn't snap to some other floor unexpectedly
        # if abs(float(ps[1]) - fh) > floor_tol:
        #     good_mask[i] = False
        # world_pts[i] = ps
        try:
            ps = pathfinder.snap_point(p)
        except Exception:
            ps = p

        xz_drift = float(np.linalg.norm(ps[[0, 2]] - p[[0, 2]]))
        if xz_drift > MAX_SNAP_XZ_M:
            good_mask[i] = False

        # Ensure we didn't snap to some other floor unexpectedly
        if abs(float(ps[1]) - fh) > floor_tol:
            good_mask[i] = False

        world_pts[i] = ps


    if not good_mask.all():
        # filter nodes that snap off-floor
        keep = np.where(good_mask)[0]
        if len(keep) == 0:
            return None, None, None, None
        valid = valid[keep]
        world_pts = world_pts[keep]
        grid_to_idx = {(int(r), int(c)): i for i, (r, c) in enumerate(valid)}

    # 8-neighborhood in grid
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1)]

    neighbors = [[] for _ in range(len(valid))]

    # Pre-allocate shortest path object for speed
    sp = habitat_sim.ShortestPath()

    # Tolerance for Euclid distance vs expected distance.
    # (Fix for C) allow both orth (1*step) and diag (sqrt(2)*step)
    dist_tol = 0.35 * step_size

    for i, (r, c) in enumerate(tqdm(valid, desc=f"Building graph @ fh≈{fh:.2f} step={step_size}")):
        r = int(r); c = int(c)
        pi = world_pts[i]
        for dr, dc in offsets:
            j = grid_to_idx.get((r + dr, c + dc))
            if j is None:
                continue
            pj = world_pts[j]

            expected = step_size * float(np.sqrt(dr * dr + dc * dc))
            euclid = float(np.linalg.norm(pi - pj))
            if abs(euclid - expected) > dist_tol:
                # geometry mismatch (can happen near borders); skip to avoid weird jumps
                continue

            if use_geodesic_check:
                sp.requested_start = pi
                sp.requested_end = pj
                found = pathfinder.find_path(sp)
                if not found:
                    continue
                gd = float(sp.geodesic_distance)
                # Mandatory (E): geodesic must be close to euclid
                if not np.isfinite(gd) or gd > geodesic_ratio_max * euclid:
                    continue

            neighbors[i].append(j)

    return td, world_pts, neighbors, valid

def rollout_paths(world_pts, neighbors, start_idxs):
    """
    Multi-agent rollout:
      - coverage reward (prefer unvisited)
      - separation penalty (avoid too close to others)
      - backtrack penalty (avoid revisits)
      - unique occupancy per step (no two agents in same node)
      - optional swap avoidance (very cheap)
    """
    n_agents = len(start_idxs)
    visited = np.zeros(len(world_pts), dtype=np.int32)
    trajectories = [[world_pts[idx].tolist()] for idx in start_idxs]
    current = list(start_idxs)
    for idx in current:
        visited[idx] += 1

    for _ in tqdm(range(1, WAYPOINTS_PER_HUMAN), desc="Rollout steps"):
        proposals = []
        for hi, cur_idx in enumerate(current):
            cands = neighbors[cur_idx] + [cur_idx]  # allow stay
            scored = []
            for ci in cands:
                cov_gain = 1.0 if visited[ci] == 0 else 1.0 / (1.0 + visited[ci])

                # separation vs other agents' CURRENT positions
                min_dist = np.inf
                for oj, other_idx in enumerate(current):
                    if oj == hi:
                        continue
                    d = np.linalg.norm(world_pts[ci] - world_pts[other_idx])
                    if d < min_dist:
                        min_dist = d
                close_penalty = 1.0 if min_dist < MIN_SEPARATION_M else 0.0

                score = (
                    COVERAGE_WEIGHT * cov_gain
                    - COLLISION_WEIGHT * close_penalty
                    - BACKTRACK_WEIGHT * visited[ci]
                )
                scored.append((score, ci))
            scored.sort(key=lambda x: x[0], reverse=True)
            proposals.append(scored)

        order = list(range(n_agents))
        random.shuffle(order)
        taken_nodes = set()
        reserved_edges = set()  # (from, to) for swap avoidance
        new_current = current[:]

        for hi in order:
            from_i = current[hi]
            for score, to_i in proposals[hi]:
                if to_i in taken_nodes:
                    continue
                if AVOID_SWAP_COLLISIONS:
                    # forbid selecting edge if reverse edge already reserved by another agent
                    if (to_i, from_i) in reserved_edges:
                        continue
                new_current[hi] = to_i
                taken_nodes.add(to_i)
                if AVOID_SWAP_COLLISIONS:
                    reserved_edges.add((from_i, to_i))
                break

        current = new_current
        for hi, ci in enumerate(current):
            visited[ci] += 1
            trajectories[hi].append(world_pts[ci].tolist())
    return trajectories

def report_final_overlaps(trajs, min_sep=MIN_SEPARATION_M):
    finals = np.array([t[-1] for t in trajs], dtype=np.float32)
    any_close = False
    for i in range(len(finals)):
        for j in range(i + 1, len(finals)):
            d = float(np.linalg.norm(finals[i] - finals[j]))
            if d < min_sep:
                print(f"[WARN] Final waypoints {i} and {j} are {d:.3f} m apart (< {min_sep} m)")
                any_close = True
    if not any_close:
        print(f"All final waypoints separated by ≥ {min_sep} m")

# -----------------------------
# Load dataset to infer floor heights
# -----------------------------
try:
    with gzip.open(episode_gz, "rt", encoding="utf-8") as f:
        data = json.load(f)

    tracks = []
    for ep in tqdm(data["episodes"], desc="Episodes for floors"):
        humans = extract_humans(ep.get("info", {}))
        for _, wpts in humans.items():
            pts = [w.get("pos") for w in wpts if w.get("pos") is not None]
            if pts:
                tracks.append(np.array(pts, dtype=np.float32))

    print(f"Tracks loaded (for floor detection): {len(tracks)}")
    all_y = np.concatenate([pts[:, 1] for pts in tracks])
    # floor_heights = floor_heights_from_y(all_y, gap=FLOOR_GAP)
    num_episodes = len(data["episodes"])
    step = num_episodes // 1000 if num_episodes > 1000 else 1
    print(f"Using step={step} for floor height detection (episodes={num_episodes})")
    floor_heights = floor_heights_from_y(all_y, gap=FLOOR_GAP, step=step) # DEBUG
except Exception as e:
    print("[WARN] social per-scan file missing:", e)

    # ---- Fallback: sample navmesh Y heights ----
    print(f"[FALLBACK] Sampling navmesh Y for floor clustering: samples={NAVMESH_Y_SAMPLES}")

    # Create a minimal sim+pathfinder (you already do this later; ok to do early here)
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = False
    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = []
    sim_tmp = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    sim_tmp.pathfinder.load_nav_mesh(navmesh_path)

    ys = []
    for _ in tqdm(range(NAVMESH_Y_SAMPLES), desc="Navmesh Y sampling"):
        p = sim_tmp.pathfinder.get_random_navigable_point()
        ys.append(float(p[1]))

    sim_tmp.close()

    all_y = np.array(ys, dtype=np.float32)
    floor_heights = floor_heights_from_y(all_y, gap=FLOOR_GAP, step=NAVMESH_Y_STEP)

print(f"Floor candidates: {[f'{fh:.2f}' for fh in floor_heights]}")
print(f"[FLOOR_DEBUG] all_y: n={len(all_y)} min={all_y.min():.3f} max={all_y.max():.3f}")
# crude bin assignment by nearest candidate (good enough for debugging)
for fh in floor_heights:
    cnt = int(np.sum(np.abs(all_y - fh) < FLOOR_GAP/2))
    print(f"[FLOOR_DEBUG] fh≈{fh:.2f}: approx_support={cnt} (within {FLOOR_GAP/2:.2f}m)")


# -----------------------------
# Simulator for navmesh queries
# -----------------------------
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene_id
sim_cfg.enable_physics = False

agent_cfg = habitat_sim.AgentConfiguration()
agent_cfg.sensor_specifications = []

sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
# sim.pathfinder.load_nav_mesh(navmesh_path)
# Load the default navmesh first (so we inherit its settings like cell_size, etc.)
sim.pathfinder.load_nav_mesh(navmesh_path)

# Inflate agent radius and rebuild navmesh in-memory
navmesh_settings = sim.pathfinder.nav_mesh_settings  # may be None if nothing loaded
if navmesh_settings is None:
    navmesh_settings = habitat_sim.nav.NavMeshSettings()
    navmesh_settings.set_defaults()

navmesh_settings.agent_radius = HUMAN_RADIUS  # <- the only thing you change
ok = sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
assert ok, "Navmesh recompute failed"


# -----------------------------
# Fine-res floor areas for human counts and plotting
# -----------------------------
floor_stats = {}
td_plot_cache = {}

for fh in floor_heights:
    td_plot, area_m2 = compute_floor_area_m2(sim.pathfinder, fh, meters_per_pixel=PLOT_METERS_PER_PIXEL)

    if MIN_FLOOR_AREA_M2 is not None and area_m2 < MIN_FLOOR_AREA_M2:
        print(f"[FLOOR_SKIP] fh≈{fh:.2f}: area {area_m2:.1f} m² < {MIN_FLOOR_AREA_M2} m²")
        continue

    floor_stats[fh] = area_m2
    td_plot_cache[fh] = td_plot

print({f"{fh:.2f}": f"{area:.1f} m^2" for fh, area in floor_stats.items()})

# -----------------------------
# Generate trajectories per floor
# -----------------------------
generated = {}

# for fh in floor_heights:
for fh in floor_stats:
    td_gen, world_pts, neighbors, valid = build_grid_and_graph(
        sim.pathfinder,
        fh,
        step_size=GEN_STEP_M,
        use_geodesic_check=USE_GEODESIC_NEIGHBOR_CHECK,
        geodesic_ratio_max=GEODESIC_RATIO_MAX,
        floor_tol=FLOOR_TOL,
    )
    if world_pts is None:
        print(f"Skipping floor ~{fh:.2f} m (no valid navmesh cells)")
        continue
    
    remaining = MAX_HUMANS_TOTAL - sum(len(v["trajectories"]) for v in generated.values())
    if remaining <= 0:
        print("Reached MAX_HUMANS_TOTAL, skipping remaining floors.")
        break


    # (A) density controls human_count; still capped
    # human_count = max(
    #     1,
    #     min(MAX_HUMANS_PER_FLOOR, int(round(floor_stats[fh] * HUMAN_DENSITY_PER_M2)))
    # )
    human_count = max(
        1,
        min(remaining, MAX_HUMANS_PER_FLOOR, int(round(floor_stats[fh] * HUMAN_DENSITY_PER_M2)))
    )


    print(f"\nFloor ~{fh:.2f} m: nodes={len(world_pts)} humans={human_count}")
    start_idxs = farthest_point_sampling(world_pts, human_count)
    trajectories = rollout_paths(world_pts, neighbors, start_idxs)

    report_final_overlaps(trajectories, min_sep=MIN_SEPARATION_M)


    generated[fh] = {
        "td_plot": td_plot_cache[fh],
        "world_pts": world_pts,
        "trajectories": trajectories,
    }

    # Coverage stats on fine map
    visited_cells = set()
    H_plot, W_plot = td_plot_cache[fh].shape
    for traj in trajectories:
        for p in traj:
            gx, gy = maps.to_grid(p[2], p[0], (H_plot, W_plot), pathfinder=sim.pathfinder)
            visited_cells.add((gx, gy))
    denom = float((td_plot_cache[fh] == maps.MAP_VALID_POINT).sum())
    coverage_ratio = (len(visited_cells) / denom) if denom > 0 else 0.0
    print(f"Coverage ratio (fine map): {coverage_ratio:.3f}")

# -----------------------------
# Plot (one color per synthetic human) using fine-res map
# -----------------------------
cmap = plt.cm.get_cmap("hsv", 512)
def color_for_track(track_i): return cmap((track_i * 53) % 512)

# for i, (fh, payload) in enumerate(generated.items()):
#     td_plot = payload["td_plot"]
#     trajs = payload["trajectories"]
#     td_rgb = maps.colorize_topdown_map(td_plot)
#     H_plot, W_plot = td_plot.shape

#     plt.figure(figsize=(10, 10))
#     plt.imshow(td_rgb, origin="upper")
#     plt.title(f"S9hNv5qa7GM floor≈{fh:.2f}m (humans={len(trajs)})")
#     plt.text(
#         5, H_plot - 10,
#         f"area ≈ {floor_stats.get(fh, 0):.1f} m²",
#         color="black",
#         fontsize=10,
#         bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
#     )

#     for hid, traj in enumerate(trajs):
#         grid_pts = [maps.to_grid(p[2], p[0], (H_plot, W_plot), pathfinder=sim.pathfinder) for p in traj]
#         if len(grid_pts) >= 2:
#             xs, ys = zip(*grid_pts)
#             plt.plot(ys, xs, "-", color=color_for_track(hid), linewidth=2, alpha=0.9)
#         elif len(grid_pts) == 1:
#             x, y = grid_pts[0]
#             plt.plot(y, x, "o", color=color_for_track(hid), markersize=4, alpha=0.9)

#     plt.axis("off")

# plt.show()
# Plot (one color per synthetic human) using fine-res map

cmap = plt.cm.get_cmap("hsv", 512)
def color_for_track(track_i): return cmap((track_i * 53) % 512)
all_overlay_paths = []

os.makedirs(VIS_DIR, exist_ok=True)  # added for the release (dir existed on author's machine)
for i, (fh, payload) in enumerate(generated.items()):
    td_plot = payload["td_plot"]              # fine topdown map from compute_floor_area_m2
    trajs = payload["trajectories"]
    td_rgb = maps.colorize_topdown_map(td_plot)
    H_plot, W_plot = td_plot.shape

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(td_rgb, origin="upper")
    ax.set_title(f"{scan_id} floor≈{fh:.2f}m (humans={len(trajs)})")
    ax.text(
        5, H_plot - 10,
        f"area ≈ {floor_stats.get(fh, 0):.1f} m²",
        color="black",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
    )

    for hid, traj in enumerate(trajs):
        grid_pts = [
            maps.to_grid(p[2], p[0], (H_plot, W_plot), pathfinder=sim.pathfinder)
            for p in traj
        ]
        if len(grid_pts) >= 2:
            xs, ys = zip(*grid_pts)
            ax.plot(ys, xs, "-", color=color_for_track(hid), linewidth=2, alpha=0.9)
        elif len(grid_pts) == 1:
            x, y = grid_pts[0]
            ax.plot(y, x, "o", color=color_for_track(hid), markersize=4, alpha=0.9)

    ax.axis("off")

    overlay_path = os.path.join(VIS_DIR, f"{scan_id}_seed_{RANDOM_SEED}_floor_{i}_humans_{len(trajs)}.png")
    raw_path = os.path.join(VIS_DIR, f"{scan_id}_seed_{RANDOM_SEED}_floor_{i}_floorplan.png")

    fig.savefig(overlay_path, bbox_inches="tight", pad_inches=0, dpi=200)  # with trajectories
    print("Saved:", overlay_path)
    all_overlay_paths.append(overlay_path)
    plt.imsave(raw_path, td_rgb, origin="upper")                           # floorplan only
    plt.close(fig)

plt.show()


# -----------------------------
# Save one episode with all synthetic pedestrians merged
# -----------------------------
base_ep = dict(data["episodes"][0])
base_info = {
    k: v
    for k, v in base_ep.get("info", {}).items()
    if not (k.startswith("human_") and "_waypoint_" in k)
}
new_info = dict(base_info)

DUMMY_YAW_RAD = 0.0  # facing forward; scalar radians
new_hid = 0
for fh, payload in generated.items():
    for traj in payload["trajectories"]:
        for wp_idx, pos in enumerate(traj):
            new_info[f"human_{new_hid}_waypoint_{wp_idx}_position"] = pos
            new_info[f"human_{new_hid}_waypoint_{wp_idx}_rotation"] = DUMMY_YAW_RAD
        new_hid += 1


new_info["human_num"] = new_hid
base_ep["info"] = new_info
filtered_eps = [base_ep]  # keep only one episode

scene_name = os.path.splitext(os.path.basename(scene_id))[0]
# split = "train" if "/train/" in episode_gz else "val"
if AUTO_DETECT_SPLIT_AND_SEED:
    splits_to_save = [scan_exist_in]
else:
    splits_to_save = ["train", "val"]  # old behavior
    
for split in splits_to_save:
# split = "val"
    out_dir = os.path.join(FALCON_DATA_ROOT, "datasets", "pointnav", "crowd-mp3d", split, "content")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene_name}.json.gz")

    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump({"episodes": filtered_eps}, f)

    print("Saved:", out_path, "humans:", new_hid)

sim.close()
