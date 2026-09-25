import os, sys, json, math, random, pickle as pkl, cv2, falcon, time
import numpy as np
import quaternion
import habitat_sim
import gzip
import copy
from habitat_sim.utils.common import quat_from_angle_axis
from dataclasses import dataclass
from omegaconf import OmegaConf
from tqdm import tqdm

from habitat_baselines.config.default import get_config
from habitat.config.read_write import read_write
from habitat.config.default_structured_configs import (
    HabitatSimEquirectangularRGBSensorConfig,
    HabitatSimEquirectangularDepthSensorConfig,
)
from habitat.gym import make_gym_from_config

# Make local packages visible
root = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [
    root,
    os.path.join(root, "habitat-baselines"),
    os.path.join(root, "habitat-lab"),
]

# ---------------------------------------------------------------------
# Machine-specific dataset roots
# ---------------------------------------------------------------------
# Configure these for your machine (originally the author's local MP3D/Falcon
# layout). Set the env vars below or edit
# the defaults. NOTE: this script must be placed at the ROOT of your Falcon
# clone (next to `habitat-baselines/`, `habitat-lab/`, `falcon/`, `data/`).
MP3D_ROOT = os.environ.get("MP3D_ROOT", "/path/to/mp3d")            # .../mp3d_habitat/mp3d
CONNECTIVITY_ROOT = os.environ.get("MP3D_CONNECTIVITY", "/path/to/Matterport3DSimulator/connectivity")
FALCON_DATA_ROOT = os.environ.get("FALCON_DATA_ROOT", "/path/to/Falcon/data")

# ---------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------
SCENE_ROOT = MP3D_ROOT
SCAN_ID = 'TbHJrupSAjP' # '8WUmhLawc2A' #'82sE5b5pLXE' #'7y3sRwLe3Va' #'759xd9YjKW5' #'5q7pvUzZiYa' #'5LpN3gDmAk7' #'2n8kARJN3HM' #'29hnd4uzFmX' #'1LXtFkjw3qL' #'1pXnuDYAj8r' #'1LXtFkjw3qL' # "HxpKQynjfin" # "17DRP5sb8fy" # "2azQ1b91cZZ" # "SN83YJsR3w2" # "S9hNv5qa7GM"
CAM_HEIGHT = 1.50
MAX_SNAP_DIST = 0.50
CAPTURE_RADIUS = 5.1
OVERLAP_THRESH = 0.2  # [m], min distance between any two agents to avoid overlap # BUG: this is hard to tune, 0.1 is too small, 0.2 is not yet tested
# HUMAN_SPEED = 10.0 / 30.0  # 10 m/s in orca_mp3d.yaml, ~0.333 m per env step
# ROBOT_HUMAN_FACTOR = 1.0  # robot moves 1x human speed
ROBOT_MOTION_FACTOR = 0.5 # smaller is slower, we tune this and freeze the human speed configured inside the yaml files
DEFAULT_STEP_DIST = 0.1 # [m]
STEP_DIST = DEFAULT_STEP_DIST * ROBOT_MOTION_FACTOR # HUMAN_SPEED * ROBOT_HUMAN_FACTOR  # meters between dense waypoints, also triggers the sim step so should be frozen given human speed
FRAME_INTERVAL_DIST = 0.5  # meters between captured frames
CAPTURE_EVERY_STEPS = round(FRAME_INTERVAL_DIST/STEP_DIST)  # capture every N env steps (~FRAME_INTERVAL_DIST m)
DEBUG_LIMIT = None # 200  # limit number of connectivity nodes
MAX_OVERLAP_RETRIES = 100 # we hope never to reach this limit
REMOVE_HUMAN_ON_PERSISTENT_OVERLAP = True  # backward compatiblility
REMOVE_AFTER_RETRIES = 10  # after N overlap retries at a waypoint, remove one human instead of failing
REMOVAL_SINK_POS_HAB = np.array([0.0, -10.0, 0.0], np.float32)  # "disappear" location
NEVER_SKIP_OVERLAP = False  # if True, always save overlap frames even if not connectivity waypoint
TELEPORT_WAYPOINT_ONLY = True  # NEW: teleport robot between connectivity waypoints, capture ONLY those

# Runtime state
REMOVED_HUMAN_AGENT_IDS = set()

# ---------------------------------------------------------------------
# Dataset selection (avoid loading ALL `content/*.json.gz`)
# ---------------------------------------------------------------------
# Habitat's PointNav dataset will enumerate and load every per-scene file under:
#   data/datasets/pointnav/crowd-mp3d/{split}/content/*.json.gz
# when `habitat.dataset.content_scenes` contains `*` (ALL_SCENES_MASK).
# This script is single-scan, so we:
#   1) auto-detect which split contains `{SCAN_ID}.json.gz`
#   2) set `habitat.dataset.content_scenes = [SCAN_ID]` before env creation
DATASET_ROOT = "data/datasets/pointnav/crowd-mp3d"
AUTO_DETECT_SPLIT = True
AUTO_ADJUST_HUMAN_UPPER_BOUND = True
MAX_HUMANS_CAP = 1200
OVERRIDE_HUMAN_RADIUS = True  # backward compatible: default on/off as you like
HUMAN_RADIUS_OVERRIDE_VALUE = 0.01  # avoid 0.0; affects navmesh/pathfinding radius


def detect_split_for_scan(scan_id: str, dataset_root: str = DATASET_ROOT) -> str:
    candidates = []
    for split in ("train", "val"):
        p = os.path.join(dataset_root, split, "content", f"{scan_id}.json.gz")
        if os.path.exists(p):
            candidates.append(split)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) == 2:
        # Ambiguous; choose train by default but keep it explicit.
        print(f"[WARN] `{scan_id}` exists in BOTH train/ and val/; defaulting to train.")
        return "train"
    raise FileNotFoundError(
        f"Could not find `{scan_id}.json.gz` in `{dataset_root}/train/content/` or "
        f"`{dataset_root}/val/content/`."
    )


split_to_save = detect_split_for_scan(SCAN_ID) if AUTO_DETECT_SPLIT else "val"


def get_max_humans_for_scan(scan_id: str, split: str, dataset_root: str = DATASET_ROOT) -> int:
    """
    Compute max required humans from the per-scan dataset file:
      `{dataset_root}/{split}/content/{scan_id}.json.gz`
    Uses `episode.info["human_num"]` when present; otherwise infers from `human_*` keys.
    """
    path = os.path.join(dataset_root, split, "content", f"{scan_id}.json.gz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing per-scan dataset: `{path}`")
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    if not episodes:
        return 0

    max_humans = 0
    for ep in episodes:
        info = ep.get("info", {}) or {}
        human_num = info.get("human_num", None)
        if human_num is None:
            # Fallback: infer from keys like human_{hid}_waypoint_{k}_position
            max_hid = -1
            for k in info.keys():
                if not k.startswith("human_"):
                    continue
                parts = k.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    max_hid = max(max_hid, int(parts[1]))
            human_num = max_hid + 1 if max_hid >= 0 else 0
        try:
            max_humans = max(max_humans, int(human_num))
        except Exception:
            pass
    return int(max_humans)


def adjust_multi_agent_cfg(cfg, num_humans: int) -> None:
    """
    Patch the merged Hydra config in-memory so we only instantiate exactly
    `num_humans` humanoids (agents 1..num_humans). Keeps YAML files untouched.
    """
    num_humans = int(max(0, num_humans))
    if num_humans > int(MAX_HUMANS_CAP):
        print(f"[WARN] num_humans={num_humans} > cap={MAX_HUMANS_CAP}; clamping.")
        num_humans = int(MAX_HUMANS_CAP)

    # -------------------------
    # Simulator agents
    # -------------------------
    cfg.habitat.simulator.agents_order = ["agent_0"] + [
        f"agent_{i}" for i in range(1, num_humans + 1)
    ]

    sim_agents = cfg.habitat.simulator.agents

    # Cycle templates from agent_1..agent_12 (these are already defined by YAML defaults).
    templates = []
    for i in range(1, 13):
        k = f"agent_{i}"
        if k in sim_agents:
            templates.append(sim_agents[k])
    if not templates:
        raise RuntimeError("No human agent templates found under cfg.habitat.simulator.agents")

    for i in range(1, num_humans + 1):
        sim_agents[f"agent_{i}"] = copy.deepcopy(templates[(i - 1) % len(templates)])

    # Prune extra agent configs (keep agent_0)
    for k in list(sim_agents.keys()):
        if not k.startswith("agent_") or k == "agent_0":
            continue
        try:
            idx = int(k.split("_")[1])
        except Exception:
            continue
        if idx > num_humans:
            del sim_agents[k]

    # -------------------------
    # Task actions (humans)
    # -------------------------
    actions = cfg.habitat.task.actions
    template_action_key = "agent_1_oracle_nav_randcoord_action_obstacle"
    if template_action_key not in actions:
        raise RuntimeError(f"Missing `{template_action_key}` in cfg.habitat.task.actions")
    action_template = actions[template_action_key]

    for i in range(1, num_humans + 1):
        actions[f"agent_{i}_oracle_nav_randcoord_action_obstacle"] = copy.deepcopy(action_template)
    for k in list(actions.keys()):
        if not k.startswith("agent_") or "_oracle_nav_randcoord_action_obstacle" not in k:
            continue
        try:
            idx = int(k.split("_")[1])
        except Exception:
            continue
        if idx > num_humans:
            del actions[k]

    # -------------------------
    # Baselines policies (humans)
    # -------------------------
    policies = cfg.habitat_baselines.rl.policy
    if "agent_1" not in policies:
        raise RuntimeError("Missing `habitat_baselines.rl.policy.agent_1`")
    policy_template = policies["agent_1"]
    for i in range(1, num_humans + 1):
        policies[f"agent_{i}"] = copy.deepcopy(policy_template)
    for k in list(policies.keys()):
        if not k.startswith("agent_"):
            continue
        try:
            idx = int(k.split("_")[1])
        except Exception:
            continue
        if idx > num_humans:
            del policies[k]

    # -------------------------
    # MultiAgentAccessMgr bookkeeping
    # -------------------------
    mgr = cfg.habitat_baselines.rl.agent
    num_agent_types = int(num_humans + 1)  # + robot
    mgr.num_agent_types = num_agent_types
    if "num_active_agents_per_type" in mgr:
        mgr.num_active_agents_per_type = [1] * num_agent_types
    if "num_pool_agents_per_type" in mgr:
        mgr.num_pool_agents_per_type = [1] * num_agent_types


def override_all_human_radii(cfg, radius: float) -> None:
    """
    Override the configured `radius` for all humanoid agents (agent_1..N if present).
    NOTE: this primarily affects navmesh/pathfinding/step_filter behavior, not the rendered mesh size.
    """
    r = float(radius)
    if not (r > 0.0 and np.isfinite(r)):
        raise ValueError(f"radius must be finite and > 0, got {radius}")
    if r < 1e-4:
        print(f"[WARN] extremely small human radius={r} may cause instability; consider >=1e-3")

    sim_agents = cfg.habitat.simulator.agents
    changed = 0
    for k in list(sim_agents.keys()):
        if not k.startswith("agent_") or k == "agent_0":
            continue
        try:
            sim_agents[k].radius = r
            changed += 1
        except Exception:
            pass
    print(f"[HUMAN_RADIUS] set radius={r} for {changed} humanoid agents")


base_dir = f"data/0_FULL_DATA/captures_v3_{split_to_save}_{SCAN_ID}"
SEED = 42 # 0 # for now both train and val use 0, and the ped path generation is train 42 val 0. if use for saving occ data we need train 42 here then
FORCE_SAME_ISLAND = True
OCC_PLACEHOLDER = f"{FALCON_DATA_ROOT}/scene_datasets/mp3d/{SCAN_ID}/{SCAN_ID}.glb"

# ---------------------------------------------------------------------
output_dir = os.path.join(base_dir, "dynamic", "rgb")
depth_dir = os.path.join(base_dir, "dynamic", "depth")
navigable_dir = os.path.join(base_dir, "traversability_mask")
static_output_dir = os.path.join(base_dir, "static", "rgb")
static_depth_dir = os.path.join(base_dir, "static", "depth")
mp3d_viewpoint_output_dir = os.path.join(base_dir, "dynamic_mp3d_viewpoints", "rgb")
mp3d_viewpoint_depth_dir = os.path.join(base_dir, "dynamic_mp3d_viewpoints", "depth")
static_mp3d_viewpoint_output_dir = os.path.join(base_dir, "static_mp3d_viewpoints", "rgb")
static_mp3d_viewpoint_depth_dir = os.path.join(base_dir, "static_mp3d_viewpoints", "depth")
overlap_rgb_dir = os.path.join(base_dir, "overlap_debug", "rgb")

for d in (
    output_dir,
    depth_dir,
    navigable_dir,
    static_output_dir,
    static_depth_dir,
    mp3d_viewpoint_output_dir,
    mp3d_viewpoint_depth_dir,
    static_mp3d_viewpoint_output_dir,
    static_mp3d_viewpoint_depth_dir,
    overlap_rgb_dir
):
    os.makedirs(d, exist_ok=False)



pkl_path = os.path.join(base_dir, f"scan{SCAN_ID}_Landmark-RxR-dynamic.pkl")
# ---------------------------------------------------------------------

T_MP2HAB = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, -1, 0, 0],
     [0, 0, 0, 1]],
    np.float32,
)
R_MP2HAB = T_MP2HAB[:3, :3]
T_HAB2MP = np.linalg.inv(T_MP2HAB)
R_HAB2MP = T_HAB2MP[:3, :3]


def load_connectivity(scan_id):
    path = os.path.join(CONNECTIVITY_ROOT, f"{scan_id}_connectivity.json")
    with open(path, "r") as f:
        data = json.load(f)
    nodes = []
    for n in data:
        if not n.get("included", True):
            continue
        M = np.array(n["pose"]).reshape(4, 4)
        x, y, z = float(M[0, 3]), float(M[1, 3]), float(M[2, 3])
        Rm = M[:3, :3]
        yaw = math.atan2(Rm[0, 2], Rm[2, 2])
        nodes.append({"hash": n["image_id"], "pos": (x, y, z), "yaw": yaw})
    return nodes


def mp3d_to_hab_pos_yaw(mp_pose):
    x, y, z, yaw = mp_pose
    hab_xyz = (T_MP2HAB @ np.array([x, y, z, 1.0], np.float32))[:3]
    f_mp = np.array([math.cos(yaw), math.sin(yaw), 0.0], np.float32)
    f_hb = R_MP2HAB @ f_mp
    yaw_hb = math.atan2(-f_hb[0], -f_hb[2]) + math.pi  # magic fix
    yaw_hb = (yaw_hb + math.pi) % (2 * math.pi) - math.pi
    return hab_xyz, yaw_hb


def hab_pos_mp_yaw_to_T(pos_hab: np.ndarray, yaw_mp: float) -> np.ndarray:
    pos_mp = (T_HAB2MP @ np.array([*pos_hab, 1.0], np.float32))[:3]
    cos, sin = math.cos(yaw_mp), math.sin(yaw_mp)
    rot_z = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], np.float32)
    R_mp = rot_z
    T_mp = np.eye(4, dtype=np.float32)
    T_mp[:3, :3] = R_mp.astype(np.float32)
    T_mp[:3, 3] = pos_mp.astype(np.float32)
    return T_mp


def make_snap_pathfinder(scene_glb):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    rgb = habitat_sim.CameraSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb]
    snap_sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    snap_sim.recompute_navmesh(snap_sim.pathfinder, habitat_sim.NavMeshSettings())
    return snap_sim.pathfinder


def snap_feet(feet, pf):
    snapped = np.array(pf.snap_point(feet), np.float32)
    dist = float(np.linalg.norm(snapped - feet))
    try:
        ok = pf.is_navigable(snapped, max_y_delta=MAX_SNAP_DIST)
    except TypeError:
        ok = pf.is_navigable(snapped)
    ok = ok and dist <= MAX_SNAP_DIST
    return ok, snapped, dist


def shortest_nav_path(pf, start: np.ndarray, end: np.ndarray):
    sp = habitat_sim.ShortestPath()
    sp.requested_start = np.array(start, np.float32)
    sp.requested_end = np.array(end, np.float32)
    found = pf.find_path(sp)
    pts = np.array(sp.points, np.float32) if sp.points else np.zeros((0, 3), np.float32)
    dist = float(sp.geodesic_distance) if found else math.inf
    if not found or not math.isfinite(dist) or len(pts) < 2:
        return False, pts, math.inf
    return True, pts, dist


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diffs = points[1:] - points[:-1]
    return float(np.linalg.norm(diffs, axis=1).sum())


def build_distance_matrix(nodes, pf):
    N = len(nodes)
    D = [[math.inf] * N for _ in range(N)]
    cache = {}
    for i in range(N):
        D[i][i] = 0.0
    for i in tqdm(range(N), desc="dist-matrix"):
        for j in range(i + 1, N):
            ok, pts, dist = shortest_nav_path(pf, nodes[i]["feet_hab"], nodes[j]["feet_hab"])
            if not ok:
                dist = math.inf
                pts = np.zeros((0, 3), np.float32)
            D[i][j] = D[j][i] = dist
            cache[(i, j)] = pts
            cache[(j, i)] = pts[::-1].copy() if len(pts) else pts
    return D, cache


def greedy_open_path(start, D):
    N = len(D)
    unvisited = set(range(N))
    unvisited.remove(start)
    order = [start]
    cur = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: D[cur][j])
        if not math.isfinite(D[cur][nxt]):
            break
        unvisited.remove(nxt)
        order.append(nxt)
        cur = nxt
    return order


def open_path_cost(order, D):
    if len(order) < 2:
        return 0.0
    return sum(D[order[k]][order[k + 1]] for k in range(len(order) - 1))


def two_opt_open_fast(order, D, time_limit_s=1.0, max_swaps=2000, eps=1e-9):
    N = len(order)
    t0 = time.perf_counter()
    swaps = 0
    while True:
        improved = False
        for i in range(1, N - 2):
            a = order[i - 1]
            b = order[i]
            for j in range(i + 1, N - 1):
                c = order[j]
                d = order[j + 1]
                delta = (D[a][c] + D[b][d]) - (D[a][b] + D[c][d])
                if delta < -eps:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    swaps += 1
                    improved = True
                    break
            if improved:
                break
        if (not improved) or swaps >= max_swaps or (time.perf_counter() - t0) > time_limit_s:
            break
    return order


def solve_open_route(nodes, pf, num_starts=30, seed=0):
    random.seed(seed)
    D, cache = build_distance_matrix(nodes, pf)
    N = len(nodes)
    starts = list(range(N))
    random.shuffle(starts)
    starts = starts[: min(num_starts, N)]
    best_order = None
    best_cost = math.inf
    for s in tqdm(starts, desc="TSP starts"):
        print("Trying start:", s)
        ord0 = greedy_open_path(s, D)
        print("  Greedy cost:", open_path_cost(ord0, D))
        ord1 = two_opt_open_fast(ord0, D)
        print("  2-opt cost:", open_path_cost(ord1, D))
        cost = open_path_cost(ord1, D)
        print("  Final cost:", cost)
        if cost < best_cost and len(ord1) == N:
            best_cost = cost
            best_order = ord1
    if best_order is None:
        best_order = list(range(N))
        best_cost = open_path_cost(best_order, D)
    return best_order, best_cost, D, cache


def stitch_world_path(nodes, order, cache, pf):
    full = []
    for k in range(len(order) - 1):
        a = order[k]
        b = order[k + 1]
        seg = cache.get((a, b))
        if seg is None or len(seg) == 0:
            ok, seg, _ = shortest_nav_path(pf, nodes[a]["feet_hab"], nodes[b]["feet_hab"])
            if not ok:
                continue
        if full and len(seg):
            if np.allclose(full[-1], seg[0]):
                seg = seg[1:]
        full.extend(seg)
    return np.array(full, np.float32)


def densify_path(points: np.ndarray, step: float):
    if len(points) < 2:
        return points
    out = [points[0]]
    for i in range(len(points) - 1):
        p0 = points[i]
        p1 = points[i + 1]
        seg = p1 - p0
        dist = float(np.linalg.norm(seg))
        if dist < 1e-5:
            continue
        direction = seg / dist
        n_steps = max(1, int(math.floor(dist / step)))
        for k in range(1, n_steps + 1):
            out.append(p0 + direction * (k * step))
    if np.linalg.norm(out[-1] - points[-1]) > 1e-5:
        out.append(points[-1])
    return np.array(out, np.float32)


def set_episode_pose(ep, feet_pos, yaw_rad):
    ep.start_position = list(feet_pos)
    ep.start_rotation = quaternion.as_float_array(quaternion.from_euler_angles(0.0, yaw_rad, 0.0))

### for static replay ###

def build_static_sim(scene_glb, height, width, depth_max):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    sim_cfg.enable_physics = False

    rgb = habitat_sim.EquirectangularSensorSpec()
    rgb.uuid = "rgb"
    rgb.sensor_type = habitat_sim.SensorType.COLOR
    rgb.resolution = [height, width]
    rgb.position = [0.0, 0.0, 0.0]
    rgb.orientation = [0.0, 0.0, 0.0]

    depth = habitat_sim.EquirectangularSensorSpec()
    depth.uuid = "depth"
    depth.sensor_type = habitat_sim.SensorType.DEPTH
    depth.resolution = [height, width]
    depth.position = [0.0, 0.0, 0.0]
    depth.orientation = [0.0, 0.0, 0.0]
    # depth.min_depth = 0.0
    depth.near = 1e-3 # it is said it should be > 0
    depth.far = depth_max
    # depth.max_depth = depth_max

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb, depth]
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))

# def build_static_sim(scene_glb, height, width, depth_max, hfov=90.0):
#     """mono pinhole debug version"""
#     sim_cfg = habitat_sim.SimulatorConfiguration()
#     sim_cfg.scene_id = scene_glb
#     sim_cfg.enable_physics = False

#     rgb = habitat_sim.CameraSensorSpec()
#     rgb.uuid = "rgb"
#     rgb.sensor_type = habitat_sim.SensorType.COLOR
#     rgb.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
#     rgb.resolution = [height, width]
#     rgb.hfov = hfov
#     rgb.position = [0.0, 0.0, 0.0]
#     rgb.orientation = [0.0, 0.0, 0.0]

#     depth = habitat_sim.CameraSensorSpec()
#     depth.uuid = "depth"
#     depth.sensor_type = habitat_sim.SensorType.DEPTH
#     depth.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
#     depth.resolution = [height, width]
#     depth.hfov = hfov
#     depth.position = [0.0, 0.0, 0.0]
#     depth.orientation = [0.0, 0.0, 0.0]
#     depth.min_depth = 0.0
#     depth.max_depth = depth_max

#     agent_cfg = habitat_sim.agent.AgentConfiguration()
#     agent_cfg.sensor_specifications = [rgb, depth]
#     return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))




def load_replay_cam_poses(pkl_path):
    with open(pkl_path, "rb") as f:
        obj = pkl.load(f)
    data_list = obj.get("data_list", [])
    poses = []
    tokens = []
    waypoint_tokens = []
    for item in data_list:
        ego2global = np.array(item["ego2global"], np.float32)
        pos_mp = ego2global[:3, 3]
        R_mp = ego2global[:3, :3]
        yaw_mp = math.atan2(R_mp[0, 2], R_mp[2, 2])
        cam_hab, yaw_hab = mp3d_to_hab_pos_yaw(
            (float(pos_mp[0]), float(pos_mp[1]), float(pos_mp[2]), yaw_mp)
        )
        yaw_hab -= math.pi  # magic fix
        yaw_hab = (yaw_hab + math.pi) % (2 * math.pi) - math.pi
        poses.append((cam_hab, yaw_hab))
        token = item.get("token") or item.get("images", {}).get("PANORAMIC", {}).get("sample_data_token")
        tokens.append(token)
        waypoint_tokens.append(item.get("waypoint_token"))
    return poses, tokens, waypoint_tokens



def run_static_replay_from_pkl(
    pkl_path,
    static_output_dir,
    static_depth_dir,
    static_mp3d_viewpoint_output_dir,
    static_mp3d_viewpoint_depth_dir,
):
    poses, tokens, waypoint_tokens = load_replay_cam_poses(pkl_path)
    if not poses:
        print("Static replay: no poses found, skipping.")
        return 0

    scene_glb = os.path.join(SCENE_ROOT, SCAN_ID, f"{SCAN_ID}.glb")
    sim = build_static_sim(scene_glb, eq.height, eq.width, eq_depth.max_depth)
    agent = sim.get_agent(0)
    rgb_sensor = sim._sensors["rgb"]
    depth_sensor = sim._sensors["depth"]

    state = habitat_sim.AgentState()
    for idx, ((cam_pos, yaw_hab), token, wp_token) in enumerate(
        tqdm(zip(poses, tokens, waypoint_tokens), total=len(poses), desc="static", unit="frame")
    ):
        state.position = cam_pos.tolist()
        state.rotation = quat_from_angle_axis(
            yaw_hab, np.array([0.0, 1.0, 0.0], np.float32)
        )
        agent.set_state(state, infer_sensor_states=False)

        rgb_sensor.draw_observation()
        depth_sensor.draw_observation()
        rgb = rgb_sensor.get_observation()
        depth = depth_sensor.get_observation()

        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if depth.ndim == 3:
            depth = depth[..., 0]

        if not token:
            token = f"{SCAN_ID}_{idx:06d}"

        rgb_path_abs = os.path.join(static_output_dir, f"{token}.png")
        depth_path_abs = os.path.join(static_depth_dir, f"{token}.npy")
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        cv2.imwrite(rgb_path_abs, rgb[..., ::-1])
        np.save(depth_path_abs, depth)

        if wp_token:
            wp_rgb_path_abs = os.path.join(static_mp3d_viewpoint_output_dir, f"{wp_token}.png")
            wp_depth_path_abs = os.path.join(static_mp3d_viewpoint_depth_dir, f"{wp_token}.npy")
            cv2.imwrite(wp_rgb_path_abs, rgb[..., ::-1])
            np.save(wp_depth_path_abs, depth)

    sim.close()
    return len(poses)


### end static replay ###


def collect_humans(sim, robot_base_hab, radius=CAPTURE_RADIUS, removed_agent_ids=None):
    humans = {}
    positions = []
    agent_ids = []
    yaws_debug = []
    removed_agent_ids = removed_agent_ids or set()
    for idx in range(1, getattr(sim, "num_articulated_agents", 1)):
        if idx in removed_agent_ids:
            continue
        agent = sim.get_agent_data(idx).articulated_agent
        pos_hab = np.array(agent.base_pos, np.float32) # shape: (3,)
        if np.linalg.norm(pos_hab[[0, 2]] - robot_base_hab[[0, 2]]) <= radius:
            humans[idx] = {"human2global": hab_pos_mp_yaw_to_T(pos_hab, 0.0)}
            positions.append(pos_hab)
            agent_ids.append(idx)
            yaws_debug.append(agent.base_rot)
    return humans, positions, agent_ids, yaws_debug


# def has_overlap(robot_pos, human_positions, radius=CAPTURE_RADIUS, thresh=OVERLAP_THRESH):
#     close_h = [p for p in human_positions if np.linalg.norm(p[[0, 2]] - robot_pos[[0, 2]]) <= radius]
#     for p in close_h:
#         if np.linalg.norm(p[[0, 2]] - robot_pos[[0, 2]]) < thresh:
#             print("Overlap with robot")
#             return True
#     for i in range(len(close_h)):
#         for j in range(i + 1, len(close_h)):
#             if np.linalg.norm(close_h[i][[0, 2]] - close_h[j][[0, 2]]) < thresh:
#                 print("Overlap between humans")
#                 return True
#     return False

def has_overlap(robot_pos, human_positions, radius=CAPTURE_RADIUS, thresh=OVERLAP_THRESH, height_thresh=1.5):
    """ consider height difference to avoid false positive overlaps on different floors """
    close_h = []
    for p in human_positions:
        if abs(p[1] - robot_pos[1]) > height_thresh:
            continue  # different floor/height
        if np.linalg.norm(p[[0, 2]] - robot_pos[[0, 2]]) <= radius:
            close_h.append(p)
    for p in close_h:
        if np.linalg.norm(p[[0, 2]] - robot_pos[[0, 2]]) < thresh:
            print("Overlap with robot")
            return True
    for i in range(len(close_h)):
        for j in range(i + 1, len(close_h)):
            if np.linalg.norm(close_h[i][[0, 2]] - close_h[j][[0, 2]]) < thresh:
                print("Overlap between humans")
                return True
    return False


def _try_get_task_from_env(core_env):
    """
    Best-effort access to Habitat task object from Gym/Habitat wrappers.
    Returns None if not available.
    """
    try:
        habitat_env = core_env._env
        return getattr(habitat_env, "task", None)
    except Exception:
        return None


def remove_human(sim, core_env, agent_id: int, sink_pos_hab: np.ndarray) -> None:
    """
    Make a humanoid "disappear" by moving it to a sink location and disabling its controller updates.
    We refer to this as "removed" (internally, the task may track `human_alive`).
    """
    agent_id = int(agent_id)
    if agent_id <= 0:
        return

    aa = sim.get_agent_data(agent_id).articulated_agent
    aa.base_pos = np.array(sink_pos_hab, np.float32)
    aa.base_rot = 0.0

    task = _try_get_task_from_env(core_env)
    if task is not None:
        # Your custom action uses `task.human_alive` to early-return; treat False as removed.
        i = agent_id - 1
        try:
            if hasattr(task, "human_alive") and i < len(task.human_alive):
                task.human_alive[i] = False
            if hasattr(task, "waypoint_reached") and i < len(task.waypoint_reached):
                task.waypoint_reached[i] = True
        except Exception:
            pass


def find_overlap_culprit_agent_id(
    robot_pos: np.ndarray,
    human_positions: list,
    human_agent_ids: list,
    thresh: float = OVERLAP_THRESH,
    height_thresh: float = 0.5,
) -> int:
    """
    If overlap is with the robot -> return that human.
    Else if overlap is between humans -> return one of that pair (the one closer to robot).
    Returns -1 if no obvious culprit.
    """
    if not human_positions:
        return -1
    rp = np.array(robot_pos, np.float32)
    hp = [np.array(p, np.float32) for p in human_positions]

    # Robot-human overlap
    for p, hid in zip(hp, human_agent_ids):
        if abs(float(p[1] - rp[1])) > float(height_thresh):
            continue
        if float(np.linalg.norm(p[[0, 2]] - rp[[0, 2]])) < float(thresh):
            return int(hid)

    # Human-human overlap (same-floor)
    best_dist = float("inf")
    best_hid = -1
    for i in range(len(hp)):
        for j in range(i + 1, len(hp)):
            if abs(float(hp[i][1] - hp[j][1])) > float(height_thresh):
                continue
            if float(np.linalg.norm(hp[i][[0, 2]] - hp[j][[0, 2]])) < float(thresh):
                di = float(np.linalg.norm(hp[i][[0, 2]] - rp[[0, 2]]))
                dj = float(np.linalg.norm(hp[j][[0, 2]] - rp[[0, 2]]))
                hid = int(human_agent_ids[i] if di <= dj else human_agent_ids[j])
                d = min(di, dj)
                if d < best_dist:
                    best_dist = d
                    best_hid = hid
    return int(best_hid)



# Load config for Spot + pano
cfg = get_config("social_nav_v2/orca_mp3d.yaml")


@dataclass
class ArmEquirectRGBConfig(HabitatSimEquirectangularRGBSensorConfig):
    uuid: str = "articulated_agent_arm_rgb"


@dataclass
class ArmEquirectDepthConfig(HabitatSimEquirectangularDepthSensorConfig):
    uuid: str = "articulated_agent_arm_depth"


eq = OmegaConf.structured(ArmEquirectRGBConfig())
eq.height = 512
eq.width = 1024

eq_depth = OmegaConf.structured(ArmEquirectDepthConfig())
eq_depth.height = eq.height
eq_depth.width = eq.width
eq_depth.max_depth = 10.0
eq_depth.normalize_depth = False

with read_write(cfg):
    cfg.habitat_baselines.num_environments = 1
    cfg.habitat.simulator.concur_render = False
    cfg.habitat_baselines.evaluate = True
    # Load ONLY the per-scene dataset file for this scan.
    cfg.habitat.dataset.split = split_to_save
    cfg.habitat.dataset.content_scenes = [SCAN_ID]
    if AUTO_ADJUST_HUMAN_UPPER_BOUND:
        max_h = get_max_humans_for_scan(SCAN_ID, split_to_save, dataset_root=DATASET_ROOT)
        print(f"[AUTO_HUMANS] scan={SCAN_ID} split={split_to_save} human_num(max)={max_h}")
        adjust_multi_agent_cfg(cfg, max_h)
    if OVERRIDE_HUMAN_RADIUS:
        override_all_human_radii(cfg, HUMAN_RADIUS_OVERRIDE_VALUE)
    cfg.habitat.simulator.agents.agent_0.sim_sensors = {
        "articulated_agent_arm_rgb": eq,
        "articulated_agent_arm_depth": eq_depth,
    }
    cfg.habitat.gym.obs_keys = [
        "agent_0_articulated_agent_arm_rgb",
        "agent_0_articulated_agent_arm_depth",
    ]

env = make_gym_from_config(cfg)
core_env = env.unwrapped._env  # RLTaskEnv
episode = core_env.current_episode
print("episode scene_id =", episode.scene_id)
print("episode human_num =", episode.info.get("human_num"))
# exit(0)

# Restrict to the target scene if possible
if core_env.episodes:
    filtered = [ep for ep in core_env.episodes if SCAN_ID in ep.scene_id]
    if filtered:
        core_env.episodes = filtered

# Pathfinder for snapping (recomputed, permissive)
scene_glb = os.path.join(SCENE_ROOT, SCAN_ID, f"{SCAN_ID}.glb")
pf = make_snap_pathfinder(scene_glb)

# ---------------------------------------------------------------------
# Build navmesh-valid nodes
# ---------------------------------------------------------------------
raw_nodes = load_connectivity(SCAN_ID)
nav_nodes = []
for n in raw_nodes:
    mp_pose = (n["pos"][0], n["pos"][1], n["pos"][2], n["yaw"])
    cam_hab, _ = mp3d_to_hab_pos_yaw(mp_pose)
    feet_raw = cam_hab - np.array([0, CAM_HEIGHT, 0], np.float32)
    ok, feet_snapped, _ = snap_feet(feet_raw, pf)
    if ok:
        nav_nodes.append(
            {
                "hash": n["hash"],
                "feet_hab": feet_snapped,
                "cam_hab": feet_snapped + np.array([0, CAM_HEIGHT, 0], np.float32),
                "yaw_mp": 0.0,  # hold yaw at 0 in MP3D frame
                "island": int(pf.get_island(feet_snapped)),
            }
        )

print(f"Navmesh-valid nodes: {len(nav_nodes)}")
if FORCE_SAME_ISLAND:
    islands = [n["island"] for n in nav_nodes if n["island"] >= 0]
    if islands:
        maj = max(set(islands), key=islands.count)
        nav_nodes = [n for n in nav_nodes if n["island"] == maj]
        print(f"After same-island filter: {len(nav_nodes)} nodes on island {maj}")
if not nav_nodes:
    raise RuntimeError("No navigable nodes after snapping.")

# ---------------------------------------------------------------------
# Solve open-route TSP on geodesic distances
# ---------------------------------------------------------------------
order, route_cost, D, path_cache = solve_open_route(nav_nodes, pf, num_starts=30, seed=SEED)
print(f"Route len {len(order)}, cost {route_cost:.2f}")
adj_dists = [D[order[i]][order[i + 1]] for i in range(len(order) - 1)]
if adj_dists:
    print(f"Adjacent geodesic distances (m): mean={np.mean(adj_dists):.3f}, "
          f"min={np.min(adj_dists):.3f}, max={np.max(adj_dists):.3f}")
world_path = stitch_world_path(nav_nodes, order, path_cache, pf)
dense_path = densify_path(world_path, STEP_DIST)
print(f"Dense waypoints: {len(dense_path)}")

# Map connectivity waypoints onto dense_path (within tolerance)
waypoint_tokens = [""] * len(dense_path)
feet_array = np.array([n["feet_hab"] for n in nav_nodes], dtype=np.float32)
hashes = [n["hash"] for n in nav_nodes]

tol = 0.05  # 5 cm tolerance
missing = []
min_dists = []
for feet, h in zip(feet_array, hashes):
    dists = np.linalg.norm(dense_path - feet[None, :], axis=1)
    idx_near = int(np.argmin(dists))
    min_d = float(dists[idx_near])
    min_dists.append(min_d)
    if min_d <= tol:
        waypoint_tokens[idx_near] = h
    else:
        missing.append((h, min_d))

if missing:
    max_d = max(m[1] for m in missing)
    raise RuntimeError(
        f"Connectivity waypoints not found in dense_path within tol={tol}. "
        f"Max min-dist={max_d:.3f}. Examples: {missing[:10]}"
    )




# ---------------------------------------------------------------------
# Run through dense waypoints, advance sim, capture every N steps
# ---------------------------------------------------------------------
# start_pos = dense_path[0]
# set_episode_pose(episode, start_pos, 0.0)
# core_env.current_episode = episode
# core_env.episode_iterator = iter([episode])
# obs = env.reset()

# step_idx = 0  # env steps (increments every waypoint tick)
# data_list = []
# capture_idx = 0

# default_action = 5  # pause index from manual control
# act_vec = np.zeros(env.action_space.shape, dtype=np.float32)
# act_vec[0] = default_action

# # Tick counter since last capture boundary
# ticks_since_capture = 0
# idx = 0
# wait_overlap_counts = 0

# if DEBUG_LIMIT is not None:
#     print(f"DEBUG: limiting dense path to first {DEBUG_LIMIT} waypoints")
#     dense_path = dense_path[:DEBUG_LIMIT] # DEBUG

# pbar = tqdm(total=len(dense_path), desc="traverse", unit="wp")
# while idx < len(dense_path):
#     wp = dense_path[idx]
#     wp_token = waypoint_tokens[idx]

#     sim = core_env._sim
#     robot = sim.get_agent_data(0).articulated_agent
#     robot.base_pos = np.array(wp, np.float32)
#     robot.base_rot = 0.0  # yaw in radians for the articulated base

#     # Advance sim (keeps humans moving)
#     obs, _, done, _ = env.step(act_vec)
#     if done:
#         obs = env.reset()

#     step_idx += 1
#     ticks_since_capture += 1

#     # If not at capture boundary, decide whether to advance along dense_path
#     if ticks_since_capture % CAPTURE_EVERY_STEPS != 0:
#         # For non-waypoint, move on; for waypoint, stay and keep ticking
#         if not wp_token:
#             idx += 1
#             pbar.update(1)
#         continue

#     # Capture boundary reached
#     robot_base = np.array(robot.base_pos, np.float32)
#     human_poses, human_positions, yaws_debug = collect_humans(sim, robot_base)

#     # overlap = has_overlap(robot_base, human_positions, radius=CAPTURE_RADIUS, thresh=0.1)
#     overlap = has_overlap(robot_base, human_positions, radius=CAPTURE_RADIUS, thresh=OVERLAP_THRESH)
#     if overlap:
#         if wp_token or NEVER_SKIP_OVERLAP:
#             # Save what we saw at this stuck waypoint (RGB only)
#             dbg_frame = obs["agent_0_articulated_agent_arm_rgb"]
#             dbg_token = wp_token or f"overlap_{step_idx:06d}"
#             rgb_dbg_path = os.path.join(
#                 overlap_rgb_dir, f"{dbg_token}_retry{wait_overlap_counts}.png"
#             )
#             cv2.imwrite(rgb_dbg_path, dbg_frame[..., ::-1])

#             wait_overlap_counts += 1
#             if wait_overlap_counts > MAX_OVERLAP_RETRIES:
#                 raise RuntimeError(
#                     f"Overlap not resolved at waypoint {wp_token} after {MAX_OVERLAP_RETRIES} retries"
#                 )
#             # stay on same idx, keep ticking
#             ticks_since_capture = 0
#             continue
#         else:
#             # skip capture and advance path
#             idx += 1
#             pbar.update(1)
#             ticks_since_capture = 0
#             continue


#     # No overlap: capture
#     frame = obs["agent_0_articulated_agent_arm_rgb"]
#     depth = obs["agent_0_articulated_agent_arm_depth"]
#     depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
#     if depth.ndim == 3:
#         depth = depth[..., 0]
#     depth_m = depth * (eq_depth.max_depth if getattr(eq_depth, "normalize_depth", False) else 1.0)

#     token = f"{SCAN_ID}_{capture_idx:06d}"
#     rgb_path_abs = os.path.join(output_dir, f"{token}.png")
#     depth_path_abs = os.path.join(depth_dir, f"{token}.npy")
#     cv2.imwrite(rgb_path_abs, frame[..., ::-1])
#     np.save(depth_path_abs, depth_m)

#     # Extra save for connectivity waypoint
#     if wp_token:
#         wp_rgb_path_abs = os.path.join(mp3d_viewpoint_output_dir, f"{wp_token}.png")
#         wp_depth_path_abs = os.path.join(mp3d_viewpoint_depth_dir, f"{wp_token}.npy")
#         cv2.imwrite(wp_rgb_path_abs, frame[..., ::-1])
#         np.save(wp_depth_path_abs, depth_m)

#     cam_center_hab = robot_base + np.array([0, CAM_HEIGHT, 0], np.float32)
#     ego2global = hab_pos_mp_yaw_to_T(cam_center_hab, 0.0)

#     data_list.append(
#         {
#             "ego2global": ego2global.astype(np.float32),
#             "human_poses": human_poses,
#             "images": {
#                 "PANORAMIC": {
#                     "img_path": os.path.relpath(rgb_path_abs, base_dir),
#                     "depth_path": os.path.relpath(depth_path_abs, base_dir),
#                     "lidar2cam": np.eye(4, dtype=np.float32),
#                     "sample_data_token": token,
#                     "timestamp": float(capture_idx),  # env steps
#                 }
#             },
#             "instances": [],
#             "occ_path": OCC_PLACEHOLDER,
#             "sample_idx": float(capture_idx),
#             "scene_idx": SCAN_ID,
#             "scene_name": SCAN_ID,
#             "scene_token": SCAN_ID,
#             "timestamp": float(capture_idx),  # env steps
#             "token": token,
#             "waypoint_token": wp_token if wp_token else None,
#         }
#     )

#     capture_idx += 1
#     ticks_since_capture = 0
#     wait_overlap_counts = 0

#     idx += 1
#     pbar.update(1)

# pbar.close()
# ---------------------------------------------------------------------
# Run through waypoints / dense waypoints
# ---------------------------------------------------------------------
# NOTE: In TELEPORT_WAYPOINT_ONLY mode:
# - we ONLY capture connectivity waypoints (nav_nodes in `order`)
# - we do ONE env.step() per waypoint (plus overlap retries)
# - we ONLY save into dynamic_mp3d_viewpoints/{rgb,depth}
# - token == waypoint token (connectivity hash)
# - orientation stays identical to previous behavior: yaw_mp = 0.0

if TELEPORT_WAYPOINT_ONLY:
    # Start at first waypoint
    start_pos = nav_nodes[order[0]]["feet_hab"]
    set_episode_pose(episode, start_pos, 0.0)
    core_env.current_episode = episode
    core_env.episode_iterator = iter([episode])
    obs = env.reset()

    step_idx = 0
    data_list = []
    capture_idx = 0
    wait_overlap_counts = 0

    default_action = 5  # pause index from manual control
    act_vec = np.zeros(env.action_space.shape, dtype=np.float32)
    act_vec[0] = default_action

    pbar = tqdm(total=len(order), desc="teleport-waypoints", unit="wp")
    for node_id in order:
        wp = nav_nodes[node_id]["feet_hab"]
        wp_token = nav_nodes[node_id]["hash"]  # token == waypoint token

        # Try until no overlap (or max retries)
        while True:
            sim = core_env._sim
            robot = sim.get_agent_data(0).articulated_agent

            # Teleport robot base to the waypoint; keep yaw fixed (matches old behavior)
            robot.base_pos = np.array(wp, np.float32)
            robot.base_rot = 0.0

            # Advance sim once (humans move, sensors update)
            obs, _, done, _ = env.step(act_vec)
            if done:
                obs = env.reset()
            step_idx += 1

            robot_base = np.array(robot.base_pos, np.float32)
            human_poses, human_positions, human_agent_ids, _ = collect_humans(
                sim, robot_base, removed_agent_ids=REMOVED_HUMAN_AGENT_IDS
            )

            overlap = has_overlap(
                robot_base, human_positions, radius=CAPTURE_RADIUS, thresh=OVERLAP_THRESH
            )

            if overlap:
                # Save overlap debug RGB (optional but kept)
                dbg_frame = obs["agent_0_articulated_agent_arm_rgb"]
                rgb_dbg_path = os.path.join(
                    overlap_rgb_dir, f"{wp_token}_retry{wait_overlap_counts}.png"
                )
                cv2.imwrite(rgb_dbg_path, dbg_frame[..., ::-1])

                wait_overlap_counts += 1
                if (
                    REMOVE_HUMAN_ON_PERSISTENT_OVERLAP
                    and wait_overlap_counts >= REMOVE_AFTER_RETRIES
                ):
                    culprit = find_overlap_culprit_agent_id(
                        robot_base,
                        human_positions,
                        human_agent_ids,
                        thresh=OVERLAP_THRESH,
                        height_thresh=0.5,
                    )
                    if culprit > 0 and culprit not in REMOVED_HUMAN_AGENT_IDS:
                        print(
                            f"[REMOVE] waypoint={wp_token} retry={wait_overlap_counts} removing agent_{culprit}"
                        )
                        remove_human(sim, core_env, culprit, REMOVAL_SINK_POS_HAB)
                        REMOVED_HUMAN_AGENT_IDS.add(int(culprit))
                        wait_overlap_counts = 0
                        continue
                if wait_overlap_counts > MAX_OVERLAP_RETRIES:
                    raise RuntimeError(
                        f"Overlap not resolved at waypoint {wp_token} after {MAX_OVERLAP_RETRIES} retries"
                    )
                # stay on same waypoint, step again
                continue

            # No overlap -> capture
            wait_overlap_counts = 0

            frame = obs["agent_0_articulated_agent_arm_rgb"]
            depth = obs["agent_0_articulated_agent_arm_depth"]
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth_m = depth * (
                eq_depth.max_depth if getattr(eq_depth, "normalize_depth", False) else 1.0
            )

            # Save ONLY to waypoint folders (no on-the-way dynamic/rgb anymore)
            wp_rgb_path_abs = os.path.join(mp3d_viewpoint_output_dir, f"{wp_token}.png")
            wp_depth_path_abs = os.path.join(mp3d_viewpoint_depth_dir, f"{wp_token}.npy")
            cv2.imwrite(wp_rgb_path_abs, frame[..., ::-1])
            np.save(wp_depth_path_abs, depth_m)

            # Pose saved in PKL: keep yaw_mp = 0.0 to match previous behavior
            cam_center_hab = robot_base + np.array([0, CAM_HEIGHT, 0], np.float32)
            ego2global = hab_pos_mp_yaw_to_T(cam_center_hab, 0.0)

            data_list.append(
                {
                    "ego2global": ego2global.astype(np.float32),
                    "human_poses": human_poses,
                    "images": {
                        "PANORAMIC": {
                            "img_path": os.path.relpath(wp_rgb_path_abs, base_dir),
                            "depth_path": os.path.relpath(wp_depth_path_abs, base_dir),
                            "lidar2cam": np.eye(4, dtype=np.float32),
                            "sample_data_token": wp_token,  # token==wp_token
                            "timestamp": float(capture_idx),
                        }
                    },
                    "instances": [],
                    "occ_path": OCC_PLACEHOLDER,
                    "sample_idx": float(capture_idx),
                    "scene_idx": SCAN_ID,
                    "scene_name": SCAN_ID,
                    "scene_token": SCAN_ID,
                    "timestamp": float(capture_idx),
                    "token": wp_token,          # Option A
                    "waypoint_token": wp_token, # always present
                }
            )

            capture_idx += 1
            break  # next waypoint

        pbar.update(1)

    pbar.close()

else:
    # -------------------------
    # ORIGINAL dense traversal
    # -------------------------
    start_pos = dense_path[0]
    set_episode_pose(episode, start_pos, 0.0)
    core_env.current_episode = episode
    core_env.episode_iterator = iter([episode])
    obs = env.reset()

    step_idx = 0  # env steps (increments every waypoint tick)
    data_list = []
    capture_idx = 0

    default_action = 5  # pause index from manual control
    act_vec = np.zeros(env.action_space.shape, dtype=np.float32)
    act_vec[0] = default_action

    ticks_since_capture = 0
    idx = 0
    wait_overlap_counts = 0

    if DEBUG_LIMIT is not None:
        print(f"DEBUG: limiting dense path to first {DEBUG_LIMIT} waypoints")
        dense_path = dense_path[:DEBUG_LIMIT]

    pbar = tqdm(total=len(dense_path), desc="traverse", unit="wp")
    while idx < len(dense_path):
        wp = dense_path[idx]
        wp_token = waypoint_tokens[idx]

        sim = core_env._sim
        robot = sim.get_agent_data(0).articulated_agent
        robot.base_pos = np.array(wp, np.float32)
        robot.base_rot = 0.0  # yaw in radians for the articulated base

        obs, _, done, _ = env.step(act_vec)
        if done:
            obs = env.reset()

        step_idx += 1
        ticks_since_capture += 1

        if ticks_since_capture % CAPTURE_EVERY_STEPS != 0:
            if not wp_token:
                idx += 1
                pbar.update(1)
            continue

        robot_base = np.array(robot.base_pos, np.float32)
        human_poses, human_positions, human_agent_ids, yaws_debug = collect_humans(
            sim, robot_base, removed_agent_ids=REMOVED_HUMAN_AGENT_IDS
        )

        overlap = has_overlap(robot_base, human_positions, radius=CAPTURE_RADIUS, thresh=OVERLAP_THRESH)
        if overlap:
            if wp_token or NEVER_SKIP_OVERLAP:
                dbg_frame = obs["agent_0_articulated_agent_arm_rgb"]
                dbg_token = wp_token or f"overlap_{step_idx:06d}"
                rgb_dbg_path = os.path.join(
                    overlap_rgb_dir, f"{dbg_token}_retry{wait_overlap_counts}.png"
                )
                cv2.imwrite(rgb_dbg_path, dbg_frame[..., ::-1])

                wait_overlap_counts += 1
                if (
                    REMOVE_HUMAN_ON_PERSISTENT_OVERLAP
                    and wait_overlap_counts >= REMOVE_AFTER_RETRIES
                ):
                    culprit = find_overlap_culprit_agent_id(
                        robot_base,
                        human_positions,
                        human_agent_ids,
                        thresh=OVERLAP_THRESH,
                        height_thresh=0.5,
                    )
                    if culprit > 0 and culprit not in REMOVED_HUMAN_AGENT_IDS:
                        print(
                            f"[REMOVE] wp={wp_token or 'dense'} retry={wait_overlap_counts} removing agent_{culprit}"
                        )
                        remove_human(sim, core_env, culprit, REMOVAL_SINK_POS_HAB)
                        REMOVED_HUMAN_AGENT_IDS.add(int(culprit))
                        wait_overlap_counts = 0
                        ticks_since_capture = 0
                        continue
                if wait_overlap_counts > MAX_OVERLAP_RETRIES:
                    raise RuntimeError(
                        f"Overlap not resolved at waypoint {wp_token} after {MAX_OVERLAP_RETRIES} retries"
                    )
                ticks_since_capture = 0
                continue
            else:
                idx += 1
                pbar.update(1)
                ticks_since_capture = 0
                continue

        frame = obs["agent_0_articulated_agent_arm_rgb"]
        depth = obs["agent_0_articulated_agent_arm_depth"]
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        if depth.ndim == 3:
            depth = depth[..., 0]
        depth_m = depth * (eq_depth.max_depth if getattr(eq_depth, "normalize_depth", False) else 1.0)

        token = f"{SCAN_ID}_{capture_idx:06d}"
        rgb_path_abs = os.path.join(output_dir, f"{token}.png")
        depth_path_abs = os.path.join(depth_dir, f"{token}.npy")
        cv2.imwrite(rgb_path_abs, frame[..., ::-1])
        np.save(depth_path_abs, depth_m)

        if wp_token:
            wp_rgb_path_abs = os.path.join(mp3d_viewpoint_output_dir, f"{wp_token}.png")
            wp_depth_path_abs = os.path.join(mp3d_viewpoint_depth_dir, f"{wp_token}.npy")
            cv2.imwrite(wp_rgb_path_abs, frame[..., ::-1])
            np.save(wp_depth_path_abs, depth_m)

        cam_center_hab = robot_base + np.array([0, CAM_HEIGHT, 0], np.float32)
        ego2global = hab_pos_mp_yaw_to_T(cam_center_hab, 0.0)

        data_list.append(
            {
                "ego2global": ego2global.astype(np.float32),
                "human_poses": human_poses,
                "images": {
                    "PANORAMIC": {
                        "img_path": os.path.relpath(rgb_path_abs, base_dir),
                        "depth_path": os.path.relpath(depth_path_abs, base_dir),
                        "lidar2cam": np.eye(4, dtype=np.float32),
                        "sample_data_token": token,
                        "timestamp": float(capture_idx),
                    }
                },
                "instances": [],
                "occ_path": OCC_PLACEHOLDER,
                "sample_idx": float(capture_idx),
                "scene_idx": SCAN_ID,
                "scene_name": SCAN_ID,
                "scene_token": SCAN_ID,
                "timestamp": float(capture_idx),
                "token": token,
                "waypoint_token": wp_token if wp_token else None,
            }
        )

        capture_idx += 1
        ticks_since_capture = 0
        wait_overlap_counts = 0

        idx += 1
        pbar.update(1)

    pbar.close()



pkl_obj = {
    "metainfo": {"dataset": "Landmark-RxR-dynamic", "version": "v1.0", "info_version": "v0", "classes": ()},
    "data_list": data_list,
}
with open(pkl_path, "wb") as f:
    pkl.dump(pkl_obj, f)

print(f"Done. Captures: {capture_idx}, route cost (m): {route_cost:.2f}, waypoints: {len(dense_path)}")
print(f"Saved pkl to {pkl_path}")
env.close()

static_count = run_static_replay_from_pkl(
    pkl_path,
    static_output_dir,
    static_depth_dir,
    static_mp3d_viewpoint_output_dir,
    static_mp3d_viewpoint_depth_dir,
)
print(f"Static done. Captures: {static_count}")

cv2.destroyAllWindows()
