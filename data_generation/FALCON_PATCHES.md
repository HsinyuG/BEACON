# BEACON modifications to the Falcon environment

BEACON's data-generation pipeline runs on top of a fork of
[Falcon](https://github.com/Zeying-Gong/Falcon) (which itself forks habitat-lab /
habitat-sim). The full fork is not bundled with this release. Instead, we provide
the three files we modified as drop-in patches under
[`falcon_patches/`](falcon_patches/), mirroring their paths in a Falcon clone:

```
falcon_patches/
├── falcon/additional_action.py
├── habitat-baselines/habitat_baselines/config/social_nav_v2/orca_mp3d.yaml
└── habitat-lab/habitat/config/benchmark/nav/socialnav_v2/orca_mp3d_task.yaml
```

Copy each file over the matching path in your Falcon clone (see the
[main README](../README.md#3️⃣-data-generation-optional)). Falcon is MIT-licensed;
these patches remain under their upstream license.

> Two of the three files add content that does not exist in upstream Falcon
> `main` (the pedestrian action class and the 200-human configs), so they are
> provided as full replacements rather than flag changes.

---

## 1. `falcon/additional_action.py`

Adds a task action class `OracleNavRandCoordAction_Obstacle_Custom` (registered
via `@registry.register_task_action`). It is a synced waypoint follower that
drives each human along the per-episode waypoint lists produced by
`create_random_peds.py`. Because the robot teleports between waypoints during
capture, humans must keep pace; two flags control this:

| Flag | Value | Effect |
|---|---|---|
| `DIRECT_TO_SYNC_GOAL` | `True` | Humans steer toward the current sync waypoint instead of the next shortest-path vertex, so they advance faster per `env.step()`. |
| `WALK_DURING_TURN` | `True` | Humans rotate while walking instead of turning in place, avoiding zero-translation stalls. |

Waypoint "reached" is decided by xz distance against `dist_thresh = 0.25` m.

## 2. `habitat-baselines/.../social_nav_v2/orca_mp3d.yaml`

Configures the crowd for data collection:

- **200 human agents** (`agent_1 … agent_200`), each using the
  `OracleNavRandCoordAction_Obstacle_Custom` action.
- **High `lin_speed` / `ang_speed` (2000)** — puts humans in a fast, effectively
  per-step-teleport mode suited to static capture.
- **Dataset** `data/datasets/pointnav/crowd-mp3d/{split}/{split}.json.gz`,
  `shuffle: False`, `num_environments: 1`.

## 3. `habitat-lab/.../benchmark/nav/socialnav_v2/orca_mp3d_task.yaml`

Declares simulator agents `agent_1 … agent_200`, reusing the humanoid templates
`human_1 … human_12` in round-robin. `auto_save_pano.py` relies on this: its
`adjust_multi_agent_cfg()` cycles the templates to instantiate exactly
`human_num` humans at runtime.

---

## Key capture parameters

Values set by the two BEACON scripts (not by the patches above), for reference:

| Parameter | Value | Where | Meaning |
|---|---|---|---|
| Human radius (sim) | `0.01` | `auto_save_pano.py` | Runtime override so humans do not block the robot during capture. |
| Human radius (planning) | `0.25` | `create_random_peds.py` | Navmesh inflation used when sampling pedestrian trajectories. |
| Goal-reach threshold (xz) | `0.25` | `additional_action.py` | Distance at which a human advances to its next waypoint. |
| Snap tolerance (xz) | `0.2` | `create_random_peds.py` | Max navmesh snap drift accepted for a grid node. |
| Robot teleport mode | `True` | `auto_save_pano.py` | Robot teleports between MP3D connectivity waypoints; only those are captured. |
