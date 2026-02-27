# Joint Axis Data Pipeline for PartPacker Training

## Overview

Provides GT joint axis data from Infinigen-Sim URDF + precompute for
the PartPacker joint prediction module (see `PartPacker/docs/joint_axis_prediction.md`).

Joint axis extraction is **integrated into `split_precompute.py`** — no separate script needed.

## Output per split

Each split directory contains:
```
{precompute}/{Factory}/{id}/{split}/
    part0.obj       # body mesh (normalized)
    part1.obj       # moving mesh (normalized)
    joints.npy      # [8, 12] joint param tensor (per-split!)
```

**Per-split joint consistency**: each `joints.npy` only includes joints whose
moving groups are in part1 for that specific split:

| Split | Part1 contains | joints.npy contains |
|-------|---------------|-------------------|
| `default` (all movable) | all moving groups | ALL joints |
| `anim0` (revolute only) | revolute groups | revolute joints only |
| `anim1` (prismatic only) | prismatic groups | prismatic joints only |
| `anim10` (joint N only) | joint N's groups | joint N only |

Example — DishwasherFactory/0 (3 joints: 1 revolute door + 2 prismatic racks):
```
default/joints.npy  → 3 joints (door + rack0 + rack1)
anim0/joints.npy    → 1 joint  (door only)
anim1/joints.npy    → 2 joints (rack0 + rack1)
anim10/joints.npy   → 1 joint  (rack0 only)
anim11/joints.npy   → 1 joint  (rack1 only)
```

## Joint Param Tensor Format

`joints.npy`: shape `[N_max=8, 12]`, float32, zero-padded:

```
dim 0-2:   axis_origin    (normalized coords, in [-0.95, 0.95])
dim 3-5:   axis_direction (unit vector)
dim 6-8:   joint type     one-hot [revolute, prismatic, continuous]
dim 9-10:  motion range   [0, max]
dim 11:    exists flag    (1.0 for real joints, 0.0 for padding)
```

## URDF Axis Extraction

`parse_urdf_joints()` in `split_precompute.py` handles:

1. **Kinematic chain traversal**: BFS from root link, composing 4x4 transforms
   through fixed joints (e.g., `l_world → l_0` offset)
2. **World-frame axis origin**: `parent_world_pos + parent_world_rot @ joint_local_xyz`
3. **World-frame axis direction**: `parent_world_rot @ axis_local` (normalized)
4. **Normalization**: `(world_origin - center) * scale` — same transform as mesh vertices
5. **Axis direction invariant**: isotropic scaling preserves unit vectors, no transform needed

## metadata.json Format

```json
{
  "factory": "DishwasherFactory",
  "identifier": "0",
  "normalize": {"center": [x, y, z], "scale": 1.23},
  "joints": [
    {
      "name": "joint_prismatic_12",
      "type": "prismatic",
      "groups": [4],
      "motion_range": 0.882304,
      "axis_origin": [0.837, 0.431, 0.689],
      "axis_direction": [1.0, 0.0, 0.0]
    },
    {
      "name": "joint_revolute_8",
      "type": "revolute",
      "groups": [1, 2],
      "motion_range": 1.57,
      "axis_origin": [1.731, 0.738, 0.0],
      "axis_direction": [0.0, 1.0, 0.0]
    }
  ],
  "splits": {
    "default": {
      "body": [0, 3],
      "moving": [1, 2, 4, 5],
      "active_joints": ["joint_prismatic_12", "joint_prismatic_13", "joint_revolute_8"],
      "n_joints": 3
    },
    "anim0": {
      "body": [0, 3, 4, 5],
      "moving": [1, 2],
      "active_joints": ["joint_revolute_8"],
      "n_joints": 1,
      "type": "revolute"
    }
  }
}
```

## Usage

```bash
cd /mnt/data/yurh/Infinigen-Sim

# Single object (joints.npy auto-generated with part0/part1)
python split_precompute.py --factory DishwasherFactory --seed 0 \
    --base /mnt/data/yurh/Infinite-Mobility --force

# All objects, 4 parallel shards
for i in 0 1 2 3; do
  python split_precompute.py --all --max_seeds 5 --shard $i/4 --force &
done
```

## Loading in PartPacker Training

```python
import numpy as np

# Load joint params for a specific split
joints = np.load("precompute/DishwasherFactory/0/default/joints.npy")  # [8, 12]
n_joints = int(joints[:, 11].sum())  # count real joints
joint_mask = joints[:, 11] > 0.5     # [8] bool mask

# For Joint Encoder input:
#   joint_params = torch.from_numpy(joints)     # [8, 12]
#   joint_mask   = torch.from_numpy(joint_mask)  # [8]
```

## Coordinate System Notes

- All coordinates in the **normalized frame**: centered at object centroid, scaled to [-0.95, 0.95]
- Axis direction is a **unit vector** — sign ambiguity exists for revolute (axis ≡ -axis)
  - Loss handles this: `L_axis = 1 - |cos(pred, gt)|`
- For prismatic joints, axis = translation direction (sign matters for range interpretation)
  - Convention: positive direction = joint opening direction
- URDF `<origin>` is relative to parent link frame
  - `parse_urdf_joints()` composes the full kinematic chain to get world-frame coords
  - Handles nested fixed joints (e.g., `l_world → l_0` with offset + rpy)
