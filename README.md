# Infinigen-Sim (infinipart)

Dual-volume topological partition pipeline for articulated objects — from procedural generation to video-conditioned 3D reconstruction.

## Overview

This project trains a network to predict **dual-volume topological partitions** of articulated objects from **video only** (V-JEPA2 encoded features). The dual volume (part0 + part1) is defined by **alternating 2-coloring** on the reduced URDF joint graph — NOT by moving/static or root/child classification.

### Dual Volume Definition

1. Classify joints as **active / passive / fixed** for the given animode
2. Merge all parts connected by **fixed joints** into single topological nodes (node contraction)
3. Reduced graph: **nodes = merged part groups**, **edges = movable joints** (active + passive)
4. Apply **bipartite 2-coloring**: adjacent nodes get opposite colors
5. part0 = one color, part1 = the other color

> This is a **topology problem**. All operations (splits, bridging, merging, joint classification) live on the URDF link/joint graph. Meshes are just geometric realization.

### Example: Lamp (chain topology)

```
Kinematic chain: base -> arm_joint -> arm -> head_joint -> head

senior_0 (active: arm_joint + head_joint):
  No fixed joints -> no merging
  Graph: base -[arm_j]- arm -[head_j]- head
  2-coloring: base=part0, arm=part1, head=part0
  Note: base and head get SAME color (both ends of chain)
```

### Example: Dishwasher (sibling topology)

```
basic_1 (active: upper_rack_joint):
  BVH: rack slides out, hits door -> door_joint = passive
  lower_rack not hit -> fixed
  Merge fixed: base + lower_rack -> node A
  Graph: rack -[rack_j]- A(base+lower) -[door_j]- door
  2-coloring: rack=part0, A=part1, door=part0
  Note: rack and door get SAME color despite one active, one passive
```

## Data Pipeline

```
Asset Generation -> Precompute -> Render -> Encode -> Train
```

1. **Asset Generation**: Infinigen-Sim procedurally generates articulated objects (URDF + per-part OBJs)
   - IS factories: 18 procedural factories in `infinigen/assets/sim_objects/`
   - PhysX factories: pre-generated in `outputs/`

2. **Precompute** (`split_precompute.py`): For each object x each animode:
   - Normalize object to unit cube
   - Classify joints via BVH collision along active trajectory
   - Merge fixed-joint parts (node contraction)
   - Bipartite 2-coloring on reduced graph
   - Export part0.obj + part1.obj + verify.png + metadata.json

3. **Render** (`render_articulation.py`): Blender renders articulation videos per animode
   - Per-frame BVH collision detection for passive joints and penetration prevention
   - 32 views per animode: 16 hemisphere + 8 orbit + 8 sweep

4. **Encode**: Videos -> V-JEPA2 features; dual volume OBJs -> PartPacker VAE -> `[B, 8192, 64]` latent

5. **Train**: V-JEPA2 video features -> PartPacker Flow DiT -> dual volume latent

## Animode System

- **basic_animode** (basic_0, basic_1, ...): one joint per animode, one per joint
- **senior_animode** (senior_0, senior_1, ...): multi-joint combinations
  - Same-chain combinations allowed
  - Randomly sampled, max 10
- Root node defaults to **part0**
- Static filtering: if all 32 views show no motion, skip animode entirely

## Joint Classification (per-animode)

| Type | Definition | Behavior |
|------|-----------|----------|
| **Active** | Defined by animode | Drives motion (trajectory: sinusoidal, one-way sinusoidal, linear, linear oscillation) |
| **Passive** | BVH collision detected along active trajectory | Pre-opens to yield; per-frame collision response during render |
| **Fixed** | No collision detected | Locked; parts merged for 2-coloring |

BVH collision classification is **per-animode per-trajectory**: each animode has its own trajectory type and split.

## Architecture

- **PartPacker VAE**: part0 + part1 each -> [B, 4096, 64] -> concat -> [B, 8192, 64]
- **PartPacker Flow DiT**: 1249.5M params, 1536 hidden_dim, 24 layers, 16 heads
- **Input**: V-JEPA2 encoded video features (video only, no 3D supervision at inference)

## Metadata (per animode)

- Trajectory type
- Envmap path + filename
- Joint classification (active / passive / fixed per joint)
- Joint 12-dim vector: `[0:3]` axis_origin, `[3:6]` axis_direction, `[6:9]` type one-hot, `[9:11]` range, `[11]` exists flag
- 2-coloring assignment
- Passive joint pre-opening angles

## Environment

- **Conda env**: `infinigen-sim`
- **Blender**: `/mnt/data/yurh/blender-3.6.0-linux-x64/blender`
- **GPU**: 4x NVIDIA L20X (143GB each)

## Project Structure

```
infinigen/assets/sim_objects/   # 18 IS procedural factories
outputs/                        # PhysX factory outputs (URDF + OBJs)
split_precompute.py             # Precompute: normalize -> classify -> 2-color -> export
render_articulation.py          # Render: Blender articulation videos
```

## References

- [Infinigen: Infinite Photorealistic Worlds Using Procedural Generation](https://arxiv.org/abs/2306.09310) (CVPR 2023)
- [Infinigen Indoors](https://arxiv.org/abs/2406.11824) (CVPR 2024)
- [PartPacker](https://arxiv.org/abs/2506.09980) — dual volume / alternating 2-coloring definition
