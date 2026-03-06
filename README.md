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

## Environment Setup

### Prerequisites

- **OS**: Ubuntu 22.04+ (tested on 22.04 and 24.04)
- **GPU**: NVIDIA GPU with CUDA support (tested on L20X 143GB, 4090 24GB)
- **Python**: 3.11 (required by both infinigen and Blender 4.2)
- **Blender**: 4.2.18 LTS (ships its own Python 3.11 + CUDA runtime for Cycles)

### Step 1: Install Blender 4.2.18

```bash
# Download and extract
cd /opt  # or any shared path
wget https://download.blender.org/release/Blender4.2/blender-4.2.18-linux-x64.tar.xz
tar -xf blender-4.2.18-linux-x64.tar.xz
rm blender-4.2.18-linux-x64.tar.xz

# Set environment variable
export BLENDER_BIN=/opt/blender-4.2.18-linux-x64/blender

# Install Python packages into Blender's bundled Python
${BLENDER_BIN} --background --python-expr "
import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install',
    'gin-config', 'trimesh', 'tqdm', 'psutil', 'opencv-python-headless',
    'matplotlib', 'scikit-image', 'scipy', 'imageio', 'scikit-learn',
    'networkx', 'pandas', 'shapely', 'geomdl', 'OpenEXR'])
"
```

### Step 2: Install Infinigen-Sim (conda)

```bash
# Create conda env with Python 3.11 (must match Blender)
conda create -n infinigen-sim python=3.11 -y
conda activate infinigen-sim

# Clone and install
git clone https://github.com/AuroraRyan0301/infinipart.git Infinigen-Sim
cd Infinigen-Sim
git checkout new_v3

# Install infinigen in editable mode (minimal, no terrain/OpenGL needed)
INFINIGEN_MINIMAL_INSTALL=True pip install -e .

# Or install just the pipeline dependencies (without infinigen package):
pip install numpy<2 trimesh matplotlib networkx scipy tqdm psutil
```

> **Note**: The `infinigen-sim` conda env is only needed for `scripts/spawn_asset.py` (IS factory spawn). The precompute script (`split_precompute.py`) only needs numpy + trimesh + matplotlib. The render script (`render_animode.py`) runs inside Blender's own Python.

### Step 3: System Dependencies

```bash
# For Blender headless rendering
apt-get install -y libgl1-mesa-glx libegl1-mesa libxrender1 libxi6 \
    libxkbcommon0 libsm6 libxext6 libgomp1

# For video encoding
apt-get install -y ffmpeg
```

### Environment Variables

All hardcoded paths can be overridden via environment variables:

```bash
# Required — set these to match your filesystem layout
export BLENDER_BIN=/path/to/blender-4.2.18-linux-x64/blender
export REPO_DIR=/path/to/Infinigen-Sim
export DATA_DIR=/path/to/data_root     # used by cluster_launch.py

# Optional — defaults are relative to DATA_DIR
# These are currently hardcoded in render_animode.py / split_precompute.py
# and may need sed replacement if your paths differ from the defaults below:
#   ENVMAP_DIR          = ${DATA_DIR}/yurh/dataset3D/envmap/indoor
#   PHYSXNET_JSON_DIR   = ${DATA_DIR}/fulian/dataset/PhysXNet/version_1/finaljson
#   PHYSXMOB_JSON_DIR   = ${DATA_DIR}/fulian/dataset/PhysX_mobility/finaljson
#   SHAPENET_BASE       = ${DATA_DIR}/yurh/dataset3D/ShapeNetCore
#   PARTNET_BASE        = ${DATA_DIR}/yurh/dataset3D/Partnet
#   PBR_TEXTURES_DIR    = ${DATA_DIR}/yurh/infinipart/pbr_textures
#   OVERLAP_MAP_PATH    = ${DATA_DIR}/yurh/infinipart/physxnet_partnet_overlap.json
```

### Required Data (read-only, ~214 GB total)

| Data | Path (default) | Size | Used by |
|------|---------------|------|---------|
| PhysXNet (32K objects) | `${DATA_DIR}/fulian/dataset/PhysXNet/version_1/` | 50 GB | setup, precompute, render |
| PhysXMobility (2K objects) | `${DATA_DIR}/fulian/dataset/PhysX_mobility/` | 3.3 GB | setup, precompute, render |
| ShapeNetCore (textures) | `${DATA_DIR}/yurh/dataset3D/ShapeNetCore/` | 121 GB | render (PhysXNet materials) |
| PartNet (MTL colors) | `${DATA_DIR}/yurh/dataset3D/Partnet/` | 8.0 GB | render (PhysXNet Tier 2 fallback) |
| Envmaps (35 HDR) | `${DATA_DIR}/yurh/dataset3D/envmap/indoor/` | 533 MB | render |
| ambientCG PBR textures | `${DATA_DIR}/yurh/infinipart/pbr_textures/` | 132 MB | render (PhysXNet Tier 3 fallback) |
| Overlap map | `${DATA_DIR}/yurh/infinipart/physxnet_partnet_overlap.json` | 1.4 MB | render |

## Pipeline Scripts

| Step | Script | Needs GPU | Description |
|------|--------|-----------|-------------|
| 0 | `setup_physxnet_scene.py` | No | PhysX only: convert flat URDF+OBJ into pipeline directory format |
| 1 | `scripts/spawn_asset.py` | Yes (Blender) | IS only: procedurally generate articulated objects |
| 2 | `split_precompute.py` | No | Topology split: joint classify → merge fixed → 2-color → export OBJs |
| 3 | `render_animode.py` | Yes (Blender Cycles) | Render articulation videos (32 views × 2 color modes) |
| 4 | `render_batch.py` | Yes | Multi-GPU batch render dispatcher (single machine) |
| 5 | `cluster_launch.py` | Yes | Multi-node cluster dispatcher (SLURM or manual) |

### Quick Start (single machine, IS factory)

```bash
cd /path/to/Infinigen-Sim
conda activate infinigen-sim

# 1. Spawn
CUDA_VISIBLE_DEVICES=0 ${BLENDER_BIN} --background --python-expr "
import sys; sys.path.insert(0, '$(pwd)')
sys.argv = ['spawn_asset', '-n', 'lamp', '-s', '0', '-exp', 'urdf', '-dir', './sim_exports']
exec(open('scripts/spawn_asset.py').read())
"

# 2. Precompute
python split_precompute.py --factory lamp --seed 0 \
    --base ./sim_exports/urdf --output_dir ./precompute_output --force

# 3. Render
CUDA_VISIBLE_DEVICES=0 ${BLENDER_BIN} --background --python render_animode.py -- \
    --metadata ./precompute_output/lamp/0/metadata.json \
    --animode all --views all --color_mode both --bg_mode both
```

### Quick Start (single machine, PhysXNet)

```bash
# 0. Setup scene directory
python setup_physxnet_scene.py --id 10000 --factory PhysXNet --source physxnet

# 2. Precompute
python split_precompute.py --factory PhysXNet --seed 10000 \
    --output_dir ./precompute_output --suffix _PhysXnet --force

# 3. Render (same as above, with metadata path)
CUDA_VISIBLE_DEVICES=0 ${BLENDER_BIN} --background --python render_animode.py -- \
    --metadata ./precompute_output/PhysXNet/10000/metadata.json \
    --animode all --views all --color_mode both --bg_mode both
```

### Multi-Node Cluster (100x 4090)

```bash
# SLURM
srun --nodes=100 --ntasks-per-node=1 --gres=gpu:1 \
    python cluster_launch.py --phase all --is_seeds 100

# Manual (on each node)
NODE_RANK=<0..99> TOTAL_NODES=100 \
    python cluster_launch.py --phase all --is_seeds 100
```

See `cluster_launch.py` for phase-by-phase execution and configuration.

### Docker / Container Image

Recommended base: **Ubuntu 22.04** (minimal). Dockerfile:

```dockerfile
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip python3.11-dev \
    libgl1-mesa-glx libegl1-mesa libxrender1 libxi6 libxkbcommon0 \
    libsm6 libxext6 libgomp1 ffmpeg git wget xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir numpy<2 trimesh matplotlib networkx scipy tqdm psutil

# Blender (download or mount from shared storage)
RUN wget -q https://download.blender.org/release/Blender4.2/blender-4.2.18-linux-x64.tar.xz \
    && tar -xf blender-4.2.18-linux-x64.tar.xz -C /opt/ \
    && rm blender-4.2.18-linux-x64.tar.xz
ENV BLENDER_BIN=/opt/blender-4.2.18-linux-x64/blender

# Blender Python deps
RUN ${BLENDER_BIN} --background --python-expr "\
import subprocess, sys; \
subprocess.check_call([sys.executable, '-m', 'pip', 'install', \
    'gin-config', 'trimesh', 'tqdm', 'psutil', 'opencv-python-headless', \
    'matplotlib', 'scikit-image', 'scipy', 'imageio', 'scikit-learn', \
    'networkx', 'pandas', 'shapely', 'geomdl', 'OpenEXR'])"
```

## Project Structure

```
infinigen/assets/sim_objects/   # 18 IS procedural factories
scripts/spawn_asset.py          # IS factory spawn entry point
setup_physxnet_scene.py         # PhysX data → pipeline directory format
split_precompute.py             # Precompute: normalize → classify → 2-color → export
render_animode.py               # Render: Blender Cycles articulation videos
render_batch.py                 # Multi-GPU batch dispatcher (single machine)
cluster_launch.py               # Multi-node cluster dispatcher
outputs/                        # PhysX factory outputs (URDF + OBJs)
sim_exports/                    # IS factory spawn outputs
precompute_output/              # Pipeline output (part0/1.obj + verify.png + videos)
```

## References

- [Infinigen: Infinite Photorealistic Worlds Using Procedural Generation](https://arxiv.org/abs/2306.09310) (CVPR 2023)
- [Infinigen Indoors](https://arxiv.org/abs/2406.11824) (CVPR 2024)
- [PartPacker](https://arxiv.org/abs/2506.09980) — dual volume / alternating 2-coloring definition
