# InfiniPart

Multi-view articulated object rendering pipeline for generating physics-aware training data. Built on [Infinigen-Sim](https://github.com/princeton-vl/infinigen) (v1.19+) and [Infinite-Mobility](https://github.com/OpenRobotLab/Infinite-Mobility).

Generates multi-view motion videos of articulated 3D objects with:
- **Positive samples**: Correct URDF-driven joint animations + custom animations (lid flip, cap detach)
- **Negative samples**: 7 types of intentionally wrong articulations (wrong joint type, wrong axis, etc.)
- **Dual-part splits**: Normalized body/moving OBJ meshes per animation mode
- **3 data sources**: Infinite-Mobility (16 factories), PhysXNet (13 factories), PhysX_mobility (5 factories)

## Quick Start (Single Node)

```bash
# 1. Clone
git clone https://github.com/AuroraRyan0301/infinipart.git
cd infinipart

# 2. Install Blender 3.6 (headless)
wget https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz
tar xf blender-3.6.0-linux-x64.tar.xz

# 3. Install Python dependencies (conda recommended)
conda create -n infinipart python=3.10 -y
conda activate infinipart
pip install trimesh scipy numpy urdfpy imageio opencv-python-headless

# 4. Pull envmaps and PBR textures (stored in Git LFS)
git lfs pull

# 5. Run full pipeline on 4 GPUs
bash run_all.sh
```

## Cluster Deployment (Multi-Node)

Designed for multi-node GPU clusters (tested on 4x L20X 143GB, targeting 100x 4090).

### run_all.sh Configuration

All parameters are set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WORK_DIR` | `/mnt/data/yurh/Infinigen-Sim` | Project root |
| `PYTHON` | `.../miniconda3/envs/partpacker_wan/bin/python` | Python executable |
| `BLENDER` | `.../blender-3.6.0-linux-x64/blender` | Blender binary |
| `N_GPUS` | `4` | GPUs per node |
| `N_SEEDS` | `50` | Seeds per IM factory |
| `SOURCE` | `all` | `im`, `physxnet`, `physxmob`, or `all` |
| `SHARD_ID` | (empty) | This node's shard index (0-based) |
| `N_SHARDS` | (empty) | Total number of shards/nodes |
| `SKIP_GENERATE` | `0` | Skip asset generation stage |
| `SKIP_POSITIVE` | `0` | Skip positive rendering stage |
| `SKIP_NEGATIVE` | `0` | Skip negative rendering stage |
| `DRY_RUN` | `0` | Print job plan without executing |

### Example: 25-node cluster with 4 GPUs each

```bash
# On each node, set SHARD_ID=0..24
# Node 0:
WORK_DIR=/path/to/infinipart PYTHON=/path/to/python BLENDER=/path/to/blender \
  SHARD_ID=0 N_SHARDS=25 N_GPUS=4 bash run_all.sh

# Node 1:
SHARD_ID=1 N_SHARDS=25 bash run_all.sh

# ... Node 24:
SHARD_ID=24 N_SHARDS=25 bash run_all.sh
```

Each node processes its shard of (factory, seed) pairs across all 3 stages. Logs are written to `pipeline_shard_K_of_N_YYYYMMDD_HHMMSS.log`.

### SLURM Example

```bash
#!/bin/bash
#SBATCH --job-name=infinipart
#SBATCH --array=0-24
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00

export SHARD_ID=$SLURM_ARRAY_TASK_ID
export N_SHARDS=25
export N_GPUS=4
export WORK_DIR=/path/to/infinipart
export PYTHON=/path/to/python
export BLENDER=/path/to/blender

cd $WORK_DIR
bash run_all.sh
```

## Pipeline Overview

### 3-Stage Pipeline (run_pipeline.py)

```
Stage 1: Asset Generation
  IM factories  → Blender generate + split_precompute.py + verify_split.py
  PhysXNet      → physxnet_loader.py (JSON → URDF + OBJ symlinks)
  PhysX_mobility → physxnet_loader.py (rewrite URDF with absolute paths)

Stage 2: Positive Rendering (render_articulation.py)
  For each (factory, seed, animode):
    → Parse URDF joints
    → Select joints by animode rules
    → Animate via Blender keyframes (URDF-driven or custom animation)
    → Render 32 views (16 hemi + 8 orbit + 8 sweep)
    → STATIC detection: delete videos with IoU ≈ 1.0

Stage 3: Negative Rendering (render_negative_samples.py)
  For each (factory, seed, joint):
    → 7 mutation types × all views
    → Export metadata.json with mutation descriptions
```

### Factory Sources

| Source | Factories | Objects | Description |
|--------|-----------|---------|-------------|
| **IM** (Infinite-Mobility) | 16 | ~50 seeds each | BeverageFridge, Bottle, Dishwasher, KitchenCabinet, Lamp, LiteDoor, Microwave, OfficeChair, Oven, Pan, Pot, Tap, Toilet, TV, Window, BarChair |
| **PhysXNet** | 13 | ~32K total | Container, Electronics, Furniture, Kitchen, Lighting, Plumbing, Tool (+ sub-categories) |
| **PhysX_mobility** | 5 | ~2K total | Electronics, Furniture, Kitchen, Plumbing, Tool |

### Animation Modes (Animodes)

Each factory defines numbered animodes that control which joints are animated:

| Animode | Selector | Example |
|---------|----------|---------|
| 0 | Revolute joints only | Door hinge |
| 1 | Prismatic joints only | Drawer slide |
| 2 | Continuous joints only | Cap rotation |
| 3 | All joints | Combined motion |
| Custom | String value | `"flip"`, `"flip_place"`, `"cap_detach"` |

**Per-factory animode table:**

| Factory | 0 | 1 | 2 | 3 | 4 |
|---------|---|---|---|---|---|
| DishwasherFactory | door (revolute) | racks (prismatic) | all | - | - |
| OvenFactory | door (revolute) | racks (prismatic) | all | - | - |
| ToiletFactory | cover (revolute) | seat (revolute) | flush (prismatic) | all | - |
| WindowFactory | pane1 (revolute) | pane2 (revolute) | sliding (prismatic) | all revolute | all |
| PotFactory | lid lift | lid rotate | all URDF | **flip** | **flip+place** |
| BottleFactory | cap lift | cap rotate | all URDF | **cap_detach** | - |
| LampFactory | arm height | bulb slide | arm rotate | all | - |
| TVFactory | tilt (revolute) | height (prismatic) | all | - | - |

Custom animations (`flip`, `flip_place`, `cap_detach`) use procedural keyframes instead of URDF joint limits.

### Negative Sample Types (7)

| Type | Mutation | Effect |
|------|----------|--------|
| `wrong_joint_type` | Swap revolute ↔ prismatic | Sliding instead of rotation or vice versa |
| `wrong_axis` | Cyclic-permute axis (X→Y→Z→X) | Motion along wrong direction |
| `wrong_direction` | Rotate axis by 45° | Tilted/skewed motion |
| `over_motion` | Scale limits by 2.5× | Exceeds physical bounds |
| `wrong_parts_moving` | Invert moving vs static parts | Body moves instead of part |
| `jitter` | Gaussian noise per frame | Jerky/noisy motion |
| `part_detach` | Large prismatic displacement | Part flies off |

### Camera Views (32 per object)

- **16 hemi views** (`hemi_00`–`hemi_15`): 4×4 grid on front hemisphere (azimuth ±67.5°, elevation 5/25/45/65°)
- **8 orbit views** (`orbit_00`–`orbit_07`): Camera travels ~180° back→front
- **8 sweep views** (`sweep_00`–`sweep_07`): Front hemisphere pans/tilts/diagonals

## Dual-Part Splits (split_precompute.py)

Pre-computes normalized body/moving OBJ meshes for each animode:

```
precompute/{Factory}/{identifier}/
    part0.obj                    # default: body (static parts)
    part1.obj                    # default: moving (all movable parts)
    anim0/part0.obj, part1.obj   # revolute joints only
    anim1/part0.obj, part1.obj   # prismatic joints only
    anim10/part0.obj, part1.obj  # joint 0 only
    anim11/part0.obj, part1.obj  # joint 1 only
    metadata.json                # split info + joint definitions
```

Features:
- Normalized to [-0.95, 0.95] bounding box
- Relative motion threshold: joints with < 10% normalized motion are skipped
- Animode deduplication: identical splits are not duplicated
- Semantic part info from PhysXNet/PhysXMob JSON

## Output Structure

### Positive Samples
```
outputs/motion_videos/{Factory}/{seed}/
    hemi_00_nobg.mp4              # animode 0, fixed view
    hemi_00_anim1_nobg.mp4        # animode 1, fixed view
    orbit_00_nobg.mp4             # animode 0, orbit view
    sweep_03_anim2_nobg.mp4       # animode 2, sweep view
```

### Negative Samples
```
outputs/negatives/{Factory}/{seed}/
    {joint_name}/{neg_type}/
        hemi_00_nobg.mp4
        orbit_00_nobg.mp4
        ...
    metadata.json                 # mutation descriptions per sample
```

Video specs: 512×512, 30fps, 4 seconds (120 frames), transparent background.

## Material System

### PBR Textures (pbr_textures/)

11 material categories from [ambientCG](https://ambientcg.com/) (CC0 license):
ceramic, concrete, fabric, leather, marble, metal, paper, plastic, rubber, stone, wood

Used by `pbr_material_system.py` for PhysXNet/PhysXMob objects. Material assignment:
1. ShapeNet texture (if available)
2. PartNet textured_objs (if available)
3. Keyword-based ambientCG PBR mapping (e.g., "handle" → metal, "cushion" → fabric)

### HDR Environment Maps (envmap/indoor/)

35 indoor HDR environment maps (2K EXR) from [Poly Haven](https://polyhaven.com/) (CC0 license):
churches, classrooms, gyms, restaurants, studios, hospitals, offices, etc.

Random envmap selection per render for lighting diversity.

## Key Scripts

| Script | Description |
|--------|-------------|
| **run_pipeline.py** | Unified 3-stage pipeline: generate → positive render → negative render |
| **run_all.sh** | Cluster deployment wrapper with sharding support |
| **render_articulation.py** | Core Blender renderer: URDF parsing, joint animation, 32-view camera, compositor |
| **render_negative_samples.py** | Negative sample renderer: 7 mutation types per joint |
| **split_precompute.py** | Dual-part split: normalization + per-animode body/moving OBJ export |
| **physxnet_loader.py** | PhysXNet/PhysXMob scene preparation (JSON → URDF + OBJ) |
| **physxnet_factory_rules.py** | PhysXNet/PhysXMob factory rules and seed→object_id mapping |
| **pbr_material_system.py** | PBR material assignment for PhysXNet/PhysXMob objects |
| **batch_generate_all.py** | IM-focused batch pipeline: generate + render + split |
| **batch_produce.py** | Production pipeline: auto-discover all 81 factories |
| **convert_partnet.py** | PartNet-Mobility → IM format converter |
| **partnet_factory_rules.py** | 40 PartNet factory rules (14 Sapien + 26 new) |
| **verify_split.py** | Visual verification of precomputed splits |
| **download_pbr_textures.sh** | Download PBR textures from ambientCG |

## Data Dependencies

### Required External Data

| Data | Path (configurable) | Description |
|------|---------------------|-------------|
| Infinite-Mobility outputs | `$BASE/outputs/{Factory}/{seed}/` | Pre-generated IM assets (OBJ + URDF) |
| PhysXNet dataset | See `physxnet_loader.py` | PhysXNet JSON + OBJ files |
| PhysX_mobility dataset | See `physxnet_loader.py` | PhysX_mobility URDF + OBJ |

### Included in Repo (via Git LFS)

| Data | Path | Size |
|------|------|------|
| HDR envmaps | `envmap/indoor/` | 533 MB (35 EXR files) |
| PBR textures | `pbr_textures/` | 132 MB (11 categories) |

## Test Results

All positive sample tests pass (90/90 jobs, 0 failures):

### IM Factories (51 jobs)
```
16 factories × all animodes = 51 jobs
Result: 51 OK, 0 FAIL
```

### PhysXNet + PhysX_mobility (39 jobs)
```
10 factories × all animodes = 39 jobs
Result: 39 OK, 0 FAIL
```

Run tests:
```bash
bash test_all_positive.sh          # IM factories
bash test_physxnet_positive.sh     # PhysXNet + PhysXMob
```

## Prerequisites

- **Python 3.10+** with trimesh, scipy, numpy, urdfpy, imageio, opencv
- **Blender 3.6** (headless, for rendering scripts)
- **NVIDIA GPU** with CUDA (tested: L20X 143GB, targeting 4090 24GB)
- **ffmpeg** for video encoding
- **Git LFS** for envmaps and PBR textures

## Credits

- [Infinigen](https://github.com/princeton-vl/infinigen) by Princeton Vision & Learning Lab
- [Infinite-Mobility](https://github.com/OpenRobotLab/Infinite-Mobility) by OpenRobotLab
- HDR envmaps from [Poly Haven](https://polyhaven.com/) (CC0)
- PBR textures from [ambientCG](https://ambientcg.com/) (CC0)
