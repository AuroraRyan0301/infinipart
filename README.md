# InfiniPart

Multi-view articulated object rendering and simulation pipeline built on [Infinigen](https://github.com/princeton-vl/infinigen) (v1.19+, Infinigen-Sim) and [Infinite-Mobility](https://github.com/OpenRobotLab/Infinite-Mobility). Generates procedural articulated 3D objects with **built-in joint definitions**, exports to URDF/MJCF/USD, and renders multi-view motion videos.

## What's New (vs Infinite-Mobility)

This repo upgrades from Infinite-Mobility (Blender 3.6, static meshes + external URDF) to **Infinigen-Sim** (Blender 4.2, native articulation):

- **17 new sim-ready articulated factories** with joints embedded in Blender Geometry Nodes
- **Kinematic compiler** that auto-extracts joint trees from geometry node graphs
- **3 simulation export formats**: URDF (PyBullet/Isaac Gym), MJCF (MuJoCo), USD (Isaac Sim)
- **Physics material system** with density, friction, and joint dynamics
- **Blender 4.2** with `bpy==4.2.0` Python bindings

Plus the proven InfiniPart rendering pipeline from Infinite-Mobility.

## Sim-Ready Object Factories (17 categories)

Located in `infinigen/assets/sim_objects/`:

| Factory | Joint Types | Description |
|---------|------------|-------------|
| BoxFactory | Hinge | Articulated box with hinged lid |
| CabinetFactory | Hinge | Cabinet with hinged doors |
| DishwasherFactory | Hinge + Sliding | Door hinge + rack slider |
| SimDoorFactory | Hinge | Door with handle articulation |
| DoorHandleFactory | Hinge + Sliding | Lever/bar handle |
| DrawerFactory | Sliding | Pullout drawer |
| FaucetFactory | Hinge | Tap handle rotation |
| LampFactory | Hinge + Sliding | Articulated arm joints |
| MicrowaveFactory | Hinge + Sliding | Door hinge + turntable |
| OvenFactory | Hinge + Sliding | Door hinge + rack slider |
| PepperGrinderFactory | Hinge | Twist grinding mechanism |
| PlierFactory | Hinge | Pivot joint for handles |
| RefrigeratorFactory | Hinge + Sliding | Door(s) + drawer shelves |
| SoapDispenserFactory | Hinge + Sliding | Pump mechanism |
| StovetopFactory | Hinge | Knob rotation + grate lift |
| ToasterFactory | Hinge + Sliding | Lever + lid |
| TrashFactory | Hinge + Sliding | Lid hinge + pedal |
| WindowFactory | Hinge + Sliding | Sash sliding + casement hinge |

## Rendering Pipeline (from Infinite-Mobility)

Also includes the InfiniPart multi-view rendering pipeline with **16 original object categories**:

- BeverageFridge, Microwave, Oven, Toilet, KitchenCabinet, Window, LiteDoor, OfficeChair, Tap, Lamp, Pot, Bottle, Dishwasher, BarChair, Pan, TV
- **Per-category animation modes (animodes)**: Independently control joint subsets
- **32 camera views per animation**: 16 fixed hemisphere views + 8 back-to-front orbits + 8 front-hemisphere sweeps
- **Moving camera support**: Animated cameras that orbit or sweep during rendering
- **Batch pipeline**: Multi-seed x multi-animode x multi-view x multi-GPU parallel rendering

## Prerequisites

- **Python 3.11** (for Infinigen-Sim conda env)
- **Blender 3.6** (headless, for InfiniPart rendering scripts)
- **NVIDIA GPU** with CUDA support (tested on L20X 143GB)
- **ffmpeg** for video encoding

## Setup

### Infinigen-Sim (sim-ready articulated assets)

```bash
# 1. Clone
git clone https://github.com/AuroraRyan0301/infinipart.git
cd infinipart

# 2. Create conda env
conda create -n infinigen-sim python=3.11 -y
conda activate infinigen-sim

# 3. Install with sim extras
INFINIGEN_MINIMAL_INSTALL=True pip install -e ".[sim]"

# 4. (Optional) Install additional deps for rendering scripts
pip install urdfpy open3d
```

### InfiniPart Rendering (Blender-based)

```bash
# 1. Download Blender 3.6
wget https://download.blender.org/release/Blender3.6/blender-3.6.0-linux-x64.tar.xz
tar xf blender-3.6.0-linux-x64.tar.xz

# 2. Update paths in scripts:
#    - BLENDER path in batch_generate_all.py and render_articulation.py
#    - BASE_DIR to your repo root
#    - Envmap HDR path (--envmap flag)
```

## Quick Start

### Generate sim-ready assets (Infinigen-Sim)

```bash
conda activate infinigen-sim

# Generate a single sim-ready refrigerator and export to URDF
python -m infinigen.core.sim.sim_factory \
  --factory RefrigeratorFactory --seed 0 \
  --output_dir outputs/sim/RefrigeratorFactory/0 \
  --export_format urdf
```

### Generate assets for rendering (Blender)

```bash
CUDA_VISIBLE_DEVICES=0 blender --background --python-use-system-env \
  --python infinigen_examples/generate_individual_assets.py -- \
  --output_folder outputs/OvenFactory -f OvenFactory -n 1 --seed 0
```

### Render single factory/seed/animode

```bash
CUDA_VISIBLE_DEVICES=0 blender --background --python-use-system-env \
  --python render_articulation.py -- \
  --factory OvenFactory --seed 0 --device 0 \
  --output_dir outputs/motion_videos/OvenFactory/0 \
  --resolution 512 --samples 32 --duration 4.0 --fps 30 \
  --animode 0 --skip_bg \
  --views hemi_00 hemi_01 hemi_02 hemi_03 \
  --moving_views orbit_00 sweep_00
```

### Batch pipeline (generate + render all)

```bash
# Generate 10 seeds + render all animodes x 32 views on 4 GPUs
python batch_generate_all.py --n_seeds 10 --n_gpus 4 --no_split

# Render only (assets already generated)
python batch_generate_all.py --n_seeds 10 --n_gpus 4 --render_only --no_split

# Single factory
python batch_generate_all.py --n_seeds 10 --n_gpus 4 --factory OvenFactory --render_only --no_split
```

## Architecture

### Infinigen-Sim Joint System

Joints are defined **inside Blender Geometry Nodes** using custom node groups:

- `nodegroup_hinge_joint` -- Revolute joints (doors, lids, knobs)
- `nodegroup_sliding_joint` -- Prismatic joints (drawers, sliders)
- `kinematic_compiler.py` -- Auto-extracts kinematic DAG from node trees
- Exporters convert to URDF/MJCF/USD with mass, inertia, collision geometry

### Animation Modes (animodes)

Each factory defines animation modes that select specific joint subsets:

| Factory | Animode 0 | Animode 1 | Animode 2 | Animode 3 | Animode 4 |
|---------|-----------|-----------|-----------|-----------|-----------|
| Oven | door (revolute) | racks (prismatic) | all | - | - |
| Toilet | cover (revolute) | seat ring (revolute) | flush (prismatic) | all | - |
| Window | pane 1 (revolute) | pane 2 (revolute) | sliding (prismatic) | all revolute | all |
| Pot | lid lift (prismatic) | lid rotate (continuous) | URDF all | flip in-place | flip+place beside |
| Lamp | arm height (prismatic) | bulb slide (prismatic) | arm rotate (revolute) | all | - |
| BarChair | height (prismatic) | spin (continuous) | all | - | - |
| TV | tilt (revolute) | height (prismatic) | all | - | - |
| Pan | lid lift (prismatic) | - | - | - | - |

Joint selectors support multiple formats:
- `("type",)` -- all significant joints of that type
- `("type", ordinal)` -- nth joint by kinematic depth (0=shallowest, -1=deepest)
- `("type", "axis", "x"|"y"|"z")` -- joints with given primary axis
- `("type", "sign", "+"|"-")` -- joints filtered by limit sign

### Camera Views (32 total)

**16 fixed hemisphere views** (`hemi_00` to `hemi_15`):
- 4x4 grid on front hemisphere (azimuth +/-67.5 deg, elevation 5/25/45/65 deg)

**8 orbit views** (`orbit_00` to `orbit_07`):
- Camera travels ~180 deg from back to front of object

**8 sweep views** (`sweep_00` to `sweep_07`):
- Camera moves within front hemisphere (horizontal pans, vertical tilts, diagonal paths)

### Output Structure

```
outputs/motion_videos/{Factory}/{seed}/
  hemi_00_nobg.mp4          # animode 0, fixed view 0
  hemi_00_anim1_nobg.mp4    # animode 1, fixed view 0
  orbit_00_nobg.mp4         # animode 0, orbit view 0
  sweep_03_anim2_nobg.mp4   # animode 2, sweep view 3
  ...
```

Each video: 512x512, 30fps, 4 seconds (120 frames), transparent background (nobg).

## Key Scripts

| Script | Description |
|--------|-------------|
| `render_articulation.py` | Core Blender rendering: URDF parsing, joint animation, multi-view camera, compositor |
| `render_articulation_video.py` | Video-specific rendering pipeline |
| `batch_generate_all.py` | Batch pipeline: generate assets + render across multiple GPUs |
| `batch_render_pipeline.py` | Lower-level batch render orchestration |
| `split_and_visualize.py` | 2-part decomposition for part-aware training data |
| `convert_partnet.py` | Convert PartNet-Mobility dataset to Infinite-Mobility format |
| `partnet_factory_rules.py` | Factory rules for 40 categories (14 Sapien + 26 PartNet) |
| `show.py` | Interactive visualization tool |
| `paralled_generate.py` | Parallel generation entry point |

## Credits

- [Infinigen](https://github.com/princeton-vl/infinigen) by Princeton Vision & Learning Lab (Infinigen-Sim articulation system)
- [Infinite-Mobility](https://github.com/OpenRobotLab/Infinite-Mobility) by OpenRobotLab (procedural articulated object generation)
