"""
Blender script: Render multi-view articulation motion videos for Infinite Mobility objects.

Parses URDF joint definitions, loads per-part OBJs with textures, animates joints
through their range of motion by directly computing per-frame transforms, and
renders from multiple camera angles.

Approach:
  - Load each per-part OBJ (centered at centroid). Apply origin offset so mesh is
    in world coordinates, then apply transform (object at 0,0,0, verts in world).
  - Parse URDF kinematic chain to find all movable joints.
  - For each frame, walk the kinematic tree and compute accumulated transforms for
    each part. Set each part's matrix_world directly.

Run with:
  CUDA_VISIBLE_DEVICES=0 /mnt/data/yurh/blender-3.6.0-linux-x64/blender \
    --background --python render_articulation.py -- \
    --factory OfficeChairFactory --seed 0 --device 0

For BottleFactory with separation animode:
  CUDA_VISIBLE_DEVICES=0 /mnt/data/yurh/blender-3.6.0-linux-x64/blender \
    --background --python render_articulation.py -- \
    --factory BottleFactory --seed 0 --device 0 --animode 1
"""

import bpy
import bmesh
import sys
import os
import json
import math
import subprocess
import shutil
import xml.etree.ElementTree as ET
from mathutils import Vector, Matrix, Euler, Quaternion
from collections import defaultdict, deque

# ── Blender version detection for API compatibility ──
BLENDER_VERSION = bpy.app.version  # e.g. (3, 6, 0) or (4, 2, 0)
BLENDER_MAJOR = BLENDER_VERSION[0]
BLENDER_MINOR = BLENDER_VERSION[1]
IS_BLENDER_4X = BLENDER_MAJOR >= 4
print(f"Blender version: {BLENDER_VERSION[0]}.{BLENDER_VERSION[1]}.{BLENDER_VERSION[2]}")

# ── Parse args after "--" ──
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--factory", required=True, help="Factory name, e.g. OfficeChairFactory")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--base", default="/mnt/data/yurh/Infinigen-Sim")
parser.add_argument("--envmap", default="/mnt/data/yurh/dataset3D/envmap/indoor/brown_photostudio_06_2k.exr")
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--duration", type=float, default=4.0, help="Video duration in seconds")
parser.add_argument("--samples", type=int, default=32, help="Cycles samples (denoiser is on, 32 is sufficient)")
parser.add_argument("--device", type=int, default=0, help="CUDA device index (after CUDA_VISIBLE_DEVICES remap)")
parser.add_argument("--animode", type=int, default=0,
                    help="Animation mode: 0=URDF limits, N>0=limits scaled by (1+N*0.5)x")
parser.add_argument("--views", nargs="+", default=["front", "side", "back", "threequarter"],
                    help="Which views to render")
parser.add_argument("--output_dir", default=None, help="Override output directory")
parser.add_argument("--png_only", action="store_true", help="Output PNG sequences only (skip ffmpeg)")
parser.add_argument("--skip_nobg", action="store_true", help="Skip transparent background renders")
parser.add_argument("--skip_bg", action="store_true", help="Skip opaque background renders (only render nobg)")
parser.add_argument("--joint_filter", default=None, help="Only animate joints matching this substring")
parser.add_argument("--moving_views", nargs="+", default=[],
                    help="Moving camera views to render (orbit_XX, sweep_XX)")
parser.add_argument("--max_bounces", type=int, default=None,
                    help="Override Cycles max bounces (default: Blender default 12)")
parser.add_argument("--engine", choices=["cycles", "eevee"], default="cycles",
                    help="Render engine: cycles (default) or eevee")
args = parser.parse_args(argv)

# ── Paths ──
FACTORY = args.factory
SEED = args.seed
BASE = args.base
SCENE_DIR = os.path.join(BASE, "outputs", FACTORY, str(SEED))
URDF_PATH = os.path.join(SCENE_DIR, "scene.urdf")
ORIGINS_PATH = os.path.join(SCENE_DIR, "origins.json")
OBJS_DIR = os.path.join(SCENE_DIR, "outputs", FACTORY, str(SEED), "objs")

if args.output_dir:
    OUT_DIR = args.output_dir
else:
    OUT_DIR = os.path.join(BASE, "outputs", "motion_test", FACTORY)
os.makedirs(OUT_DIR, exist_ok=True)

NUM_FRAMES = int(args.fps * args.duration)

VIEW_CONFIGS = {
    "front": (25, 0),
    "side": (25, 90),
    "back": (25, 180),
    "threequarter": (25, 45),
}

# 16 fixed views distributed on front hemisphere (azimuth -90° to +90°)
# 4 elevation rings × 4 azimuth columns
_HEMI_ELEVS = [5, 25, 45, 65]
_HEMI_AZIMS = [-67.5, -22.5, 22.5, 67.5]
for _i, _elev in enumerate(_HEMI_ELEVS):
    for _j, _azim in enumerate(_HEMI_AZIMS):
        VIEW_CONFIGS[f"hemi_{_i*4+_j:02d}"] = (_elev, _azim)

# Moving camera views: (start_elev, start_azim, end_elev, end_azim)
# Linear interpolation in spherical coords over the animation duration.
MOVING_VIEW_CONFIGS = {
    # 8 back-to-front orbits (camera travels ~180° around the object)
    "orbit_00": (10, 180, 10, 0),      # low, via +Y side
    "orbit_01": (10, -180, 10, 0),     # low, via -Y side
    "orbit_02": (30, 180, 30, 0),      # mid, via +Y
    "orbit_03": (30, -180, 30, 0),     # mid, via -Y
    "orbit_04": (50, 150, 15, 0),      # descending via +Y
    "orbit_05": (50, -150, 15, 0),     # descending via -Y
    "orbit_06": (15, 170, 50, 0),      # ascending via +Y
    "orbit_07": (15, -170, 50, 0),     # ascending via -Y
    # 8 front hemisphere sweeps (camera stays in front hemisphere)
    "sweep_00": (15, -80, 15, 80),     # horizontal pan, low
    "sweep_01": (35, -75, 35, 75),     # horizontal pan, mid
    "sweep_02": (55, -60, 55, 60),     # horizontal pan, high
    "sweep_03": (25, 80, 25, -80),     # reverse horizontal pan
    "sweep_04": (5, 0, 70, 0),         # vertical tilt, center
    "sweep_05": (5, -45, 65, -45),     # vertical tilt, left
    "sweep_06": (5, 45, 65, 45),       # vertical tilt, right
    "sweep_07": (10, -60, 55, 60),     # diagonal sweep
}

# Find ffmpeg binary
FFMPEG_BIN = None
try:
    result = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
    if result.returncode == 0:
        FFMPEG_BIN = result.stdout.strip()
except:
    pass
if not FFMPEG_BIN:
    try:
        import imageio_ffmpeg
        FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
    except:
        pass
if not FFMPEG_BIN:
    # Fallback: search common conda/pip locations
    import glob
    candidates = glob.glob("/mnt/data/yurh/miniconda3/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*")
    candidates += glob.glob("/mnt/data/yurh/miniconda3/envs/*/bin/ffmpeg")
    candidates += glob.glob("/mnt/data/yurh/miniconda3/bin/ffmpeg")
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            FFMPEG_BIN = c
            break


# ═══════════════════════════════════════════════════════════════
# URDF Parsing
# ═══════════════════════════════════════════════════════════════

class URDFJoint:
    def __init__(self, name, jtype, parent_link, child_link, axis, origin_xyz, origin_rpy, lower, upper):
        self.name = name
        self.jtype = jtype
        self.parent_link = parent_link
        self.child_link = child_link
        self.axis = axis
        self.origin_xyz = origin_xyz
        self.origin_rpy = origin_rpy
        self.lower = lower
        self.upper = upper

    def __repr__(self):
        return (f"URDFJoint({self.name}, {self.jtype}, "
                f"{self.parent_link}->{self.child_link}, "
                f"axis={[round(a,4) for a in self.axis]}, "
                f"limits=[{self.lower:.4f},{self.upper:.4f}])")


def parse_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for link_el in root.findall("link"):
        name = link_el.get("name")
        visual = link_el.find("visual")
        mesh_file = None
        if visual is not None:
            geom = visual.find("geometry")
            if geom is not None:
                mesh_el = geom.find("mesh")
                if mesh_el is not None:
                    mesh_file = mesh_el.get("filename")
        part_idx = None
        if name.startswith("l_") and name[2:].isdigit():
            part_idx = int(name[2:])
        links[name] = {"name": name, "mesh_file": mesh_file, "part_idx": part_idx}

    joints = []
    for joint_el in root.findall("joint"):
        name = joint_el.get("name")
        jtype = joint_el.get("type")
        parent_el = joint_el.find("parent")
        child_el = joint_el.find("child")
        parent_link = parent_el.get("link") if parent_el is not None else None
        child_link = child_el.get("link") if child_el is not None else None

        axis_el = joint_el.find("axis")
        axis = [1.0, 0.0, 0.0]
        if axis_el is not None:
            axis = [float(x) for x in axis_el.get("xyz").split()]

        origin_el = joint_el.find("origin")
        origin_xyz = [0.0, 0.0, 0.0]
        origin_rpy = [0.0, 0.0, 0.0]
        if origin_el is not None:
            origin_xyz = [float(x) for x in origin_el.get("xyz", "0 0 0").split()]
            origin_rpy = [float(x) for x in origin_el.get("rpy", "0 0 0").split()]

        limit_el = joint_el.find("limit")
        lower, upper = 0.0, 0.0
        if limit_el is not None:
            lower = float(limit_el.get("lower", "0"))
            upper = float(limit_el.get("upper", "0"))

        joints.append(URDFJoint(name, jtype, parent_link, child_link,
                                axis, origin_xyz, origin_rpy, lower, upper))

    return links, joints


def build_kinematic_tree(joints):
    """Build parent/children maps from URDF joints."""
    parent_map = {}  # child_link -> (parent_link, joint)
    children_map = defaultdict(list)  # parent_link -> [(child_link, joint), ...]
    for j in joints:
        parent_map[j.child_link] = (j.parent_link, j)
        children_map[j.parent_link].append((j.child_link, j))
    return parent_map, children_map


# ═══════════════════════════════════════════════════════════════
# Factory-specific rules: which parts are "moving" (animated)
# ═══════════════════════════════════════════════════════════════

FACTORY_RULES = {
    # ── Door-type: revolute=door, prismatic=racks/drawers ──
    "DishwasherFactory": {
        "moving_parts": {"door_part", "handle_part"},
        "animode_joints": {
            0: [("revolute",)],                        # base: door open
            1: [("prismatic",)],                       # base: racks slide
            2: [("revolute",), ("prismatic",)],        # senior: all
        },
    },
    "BeverageFridgeFactory": {
        "moving_parts": {"door_part"},
        "animode_joints": {
            0: [("revolute",)],                        # base: door open
            1: [("prismatic",)],                       # base: racks slide
            2: [("revolute",), ("prismatic",)],        # senior: all
        },
    },
    "MicrowaveFactory": {
        "moving_parts": {"door_part"},
        "animode_joints": {
            0: [("revolute",)],                        # base: door open
            1: [("continuous",)],                      # base: plate/turntable
            2: [("revolute",), ("continuous",)],       # senior: all
        },
    },
    "OvenFactory": {
        "moving_parts": {"door_part"},
        "animode_joints": {
            0: [("revolute",)],                        # base: door + knobs
            1: [("prismatic",)],                       # base: racks slide
            2: [("revolute",), ("prismatic",)],        # senior: all
        },
    },
    "KitchenCabinetFactory": {
        "moving_parts": {"door_part", "drawer_part"},
        "animode_joints": {
            0: [("revolute",)],                        # base: doors
            1: [("prismatic",)],                       # base: drawers
            2: [("revolute",), ("prismatic",)],        # senior: all
        },
    },
    # ── Toilet/Window/Door ──
    "ToiletFactory": {
        "moving_parts": {"toilet_cover_part", "toilet_seat_part"},
        "animode_joints": {
            0: [("revolute", -1)],                     # base: cover/lid only (last revolute)
            1: [("revolute", 0)],                      # base: seat ring only (first revolute)
            2: [("prismatic",)],                       # base: flush lever
            3: [("revolute",), ("prismatic",)],        # senior: all
        },
    },
    "WindowFactory": {
        "moving_parts": {"panel_part", "shutter_part", "curtain_part",
                          "curtain_hold_part", "sash_part", "pane_part"},
        "animode_joints": {
            0: [("revolute", 0)],                      # base: single pane (first)
            1: [("revolute", -1)],                     # base: single pane (last)
            2: [("prismatic",)],                       # base: sliding (if exists)
            3: [("revolute",)],                        # senior: all panes rotate
            4: [("revolute",), ("prismatic",)],        # senior: all joints
        },
    },
    # ── OfficeChair: height + rotation ──
    "OfficeChairFactory": {
        "moving_parts": {"seat_part", "chair_back_part", "back_part",
                         "leg_wheeled_upper_part", "chair_arm_part"},
        "exclude_parts": {"unknown_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # base: height
            1: [("revolute",)],                        # base: back tilt (some seeds)
            2: [("prismatic",), ("revolute",)],        # senior: all
        },
    },
    # ── Tap: handles + spout ──
    "TapFactory": {
        "moving_parts": {"handle_2_part", "handle_3_part", "handle_part",
                          "spout_part", "tap_7_part"},
        "animode_joints": {
            0: [("revolute",)],                        # base: handles
            1: [("continuous",)],                      # base: spout rotation
            2: [("revolute",), ("continuous",)],       # senior: all
        },
    },
    # ── Lamp: per-joint (distinct functional roles) ──
    "LampFactory": {
        "moving_parts": {"lamp_head_part", "bulb_part", "shade_part",
                         "bulb_rack_1_part", "bulb_rack_2_part", "bulb_rack_3_part",
                         "bulb_rack_4_part", "bulb_rack_5_part",
                         "lamp_leg_upper_part", "lamp_support_curve_part",
                         "lamp_connector_part", "lamp_leg_seg_part"},
        "animode_joints": {
            0: [("prismatic", 0)],                     # base: arm height
            1: [("prismatic", -1)],                    # base: bulb slide
            2: [("revolute", 0)],                      # base: arm rotation
            3: [("prismatic",), ("revolute",)],        # senior: all
        },
    },
    # ── Small objects: prismatic=lift, continuous=rotation ──
    "PotFactory": {
        "moving_parts": {"lid_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # base: lid lift
            1: [("continuous",)],                      # base: lid rotation
            2: [("prismatic",), ("continuous",)],      # senior: URDF joints
            3: "flip",                                 # senior: lid flips 180° in place (round-trip)
            4: "flip_place",                             # senior: lid flips + placed beside pot
        },
    },
    "BottleFactory": {
        "moving_parts": {"cap_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # base: cap lift
            1: [("continuous",)],                      # base: cap rotation
            2: [("prismatic",), ("continuous",)],      # senior: all
            3: "cap_detach",                           # senior: cap unscrews + flies off
        },
    },
    # ── Bar Chair: height + seat spin (like OfficeChair) ──
    "BarChairFactory": {
        "moving_parts": {"bar_seat_1_part", "bar_seat_2_part", "bar_seat_3_part",
                         "leg_wheeled_upper_part"},
        "exclude_parts": {"unknown_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # base: height adjust
            1: [("continuous",)],                      # base: seat spin + wheel rotation
            2: [("prismatic",), ("continuous",)],      # senior: all
        },
    },
    # ── Pan: simple lid lift ──
    "PanFactory": {
        "moving_parts": {"lid_part"},
        "exclude_parts": {"unknown_part"},
        "animode_joints": {
            0: [("prismatic",)],                       # base: lid lift
        },
    },
    # ── TV: screen tilt + height adjust ──
    "TVFactory": {
        "moving_parts": {"connector_part", "screen_part", "button_part"},
        "exclude_parts": {"unknown_part"},
        "animode_joints": {
            0: [("revolute",)],                        # base: screen tilt
            1: [("prismatic",)],                       # base: height adjust
            2: [("revolute",), ("prismatic",)],        # senior: all
        },
    },

    # ═══════════════════════════════════════════════════════════════
    # Infinigen-Sim sim_objects (17 new factories)
    # These use nodegroup_hinge_joint (=revolute) and
    # nodegroup_sliding_joint (=prismatic) in their geometry nodes.
    # Part labels come from nodegroup_add_jointed_geometry_metadata.
    # ═══════════════════════════════════════════════════════════════

    # ── SimDoorFactory: door swings open (hinge) ──
    "SimDoorFactory": {
        "moving_parts": set(),  # all parts animated via URDF
        "animode_joints": {
            0: [("revolute",)],                        # base: door swings
        },
    },
    # ── DoorHandleFactory: handle turns (hinge) + lock slides ──
    "DoorHandleFactory": {
        "moving_parts": set(),
        "animode_joints": {
            0: [("revolute",)],                        # base: lever turn
            1: [("prismatic",)],                       # base: lock slide
            2: [("revolute",), ("prismatic",)],        # senior: all
        },
    },
    # ── DrawerFactory: drawer slides out (prismatic) ──
    "DrawerFactory": {
        "moving_parts": set(),
        "animode_joints": {
            0: [("prismatic",)],                       # base: drawer slide
        },
    },
    # ── BoxFactory: lids/flaps hinge open ──
    "BoxFactory": {
        "moving_parts": set(),
        "animode_joints": {
            0: [("revolute", 0)],                      # base: first lid/flap
            1: [("revolute", -1)],                     # base: last lid/flap
            2: [("revolute",)],                        # senior: all flaps
        },
    },
    # ── CabinetFactory: doors (hinge) + drawers (slide) ──
    "CabinetFactory": {
        "moving_parts": set(),
        "animode_joints": {
            0: [("revolute",)],                        # base: doors
            1: [("prismatic",)],                       # base: drawers
            2: [("revolute",), ("prismatic",)],        # senior: all
        },
    },
    # ── RefrigeratorFactory: doors (hinge) + drawers (slide) ──
    "RefrigeratorFactory": {
        "moving_parts": set(),
        "animode_joints": {
            0: [("revolute", 0)],                      # base: main door
            1: [("revolute", -1)],                     # base: freezer door
            2: [("prismatic",)],                       # base: drawers slide
            3: [("revolute",)],                        # senior: all doors
            4: [("revolute",), ("prismatic",)],        # senior: all joints
        },
    },
    # ── FaucetFactory: handles turn (hinge), spout rotates ──
    "FaucetFactory": {
        "moving_parts": set(),
        "animode_joints": {
            0: [("revolute", 0)],                      # base: first handle
            1: [("revolute", -1)],                     # base: spout or last handle
            2: [("revolute",)],                        # senior: all hinges
        },
    },
    # ── StovetopFactory: knobs turn (hinge) ──
    "StovetopFactory": {
        "moving_parts": set(),
        "animode_joints": {
            0: [("revolute",)],                        # base: knobs turn
        },
    },
    # ── ToasterFactory: slider (prismatic), knobs (hinge), button (prismatic) ──
    "ToasterFactory": {
        "moving_parts": set(),
        "animode_joints": {
            0: [("prismatic",)],                       # base: slider/button push
            1: [("revolute",)],                        # base: knobs turn
            2: [("revolute",), ("prismatic",)],        # senior: all
        },
    },
    # ── PepperGrinderFactory: lid rotates (hinge) ──
    "PepperGrinderFactory": {
        "moving_parts": set(),
        "animode_joints": {
            0: [("revolute",)],                        # base: lid twist
        },
    },
    # ── PlierFactory: arms rotate around pivot (hinge) ──
    "PlierFactory": {
        "moving_parts": set(),
        "animode_joints": {
            0: [("revolute",)],                        # base: arms open/close
        },
    },
    # ── SoapDispenserFactory: nozzle push (slide) + cap rotate (hinge) ──
    "SoapDispenserFactory": {
        "moving_parts": set(),
        "animode_joints": {
            0: [("prismatic",)],                       # base: pump press
            1: [("revolute",)],                        # base: cap rotate
            2: [("revolute",), ("prismatic",)],        # senior: all
        },
    },
    # ── TrashFactory: lid (hinge), pedal (hinge), flaps (hinge), drawer (slide) ──
    "TrashFactory": {
        "moving_parts": set(),
        "animode_joints": {
            0: [("revolute", 0)],                      # base: lid/flap open
            1: [("revolute", -1)],                     # base: pedal press
            2: [("prismatic",)],                       # base: drawer slide
            3: [("revolute",)],                        # senior: all hinges
            4: [("revolute",), ("prismatic",)],        # senior: all joints
        },
    },
    # Note: LampFactory, DishwasherFactory, OvenFactory, MicrowaveFactory, WindowFactory
    # already have entries above (from IM). Those rules work for Infinigen-Sim too since
    # they select joints by TYPE (revolute/prismatic), not by part name.
    # The joint types are the same: hinge_joint->revolute, sliding_joint->prismatic.
}

# Merge PartNet-Mobility factory rules
# Add script directory to path so partnet_factory_rules can be found
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
try:
    from partnet_factory_rules import merge_render_rules, PARTNET_ANIMODES
    FACTORY_RULES = merge_render_rules(FACTORY_RULES)
except ImportError:
    pass

# Merge PhysXNet + PhysX_mobility factory rules
try:
    from physxnet_factory_rules import (
        merge_render_rules as physxnet_merge_render_rules,
        ALL_ANIMODES as PHYSXNET_ALL_ANIMODES,
        ALL_MATERIAL_DEFAULTS as PHYSXNET_ALL_MATERIAL_DEFAULTS,
    )
    FACTORY_RULES = physxnet_merge_render_rules(FACTORY_RULES)
except ImportError:
    PHYSXNET_ALL_ANIMODES = {}
    PHYSXNET_ALL_MATERIAL_DEFAULTS = {}

# ── PhysXNet / PhysX_mobility generic fallback rules ──
_PHYSXNET_GENERIC_RULE = {
    "moving_parts": set(),  # no name filtering - animate all
    "animode_joints": {
        0: [("revolute",)],
        1: [("prismatic",)],
        2: [("continuous",)],
        3: [("revolute",), ("prismatic",), ("continuous",)],
    },
}
# Auto-register any *PhysXNetFactory or *PhysXMobilityFactory not already in rules
import re as _re
class _PhysXNetRuleProxy(dict):
    """Fallback: if factory name matches PhysXNet/PhysXMobility pattern, use generic rule."""
    def __init__(self, base):
        super().__init__(base)
    def __contains__(self, key):
        if super().__contains__(key):
            return True
        return bool(_re.search(r'Phys[XxNn].*Factory$', str(key)))
    def get(self, key, default=None):
        if super().__contains__(key):
            return super().get(key, default)
        if _re.search(r'Phys[XxNn].*Factory$', str(key)):
            return _PHYSXNET_GENERIC_RULE
        return default
    def __getitem__(self, key):
        if super().__contains__(key):
            return super().__getitem__(key)
        if _re.search(r'Phys[XxNn].*Factory$', str(key)):
            return _PHYSXNET_GENERIC_RULE
        raise KeyError(key)

FACTORY_RULES = _PhysXNetRuleProxy(FACTORY_RULES)


# ═══════════════════════════════════════════════════════════════
# PartNet Material Enhancement
# ═══════════════════════════════════════════════════════════════

# Per-category default PBR values: (metallic, roughness)
# Parts matching specific keywords get overrides.
PARTNET_MATERIAL_DEFAULTS = {
    # ── metallic objects ──
    # Scissors: both parts are "leg_part" — the whole thing is metal
    "ScissorsFactory":              {"default": (0.90, 0.25), "leg": (0.95, 0.22)},
    "PliersFactory":                {"default": (0.85, 0.30), "rotation_blade": (0.90, 0.20), "leg": (0.85, 0.28)},
    "StaplerFactory":               {"default": (0.80, 0.30), "lid": (0.82, 0.28), "stapler_body": (0.78, 0.32)},
    "EyeglassesFactory":            {"default": (0.70, 0.25), "leg": (0.75, 0.22)},
    "FaucetSapienFactory":          {"default": (0.90, 0.20), "spout": (0.92, 0.18), "stem": (0.88, 0.22), "switch": (0.85, 0.25)},
    "TapFactory":                   {"default": (0.90, 0.20)},
    # ── appliances - metal exterior ──
    "DishwasherSapienFactory":      {"default": (0.70, 0.30), "rotation_door": (0.75, 0.25), "shelf": (0.72, 0.28), "button": (0.50, 0.40), "knob": (0.80, 0.25)},
    "MicrowaveSapienFactory":       {"default": (0.65, 0.30), "door": (0.70, 0.25), "button": (0.50, 0.40), "knob": (0.60, 0.35), "rotation_tray": (0.40, 0.35)},
    "OvenSapienFactory":            {"default": (0.60, 0.35), "door": (0.65, 0.30), "knob": (0.80, 0.25), "button": (0.50, 0.40), "tray": (0.55, 0.30)},
    "RefrigeratorSapienFactory":    {"default": (0.70, 0.25), "door": (0.75, 0.22)},
    "WashingMachineFactory":        {"default": (0.65, 0.30), "door": (0.50, 0.35), "button": (0.50, 0.40), "knob": (0.60, 0.35)},
    "SafeFactory":                  {"default": (0.80, 0.30), "door": (0.82, 0.28), "handle": (0.85, 0.22), "knob": (0.85, 0.22), "button": (0.50, 0.40)},
    "CoffeeMachineFactory":         {"default": (0.40, 0.40), "button": (0.50, 0.35), "knob": (0.60, 0.30), "lever": (0.55, 0.35), "portafilter": (0.80, 0.25), "rotor": (0.50, 0.35)},
    "KitchenPotSapienFactory":      {"default": (0.80, 0.25), "lid": (0.85, 0.22), "button": (0.50, 0.40)},
    "KettleFactory":                {"default": (0.70, 0.30), "lid": (0.75, 0.25), "rotation_lid": (0.75, 0.25), "handle": (0.60, 0.35), "button": (0.50, 0.40)},
    # ── furniture - wood/plastic ──
    "StorageFurnitureSapienFactory":{"default": (0.0, 0.55), "rotation_door": (0.0, 0.50), "translation_door": (0.0, 0.50), "drawer": (0.0, 0.50), "board": (0.0, 0.55), "caster": (0.30, 0.40), "wheel": (0.30, 0.40)},
    "TableSapienFactory":           {"default": (0.0, 0.55), "drawer": (0.0, 0.50), "door": (0.0, 0.50), "handle": (0.70, 0.30), "caster": (0.30, 0.40), "wheel": (0.30, 0.40)},
    "ChairSapienFactory":           {"default": (0.0, 0.50), "wheel": (0.30, 0.40), "caster": (0.30, 0.40), "seat": (0.0, 0.60), "lever": (0.60, 0.35), "knob": (0.60, 0.35)},
    # ── household ──
    "BottleSapienFactory":          {"default": (0.0, 0.40), "rotation_lid": (0.30, 0.35), "translation_lid": (0.30, 0.35)},
    "LampSapienFactory":            {"default": (0.0, 0.50), "head": (0.0, 0.55), "rotation_bar": (0.60, 0.30), "translation_bar": (0.60, 0.30), "fastener": (0.70, 0.25), "button": (0.0, 0.45)},
    "ToiletSapienFactory":          {"default": (0.0, 0.35), "lid": (0.0, 0.30), "seat": (0.0, 0.30), "button": (0.50, 0.40), "lever": (0.70, 0.30)},
    "WindowSapienFactory":          {"default": (0.0, 0.40), "rotation_window": (0.0, 0.35), "translation_window": (0.0, 0.35)},
    "DoorSapienFactory":            {"default": (0.0, 0.55), "rotation_door": (0.0, 0.50), "translation_door": (0.0, 0.50)},
    "FanFactory":                   {"default": (0.0, 0.45), "rotor": (0.0, 0.40), "button": (0.0, 0.45), "slider": (0.30, 0.40)},
    "GlobeFactory":                 {"default": (0.0, 0.45), "sphere": (0.0, 0.35), "circle": (0.60, 0.30)},
    "TrashCanFactory":              {"default": (0.60, 0.35), "cover_lid": (0.65, 0.30), "lid": (0.65, 0.30), "foot_pad": (0.50, 0.40), "wheel": (0.30, 0.50)},
    "LaptopFactory":                {"default": (0.30, 0.40), "screen": (0.0, 0.30)},
    "DisplayFactory":               {"default": (0.0, 0.35), "screen": (0.0, 0.25), "button": (0.50, 0.35)},
    "BucketFactory":                {"default": (0.0, 0.50), "handle": (0.70, 0.30)},
    "BoxFactory":                   {"default": (0.0, 0.55), "rotation_lid": (0.0, 0.50), "drawer": (0.0, 0.50), "lock": (0.70, 0.30), "handle": (0.70, 0.30)},
    "PenFactory":                   {"default": (0.40, 0.35), "cap": (0.50, 0.30), "button": (0.50, 0.35)},
    "USBFactory":                   {"default": (0.30, 0.40), "lid": (0.40, 0.35), "usb_rotation": (0.35, 0.38)},
    "CartFactory":                  {"default": (0.60, 0.35), "wheel": (0.30, 0.50), "caster": (0.30, 0.50), "steering_wheel": (0.40, 0.40)},
    "SuitcaseFactory":              {"default": (0.0, 0.55), "translation_handle": (0.50, 0.35), "rotation_handle": (0.50, 0.35), "lock": (0.70, 0.30), "caster": (0.30, 0.40), "wheel": (0.30, 0.40)},
    "FoldingChairFactory":          {"default": (0.50, 0.40), "leg": (0.55, 0.38), "seat": (0.0, 0.55)},
    "MouseFactory":                 {"default": (0.0, 0.45), "wheel": (0.30, 0.35), "ball": (0.20, 0.40), "button": (0.0, 0.42)},
    "SwitchFactory":                {"default": (0.0, 0.45), "switch": (0.0, 0.40), "button": (0.0, 0.40)},
    "KeyboardFactory":              {"default": (0.0, 0.55), "key": (0.0, 0.50), "tilt_leg": (0.30, 0.40)},
    "PrinterFactory":               {"default": (0.0, 0.50), "button": (0.0, 0.45), "drawer": (0.0, 0.48), "toggle_button": (0.0, 0.45)},
    "RemoteFactory":                {"default": (0.0, 0.50), "button": (0.0, 0.45), "rotation_button": (0.0, 0.43), "knob": (0.30, 0.40)},
    "PhoneFactory":                 {"default": (0.0, 0.40), "button": (0.0, 0.35), "rotation_lid": (0.0, 0.38), "slider": (0.0, 0.38)},
}

# Map PartNet Sapien factories to IM factories whose textures can be borrowed
PARTNET_TEXTURE_DONORS = {
    "StorageFurnitureSapienFactory": "KitchenCabinetFactory",
    "DishwasherSapienFactory":       "DishwasherFactory",
    "MicrowaveSapienFactory":        "MicrowaveFactory",
    "OvenSapienFactory":             "OvenFactory",
    "RefrigeratorSapienFactory":     "BeverageFridgeFactory",
    "ToiletSapienFactory":           "ToiletFactory",
    "WindowSapienFactory":           "WindowFactory",
    "DoorSapienFactory":             "LiteDoorFactory",
    "LampSapienFactory":             "LampFactory",
    "BottleSapienFactory":           "BottleFactory",
    "KitchenPotSapienFactory":       "PotFactory",
    "FaucetSapienFactory":           "TapFactory",
}


def _find_principled_bsdf(mat):
    """Find the Principled BSDF node in a material's node tree."""
    if mat is None or not mat.use_nodes or mat.node_tree is None:
        return None
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            return node
    return None


def _collect_donor_textures(donor_factory, base_dir):
    """Collect DIFFUSE texture paths from an IM factory's outputs."""
    import glob
    factory_dir = os.path.join(base_dir, "outputs", donor_factory)
    if not os.path.isdir(factory_dir):
        return []
    textures = []
    pattern = os.path.join(factory_dir, "*/outputs", donor_factory, "*/objs/*", "*_DIFFUSE.png")
    for p in glob.glob(pattern):
        textures.append(p)
    return textures


def _collect_donor_texture_sets(donor_factory, base_dir):
    """Collect (DIFFUSE, ROUGHNESS, NORMAL) texture triplets from an IM factory."""
    import glob
    factory_dir = os.path.join(base_dir, "outputs", donor_factory)
    if not os.path.isdir(factory_dir):
        return []
    sets = []
    pattern = os.path.join(factory_dir, "*/outputs", donor_factory, "*/objs/*", "*_DIFFUSE.png")
    for diffuse_path in glob.glob(pattern):
        prefix = diffuse_path.rsplit("_DIFFUSE.png", 1)[0]
        roughness = prefix + "_ROUGHNESS.png"
        normal = prefix + "_NORMAL.png"
        sets.append({
            "diffuse": diffuse_path,
            "roughness": roughness if os.path.exists(roughness) else None,
            "normal": normal if os.path.exists(normal) else None,
        })
    return sets


def _apply_texture_to_material(mat, texture_set):
    """Replace a material's base color with a donor DIFFUSE texture and add ROUGHNESS/NORMAL."""
    if mat is None or not mat.use_nodes:
        return
    tree = mat.node_tree
    bsdf = _find_principled_bsdf(mat)
    if bsdf is None:
        return

    # Load and connect DIFFUSE texture
    diffuse_path = texture_set["diffuse"]
    try:
        img = bpy.data.images.load(diffuse_path)
    except RuntimeError:
        return

    tex_node = tree.nodes.new('ShaderNodeTexImage')
    tex_node.image = img
    tex_node.location = (bsdf.location[0] - 300, bsdf.location[1])
    tree.links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])

    # ROUGHNESS texture
    if texture_set.get("roughness"):
        try:
            rough_img = bpy.data.images.load(texture_set["roughness"])
            rough_img.colorspace_settings.name = 'Non-Color'
            rough_node = tree.nodes.new('ShaderNodeTexImage')
            rough_node.image = rough_img
            rough_node.location = (bsdf.location[0] - 300, bsdf.location[1] - 280)
            tree.links.new(rough_node.outputs['Color'], bsdf.inputs['Roughness'])
        except RuntimeError:
            pass

    # NORMAL texture
    if texture_set.get("normal"):
        try:
            norm_img = bpy.data.images.load(texture_set["normal"])
            norm_img.colorspace_settings.name = 'Non-Color'
            norm_node = tree.nodes.new('ShaderNodeTexImage')
            norm_node.image = norm_img
            norm_node.location = (bsdf.location[0] - 300, bsdf.location[1] - 560)
            normal_map = tree.nodes.new('ShaderNodeNormalMap')
            normal_map.location = (bsdf.location[0] - 50, bsdf.location[1] - 560)
            tree.links.new(norm_node.outputs['Color'], normal_map.inputs['Color'])
            tree.links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])
        except RuntimeError:
            pass


def enhance_partnet_materials(factory_name, seed, part_objects, base_dir):
    """Enhance materials for PartNet-converted objects.

    - Sets metallic/roughness on Principled BSDF nodes based on category + part semantics
    - For Sapien factories with IM counterparts, optionally replaces textures with IM ones
    """
    import random as _rng

    rules = PARTNET_MATERIAL_DEFAULTS.get(factory_name)
    if rules is None:
        # Not a PartNet factory or no rules defined - apply generic roughness
        return

    # Load data_infos for part name mapping
    factory_dir = os.path.join(base_dir, "outputs", factory_name)
    data_infos_path = os.path.join(factory_dir, f"data_infos_{seed}.json")
    part_names = {}  # part_idx -> part_name
    if os.path.exists(data_infos_path):
        with open(data_infos_path) as f:
            data_infos = json.load(f)
        if data_infos:
            for part in data_infos[0].get("part", []):
                pname = part.get("part_name", "")
                fname = part.get("file_name", "")
                # Extract index from filename like "0.obj"
                try:
                    pidx = int(fname.replace(".obj", ""))
                    part_names[pidx] = pname
                except ValueError:
                    pass

    # Seed-based RNG for reproducible randomization
    mat_rng = _rng.Random(hash((factory_name, seed)))

    # Collect donor textures if available
    donor_factory = PARTNET_TEXTURE_DONORS.get(factory_name)
    donor_sets = []
    if donor_factory:
        donor_sets = _collect_donor_texture_sets(donor_factory, base_dir)

    enhanced_count = 0
    texture_count = 0

    for part_idx, obj in part_objects.items():
        pname = part_names.get(part_idx, "")

        # Determine metallic/roughness from rules
        metallic, roughness = rules.get("default", (0.0, 0.50))
        # Check part-name-specific overrides
        for keyword, (m, r) in rules.items():
            if keyword == "default":
                continue
            if keyword in pname:
                metallic, roughness = m, r
                break

        # Add slight per-seed variation
        metallic = max(0, min(1, metallic + mat_rng.uniform(-0.05, 0.05)))
        roughness = max(0, min(1, roughness + mat_rng.uniform(-0.08, 0.08)))

        # Apply to all materials on the object
        for mat_slot in obj.material_slots:
            mat = mat_slot.material
            bsdf = _find_principled_bsdf(mat)
            if bsdf is None:
                continue

            bsdf.inputs['Metallic'].default_value = metallic
            bsdf.inputs['Roughness'].default_value = roughness

            # Slight specular adjustment for metallic surfaces
            if metallic > 0.5:
                # Blender 4.0+ renamed "Specular" to "Specular IOR Level"
                spec_name = 'Specular IOR Level' if IS_BLENDER_4X else 'Specular'
                if spec_name in bsdf.inputs:
                    bsdf.inputs[spec_name].default_value = 0.5 + metallic * 0.3

            enhanced_count += 1

        # Texture replacement for Sapien factories (50% chance per part per seed)
        if donor_sets and mat_rng.random() < 0.5:
            tex_set = mat_rng.choice(donor_sets)
            # Only replace if the part has at least one material
            if obj.material_slots:
                mat = obj.material_slots[0].material
                _apply_texture_to_material(mat, tex_set)
                texture_count += 1

    if enhanced_count > 0:
        print(f"  Materials enhanced: {enhanced_count} slots "
              f"(metallic/roughness), {texture_count} texture replacements")


def _enhance_physxnet_materials(factory_name, seed, part_objects):
    """Enhance materials for PhysXNet/PhysX_mobility factories.

    Uses PBR texture library with box projection mapping, with PartNet
    color extraction for 1,081 overlapping objects. Falls back to flat
    PBR values when textures are unavailable.

    Material sources (priority):
      1. ShapeNet textures (future — placeholder)
      2. PartNet colors + PBR overlay (for overlap objects)
      3. ambientCG PBR textures with box projection (default)

    IMPORTANT: part_objects keys are GROUP indices (from URDF link l_N),
    NOT raw part labels. We must map group_idx -> part_labels via group_info.
    """
    try:
        from physxnet_loader import (
            is_physxnet_factory, load_physxnet_json, load_physxmob_json,
            seed_to_object_id,
        )
        from physxnet_factory_rules import factory_dataset
    except ImportError:
        return

    if not is_physxnet_factory(factory_name):
        return

    ds = factory_dataset(factory_name)

    # Load JSON data for this object
    json_data = None
    obj_id = str(seed)
    if ds == "physxmob":
        json_data = load_physxmob_json(obj_id)
    else:
        json_data = load_physxnet_json(obj_id)

    if json_data is None:
        mapped_id, _ = seed_to_object_id(factory_name, seed, dataset=ds)
        if mapped_id is None:
            return
        obj_id = str(mapped_id)
        if ds == "physxmob":
            json_data = load_physxmob_json(obj_id)
        else:
            json_data = load_physxnet_json(obj_id)

    if json_data is None:
        return

    # Use new PBR material system
    try:
        from pbr_material_system import apply_pbr_materials
        apply_pbr_materials(factory_name, seed, part_objects, json_data)
    except ImportError:
        # Fallback to flat PBR if pbr_material_system not available
        _enhance_physxnet_materials_flat(factory_name, seed, part_objects, json_data)


def _enhance_physxnet_materials_flat(factory_name, seed, part_objects, json_data):
    """Flat PBR fallback (no textures) — used when pbr_material_system unavailable."""
    import random as _rng

    from physxnet_loader import get_pbr_for_material

    parts = json_data.get("parts", [])
    label_to_info = {}
    for p in parts:
        label = p.get("label")
        if label is not None:
            label_to_info[label] = p

    group_info = json_data.get("group_info", {})
    group_to_labels = {}
    for gid_str, val in group_info.items():
        gid = int(gid_str)
        if isinstance(val, list) and len(val) >= 4 and isinstance(val[-1], str) and val[-1] in ('A', 'B', 'C', 'D', 'CB'):
            labels = val[0] if isinstance(val[0], list) else [val[0]]
            group_to_labels[gid] = labels
        elif isinstance(val, list):
            group_to_labels[gid] = [x for x in val if isinstance(x, int)]

    mat_rng = _rng.Random(hash((factory_name, seed)))
    enhanced_count = 0

    for part_idx, obj in part_objects.items():
        part_labels = group_to_labels.get(part_idx, [part_idx])
        info = {}
        for lbl in part_labels:
            info = label_to_info.get(lbl, {})
            if info:
                break
        material_name = info.get("material", "Unknown")
        metallic, roughness, color = get_pbr_for_material(material_name)
        metallic = max(0, min(1, metallic + mat_rng.uniform(-0.05, 0.05)))
        roughness = max(0, min(1, roughness + mat_rng.uniform(-0.08, 0.08)))

        for mat_slot in obj.material_slots:
            mat = mat_slot.material
            bsdf = _find_principled_bsdf(mat)
            if bsdf is None:
                continue
            bc_input = bsdf.inputs['Base Color']
            if bc_input.is_linked:
                for link in list(mat.node_tree.links):
                    if link.to_socket == bc_input:
                        mat.node_tree.links.remove(link)
            bsdf.inputs['Metallic'].default_value = metallic
            bsdf.inputs['Roughness'].default_value = roughness
            bsdf.inputs['Base Color'].default_value = (*color, 1.0)
            if metallic > 0.5:
                bsdf.inputs['Specular'].default_value = 0.5 + metallic * 0.3
            enhanced_count += 1

        if not obj.material_slots:
            mat = bpy.data.materials.new(name=f"physxnet_mat_{part_idx}")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs['Metallic'].default_value = metallic
                bsdf.inputs['Roughness'].default_value = roughness
                bsdf.inputs['Base Color'].default_value = (*color, 1.0)
                if metallic > 0.5:
                    bsdf.inputs['Specular'].default_value = 0.5 + metallic * 0.3
            obj.data.materials.append(mat)
            enhanced_count += 1

    if enhanced_count > 0:
        print(f"  PhysXNet materials (flat fallback): {enhanced_count} parts")


def get_moving_part_indices(factory_name, scene_dir):
    """Determine which part indices are 'moving' using FACTORY_RULES + data_infos.

    Returns set of int part indices, or None (animate all) if no rules or data_infos.
    """
    rules = FACTORY_RULES.get(factory_name)
    if rules is None:
        return None  # No rules: animate all parts

    moving_names = rules["moving_parts"]
    exclude_names = rules.get("exclude_parts", set()) | {"unknown_part"}

    # Find data_infos file
    factory_dir = os.path.dirname(scene_dir)  # e.g. outputs/BottleFactory
    seed = os.path.basename(scene_dir)
    data_infos_path = os.path.join(factory_dir, f"data_infos_{seed}.json")

    if not os.path.exists(data_infos_path):
        print(f"  WARNING: No data_infos at {data_infos_path}, using URDF fallback")
        return None

    with open(data_infos_path) as f:
        data_infos = json.load(f)

    if not data_infos:
        return None

    # Map part_name → part_idx from first instance
    moving_indices = set()
    exclude_indices = set()
    for part in data_infos[0]["part"]:
        pname = part["part_name"]
        pidx = int(os.path.splitext(part["file_name"])[0])
        if pname in moving_names:
            moving_indices.add(pidx)
        if pname in exclude_names:
            exclude_indices.add(pidx)

    # If no parts matched, fall back to animate all (avoid silent static scene)
    if not moving_indices:
        print(f"  WARNING: No parts matched FACTORY_RULES moving_parts, animating all")
        return None

    print(f"  FACTORY_RULES: moving indices = {sorted(moving_indices)}")
    if exclude_indices:
        print(f"  FACTORY_RULES: excluded indices = {sorted(exclude_indices)}")

    return moving_indices, exclude_indices


# ═══════════════════════════════════════════════════════════════
# Blender Helpers
# ═══════════════════════════════════════════════════════════════

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.images:
        if block.users == 0:
            bpy.data.images.remove(block)


def setup_render_engine(engine="cycles"):
    """Set up render engine: cycles (with GPU) or eevee."""
    scene = bpy.context.scene
    if engine == "eevee":
        scene.render.engine = 'BLENDER_EEVEE'
        scene.eevee.taa_render_samples = 64
        scene.eevee.use_gtao = True       # ambient occlusion
        scene.eevee.use_ssr = True         # screen-space reflections
        scene.eevee.use_soft_shadows = True
        print(f"  Engine: EEVEE (64 TAA samples, AO+SSR+soft shadows)")
        return

    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    # OptiX uses RT cores for hardware-accelerated ray tracing (2-3x faster)
    # Fall back to CUDA if OptiX is not available
    for backend in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            usable = [d for d in prefs.devices if d.type == backend]
            if usable:
                for dev in prefs.devices:
                    dev.use = (dev.type == backend)
                    if dev.use:
                        print(f"  GPU ({backend}): {dev.name}")
                break
        except Exception:
            continue
    scene.cycles.device = 'GPU'


def setup_render_settings(resolution, fps, num_frames, samples, transparent=False,
                          max_bounces=None, engine="cycles"):
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = num_frames

    if engine == "eevee":
        # EEVEE settings already configured in setup_render_engine
        pass
    else:
        # Cycles settings
        scene.cycles.samples = samples
        scene.cycles.use_denoising = True
        scene.cycles.denoiser = 'OPENIMAGEDENOISE'
        try:
            scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
        except TypeError:
            pass
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.01

        # Bounce limits
        if max_bounces is not None:
            scene.cycles.max_bounces = max_bounces
            scene.cycles.diffuse_bounces = min(max_bounces, 2)
            scene.cycles.glossy_bounces = min(max_bounces, 2)
            scene.cycles.transmission_bounces = min(max_bounces, 2)
            scene.cycles.transparent_max_bounces = min(max_bounces, 4)
            scene.cycles.volume_bounces = 0
            print(f"  Bounces: max={max_bounces}, diffuse={scene.cycles.diffuse_bounces}, "
                  f"glossy={scene.cycles.glossy_bounces}, transmission={scene.cycles.transmission_bounces}")

    # Keep textures, shaders, BVH, and light data in memory between frames
    scene.render.use_persistent_data = True
    scene.render.film_transparent = transparent
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    # Color management: Filmic for realistic HDR tonemapping
    # Blender 4.x: "Filmic" is now under "AgX" by default, but "Filmic" still exists
    try:
        scene.view_settings.view_transform = 'Filmic'
    except TypeError:
        scene.view_settings.view_transform = 'AgX'  # Blender 4.x default
    try:
        scene.view_settings.look = 'Medium Contrast'
    except TypeError:
        scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def setup_envmap_lighting(envmap_path, strength=1.0):
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    for n in nodes:
        nodes.remove(n)

    bg = nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = strength
    env_tex = nodes.new('ShaderNodeTexEnvironment')
    if os.path.exists(envmap_path):
        env_tex.image = bpy.data.images.load(envmap_path)
        print(f"  Envmap loaded: {envmap_path}")
    else:
        print(f"  WARNING: envmap not found: {envmap_path}")
    output = nodes.new('ShaderNodeOutputWorld')
    links.new(env_tex.outputs['Color'], bg.inputs['Color'])
    links.new(bg.outputs['Background'], output.inputs['Surface'])


def import_obj_with_textures(filepath):
    """Import OBJ file into Blender.

    The per-part OBJ files are already in Z-up convention (matching Blender).
    Blender's OBJ importer always applies axis conversion as an OBJECT rotation
    (rotation_euler), not to vertex data. Even with axis_forward='-Y', axis_up='Z',
    it adds a spurious Rz(180°). We must reset rotation_euler=(0,0,0) after import
    so that transform_apply only bakes the location (origin offset) into vertices.

    Blender 4.x replaced import_scene.obj with wm.obj_import (C++ importer).
    We detect the version and use the appropriate API.
    """
    existing = set(bpy.data.objects.keys())

    if IS_BLENDER_4X:
        # Blender 4.x: new C++ OBJ importer
        bpy.ops.wm.obj_import(
            filepath=filepath,
            forward_axis='NEGATIVE_Y',
            up_axis='Z',
        )
    else:
        # Blender 3.x: legacy Python OBJ importer
        bpy.ops.import_scene.obj(
            filepath=filepath, use_edges=False, use_smooth_groups=True,
            axis_forward='-Y', axis_up='Z',
        )

    new_objs = [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in existing]
    for obj in new_objs:
        if obj.type == 'MESH':
            # Reset the spurious rotation from the importer
            obj.rotation_euler = (0, 0, 0)
            if IS_BLENDER_4X:
                # Blender 4.x: use shade_smooth attribute instead of per-polygon flag
                obj.data.shade_smooth()
            else:
                for poly in obj.data.polygons:
                    poly.use_smooth = True
    return new_objs


def join_objects(objects, name="joined"):
    if not objects:
        return None
    if len(objects) == 1:
        objects[0].name = name
        return objects[0]
    bpy.ops.object.select_all(action='DESELECT')
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    joined = bpy.context.active_object
    joined.name = name
    return joined


def compute_scene_bounds(objects):
    """Compute bounding box of all mesh objects in world space."""
    all_coords = []
    for obj in objects:
        if obj.type == 'MESH':
            for v in obj.data.vertices:
                world_co = obj.matrix_world @ v.co
                all_coords.append((world_co.x, world_co.y, world_co.z))
    if not all_coords:
        return [0, 0, 0], 1.0
    import numpy as np
    coords = np.array(all_coords)
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = np.linalg.norm(bbox_max - bbox_min)
    return center.tolist(), extent


def create_camera(name, center, distance, elev_deg, azim_deg, lens=50):
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.clip_start = 0.001
    cam_data.clip_end = 100
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)

    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    x = distance * math.cos(elev) * math.cos(azim) + center[0]
    y = distance * math.cos(elev) * math.sin(azim) + center[1]
    z = distance * math.sin(elev) + center[2]

    cam.location = (x, y, z)
    direction = Vector(center) - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    return cam


def create_animated_camera(name, center, distance, start_elev, start_azim,
                           end_elev, end_azim, num_frames, lens=50):
    """Create a camera that orbits from start to end position over the animation.

    Interpolates linearly in spherical coordinates, sets per-frame keyframes.
    """
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.clip_start = 0.001
    cam_data.clip_end = 100
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)

    for frame in range(1, num_frames + 1):
        t = (frame - 1) / max(num_frames - 1, 1)
        elev_deg = start_elev + (end_elev - start_elev) * t
        azim_deg = start_azim + (end_azim - start_azim) * t

        elev = math.radians(elev_deg)
        azim = math.radians(azim_deg)
        x = distance * math.cos(elev) * math.cos(azim) + center[0]
        y = distance * math.cos(elev) * math.sin(azim) + center[1]
        z = distance * math.sin(elev) + center[2]

        cam.location = (x, y, z)
        direction = Vector(center) - cam.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam.rotation_euler = rot_quat.to_euler()

        cam.keyframe_insert(data_path="location", frame=frame)
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Linear interpolation for smooth constant-speed motion
    if cam.animation_data and cam.animation_data.action:
        for fc in cam.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'

    return cam


# ═══════════════════════════════════════════════════════════════
# Transform Math
# ═══════════════════════════════════════════════════════════════

def mat_translate(x, y, z):
    m = Matrix.Identity(4)
    m[0][3] = x
    m[1][3] = y
    m[2][3] = z
    return m


def mat_rotate_axis_angle(axis, angle):
    """4x4 rotation matrix around an axis by an angle (radians)."""
    v = Vector(axis).normalized()
    if v.length < 1e-10:
        return Matrix.Identity(4)
    q = Quaternion(v, angle)
    return q.to_matrix().to_4x4()


# ═══════════════════════════════════════════════════════════════
# Kinematic Forward Kinematics
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
# Animation: use URDF joint limits directly
# ═══════════════════════════════════════════════════════════════
#
# Each joint animates from 0 → URDF limit → 0 (sinusoidal round trip).
# Which joints animate is controlled by animode_joints in FACTORY_RULES.

_CURRENT_ANIMODE = 0  # set by main() before animation


def compute_joint_value(joint, frame, num_frames):
    """Compute joint value at given frame using URDF limits directly.

    Simple round-trip: 0 → target → 0 via sin(t * pi).
    """
    if joint.jtype == "fixed":
        return 0.0

    # Pick target from URDF limits (whichever end has larger absolute value)
    if joint.jtype in ("prismatic", "revolute"):
        target = joint.upper if abs(joint.upper) >= abs(joint.lower) else joint.lower
    elif joint.jtype == "continuous":
        if abs(joint.upper - joint.lower) > 1e-6:
            target = joint.upper if abs(joint.upper) >= abs(joint.lower) else joint.lower
        else:
            target = 1.0
    else:
        return 0.0

    # Simple sinusoidal round-trip: 0 → target → 0
    t = (frame - 1) / max(num_frames - 1, 1)
    return math.sin(t * math.pi) * target


def compute_joint_local_transform(joint, q_value):
    """Compute the 4x4 local transform for a joint given its parameter value q.

    The URDF joint transform is:
      T = T_origin * T_joint(q)

    Where T_origin is the fixed offset from parent to child frame,
    and T_joint(q) is the joint motion (rotation or translation).

    Returns 4x4 Matrix.
    """
    # Origin transform (fixed offset from parent)
    ox, oy, oz = joint.origin_xyz
    T_origin = mat_translate(ox, oy, oz)

    # Joint motion transform
    if joint.jtype == "fixed":
        T_joint = Matrix.Identity(4)
    elif joint.jtype in ("revolute", "continuous"):
        T_joint = mat_rotate_axis_angle(joint.axis, q_value)
    elif joint.jtype == "prismatic":
        ax = Vector(joint.axis).normalized()
        T_joint = mat_translate(ax.x * q_value, ax.y * q_value, ax.z * q_value)
    else:
        T_joint = Matrix.Identity(4)

    return T_origin @ T_joint


def forward_kinematics(links, joints, parent_map, children_map, frame, num_frames):
    """Compute world-space transforms for all links at a given frame.

    Uses BFS from root links. Returns dict: link_name -> 4x4 Matrix (world transform).
    """
    link_transforms = {}

    # Find root: l_world or links that are only parents
    all_children = set(j.child_link for j in joints)
    all_parents = set(j.parent_link for j in joints)
    roots = all_parents - all_children

    # Initialize roots at identity
    for r in roots:
        link_transforms[r] = Matrix.Identity(4)

    # BFS
    queue = deque(roots)
    visited = set()

    while queue:
        link_name = queue.popleft()
        if link_name in visited:
            continue
        visited.add(link_name)

        parent_T = link_transforms.get(link_name, Matrix.Identity(4))

        for child_link, joint in children_map.get(link_name, []):
            # Compute joint value
            q = compute_joint_value(joint, frame, num_frames)

            # Compute local transform
            T_local = compute_joint_local_transform(joint, q)

            # Child world transform = parent * local
            child_T = parent_T @ T_local
            link_transforms[child_link] = child_T

            queue.append(child_link)

    return link_transforms


# ═══════════════════════════════════════════════════════════════
# Part Loading
# ═══════════════════════════════════════════════════════════════

def load_scene_parts(objs_dir, origins, exclude_indices=None):
    """Load all per-part OBJs and translate them to world position.

    Each OBJ is centroid-subtracted. We set the object location to the origin
    and then apply the transform so the mesh vertices are in world coordinates
    and the object origin is at (0,0,0).

    Returns dict: part_idx -> Blender object
    """
    if exclude_indices is None:
        exclude_indices = set()

    part_objects = {}

    for part_idx_str, origin in origins.items():
        if part_idx_str == "world":
            continue

        part_idx = int(part_idx_str)

        if part_idx in exclude_indices:
            print(f"  SKIP part {part_idx}: excluded")
            continue

        part_dir = os.path.join(objs_dir, str(part_idx))
        obj_path = os.path.join(part_dir, f"{part_idx}.obj")

        if not os.path.exists(obj_path):
            print(f"  SKIP part {part_idx}: OBJ not found")
            continue

        imported = import_obj_with_textures(obj_path)
        if not imported:
            continue

        # Load extra OBJs for multi-visual links (e.g. {idx}_extra_0.obj)
        extra_idx = 0
        while True:
            extra_path = os.path.join(part_dir, f"{part_idx}_extra_{extra_idx}.obj")
            if not os.path.exists(extra_path):
                break
            extra_imported = import_obj_with_textures(extra_path)
            if extra_imported:
                imported.extend(extra_imported)
            extra_idx += 1
        if extra_idx > 0:
            print(f"  Part {part_idx}: loaded {extra_idx} extra OBJ(s)")

        obj = join_objects(imported, name=f"part_{part_idx}")
        if obj is None:
            continue

        # Move to world position
        obj.location = Vector(origin)

        # Apply transform: mesh vertices become world-space, object origin at (0,0,0)
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)

        part_objects[part_idx] = obj

    print(f"  Loaded {len(part_objects)} parts")
    return part_objects


# ═══════════════════════════════════════════════════════════════
# Animation via Direct Matrix Setting
# ═══════════════════════════════════════════════════════════════

def animate_parts(part_objects, links, joints, parent_map, children_map, origins, num_frames,
                   moving_indices=None):
    """Animate moving parts by computing forward kinematics per frame.

    Only parts in moving_indices receive animated transforms. Body parts stay static
    at their rest-pose position (identity matrix_world after transform_apply).
    """
    print(f"\nComputing animation for {num_frames} frames...")

    # Get link -> part_idx mapping
    link_part_map = {}  # link_name -> part_idx
    for link_name, info in links.items():
        if info["part_idx"] is not None:
            link_part_map[link_name] = info["part_idx"]

    # Determine which links are animated
    # When animode is active, ALL parts may be animated (animode picks the joints,
    # and the affected parts follow). So treat all parts as potentially animated.
    animode_joints_cfg_pre = FACTORY_RULES.get(FACTORY, {}).get("animode_joints", {})
    animode_active_pre = (_CURRENT_ANIMODE >= 10 or _CURRENT_ANIMODE in animode_joints_cfg_pre)

    animated_links = {}
    for link_name, part_idx in link_part_map.items():
        if moving_indices is None or animode_active_pre or part_idx in moving_indices:
            animated_links[link_name] = part_idx

    body_links = {ln: pi for ln, pi in link_part_map.items() if ln not in animated_links}

    print(f"  Animated parts: {sorted(animated_links.values())}")
    print(f"  Static parts: {sorted(body_links.values())}")

    # Determine which joints should be animated
    # When animode filtering is active (type-based or per-joint), we need ALL
    # non-fixed joints as candidates so the filter can select from the full set.
    # Otherwise, restrict to joints on paths to moving parts.
    animode_joints_cfg = FACTORY_RULES.get(FACTORY, {}).get("animode_joints", {})
    animode_active = (_CURRENT_ANIMODE >= 10 or _CURRENT_ANIMODE in animode_joints_cfg)

    all_nonfixed = {j.name for j in joints if j.jtype != "fixed"}

    if animode_active or moving_indices is None:
        # Use all non-fixed joints as candidates for animode filtering
        animated_joint_names = all_nonfixed
    else:
        # BFS backwards from each moving link to root, collecting joints on the path
        animated_joint_names = set()
        for link_name, part_idx in animated_links.items():
            curr = link_name
            while curr in parent_map:
                parent_link, joint = parent_map[curr]
                animated_joint_names.add(joint.name)
                curr = parent_link

    print(f"  Animated joints: {sorted(animated_joint_names)}")

    frozen_joints = set()  # descendant joints kept at q=0 for rigid following

    if _CURRENT_ANIMODE >= 10:
        # Per-joint mode: animate only the Nth movable joint (0-indexed)
        joint_index = _CURRENT_ANIMODE - 10
        joint_by_name = {j.name: j for j in joints}
        movable_joints = sorted([
            jname for jname in animated_joint_names
            if joint_by_name.get(jname) and joint_by_name[jname].jtype != "fixed"
        ])
        if joint_index < len(movable_joints):
            animated_joint_names = {movable_joints[joint_index]}
            print(f"  Animode {_CURRENT_ANIMODE}: per-joint mode, joint[{joint_index}] = {movable_joints[joint_index]}")
        else:
            print(f"  Animode {_CURRENT_ANIMODE}: per-joint mode, joint[{joint_index}] out of range ({len(movable_joints)} movable joints), skipping")
            return False
    elif _CURRENT_ANIMODE in animode_joints_cfg:
        selectors = animode_joints_cfg[_CURRENT_ANIMODE]
        joint_by_name = {j.name: j for j in joints}

        # Filter: exclude joints with negligible range
        MIN_PRISMATIC = 0.005  # 5mm
        MIN_ROTARY = 0.05     # ~3 degrees
        significant = set()
        for jname in animated_joint_names:
            j = joint_by_name.get(jname)
            if not j or j.jtype == "fixed":
                continue
            rng = abs(j.upper - j.lower)
            if j.jtype == "prismatic" and rng < MIN_PRISMATIC:
                continue
            if j.jtype in ("revolute", "continuous") and rng < MIN_ROTARY:
                continue
            significant.add(jname)

        # Resolve selectors
        # Formats: ("type",)                       → all significant joints of type
        #          ("type", ordinal)                → nth joint by depth
        #          ("type", "axis", "x"|"y"|"z")   → joints with given primary axis
        #          ("type", "sign", "+"|"-")        → joints by limit sign (target value)
        selected = set()
        for sel in selectors:
            if len(sel) == 1:
                # All significant joints of this type
                sel_type = sel[0]
                matched = {n for n in significant if joint_by_name[n].jtype == sel_type}
                selected |= matched
            elif len(sel) == 3 and sel[1] == "axis":
                # Filter by rotation/translation axis
                sel_type, _, axis_name = sel
                axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name.lower()]
                for jname in significant:
                    j = joint_by_name[jname]
                    if j.jtype == sel_type and abs(j.axis[axis_idx]) > 0.5:
                        selected.add(jname)
            elif len(sel) == 3 and sel[1] == "sign":
                # Filter by limit sign: "+" = target > 0, "-" = target < 0
                sel_type, _, sign = sel
                for jname in significant:
                    j = joint_by_name[jname]
                    if j.jtype != sel_type:
                        continue
                    target = j.upper if abs(j.upper) >= abs(j.lower) else j.lower
                    if sign == "+" and target > 0:
                        selected.add(jname)
                    elif sign == "-" and target < 0:
                        selected.add(jname)
            elif len(sel) == 2:
                # Specific joint by type + depth ordinal
                sel_type, sel_ord = sel
                candidates = []
                for jname in significant:
                    j = joint_by_name[jname]
                    if j.jtype == sel_type:
                        depth = 0
                        curr = j.child_link
                        while curr in parent_map:
                            depth += 1
                            curr = parent_map[curr][0]
                        candidates.append((depth, jname))
                candidates.sort()  # (depth, name) for deterministic order
                if candidates:
                    selected.add(candidates[sel_ord][1])

        if selected:
            # Prune descendant joints: if joint B is a kinematic descendant
            # of joint A, and both are selected with a single-type selector,
            # freeze B at q=0 so its parts follow A rigidly (avoids compound
            # rotation artifacts in hierarchical URDFs like lamps with
            # arm + head joints).
            # Frozen joints stay in animated_joint_names (so collision code
            # doesn't treat them as passive) but are excluded from FK animation.
            frozen_joints = set()
            if len(selectors) == 1 and len(selected) > 1:
                for jname in list(selected):
                    j = joint_by_name[jname]
                    curr = j.parent_link
                    while curr in parent_map:
                        ancestor_link, ancestor_joint = parent_map[curr]
                        if ancestor_joint.name in selected:
                            frozen_joints.add(jname)
                            break
                        curr = ancestor_link
                if frozen_joints:
                    print(f"  Frozen descendant joints (q=0, rigid follow): "
                          f"{sorted(frozen_joints)}")

            animated_joint_names = selected
            print(f"  Animode {_CURRENT_ANIMODE}: {sorted(selected)}")
        else:
            print(f"  Animode {_CURRENT_ANIMODE}: no matching joints, skipping render")
            return False

    # Compute rest transforms (all joints at q=0)
    rest_transforms = forward_kinematics_at_q(links, joints, parent_map, children_map, q_values=None)

    # For each frame, compute current transforms
    for frame in range(1, num_frames + 1):
        bpy.context.scene.frame_set(frame)

        # Compute FK at this frame, only animating joints on paths to moving parts
        # Frozen joints (pruned descendants) stay at q=0 for rigid following
        active_for_fk = animated_joint_names - frozen_joints if frozen_joints else animated_joint_names
        link_T = forward_kinematics_selective(
            links, joints, parent_map, children_map,
            frame, num_frames, active_for_fk,
        )

        # Apply transforms ONLY to animated (moving) parts
        for link_name, part_idx in animated_links.items():
            if part_idx not in part_objects:
                continue

            obj = part_objects[part_idx]

            if link_name not in link_T:
                continue

            T_current = link_T[link_name]
            T_rest = rest_transforms.get(link_name, Matrix.Identity(4))

            try:
                T_rest_inv = T_rest.inverted()
            except:
                T_rest_inv = Matrix.Identity(4)

            delta = T_current @ T_rest_inv

            obj.matrix_world = delta

            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            obj.keyframe_insert(data_path="scale", frame=frame)

    # Smooth interpolation
    for part_idx, obj in part_objects.items():
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

    print(f"  Animation set for {len(animated_links)} moving parts")
    return animated_joint_names


# ═══════════════════════════════════════════════════════════════
# Bullet Rigid Body Physics for Collision Response
# ═══════════════════════════════════════════════════════════════

def setup_rigid_body_physics(part_objects, links, joints, parent_map, children_map,
                              animated_joint_names, num_frames):
    """Setup Bullet rigid body physics so animated parts push non-animated parts.

    When >=2 movable joints and some are not animated (e.g. door in prismatic
    mode), Bullet simulates collision response: drawers push the door open
    via its hinge constraint.

    Key principle (Bullet docs): object origins must be at joint pivot points
    for correct constraint behavior. After load_scene_parts() all origins are
    at (0,0,0) with vertices in world space. We relocate ACTIVE objects'
    origins to their joint pivot before adding rigid bodies and constraints.
    """
    import math as _math

    # --- helpers ---
    link_part_map = {}
    for ln, info in links.items():
        if info["part_idx"] is not None:
            link_part_map[ln] = info["part_idx"]

    def _descendant_parts(joint):
        """All part indices reachable from joint's child via fixed joints."""
        parts = []
        q = deque([joint.child_link])
        vis = set()
        while q:
            ln = q.popleft()
            if ln in vis:
                continue
            vis.add(ln)
            if ln in link_part_map:
                parts.append(link_part_map[ln])
            for cln, cj in children_map.get(ln, []):
                if cj.jtype == "fixed":
                    q.append(cln)
        return parts

    # --- classify joints & parts ---
    passive_joints = [j for j in joints if j.jtype != "fixed" and j.name not in animated_joint_names]
    animated_joints = [j for j in joints if j.jtype != "fixed" and j.name in animated_joint_names]

    if not passive_joints:
        print("  No passive joints for Bullet physics")
        return

    passive_joint_parts = {j.name: _descendant_parts(j) for j in passive_joints}

    animated_part_set = set()
    for j in animated_joints:
        animated_part_set.update(_descendant_parts(j))

    all_movable = set()
    for j in joints:
        if j.jtype != "fixed":
            all_movable.update(_descendant_parts(j))
    static_parts = set(part_objects.keys()) - all_movable

    # FK at rest (all q=0) — needed for pivot positions
    rest_fk = forward_kinematics_at_q(links, joints, parent_map, children_map)

    print(f"\n  Bullet rigid body physics:")
    print(f"    Passive joints: {[j.name for j in passive_joints]}")
    print(f"    Animated parts: {sorted(animated_part_set)}")
    print(f"    Static parts:   {sorted(static_parts)}")

    # --- ensure rigid body world exists ---
    scene = bpy.context.scene
    if scene.rigidbody_world is None:
        bpy.ops.rigidbody.world_add()

    def _add_rb(obj):
        """Select + activate obj, then add rigid body."""
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_add()
        obj.select_set(False)

    # --- Step 1: PASSIVE KINEMATIC for animated (keyframed) parts ---
    for pi in sorted(animated_part_set):
        if pi not in part_objects:
            continue
        obj = part_objects[pi]
        _add_rb(obj)
        rb = obj.rigid_body
        rb.type = 'PASSIVE'
        rb.kinematic = True
        rb.collision_shape = 'MESH'
        rb.mesh_source = 'DEFORM'
        rb.friction = 0.5
        rb.restitution = 0.0
        rb.collision_margin = 0.001
        print(f"    Part {pi}: PASSIVE KINEMATIC (animated)")

    # --- Step 2: PASSIVE for static parts ---
    for pi in sorted(static_parts):
        if pi not in part_objects:
            continue
        obj = part_objects[pi]
        _add_rb(obj)
        rb = obj.rigid_body
        rb.type = 'PASSIVE'
        rb.kinematic = False
        rb.collision_shape = 'MESH'
        rb.mesh_source = 'DEFORM'
        rb.friction = 0.5
        rb.restitution = 0.0
        rb.collision_margin = 0.001
        print(f"    Part {pi}: PASSIVE (static)")

    # Pick anchor for constraints (first static part, fallback to animated)
    anchor_obj = None
    for pi in sorted(static_parts):
        if pi in part_objects and part_objects[pi].rigid_body:
            anchor_obj = part_objects[pi]
            break
    if anchor_obj is None:
        for pi in sorted(animated_part_set):
            if pi in part_objects and part_objects[pi].rigid_body:
                anchor_obj = part_objects[pi]
                break
    if anchor_obj is None:
        # Create tiny invisible anchor
        bpy.ops.mesh.primitive_cube_add(size=0.001, location=(0, 0, 0))
        anchor_obj = bpy.context.active_object
        anchor_obj.name = "physics_anchor"
        anchor_obj.hide_render = True
        _add_rb(anchor_obj)
        anchor_obj.rigid_body.type = 'PASSIVE'
        anchor_obj.rigid_body.collision_shape = 'BOX'

    print(f"    Anchor: {anchor_obj.name}")

    # --- Step 3: ACTIVE for each passive joint's parts + constraint ---
    for j in passive_joints:
        child_parts = passive_joint_parts[j.name]
        if not child_parts:
            continue

        # World-space pivot = parent_transform @ joint_origin
        parent_T = rest_fk.get(j.parent_link, Matrix.Identity(4))
        pivot_world = parent_T @ Vector(j.origin_xyz)

        # Joint axis in world space
        axis_world = (parent_T.to_3x3() @ Vector(j.axis).normalized()).normalized()

        print(f"    Joint {j.name} ({j.jtype}):")
        print(f"      Pivot:  [{pivot_world.x:.4f}, {pivot_world.y:.4f}, {pivot_world.z:.4f}]")
        print(f"      Axis:   [{axis_world.x:.4f}, {axis_world.y:.4f}, {axis_world.z:.4f}]")
        print(f"      Limits: [{j.lower:.4f}, {j.upper:.4f}]")

        # Relocate each descendant part's origin to pivot
        for cpi in child_parts:
            if cpi not in part_objects:
                continue
            obj = part_objects[cpi]
            # Clear keyframes set by animate_parts() — ACTIVE rigid body is
            # driven by physics, not animation.  Old keyframes would pull the
            # object back to location=(0,0,0) and fight the physics sim.
            obj.animation_data_clear()
            mesh = obj.data
            # Shift all vertices so they're relative to pivot
            for v in mesh.vertices:
                v.co -= pivot_world
            mesh.update()
            # Set Blender object location to pivot (origin = pivot)
            obj.location = pivot_world

        # If multiple parts under this joint, join into one rigid body
        objs_to_join = [part_objects[cpi] for cpi in child_parts if cpi in part_objects]
        if len(objs_to_join) > 1:
            bpy.ops.object.select_all(action='DESELECT')
            for o in objs_to_join:
                o.select_set(True)
            bpy.context.view_layer.objects.active = objs_to_join[0]
            bpy.ops.object.join()
            physics_obj = objs_to_join[0]
            # Update part_objects mapping
            for cpi in child_parts:
                if cpi in part_objects:
                    part_objects[cpi] = physics_obj
            print(f"      Joined {len(objs_to_join)} parts into {physics_obj.name}")
        else:
            physics_obj = objs_to_join[0] if objs_to_join else None

        if physics_obj is None:
            continue

        # Add ACTIVE rigid body
        _add_rb(physics_obj)
        rb = physics_obj.rigid_body
        rb.type = 'ACTIVE'
        rb.mass = 1.0
        rb.collision_shape = 'CONVEX_HULL'  # more stable than MESH for dynamic objects
        rb.friction = 0.0       # zero friction: only normal push, no tangential drag-back
        rb.restitution = 0.0
        rb.linear_damping = 0.8
        rb.angular_damping = 0.8   # moderate: door slows after push but stays open
        rb.collision_margin = 0.001
        rb.use_deactivation = False
        print(f"      Parts {child_parts}: ACTIVE, origin at pivot")

        # Determine constraint type
        if j.jtype in ("revolute", "continuous"):
            constraint_type = 'HINGE'
        elif j.jtype == "prismatic":
            constraint_type = 'SLIDER'
        else:
            continue

        # Create constraint Empty at pivot
        empty = bpy.data.objects.new(f"rbc_{j.name}", None)
        bpy.context.collection.objects.link(empty)
        empty.location = pivot_world
        empty.empty_display_size = 0.05

        # Orient Empty: HINGE uses Z-axis, SLIDER uses X-axis
        if constraint_type == 'HINGE':
            default_dir = Vector((0, 0, 1))
        else:
            default_dir = Vector((1, 0, 0))

        cross = default_dir.cross(axis_world)
        if cross.length > 1e-6:
            rot_quat = default_dir.rotation_difference(axis_world)
            empty.rotation_euler = rot_quat.to_euler()
        elif default_dir.dot(axis_world) < 0:
            # Anti-parallel: 180-degree flip
            if constraint_type == 'HINGE':
                empty.rotation_euler = (_math.pi, 0, 0)
            else:
                empty.rotation_euler = (0, 0, _math.pi)

        # Add rigid body constraint
        bpy.ops.object.select_all(action='DESELECT')
        empty.select_set(True)
        bpy.context.view_layer.objects.active = empty
        bpy.ops.rigidbody.constraint_add()
        empty.select_set(False)

        rbc = empty.rigid_body_constraint
        rbc.type = constraint_type
        rbc.object1 = anchor_obj
        rbc.object2 = physics_obj
        rbc.use_breaking = False
        rbc.disable_collisions = False

        # Set limits
        if constraint_type == 'HINGE':
            rbc.use_limit_ang_z = True
            rbc.limit_ang_z_lower = j.lower
            rbc.limit_ang_z_upper = j.upper
        elif constraint_type == 'SLIDER':
            rbc.use_limit_lin_x = True
            rbc.limit_lin_x_lower = j.lower
            rbc.limit_lin_x_upper = j.upper

        print(f"      Constraint: {constraint_type}, limits=[{j.lower:.4f}, {j.upper:.4f}]")

    # --- Step 4: Simulation parameters ---
    # High precision: more substeps = smaller timestep = less constraint drift
    scene.gravity = (0, 0, 0)  # no gravity — only collision response
    rw = scene.rigidbody_world
    rw.substeps_per_frame = 120   # 120 substeps @ 30fps = 3600 steps/sec
    rw.solver_iterations = 200    # more iterations per substep for tighter constraints
    rw.point_cache.frame_start = 1
    rw.point_cache.frame_end = num_frames

    # Bake physics
    print(f"    Baking physics ({num_frames} frames)...")
    bpy.ops.ptcache.bake({"point_cache": rw.point_cache}, bake=True)
    print(f"    Physics bake done")

    # --- Step 5: Post-process — enforce monotonic opening ---
    # PASSIVE KINEMATIC drawers ignore collision (follow keyframes exactly).
    # When retracting, they sweep through the open door from behind and push
    # it closed.  Fix: read baked rotation per frame, clamp to running max,
    # then re-keyframe the object without rigid body.
    for j in passive_joints:
        child_parts = passive_joint_parts[j.name]
        objs = [part_objects[cpi] for cpi in child_parts if cpi in part_objects]
        if not objs:
            continue
        physics_obj = objs[0]

        # Read baked transforms
        baked_matrices = []
        for frame in range(1, num_frames + 1):
            bpy.context.scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = physics_obj.evaluated_get(depsgraph)
            baked_matrices.append(eval_obj.matrix_world.copy())

        # Compute monotonic-max rotation angle around hinge axis
        # For revolute joints the rotation is around the joint axis.
        # We decompose matrix → (location, euler) relative to pivot.
        pivot = physics_obj.location.copy()  # origin was set to pivot
        max_angles = {}   # axis_index -> running max
        clamped_eulers = []

        for mat in baked_matrices:
            euler = mat.to_euler()
            clamped = list(euler)
            for ax_i in range(3):
                prev_max = max_angles.get(ax_i, 0.0)
                if abs(clamped[ax_i]) > abs(prev_max):
                    max_angles[ax_i] = clamped[ax_i]
                else:
                    clamped[ax_i] = prev_max
            clamped_eulers.append(clamped)

        # Free physics cache and remove rigid body from this object
        bpy.ops.ptcache.free_bake({"point_cache": rw.point_cache})
        bpy.context.view_layer.objects.active = physics_obj
        physics_obj.select_set(True)
        bpy.ops.rigidbody.object_remove()
        physics_obj.select_set(False)

        # Re-keyframe with clamped rotation (location stays at pivot)
        for frame_i, euler_vals in enumerate(clamped_eulers):
            frame = frame_i + 1
            physics_obj.location = pivot
            physics_obj.rotation_euler = Euler(euler_vals)
            physics_obj.keyframe_insert(data_path="location", frame=frame)
            physics_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

        # Linear interpolation
        if physics_obj.animation_data and physics_obj.animation_data.action:
            for fc in physics_obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

        print(f"    Post-process {j.name}: clamped to monotonic opening")

    print(f"    Bullet physics setup complete")


# ═══════════════════════════════════════════════════════════════
# Kinematic Collision Response (BVHTree + state memory)
# ═══════════════════════════════════════════════════════════════

def setup_collision_response(part_objects, links, joints, parent_map, children_map,
                              animated_joint_names, num_frames):
    """Kinematic collision avoidance with one-way animation and anticipation.

    Three phases:
      0. Re-animate animated joints with one-way half-speed curve (no round-trip).
         sin(t * pi/2) * target: smooth 0 → target over full duration.
      1. Raw collision detection with state memory — binary-search per frame for
         the minimum passive-joint angle that clears collision.  q never decreases.
      2. Anticipation smoothing — spread each angle jump backward over LEAD frames
         so passive parts begin opening before the collision frame.
      3. Keyframe passive parts with the smoothed trajectory.

    Deterministic, no solver drift, generalises to any joint type.
    """
    import time as _time
    from mathutils.bvhtree import BVHTree

    t0 = _time.time()

    # ── Setup maps ──
    link_part_map = {}
    for ln, info in links.items():
        if info["part_idx"] is not None:
            link_part_map[ln] = info["part_idx"]
    part_link_map = {pi: ln for ln, pi in link_part_map.items()}

    def _descendant_parts(joint):
        parts = []
        q = deque([joint.child_link])
        vis = set()
        while q:
            ln = q.popleft()
            if ln in vis:
                continue
            vis.add(ln)
            if ln in link_part_map:
                parts.append(link_part_map[ln])
            for cln, cj in children_map.get(ln, []):
                if cj.jtype == "fixed":
                    q.append(cln)
        return parts

    passive_joints = [j for j in joints if j.jtype != "fixed" and j.name not in animated_joint_names]
    if not passive_joints:
        print("  No passive joints — collision response skipped")
        return

    passive_joint_parts = {j.name: _descendant_parts(j) for j in passive_joints}

    animated_joints = [j for j in joints if j.name in animated_joint_names and j.jtype != "fixed"]
    animated_part_indices = set()
    for j in animated_joints:
        animated_part_indices.update(_descendant_parts(j))

    rest_fk = forward_kinematics_at_q(links, joints, parent_map, children_map)

    print(f"\n  Kinematic collision response (anticipation + one-way):")
    print(f"    Passive joints: {[j.name for j in passive_joints]}")
    print(f"    Animated joints: {[j.name for j in animated_joints]}")
    print(f"    Animated parts: {sorted(animated_part_indices)}")

    # ── Phase 0: Re-animate with one-way half-speed curve ──
    # Replace round-trip sin(t*pi) with one-way sin(t*pi/2)
    print(f"    Phase 0: Re-animating {len(animated_joints)} joints with one-way curve...")

    for api in animated_part_indices:
        if api not in part_objects:
            continue
        obj = part_objects[api]
        if obj.animation_data:
            obj.animation_data_clear()

    for frame in range(1, num_frames + 1):
        bpy.context.scene.frame_set(frame)
        t = (frame - 1) / max(num_frames - 1, 1)

        q_anim = {}
        for aj in animated_joints:
            if aj.jtype in ("prismatic", "revolute"):
                tgt = aj.upper if abs(aj.upper) >= abs(aj.lower) else aj.lower
            elif aj.jtype == "continuous":
                tgt = (aj.upper if abs(aj.upper) >= abs(aj.lower) else aj.lower) \
                      if abs(aj.upper - aj.lower) > 1e-6 else 1.0
            else:
                continue
            q_anim[aj.name] = math.sin(t * math.pi / 2) * tgt

        fk = forward_kinematics_at_q(links, joints, parent_map, children_map, q_anim)

        for api in animated_part_indices:
            if api not in part_objects:
                continue
            ln = part_link_map.get(api)
            if not ln:
                continue
            T_cur = fk.get(ln, Matrix.Identity(4))
            T_rest = rest_fk.get(ln, Matrix.Identity(4))
            try:
                delta = T_cur @ T_rest.inverted()
            except Exception:
                delta = Matrix.Identity(4)
            obj = part_objects[api]
            obj.matrix_world = delta
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            obj.keyframe_insert(data_path="scale", frame=frame)

    for api in animated_part_indices:
        if api not in part_objects:
            continue
        obj = part_objects[api]
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

    # ── Pre-cache passive mesh data ──
    passive_mesh_cache = {}
    for j in passive_joints:
        for cpi in passive_joint_parts[j.name]:
            if cpi not in part_objects:
                continue
            mesh = part_objects[cpi].data
            passive_mesh_cache[cpi] = (
                [Vector(v.co) for v in mesh.vertices],
                [list(p.vertices) for p in mesh.polygons],
            )

    LEAD_FRAMES = 10   # anticipation: spread jumps backward over this many frames

    # ── Phase 1+2+3 per passive joint ──
    for j in passive_joints:
        child_parts = passive_joint_parts[j.name]
        if not child_parts:
            continue

        target = j.upper if abs(j.upper) >= abs(j.lower) else j.lower
        print(f"    {j.name} ({j.jtype}): target={target:.4f}, parts={child_parts}")

        # ── Compute adaptive PROXIMITY from rest-state distance ──
        # At rest (frame 1, q=0), measure minimum distance between animated
        # and passive meshes. Set PROXIMITY to 80% of rest distance so that
        # collision is only flagged when the gap is mostly closed.
        bpy.context.scene.frame_set(1)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        animated_rest_verts = []
        for api in animated_part_indices:
            if api not in part_objects:
                continue
            aobj = part_objects[api].evaluated_get(depsgraph)
            for v in aobj.data.vertices:
                animated_rest_verts.append(aobj.matrix_world @ v.co)

        rest_min_dist = float('inf')
        for cpi in child_parts:
            if cpi not in passive_mesh_cache:
                continue
            verts, polys = passive_mesh_cache[cpi]
            pbvh = BVHTree.FromPolygons(verts, polys)   # passive at rest (q=0)
            for av in animated_rest_verts:
                result = pbvh.find_nearest(av)
                if result[3] is not None:
                    rest_min_dist = min(rest_min_dist, result[3])

        if rest_min_dist < 0.005:
            PROXIMITY = 0.002   # objects nearly overlap at rest — tiny threshold
        else:
            PROXIMITY = min(rest_min_dist * 0.15, 0.01)   # 15% of rest gap, max 1cm
        print(f"      Rest min distance: {rest_min_dist:.4f}, adaptive PROXIMITY: {PROXIMITY:.4f}")

        # ── Phase 1: Raw collision detection with state memory ──
        q_raw = [0.0] * (num_frames + 1)   # 1-indexed
        q_current = 0.0
        collision_frames = 0

        for frame in range(1, num_frames + 1):
            bpy.context.scene.frame_set(frame)
            depsgraph = bpy.context.evaluated_depsgraph_get()

            # Cache animated verts for this frame (one-way keyframes)
            animated_world_verts = []
            for api in animated_part_indices:
                if api not in part_objects:
                    continue
                aobj = part_objects[api].evaluated_get(depsgraph)
                mat_w = aobj.matrix_world
                for v in aobj.data.vertices:
                    animated_world_verts.append(mat_w @ v.co)

            def _check_collision(q_val):
                q_values = {j.name: q_val}
                fk_q = forward_kinematics_at_q(links, joints, parent_map, children_map, q_values)
                for cpi in child_parts:
                    if cpi not in passive_mesh_cache:
                        continue
                    ln = part_link_map.get(cpi)
                    if not ln:
                        continue
                    T_cur = fk_q.get(ln, Matrix.Identity(4))
                    T_rest = rest_fk.get(ln, Matrix.Identity(4))
                    try:
                        delta = T_cur @ T_rest.inverted()
                    except Exception:
                        delta = Matrix.Identity(4)
                    verts, polys = passive_mesh_cache[cpi]
                    moved = [delta @ v for v in verts]
                    pbvh = BVHTree.FromPolygons(moved, polys)
                    for av in animated_world_verts:
                        result = pbvh.find_nearest(av)
                        if result[3] is not None and result[3] < PROXIMITY:
                            return True
                return False

            if not _check_collision(q_current):
                q_raw[frame] = q_current
            else:
                collision_frames += 1
                q_lo = q_current
                q_hi = target
                for _ in range(12):
                    q_mid = (q_lo + q_hi) / 2.0
                    if _check_collision(q_mid):
                        q_lo = q_mid
                    else:
                        q_hi = q_mid
                q_safe = q_hi + abs(target) * 0.02   # small clearance
                if target > 0:
                    q_safe = min(q_safe, target)
                else:
                    q_safe = max(q_safe, target)
                q_current = q_safe
                q_raw[frame] = q_current

        print(f"      Phase 1: {collision_frames}/{num_frames} collisions, final q={q_current:.4f}")

        # ── Phase 2: Anticipation smoothing ──
        # Find every frame where q_raw jumps, spread the increase backward.
        q_smooth = list(q_raw)

        jumps = []
        for f in range(2, num_frames + 1):
            if q_raw[f] > q_raw[f - 1] + 1e-6:
                jumps.append((f, q_raw[f - 1], q_raw[f]))

        for jump_frame, q_before, q_after in jumps:
            lead_start = max(1, jump_frame - LEAD_FRAMES)
            span = max(jump_frame - lead_start, 1)
            for f in range(lead_start, jump_frame):
                frac = (f - lead_start) / span
                interp = q_before + (q_after - q_before) * math.sin(frac * math.pi / 2)
                q_smooth[f] = max(q_smooth[f], interp)

        # Enforce monotonicity
        for f in range(2, num_frames + 1):
            q_smooth[f] = max(q_smooth[f], q_smooth[f - 1])

        lead_start_first = max(1, jumps[0][0] - LEAD_FRAMES) if jumps else None

        print(f"      Phase 2: anticipation from frame {lead_start_first or 'N/A'}, "
              f"final q={q_smooth[num_frames]:.4f}")

        # ── Phase 3: Keyframe passive parts (every frame for clean interpolation) ──
        for frame in range(1, num_frames + 1):
            q_val = q_smooth[frame]
            bpy.context.scene.frame_set(frame)

            if abs(q_val) < 1e-6:
                # Identity — rest pose
                for cpi in child_parts:
                    if cpi not in part_objects:
                        continue
                    obj = part_objects[cpi]
                    obj.matrix_world = Matrix.Identity(4)
                    obj.keyframe_insert(data_path="location", frame=frame)
                    obj.keyframe_insert(data_path="rotation_euler", frame=frame)
                    obj.keyframe_insert(data_path="scale", frame=frame)
            else:
                q_values = {j.name: q_val}
                fk_q = forward_kinematics_at_q(links, joints, parent_map, children_map, q_values)
                for cpi in child_parts:
                    if cpi not in part_objects:
                        continue
                    ln = part_link_map.get(cpi)
                    if not ln:
                        continue
                    T_cur = fk_q.get(ln, Matrix.Identity(4))
                    T_rest = rest_fk.get(ln, Matrix.Identity(4))
                    try:
                        delta = T_cur @ T_rest.inverted()
                    except Exception:
                        delta = Matrix.Identity(4)
                    obj = part_objects[cpi]
                    obj.matrix_world = delta
                    obj.keyframe_insert(data_path="location", frame=frame)
                    obj.keyframe_insert(data_path="rotation_euler", frame=frame)
                    obj.keyframe_insert(data_path="scale", frame=frame)

        # Linear interpolation for passive parts
        for cpi in child_parts:
            if cpi not in part_objects:
                continue
            obj = part_objects[cpi]
            if obj.animation_data and obj.animation_data.action:
                for fc in obj.animation_data.action.fcurves:
                    for kp in fc.keyframe_points:
                        kp.interpolation = 'LINEAR'

        print(f"      Phase 3: keyframed {num_frames} frames for parts {child_parts}")

    elapsed = _time.time() - t0
    print(f"  Collision response done ({elapsed:.2f}s)")


def animate_flip(part_objects, moving_indices, num_frames):
    """Custom animation: lid flips 180° in place (round-trip).

    Lid lifts, rotates 180° showing underside, then returns to rest.
    """
    print(f"\nCustom flip (round-trip) animation for {num_frames} frames...")

    moving_objs = [part_objects[i] for i in (moving_indices or part_objects.keys())
                   if i in part_objects]
    if not moving_objs:
        print("  WARNING: no moving objects for flip")
        return

    all_coords = []
    for obj in moving_objs:
        for v in obj.data.vertices:
            all_coords.append(obj.matrix_world @ v.co)
    if not all_coords:
        return

    min_x = min(v.x for v in all_coords)
    max_x = max(v.x for v in all_coords)
    min_y = min(v.y for v in all_coords)
    max_y = max(v.y for v in all_coords)
    min_z = min(v.z for v in all_coords)
    max_z = max(v.z for v in all_coords)

    lid_height = max_z - min_z
    center_z = (min_z + max_z) / 2.0
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    pivot = Vector((center_x, center_y, center_z))
    lift_amount = lid_height * 1.5

    print(f"  Lid bbox: x=[{min_x:.3f},{max_x:.3f}] z=[{min_z:.3f},{max_z:.3f}]")
    print(f"  Pivot: [{pivot.x:.3f}, {pivot.y:.3f}, {pivot.z:.3f}], Lift: {lift_amount:.3f}")

    for frame in range(1, num_frames + 1):
        bpy.context.scene.frame_set(frame)
        t = (frame - 1) / max(num_frames - 1, 1)
        phase = math.sin(t * math.pi)  # 0 → 1 → 0

        lift = Matrix.Translation(Vector((0, 0, lift_amount * phase)))
        angle = math.pi * phase
        to_pivot = Matrix.Translation(-pivot)
        rot = Matrix.Rotation(angle, 4, 'Y')
        from_pivot = Matrix.Translation(pivot)
        flip = from_pivot @ rot @ to_pivot
        transform = lift @ flip

        for obj in moving_objs:
            obj.matrix_world = transform
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            obj.keyframe_insert(data_path="scale", frame=frame)

    for obj in moving_objs:
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

    print(f"  Flip (round-trip) animation set for {len(moving_objs)} parts")


def animate_flip_place(part_objects, moving_indices, num_frames):
    """Custom animation: lid lifts, flips 180°, and is placed inverted beside the pot.

    One-way motion: lid ends up upside-down next to the pot body.
    """
    print(f"\nCustom flip animation for {num_frames} frames...")

    # Collect all moving part objects
    moving_objs = [part_objects[i] for i in (moving_indices or part_objects.keys())
                   if i in part_objects]
    if not moving_objs:
        print("  WARNING: no moving objects for flip")
        return

    # Compute bounding box of moving parts (lid)
    all_coords = []
    for obj in moving_objs:
        for v in obj.data.vertices:
            co = obj.matrix_world @ v.co
            all_coords.append(co)

    if not all_coords:
        return

    min_x = min(v.x for v in all_coords)
    max_x = max(v.x for v in all_coords)
    min_y = min(v.y for v in all_coords)
    max_y = max(v.y for v in all_coords)
    min_z = min(v.z for v in all_coords)
    max_z = max(v.z for v in all_coords)

    lid_height = max_z - min_z
    center_z = (min_z + max_z) / 2.0
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    rest_pos = Vector((center_x, center_y, center_z))

    # Final position: inverted, placed to the right (-Y for front camera view)
    lid_radius = max(max_x - min_x, max_y - min_y) / 2.0
    side_dist = lid_radius * 2.5  # clear of pot body
    final_z = lid_height / 2.0    # inverted lid center height on ground
    final_pos = Vector((center_x, center_y - side_dist, final_z))

    # Arc height for clearance during flip
    arc_height = lid_radius

    def smoothstep(edge0, edge1, x):
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)

    print(f"  Lid bbox: x=[{min_x:.3f},{max_x:.3f}] z=[{min_z:.3f},{max_z:.3f}]")
    print(f"  Rest: [{rest_pos.x:.3f}, {rest_pos.y:.3f}, {rest_pos.z:.3f}]")
    print(f"  Final: [{final_pos.x:.3f}, {final_pos.y:.3f}, {final_pos.z:.3f}]")
    print(f"  Arc height: {arc_height:.3f}, side_dist: {side_dist:.3f}")

    for frame in range(1, num_frames + 1):
        bpy.context.scene.frame_set(frame)
        t = (frame - 1) / max(num_frames - 1, 1)

        # Rotation: 0 → 180° (flip around Y)
        rot_t = smoothstep(0.05, 0.65, t)
        angle = math.pi * rot_t

        # Horizontal displacement: slide to the side
        slide_t = smoothstep(0.15, 0.85, t)
        offset_y = -side_dist * slide_t

        # Vertical: arc up then descend to final_z
        # sin arc for clearance + linear descent to final height
        z_arc = arc_height * math.sin(math.pi * t)
        z_descent = (final_z - center_z) * smoothstep(0.3, 1.0, t)
        z_offset = z_arc + z_descent

        # Transform: translate to origin, rotate, translate to target
        target_pos = rest_pos + Vector((0, offset_y, z_offset))
        to_origin = Matrix.Translation(-rest_pos)
        rot = Matrix.Rotation(angle, 4, 'Y')
        to_target = Matrix.Translation(target_pos)
        transform = to_target @ rot @ to_origin

        for obj in moving_objs:
            obj.matrix_world = transform
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            obj.keyframe_insert(data_path="scale", frame=frame)

    # Linear interpolation
    for obj in moving_objs:
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

    print(f"  Flip animation set for {len(moving_objs)} parts")


def animate_cap_detach(part_objects, moving_indices, num_frames):
    """Custom animation: cap unscrews and flies off the bottle/pot.

    One-way motion: cap rotates (unscrewing) while lifting upward,
    then separates completely from the body.
    """
    print(f"\nCustom cap_detach animation for {num_frames} frames...")

    moving_objs = [part_objects[i] for i in (moving_indices or part_objects.keys())
                   if i in part_objects]
    body_objs = [part_objects[i] for i in part_objects.keys()
                 if i not in (moving_indices or set())]
    if not moving_objs:
        print("  WARNING: no moving objects for cap_detach")
        return

    # Bounding box of cap (moving parts)
    cap_coords = []
    for obj in moving_objs:
        for v in obj.data.vertices:
            cap_coords.append(obj.matrix_world @ v.co)
    if not cap_coords:
        return

    cap_min_z = min(v.z for v in cap_coords)
    cap_max_z = max(v.z for v in cap_coords)
    cap_cx = (min(v.x for v in cap_coords) + max(v.x for v in cap_coords)) / 2.0
    cap_cy = (min(v.y for v in cap_coords) + max(v.y for v in cap_coords)) / 2.0
    cap_cz = (cap_min_z + cap_max_z) / 2.0
    cap_height = cap_max_z - cap_min_z
    pivot = Vector((cap_cx, cap_cy, cap_cz))

    # Bounding box of body to determine fly-off distance
    body_coords = []
    for obj in body_objs:
        for v in obj.data.vertices:
            body_coords.append(obj.matrix_world @ v.co)
    if body_coords:
        body_max_z = max(v.z for v in body_coords)
        body_extent = max(
            max(v.x for v in body_coords) - min(v.x for v in body_coords),
            max(v.y for v in body_coords) - min(v.y for v in body_coords),
            max(v.z for v in body_coords) - min(v.z for v in body_coords),
        )
    else:
        body_max_z = cap_max_z
        body_extent = cap_height * 5

    # Fly-off parameters
    lift_distance = body_extent * 1.2   # fly up 1.2x body height
    n_rotations = 3.0                   # 3 full unscrewing turns
    total_rotation = n_rotations * 2 * math.pi

    print(f"  Cap bbox z=[{cap_min_z:.3f},{cap_max_z:.3f}], height={cap_height:.3f}")
    print(f"  Pivot: [{pivot.x:.3f}, {pivot.y:.3f}, {pivot.z:.3f}]")
    print(f"  Lift distance: {lift_distance:.3f}, Rotations: {n_rotations}")

    def smoothstep(edge0, edge1, x):
        t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
        return t * t * (3.0 - 2.0 * t)

    for frame in range(1, num_frames + 1):
        bpy.context.scene.frame_set(frame)
        t = (frame - 1) / max(num_frames - 1, 1)

        # Rotation: accelerating unscrew (starts slow, speeds up)
        rot_progress = smoothstep(0.0, 0.8, t)
        angle = total_rotation * rot_progress

        # Vertical lift: starts after initial unscrew, accelerates
        lift_progress = smoothstep(0.1, 1.0, t)
        z_offset = lift_distance * lift_progress

        # Build transform: rotate around Z at pivot, then lift upward
        to_pivot = Matrix.Translation(-pivot)
        rot = Matrix.Rotation(angle, 4, 'Z')
        from_pivot = Matrix.Translation(pivot)
        lift = Matrix.Translation(Vector((0, 0, z_offset)))
        transform = lift @ from_pivot @ rot @ to_pivot

        for obj in moving_objs:
            obj.matrix_world = transform
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            obj.keyframe_insert(data_path="scale", frame=frame)

    # Linear interpolation
    for obj in moving_objs:
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

    print(f"  Cap detach animation set for {len(moving_objs)} parts")


def forward_kinematics_selective(links, joints, parent_map, children_map,
                                  frame, num_frames, animated_joint_names):
    """Compute FK where only joints in animated_joint_names get nonzero q values.

    Joints NOT in animated_joint_names are treated as q=0 (rest position).
    This prevents body-to-body joints from animating.
    """
    link_transforms = {}

    all_children = set(j.child_link for j in joints)
    all_parents = set(j.parent_link for j in joints)
    roots = all_parents - all_children

    for r in roots:
        link_transforms[r] = Matrix.Identity(4)

    queue = deque(roots)
    visited = set()

    while queue:
        link_name = queue.popleft()
        if link_name in visited:
            continue
        visited.add(link_name)

        parent_T = link_transforms.get(link_name, Matrix.Identity(4))

        for child_link, joint in children_map.get(link_name, []):
            # Only animate joints on paths to moving parts
            if joint.name in animated_joint_names:
                q = compute_joint_value(joint, frame, num_frames)
            else:
                q = 0.0

            T_local = compute_joint_local_transform(joint, q)
            child_T = parent_T @ T_local
            link_transforms[child_link] = child_T

            queue.append(child_link)

    return link_transforms


def forward_kinematics_at_q(links, joints, parent_map, children_map, q_values=None):
    """Compute FK with specified q values (or all zeros if None).

    q_values: dict mapping joint_name -> q_value. If None, all q = 0.
    """
    link_transforms = {}

    all_children = set(j.child_link for j in joints)
    all_parents = set(j.parent_link for j in joints)
    roots = all_parents - all_children

    for r in roots:
        link_transforms[r] = Matrix.Identity(4)

    queue = deque(roots)
    visited = set()

    while queue:
        link_name = queue.popleft()
        if link_name in visited:
            continue
        visited.add(link_name)

        parent_T = link_transforms.get(link_name, Matrix.Identity(4))

        for child_link, joint in children_map.get(link_name, []):
            q = 0.0
            if q_values and joint.name in q_values:
                q = q_values[joint.name]

            T_local = compute_joint_local_transform(joint, q)
            child_T = parent_T @ T_local
            link_transforms[child_link] = child_T

            queue.append(child_link)

    return link_transforms



# ═══════════════════════════════════════════════════════════════
# Rendering
# ═══════════════════════════════════════════════════════════════

def setup_compositor_dual_output(bg_dir):
    """Set up compositor to output bg version via File Output node.

    Renders with film_transparent=True. The nobg RGBA output goes through
    the Composite node (saved to scene.render.filepath). The bg version
    composites the Environment pass behind the transparent render via
    Alpha Over, output through a File Output node.
    Single render pass produces both outputs — no double rendering.
    """
    scene = bpy.context.scene
    scene.render.film_transparent = True

    # Enable environment pass for background compositing
    bpy.context.view_layer.use_pass_environment = True

    scene.use_nodes = True
    tree = scene.node_tree
    nodes = tree.nodes
    links = tree.links

    for n in nodes:
        nodes.remove(n)

    # Render Layers
    rl = nodes.new('CompositorNodeRLayers')
    rl.location = (0, 0)

    # Composite node -> nobg output (saved to scene.render.filepath)
    composite = nodes.new('CompositorNodeComposite')
    composite.location = (600, 0)
    links.new(rl.outputs['Image'], composite.inputs['Image'])

    # Alpha Over: environment background + transparent render foreground
    alpha_over = nodes.new('CompositorNodeAlphaOver')
    alpha_over.location = (300, 200)
    links.new(rl.outputs['Env'], alpha_over.inputs[1])   # background
    links.new(rl.outputs['Image'], alpha_over.inputs[2])  # foreground

    # File Output node -> bg output
    bg_out = nodes.new('CompositorNodeOutputFile')
    bg_out.base_path = bg_dir
    bg_out.format.file_format = 'PNG'
    bg_out.format.color_mode = 'RGBA'
    bg_out.file_slots[0].path = "frame_"
    bg_out.location = (600, 200)
    links.new(alpha_over.outputs['Image'], bg_out.inputs[0])


def cleanup_compositor():
    """Reset compositor to default state."""
    scene = bpy.context.scene
    scene.use_nodes = False
    bpy.context.view_layer.use_pass_environment = False


def render_view(out_dir, view_name, num_frames, center, distance, elev_deg, azim_deg,
                render_bg=True, render_nobg=True, animode_suffix=""):
    """Render a single view, outputting bg and/or nobg in one render pass.

    When both are needed, uses compositor to produce both from a single render.
    This halves the render time compared to rendering bg and nobg separately.
    """
    if not render_bg and not render_nobg:
        return None, None

    vname = f"{view_name}{animode_suffix}"
    nobg_dir = os.path.join(out_dir, f"{vname}_nobg") if render_nobg else None
    bg_dir = os.path.join(out_dir, f"{vname}_bg") if render_bg else None

    if nobg_dir:
        os.makedirs(nobg_dir, exist_ok=True)
    if bg_dir:
        os.makedirs(bg_dir, exist_ok=True)

    scene = bpy.context.scene

    if render_bg and render_nobg:
        # Single render: nobg via Composite, bg via File Output + Alpha Over
        setup_compositor_dual_output(bg_dir)
        scene.render.filepath = os.path.join(nobg_dir, "frame_")
    elif render_nobg:
        scene.render.film_transparent = True
        scene.use_nodes = False
        scene.render.filepath = os.path.join(nobg_dir, "frame_")
    else:  # only bg
        scene.render.film_transparent = False
        scene.use_nodes = False
        scene.render.filepath = os.path.join(bg_dir, "frame_")

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    cam = create_camera(f"cam_{vname}", center, distance, elev_deg, azim_deg)
    scene.camera = cam

    mode = "bg+nobg" if (render_bg and render_nobg) else ("nobg" if render_nobg else "bg")
    print(f"\n  Rendering {vname} ({mode}): {num_frames} frames")
    bpy.ops.render.render(animation=True)

    bpy.data.objects.remove(cam, do_unlink=True)
    if render_bg and render_nobg:
        cleanup_compositor()

    return nobg_dir, bg_dir


def render_moving_view(out_dir, view_name, num_frames, center, distance,
                       start_elev, start_azim, end_elev, end_azim,
                       render_bg=True, render_nobg=True, animode_suffix=""):
    """Render a moving camera view that orbits from start to end position."""
    if not render_bg and not render_nobg:
        return None, None

    vname = f"{view_name}{animode_suffix}"
    nobg_dir = os.path.join(out_dir, f"{vname}_nobg") if render_nobg else None
    bg_dir = os.path.join(out_dir, f"{vname}_bg") if render_bg else None

    if nobg_dir:
        os.makedirs(nobg_dir, exist_ok=True)
    if bg_dir:
        os.makedirs(bg_dir, exist_ok=True)

    scene = bpy.context.scene

    if render_bg and render_nobg:
        setup_compositor_dual_output(bg_dir)
        scene.render.filepath = os.path.join(nobg_dir, "frame_")
    elif render_nobg:
        scene.render.film_transparent = True
        scene.use_nodes = False
        scene.render.filepath = os.path.join(nobg_dir, "frame_")
    else:
        scene.render.film_transparent = False
        scene.use_nodes = False
        scene.render.filepath = os.path.join(bg_dir, "frame_")

    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    cam = create_animated_camera(f"cam_{vname}", center, distance,
                                 start_elev, start_azim, end_elev, end_azim,
                                 num_frames)
    scene.camera = cam

    mode = "bg+nobg" if (render_bg and render_nobg) else ("nobg" if render_nobg else "bg")
    print(f"\n  Rendering {vname} (moving, {mode}): {num_frames} frames")
    print(f"    ({start_elev}°,{start_azim}°) -> ({end_elev}°,{end_azim}°)")
    bpy.ops.render.render(animation=True)

    bpy.data.objects.remove(cam, do_unlink=True)
    if render_bg and render_nobg:
        cleanup_compositor()

    return nobg_dir, bg_dir


def check_static_video(frame_dir, num_frames, iou_threshold=0.98, mse_threshold=0.001):
    """Check if rendered video is static (no meaningful motion).

    Compares first frame vs midpoint frame using:
      1. Alpha mask IoU (silhouette overlap) — catches shape/position changes
      2. RGB MSE on visible pixels — catches subtle color/lighting changes

    Returns True if the video should be considered static and deleted.
    """
    import numpy as np

    mid = max(1, num_frames // 2)
    first_path = os.path.join(frame_dir, "frame_0001.png")
    mid_path = os.path.join(frame_dir, f"frame_{mid:04d}.png")

    if not os.path.exists(first_path) or not os.path.exists(mid_path):
        return False

    img1 = bpy.data.images.load(first_path)
    img2 = bpy.data.images.load(mid_path)

    px1 = list(img1.pixels[:])
    px2 = list(img2.pixels[:])
    w, h = img1.size[0], img1.size[1]

    bpy.data.images.remove(img1)
    bpy.data.images.remove(img2)

    px1 = np.array(px1, dtype=np.float32).reshape(h * w, 4)
    px2 = np.array(px2, dtype=np.float32).reshape(h * w, 4)

    # Alpha mask IoU
    mask1 = px1[:, 3] > 0.5
    mask2 = px2[:, 3] > 0.5

    union = np.sum(mask1 | mask2)
    if union == 0:
        return True  # both frames empty → static

    iou = float(np.sum(mask1 & mask2)) / float(union)
    if iou < iou_threshold:
        return False  # silhouette changed → not static

    # RGB MSE on visible pixels
    visible = mask1 | mask2
    n_vis = int(np.sum(visible))
    if n_vis > 0:
        mse = float(np.mean((px1[visible, :3] - px2[visible, :3]) ** 2))
    else:
        mse = 0.0

    is_static = mse < mse_threshold
    if is_static:
        print(f"  STATIC detected: IoU={iou:.4f}, MSE={mse:.6f} ({frame_dir})")
    return is_static


def frames_to_video(frame_dir, output_mp4, fps):
    if not FFMPEG_BIN:
        print(f"  WARNING: ffmpeg not found, skipping video")
        return False

    frame_pattern = os.path.join(frame_dir, "frame_%04d.png")
    cmd = [
        FFMPEG_BIN, "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        output_mp4,
    ]
    print(f"  ffmpeg -> {output_mp4}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr[:500]}")
            return False
        return True
    except Exception as e:
        print(f"  ffmpeg failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"Rendering articulation: {FACTORY} seed={SEED}")
    print(f"{'='*60}")
    print(f"  URDF: {URDF_PATH}")
    print(f"  OBJs: {OBJS_DIR}")
    print(f"  Output: {OUT_DIR}")
    print(f"  Frames: {NUM_FRAMES} ({args.duration}s @ {args.fps}fps)")
    print(f"  Res: {args.resolution}, Samples: {args.samples}")
    print(f"  Animode: {args.animode}, ffmpeg: {FFMPEG_BIN}")

    for p, label in [(URDF_PATH, "URDF"), (ORIGINS_PATH, "Origins"), (OBJS_DIR, "OBJs")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} not found: {p}")
            return

    import time as _time
    _timers = {}
    def _tick(name):
        _timers[name] = _time.time()
    def _tock(name):
        elapsed = _time.time() - _timers[name]
        print(f"  [TIMER] {name}: {elapsed:.1f}s")
        _timers[name] = elapsed  # store elapsed for summary

    _tick("total")

    # Parse URDF
    _tick("parse_urdf")
    print("\nParsing URDF...")
    links, joints = parse_urdf(URDF_PATH)
    parent_map, children_map = build_kinematic_tree(joints)

    movable = [j for j in joints if j.jtype in ("revolute", "prismatic", "continuous")]
    print(f"  Links: {len(links)}, Joints: {len(joints)} ({len(movable)} movable)")
    for j in movable:
        print(f"    {j}")

    # Determine moving part indices from FACTORY_RULES + data_infos
    moving_result = get_moving_part_indices(FACTORY, SCENE_DIR)
    moving_indices = None
    exclude_indices = set()
    if moving_result is not None:
        moving_indices, exclude_indices = moving_result
    else:
        print("  No FACTORY_RULES: animating all parts")

    # Load origins
    with open(ORIGINS_PATH) as f:
        origins = json.load(f)
    _tock("parse_urdf")

    # Clear scene
    clear_scene()

    # GPU rendering
    _tick("setup_engine")
    setup_render_engine(args.engine)
    setup_render_settings(args.resolution, args.fps, NUM_FRAMES, args.samples,
                          max_bounces=args.max_bounces, engine=args.engine)

    # Envmap lighting
    setup_envmap_lighting(args.envmap, strength=1.0)
    _tock("setup_engine")

    # Load parts (skip excluded indices like unknown_part)
    _tick("load_objs")
    print("\nLoading per-part OBJs...")
    part_objects = load_scene_parts(OBJS_DIR, origins, exclude_indices=exclude_indices)
    if not part_objects:
        print("ERROR: No parts loaded!")
        return

    # Remove orphan parts: loaded from origins.json but not in URDF
    urdf_part_indices = {info["part_idx"] for info in links.values() if info["part_idx"] is not None}
    orphan_indices = set(part_objects.keys()) - urdf_part_indices
    for pidx in sorted(orphan_indices):
        obj = part_objects.pop(pidx)
        bpy.data.objects.remove(obj, do_unlink=True)
        print(f"  Removed orphan part {pidx} (in origins but not in URDF)")
    if orphan_indices:
        print(f"  {len(orphan_indices)} orphan part(s) removed")

    _tock("load_objs")

    # Enhance materials for PartNet factories (metallic, roughness, textures)
    _tick("materials")
    enhance_partnet_materials(FACTORY, args.seed, part_objects, args.base)

    # Enhance materials for PhysXNet/PhysX_mobility factories
    _enhance_physxnet_materials(FACTORY, args.seed, part_objects)
    _tock("materials")

    # Compute scene bounds at rest
    all_objs = list(part_objects.values())
    center, extent = compute_scene_bounds(all_objs)
    distance = extent * 1.8
    print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  Extent: {extent:.3f}, CamDist: {distance:.3f}")

    # Set up animation mode
    global _CURRENT_ANIMODE
    _CURRENT_ANIMODE = args.animode
    animode_suffix = f"_anim{args.animode}" if args.animode > 0 else ""
    scale = 1.0 + args.animode * 0.5 if args.animode > 0 else 1.0
    print(f"  Animode: {args.animode} (limit_scale={scale:.1f}x)")

    # Animate
    _tick("animate")
    animated_joint_names = None
    animode_cfg = FACTORY_RULES.get(FACTORY, {}).get("animode_joints", {})
    custom_anim = animode_cfg.get(args.animode)
    if custom_anim == "flip":
        animate_flip(part_objects, moving_indices, NUM_FRAMES)
    elif custom_anim == "flip_place":
        animate_flip_place(part_objects, moving_indices, NUM_FRAMES)
    elif custom_anim == "cap_detach":
        animate_cap_detach(part_objects, moving_indices, NUM_FRAMES)
    elif movable:
        anim_result = animate_parts(part_objects, links, joints, parent_map, children_map, origins, NUM_FRAMES,
                                    moving_indices=moving_indices)
        if anim_result is False:
            print("  No matching joints for this animode - exiting without rendering")
            sys.exit(0)
        animated_joint_names = anim_result
    else:
        print("  No movable joints - rendering static scene")
    _tock("animate")

    # Collision response: when >=2 movable joints and some are not animated,
    # use BVHTree kinematic collision avoidance (e.g. drawer pushes door open).
    _tick("collision")
    if animated_joint_names is not None and len(movable) >= 2:
        non_animated_movable = [j for j in movable if j.name not in animated_joint_names]
        if non_animated_movable:
            setup_collision_response(
                part_objects, links, joints, parent_map, children_map,
                animated_joint_names, NUM_FRAMES,
            )
    _tock("collision")

    # Render views (single render produces both bg and nobg via compositor)
    _tick("render_all")
    for view_name in args.views:
        if view_name not in VIEW_CONFIGS:
            print(f"  WARNING: Unknown view '{view_name}'")
            continue

        elev_deg, azim_deg = VIEW_CONFIGS[view_name]

        nobg_dir, bg_dir = render_view(
            OUT_DIR, view_name, NUM_FRAMES,
            center, distance, elev_deg, azim_deg,
            render_bg=not args.skip_bg,
            render_nobg=not args.skip_nobg,
            animode_suffix=animode_suffix,
        )

        # Static video filter: delete frames with no meaningful motion
        if nobg_dir and check_static_video(nobg_dir, NUM_FRAMES):
            shutil.rmtree(nobg_dir)
            nobg_dir = None
        if bg_dir and check_static_video(bg_dir, NUM_FRAMES):
            shutil.rmtree(bg_dir)
            bg_dir = None

        if not args.png_only:
            if nobg_dir:
                mp4_nobg = os.path.join(OUT_DIR, f"{view_name}{animode_suffix}_nobg.mp4")
                frames_to_video(nobg_dir, mp4_nobg, args.fps)
            if bg_dir:
                mp4_bg = os.path.join(OUT_DIR, f"{view_name}{animode_suffix}_bg.mp4")
                frames_to_video(bg_dir, mp4_bg, args.fps)

    # Render moving views
    for view_name in args.moving_views:
        if view_name not in MOVING_VIEW_CONFIGS:
            print(f"  WARNING: Unknown moving view '{view_name}'")
            continue

        start_elev, start_azim, end_elev, end_azim = MOVING_VIEW_CONFIGS[view_name]

        nobg_dir, bg_dir = render_moving_view(
            OUT_DIR, view_name, NUM_FRAMES,
            center, distance, start_elev, start_azim, end_elev, end_azim,
            render_bg=not args.skip_bg,
            render_nobg=not args.skip_nobg,
            animode_suffix=animode_suffix,
        )

        # Static video filter for moving views
        if nobg_dir and check_static_video(nobg_dir, NUM_FRAMES):
            shutil.rmtree(nobg_dir)
            nobg_dir = None
        if bg_dir and check_static_video(bg_dir, NUM_FRAMES):
            shutil.rmtree(bg_dir)
            bg_dir = None

        if not args.png_only:
            if nobg_dir:
                mp4_nobg = os.path.join(OUT_DIR, f"{view_name}{animode_suffix}_nobg.mp4")
                frames_to_video(nobg_dir, mp4_nobg, args.fps)
            if bg_dir:
                mp4_bg = os.path.join(OUT_DIR, f"{view_name}{animode_suffix}_bg.mp4")
                frames_to_video(bg_dir, mp4_bg, args.fps)

    _tock("render_all")
    _tock("total")

    # Print timing summary
    print(f"\n{'='*60}")
    print(f"TIMING SUMMARY ({args.engine}, bounces={args.max_bounces or 'default'}):")
    print(f"{'='*60}")
    for name in ["parse_urdf", "setup_engine", "load_objs", "materials",
                  "animate", "collision", "render_all", "total"]:
        if name in _timers and isinstance(_timers[name], float):
            pct = _timers[name] / _timers["total"] * 100 if _timers["total"] > 0 else 0
            print(f"  {name:20s}: {_timers[name]:8.1f}s  ({pct:5.1f}%)")
    n_views = len(args.views) + len(args.moving_views)
    if n_views > 0 and "render_all" in _timers:
        print(f"  {'per_view':20s}: {_timers['render_all']/n_views:8.1f}s  ({NUM_FRAMES} frames)")
        print(f"  {'per_frame':20s}: {_timers['render_all']/n_views/NUM_FRAMES:8.2f}s")
    print(f"{'='*60}")
    print(f"DONE! Output: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
