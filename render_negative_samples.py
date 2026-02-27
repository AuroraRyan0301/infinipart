"""
Blender script: Render negative-sample articulation videos for reward model training.

Generates videos with intentionally WRONG articulations (wrong joint type, wrong axis,
wrong direction, over-motion, wrong parts moving, jittery motion, part detach) alongside the correct
articulation rendered by render_articulation.py.  These serve as negative examples for
training a reward model to distinguish correct from incorrect joint motion.

This script is standalone — it imports utility functions from render_articulation.py
but has its own CLI, paths, and main loop.  Zero modifications to the original script.

Run with:
  CUDA_VISIBLE_DEVICES=0 blender --background --python render_negative_samples.py -- \
    --factory BottleFactory --seed 0 --device 0 \
    --neg_types wrong_joint_type wrong_axis over_motion \
    --neg_views front threequarter
"""

import bpy
import sys
import os
import json
import math
import copy
import random

from mathutils import Vector, Matrix
from collections import defaultdict, deque

# ── Parse our own args FIRST (before importing render_articulation) ──
_orig_argv = sys.argv[:]
own_argv = sys.argv
if "--" in own_argv:
    own_argv = own_argv[own_argv.index("--") + 1:]
else:
    own_argv = []

import argparse

parser = argparse.ArgumentParser(description="Render negative articulation samples")
# Base args (same as render_articulation.py)
parser.add_argument("--factory", required=True, help="Factory name, e.g. OfficeChairFactory")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--base", default="/mnt/data/yurh/Infinite-Mobility",
                    help="Base dir with outputs/ (auto-detected for PhysXNet)")
parser.add_argument("--envmap", default="/mnt/data/yurh/dataset3D/envmap/indoor/brown_photostudio_06_2k.exr")
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--duration", type=float, default=4.0, help="Video duration in seconds")
parser.add_argument("--samples", type=int, default=32, help="Cycles samples")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--animode", type=int, default=0,
                    help="Animation mode: 0=URDF limits, N>0=limits scaled by (1+N*0.5)x")
parser.add_argument("--output_dir", default=None, help="Override output directory")
parser.add_argument("--png_only", action="store_true", help="Output PNG sequences only (skip ffmpeg)")
parser.add_argument("--skip_nobg", action="store_true", help="Skip transparent background renders")
parser.add_argument("--skip_bg", action="store_true", help="Skip opaque background renders")
# Negative-specific args
parser.add_argument("--neg_types", nargs="+",
                    default=["wrong_joint_type", "wrong_axis", "wrong_direction",
                             "over_motion", "wrong_parts_moving", "jitter",
                             "part_detach"],
                    help="Which negative types to render")
parser.add_argument("--neg_views", nargs="+", default=["front", "threequarter"],
                    help="Which views to render for negative samples")
parser.add_argument("--neg_scale_factor", type=float, default=2.5,
                    help="Over-motion limit multiplier")
parser.add_argument("--neg_direction_offset_deg", type=float, default=45.0,
                    help="Angular offset (degrees) added to joint axis for wrong_direction")
parser.add_argument("--neg_jitter_std", type=float, default=0.3,
                    help="Jitter noise std as fraction of target magnitude")
args = parser.parse_args(own_argv)

# ── Import from render_articulation (need dummy args for its module-level argparse) ──
sys.argv = ['blender', '--', '--factory', '_dummy', '--output_dir', '/tmp/_neg_import_dummy']
# Add the script directory to sys.path so the import works
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from render_articulation import (
    URDFJoint, parse_urdf, build_kinematic_tree,
    compute_joint_value, compute_joint_local_transform,
    forward_kinematics_at_q,
    clear_scene, setup_render_engine, setup_render_settings,
    setup_envmap_lighting, load_scene_parts,
    compute_scene_bounds, render_view, render_moving_view, frames_to_video,
    get_moving_part_indices,
    FACTORY_RULES, VIEW_CONFIGS, MOVING_VIEW_CONFIGS,
    mat_translate, mat_rotate_axis_angle,
)

sys.argv = _orig_argv  # restore

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
NEG_DIR = os.path.join(OUT_DIR, "negatives")
os.makedirs(NEG_DIR, exist_ok=True)

NUM_FRAMES = int(args.fps * args.duration)

NEGATIVE_TYPES = [
    "wrong_joint_type",
    "wrong_axis",
    "wrong_direction",
    "over_motion",
    "wrong_parts_moving",
    "jitter",
    "part_detach",
]


# ═══════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════

def clear_animation_data(part_objects):
    """Remove all animation keyframes from part objects, reset to rest pose."""
    for obj in part_objects.values():
        if obj.animation_data:
            obj.animation_data_clear()
        obj.matrix_world = Matrix.Identity(4)
        obj.location = (0, 0, 0)
        obj.rotation_euler = (0, 0, 0)
        obj.scale = (1, 1, 1)


def get_animated_joint_names(joints, parent_map, links, moving_indices,
                             factory_name, animode):
    """Determine which joint names should be animated.

    Replicates the selection logic from render_articulation.animate_parts
    (lines 867-958) as a standalone function.

    Returns set of joint name strings.
    """
    # Build link -> part_idx map
    link_part_map = {}
    for link_name, info in links.items():
        if info["part_idx"] is not None:
            link_part_map[link_name] = info["part_idx"]

    # Determine animated links
    animated_links = {}
    for link_name, part_idx in link_part_map.items():
        if moving_indices is None or part_idx in moving_indices:
            animated_links[link_name] = part_idx

    # Trace back from animated links to root, collecting joints
    animated_joint_names = set()
    if moving_indices is not None:
        for link_name in animated_links:
            curr = link_name
            while curr in parent_map:
                parent_link, joint = parent_map[curr]
                animated_joint_names.add(joint.name)
                curr = parent_link
    else:
        animated_joint_names = {j.name for j in joints if j.jtype != "fixed"}

    # Per-animode filtering from FACTORY_RULES
    animode_joints_cfg = FACTORY_RULES.get(factory_name, {}).get("animode_joints", {})
    if animode not in animode_joints_cfg:
        return animated_joint_names

    selectors = animode_joints_cfg[animode]
    if isinstance(selectors, str):
        # Custom animation like "flip" — return as-is
        return animated_joint_names

    joint_by_name = {j.name: j for j in joints}

    # Filter: exclude joints with negligible range
    MIN_PRISMATIC = 0.005
    MIN_ROTARY = 0.05
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
    selected = set()
    for sel in selectors:
        if len(sel) == 1:
            sel_type = sel[0]
            matched = {n for n in significant if joint_by_name[n].jtype == sel_type}
            selected |= matched
        elif len(sel) == 3 and sel[1] == "axis":
            sel_type, _, axis_name = sel
            axis_idx = {"x": 0, "y": 1, "z": 2}[axis_name.lower()]
            for jname in significant:
                j = joint_by_name[jname]
                if j.jtype == sel_type and abs(j.axis[axis_idx]) > 0.5:
                    selected.add(jname)
        elif len(sel) == 3 and sel[1] == "sign":
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
            candidates.sort()
            if candidates:
                selected.add(candidates[sel_ord][1])

    return selected if selected else animated_joint_names


# ═══════════════════════════════════════════════════════════════
# Mutators — each modifies a deep-copied joint list in place
# ═══════════════════════════════════════════════════════════════

def mutate_wrong_joint_type(joints, animated_joint_names):
    """Swap revolute/continuous <-> prismatic for animated joints.

    When converting revolute (radians) to prismatic (meters), scale limits by 0.1
    so the sliding magnitude is visually plausible.  Reverse for prismatic->revolute.
    """
    for j in joints:
        if j.name not in animated_joint_names or j.jtype == "fixed":
            continue
        if j.jtype in ("revolute", "continuous"):
            j.jtype = "prismatic"
            j.lower *= 0.1
            j.upper *= 0.1
        elif j.jtype == "prismatic":
            j.jtype = "revolute"
            j.lower *= 10.0
            j.upper *= 10.0


def mutate_wrong_axis(joints, animated_joint_names):
    """Cyclic-permute the dominant joint axis (X->Y, Y->Z, Z->X)."""
    for j in joints:
        if j.name not in animated_joint_names or j.jtype == "fixed":
            continue
        dominant = max(range(3), key=lambda i: abs(j.axis[i]))
        new_dominant = (dominant + 1) % 3
        sign = 1.0 if j.axis[dominant] > 0 else -1.0
        j.axis = [0.0, 0.0, 0.0]
        j.axis[new_dominant] = sign


def mutate_wrong_direction(joints, animated_joint_names, offset_deg=45.0):
    """Add an angular offset to the joint axis direction.

    Rotates the axis vector by *offset_deg* degrees around an arbitrary
    perpendicular direction, producing a visibly wrong motion direction
    while keeping the same joint type and limits.
    """
    offset_rad = math.radians(offset_deg)
    for j in joints:
        if j.name not in animated_joint_names or j.jtype == "fixed":
            continue
        axis = Vector(j.axis).normalized()
        # Find a perpendicular vector to rotate around
        # Pick the world axis least aligned with the joint axis
        candidates = [Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))]
        perp_seed = min(candidates, key=lambda v: abs(axis.dot(v)))
        perp = axis.cross(perp_seed).normalized()
        # Rotate the axis around the perpendicular by offset_deg
        rot_matrix = Matrix.Rotation(offset_rad, 3, perp)
        new_axis = rot_matrix @ axis
        j.axis = [new_axis.x, new_axis.y, new_axis.z]


def mutate_over_motion(joints, animated_joint_names, scale_factor=2.5):
    """Scale joint limits beyond URDF bounds."""
    for j in joints:
        if j.name not in animated_joint_names or j.jtype == "fixed":
            continue
        j.lower *= scale_factor
        j.upper *= scale_factor


def mutate_part_detach(joints, animated_joint_names, detach_distance=0.5):
    """Make animated parts fly off by converting to prismatic with large displacement.

    The part translates along the joint axis direction (or upward for revolute joints)
    and separates completely from the body, simulating a detached/broken part.
    For revolute/continuous joints: axis is kept (hinge direction = fly-off direction).
    For prismatic joints: existing axis is used with amplified range.
    """
    for j in joints:
        if j.name not in animated_joint_names or j.jtype == "fixed":
            continue
        # Convert to prismatic with large travel distance
        original_type = j.jtype
        j.jtype = "prismatic"
        # Use the joint axis as fly-off direction
        # Set limits so the part travels detach_distance away (one-way)
        j.lower = 0.0
        j.upper = detach_distance


def invert_moving_indices(all_part_indices, original_moving_indices):
    """Return the complement of moving_indices (body parts become movers).

    Returns None if inversion is not possible.
    """
    if original_moving_indices is None:
        return None
    inverted = all_part_indices - original_moving_indices
    if not inverted:
        return None
    return inverted


def compute_joint_value_jittery(joint, frame, num_frames, rng, jitter_std=0.3):
    """Like compute_joint_value but with additive Gaussian noise per frame."""
    base_val = compute_joint_value(joint, frame, num_frames)
    if joint.jtype == "fixed" or abs(base_val) < 1e-10:
        return base_val
    target = joint.upper if abs(joint.upper) >= abs(joint.lower) else joint.lower
    noise = rng.gauss(0, abs(target) * jitter_std)
    return base_val + noise


# ═══════════════════════════════════════════════════════════════
# Forward Kinematics with custom value function
# ═══════════════════════════════════════════════════════════════

def forward_kinematics_selective_custom(links, joints, parent_map, children_map,
                                        frame, num_frames, animated_joint_names,
                                        value_fn=None):
    """BFS forward kinematics with optional custom joint value function.

    Like render_articulation.forward_kinematics_selective but accepts value_fn
    for jitter/noise injection.
    """
    if value_fn is None:
        value_fn = compute_joint_value

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
            if joint.name in animated_joint_names:
                q = value_fn(joint, frame, num_frames)
            else:
                q = 0.0

            T_local = compute_joint_local_transform(joint, q)
            child_T = parent_T @ T_local
            link_transforms[child_link] = child_T
            queue.append(child_link)

    return link_transforms


# ═══════════════════════════════════════════════════════════════
# Animation for negative samples
# ═══════════════════════════════════════════════════════════════

def animate_parts_negative(part_objects, links, joints, parent_map, children_map,
                           origins, num_frames, moving_indices,
                           animated_joint_names, value_fn=None):
    """Animate parts using (potentially mutated) joint parameters.

    A slim variant of render_articulation.animate_parts that:
      - Takes pre-computed animated_joint_names (no FACTORY_RULES re-evaluation)
      - Accepts optional value_fn for jitter/noise
      - Works with mutated joints / kinematic tree
    """
    print(f"\n  Computing negative animation for {num_frames} frames...")

    link_part_map = {}
    for link_name, info in links.items():
        if info["part_idx"] is not None:
            link_part_map[link_name] = info["part_idx"]

    animated_links = {}
    body_links = {}
    for link_name, part_idx in link_part_map.items():
        if moving_indices is None or part_idx in moving_indices:
            animated_links[link_name] = part_idx
        else:
            body_links[link_name] = part_idx

    print(f"    Animated parts: {sorted(animated_links.values())}")
    print(f"    Static parts:   {sorted(body_links.values())}")
    print(f"    Animated joints: {sorted(animated_joint_names)}")

    # Rest transforms (all q=0) using the (mutated) joints
    rest_transforms = forward_kinematics_at_q(
        links, joints, parent_map, children_map, q_values=None)

    for frame in range(1, num_frames + 1):
        bpy.context.scene.frame_set(frame)

        link_T = forward_kinematics_selective_custom(
            links, joints, parent_map, children_map,
            frame, num_frames, animated_joint_names,
            value_fn=value_fn,
        )

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
            except Exception:
                T_rest_inv = Matrix.Identity(4)

            delta = T_current @ T_rest_inv
            obj.matrix_world = delta

            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)
            obj.keyframe_insert(data_path="scale", frame=frame)

    # Linear interpolation for all keyframed objects
    for obj in part_objects.values():
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'

    print(f"    Animation set for {len(animated_links)} moving parts")


# ═══════════════════════════════════════════════════════════════
# Render helpers
# ═══════════════════════════════════════════════════════════════

def render_neg_views(neg_dir, neg_views, num_frames, center, distance,
                     render_bg, render_nobg, animode_suffix, fps, png_only):
    """Render the specified views and optionally encode to MP4.

    Supports both static views (VIEW_CONFIGS: hemi_*, front, etc.) and
    moving views (MOVING_VIEW_CONFIGS: orbit_*, sweep_*).
    """
    for view_name in neg_views:
        if view_name in VIEW_CONFIGS:
            elev_deg, azim_deg = VIEW_CONFIGS[view_name]
            nobg_dir, bg_dir = render_view(
                neg_dir, view_name, num_frames,
                center, distance, elev_deg, azim_deg,
                render_bg=render_bg,
                render_nobg=render_nobg,
                animode_suffix=animode_suffix,
            )
        elif view_name in MOVING_VIEW_CONFIGS:
            start_elev, start_azim, end_elev, end_azim = MOVING_VIEW_CONFIGS[view_name]
            nobg_dir, bg_dir = render_moving_view(
                neg_dir, view_name, num_frames,
                center, distance, start_elev, start_azim, end_elev, end_azim,
                render_bg=render_bg,
                render_nobg=render_nobg,
                animode_suffix=animode_suffix,
            )
        else:
            print(f"    WARNING: Unknown view '{view_name}', skipping")
            continue

        if not png_only:
            vname = f"{view_name}{animode_suffix}"
            if nobg_dir:
                frames_to_video(nobg_dir,
                                os.path.join(neg_dir, f"{vname}_nobg.mp4"), fps)
            if bg_dir:
                frames_to_video(bg_dir,
                                os.path.join(neg_dir, f"{vname}_bg.mp4"), fps)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print(f"Rendering NEGATIVE samples: {FACTORY} seed={SEED}")
    print(f"{'='*60}")
    print(f"  URDF:       {URDF_PATH}")
    print(f"  OBJs:       {OBJS_DIR}")
    print(f"  Output:     {NEG_DIR}")
    print(f"  Frames:     {NUM_FRAMES} ({args.duration}s @ {args.fps}fps)")
    print(f"  Res:        {args.resolution}, Samples: {args.samples}")
    print(f"  Animode:    {args.animode}")
    print(f"  Neg types:  {args.neg_types}")
    print(f"  Neg views:  {args.neg_views}")

    for p, label in [(URDF_PATH, "URDF"), (ORIGINS_PATH, "Origins"), (OBJS_DIR, "OBJs")]:
        if not os.path.exists(p):
            print(f"ERROR: {label} not found: {p}")
            return

    # ── Parse URDF ──
    print("\nParsing URDF...")
    links, joints = parse_urdf(URDF_PATH)
    parent_map, children_map = build_kinematic_tree(joints)

    movable = [j for j in joints if j.jtype in ("revolute", "prismatic", "continuous")]
    print(f"  Links: {len(links)}, Joints: {len(joints)} ({len(movable)} movable)")
    for j in movable:
        print(f"    {j}")

    if not movable:
        print("  No movable joints — nothing to generate negatives for.")
        return

    # ── Check for custom animations (skip negatives) ──
    animode_cfg = FACTORY_RULES.get(FACTORY, {}).get("animode_joints", {})
    custom_anim = animode_cfg.get(args.animode)
    if isinstance(custom_anim, str):
        print(f"  Animode {args.animode} uses custom animation '{custom_anim}' — "
              f"negatives not supported for custom animations.")
        return

    # ── Determine moving parts ──
    moving_result = get_moving_part_indices(FACTORY, SCENE_DIR)
    moving_indices = None
    exclude_indices = set()
    if moving_result is not None:
        moving_indices, exclude_indices = moving_result
    else:
        print("  No FACTORY_RULES: animating all parts")

    # ── Load origins ──
    with open(ORIGINS_PATH) as f:
        origins = json.load(f)

    # ── Setup Blender scene ──
    clear_scene()
    setup_render_engine()
    setup_render_settings(args.resolution, args.fps, NUM_FRAMES, args.samples)
    setup_envmap_lighting(args.envmap, strength=1.0)

    # ── Load parts ──
    print("\nLoading per-part OBJs...")
    part_objects = load_scene_parts(OBJS_DIR, origins, exclude_indices=exclude_indices)
    if not part_objects:
        print("ERROR: No parts loaded!")
        return

    # Remove orphan parts not in URDF
    urdf_part_indices = {info["part_idx"] for info in links.values()
                         if info["part_idx"] is not None}
    orphan_indices = set(part_objects.keys()) - urdf_part_indices
    for pidx in sorted(orphan_indices):
        obj = part_objects.pop(pidx)
        bpy.data.objects.remove(obj, do_unlink=True)
    if orphan_indices:
        print(f"  Removed {len(orphan_indices)} orphan part(s)")

    # ── Scene bounds ──
    all_objs = list(part_objects.values())
    center, extent = compute_scene_bounds(all_objs)
    distance = extent * 1.8
    print(f"  Center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  Extent: {extent:.3f}, CamDist: {distance:.3f}")

    # ── Collect all movable joints ──
    all_movable = [j for j in joints if j.jtype in ("revolute", "prismatic", "continuous")]
    # Filter negligible-range joints
    MIN_PRISMATIC = 0.005
    MIN_ROTARY = 0.05
    significant_joints = []
    for j in all_movable:
        rng_val = abs(j.upper - j.lower)
        if j.jtype == "prismatic" and rng_val < MIN_PRISMATIC:
            continue
        if j.jtype in ("revolute", "continuous") and rng_val < MIN_ROTARY:
            continue
        significant_joints.append(j)

    if not significant_joints:
        print("  No significant movable joints — nothing to perturb.")
        return

    print(f"  Significant joints ({len(significant_joints)}):")
    for j in significant_joints:
        print(f"    {j}")

    # ── Rendering config ──
    render_bg = not args.skip_bg
    render_nobg = not args.skip_nobg
    rng = random.Random(SEED)

    metadata = []

    # ═══════════════════════════════════════════════════════════
    # Outer loop: each joint as its own "animode"
    # Inner loop: 6 negative types per joint
    # ═══════════════════════════════════════════════════════════

    for joint_idx, target_joint in enumerate(significant_joints):
        jname = target_joint.name
        animated_joint_names = {jname}

        print(f"\n{'═'*60}")
        print(f"  Joint {joint_idx}/{len(significant_joints)}: {jname} "
              f"({target_joint.jtype}, [{target_joint.lower:.4f}, {target_joint.upper:.4f}])")
        print(f"{'═'*60}")

        joint_dir = os.path.join(NEG_DIR, jname)
        os.makedirs(joint_dir, exist_ok=True)
        animode_suffix = f"_{jname}"

        for neg_type in args.neg_types:
            if neg_type not in NEGATIVE_TYPES:
                print(f"\n  WARNING: Unknown negative type '{neg_type}', skipping")
                continue

            print(f"\n{'─'*50}")
            print(f"  [{jname}] Negative: {neg_type}")
            print(f"{'─'*50}")

            neg_out = os.path.join(joint_dir, neg_type)
            os.makedirs(neg_out, exist_ok=True)

            # Reset animation state
            clear_animation_data(part_objects)

            # Prepare mutated joints / params
            use_joints = joints
            use_parent_map = parent_map
            use_children_map = children_map
            use_moving_indices = moving_indices
            value_fn = None
            meta_info = {"joint": jname, "joint_type": target_joint.jtype,
                         "neg_type": neg_type, "views": args.neg_views}

            if neg_type == "wrong_joint_type":
                mut_joints = copy.deepcopy(joints)
                mutate_wrong_joint_type(mut_joints, animated_joint_names)
                use_parent_map, use_children_map = build_kinematic_tree(mut_joints)
                use_joints = mut_joints
                changes = []
                for mj, oj in zip(mut_joints, joints):
                    if mj.jtype != oj.jtype:
                        changes.append({"joint": mj.name,
                                        "original_type": oj.jtype,
                                        "new_type": mj.jtype})
                meta_info["description"] = "Joint type swapped (revolute<->prismatic)"
                meta_info["changes"] = changes

            elif neg_type == "wrong_axis":
                mut_joints = copy.deepcopy(joints)
                mutate_wrong_axis(mut_joints, animated_joint_names)
                use_parent_map, use_children_map = build_kinematic_tree(mut_joints)
                use_joints = mut_joints
                changes = []
                for mj, oj in zip(mut_joints, joints):
                    if mj.axis != oj.axis:
                        changes.append({"joint": mj.name,
                                        "original_axis": list(oj.axis),
                                        "new_axis": list(mj.axis)})
                meta_info["description"] = "Joint axis permuted to wrong direction"
                meta_info["changes"] = changes

            elif neg_type == "wrong_direction":
                mut_joints = copy.deepcopy(joints)
                _offset = args.neg_direction_offset_deg
                mutate_wrong_direction(mut_joints, animated_joint_names,
                                       offset_deg=_offset)
                use_parent_map, use_children_map = build_kinematic_tree(mut_joints)
                use_joints = mut_joints
                changes = []
                for mj, oj in zip(mut_joints, joints):
                    if list(mj.axis) != list(oj.axis):
                        changes.append({"joint": mj.name,
                                        "original_axis": list(oj.axis),
                                        "new_axis": list(mj.axis)})
                meta_info["description"] = (f"Joint axis offset by "
                                            f"{_offset} degrees")
                meta_info["offset_deg"] = _offset
                meta_info["changes"] = changes

            elif neg_type == "over_motion":
                mut_joints = copy.deepcopy(joints)
                mutate_over_motion(mut_joints, animated_joint_names,
                                   args.neg_scale_factor)
                use_parent_map, use_children_map = build_kinematic_tree(mut_joints)
                use_joints = mut_joints
                meta_info["description"] = (f"Joint limits scaled by "
                                            f"{args.neg_scale_factor}x")
                meta_info["scale_factor"] = args.neg_scale_factor

            elif neg_type == "wrong_parts_moving":
                all_indices = set(part_objects.keys())
                inverted = invert_moving_indices(all_indices, moving_indices)
                if inverted is None:
                    print("    Skipping: cannot invert moving indices")
                    continue
                use_moving_indices = inverted
                meta_info["description"] = "Body parts move instead of articulated parts"
                meta_info["original_moving"] = (sorted(moving_indices)
                                                if moving_indices else "all")
                meta_info["negative_moving"] = sorted(inverted)

            elif neg_type == "jitter":
                _rng = random.Random(rng.randint(0, 2**32))
                _std = args.neg_jitter_std
                value_fn = lambda joint, frame, nf, _r=_rng, _s=_std: \
                    compute_joint_value_jittery(joint, frame, nf, _r, _s)
                meta_info["description"] = (f"Random jitter noise (std="
                                            f"{args.neg_jitter_std})")
                meta_info["jitter_std"] = args.neg_jitter_std

            elif neg_type == "part_detach":
                mut_joints = copy.deepcopy(joints)
                # Compute detach distance from scene extent
                all_objs_list = list(part_objects.values())
                _, ext = compute_scene_bounds(all_objs_list)
                detach_dist = ext * 0.8  # fly off ~80% of scene extent
                original_types = {j.name: j.jtype for j in joints
                                  if j.name in animated_joint_names}
                mutate_part_detach(mut_joints, animated_joint_names,
                                   detach_distance=detach_dist)
                use_parent_map, use_children_map = build_kinematic_tree(mut_joints)
                use_joints = mut_joints
                meta_info["description"] = (f"Part detaches and flies away "
                                            f"(dist={detach_dist:.3f})")
                meta_info["detach_distance"] = detach_dist
                meta_info["original_joint_types"] = original_types

            # ── Animate: only the target joint ──
            animate_parts_negative(
                part_objects, links, use_joints, use_parent_map, use_children_map,
                origins, NUM_FRAMES, use_moving_indices,
                animated_joint_names, value_fn=value_fn,
            )

            # ── Render views ──
            render_neg_views(neg_out, args.neg_views, NUM_FRAMES, center, distance,
                             render_bg, render_nobg, "",
                             args.fps, args.png_only)

            metadata.append(meta_info)

    # ── Save metadata ──
    meta_path = os.path.join(NEG_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved to {meta_path}")

    n_joints = len(significant_joints)
    n_types = len([m for m in metadata])
    print(f"\n{'='*60}")
    print(f"DONE! Negative samples: {NEG_DIR}")
    print(f"  {n_joints} joints x {len(args.neg_types)} neg types = {n_types} combos")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
