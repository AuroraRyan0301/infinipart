"""
Blender script: Render negative-sample articulation videos.

Generates videos with intentionally WRONG articulations for reward model training.
Uses the same pipeline as render_animode.py but with mutated joint parameters
and NO BVH collision detection (negative samples intentionally interpenetrate).

6 negative types:
  1. wrong_joint_type   — swap revolute <-> prismatic
  2. wrong_axis         — cyclic-permute dominant axis (X->Y, Y->Z, Z->X)
  3. wrong_direction    — rotate axis by 45° around perpendicular
  4. over_motion        — 360° rotation / 10x prismatic range
  5. wrong_parts_moving — invert which parts are animated vs static
  6. jitter             — additive Gaussian noise per frame

Output layout:
  {output_dir}/{factory}/{seed}/{animode}/negatives/{neg_type}/{view}_{suffix}.mp4

Usage:
  CUDA_VISIBLE_DEVICES=0 blender --background --python render_negative.py -- \
    --metadata ./precompute_output/lamp/0/metadata.json \
    --animode basic_0 \
    --neg_types wrong_joint_type wrong_axis over_motion \
    --views hemi_05 \
    --color_mode both --bg_mode both
"""

import sys
import os
import copy
import json
import math
import random

# ── parse args before any Blender imports ──
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

import argparse

parser = argparse.ArgumentParser(description="Render negative articulation samples")
parser.add_argument("--metadata", required=True, help="Path to metadata.json")
parser.add_argument("--animode", default="all", help="Animode name or 'all'")
parser.add_argument("--neg_types", nargs="+",
                    default=["wrong_joint_type", "wrong_axis", "wrong_direction",
                             "over_motion", "wrong_parts_moving", "jitter"],
                    help="Negative types to render")
parser.add_argument("--views", nargs="+", default=["hemi_05"])
parser.add_argument("--color_mode", default="both",
                    choices=["realistic", "group", "part", "both"])
parser.add_argument("--bg_mode", default="both",
                    choices=["nobg", "withbg", "both"])
parser.add_argument("--resolution", type=int, default=512)
parser.add_argument("--samples", type=int, default=16)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--duration", type=float, default=3.0)
parser.add_argument("--cam_distance", type=float, default=1.4)
parser.add_argument("--output_dir", default="./precompute_nega_output",
                    help="Output root directory (default: ./precompute_nega_output)")
parser.add_argument("--skip_existing", action="store_true")
# Negative-specific params
parser.add_argument("--neg_direction_offset_deg", type=float, default=45.0)
parser.add_argument("--neg_jitter_std", type=float, default=0.3)
parser.add_argument("--neg_over_scale", type=float, default=2.5)

args = parser.parse_args(argv)

# ── Now import Blender + render_animode ──
import bpy
from mathutils import Vector, Matrix
from collections import deque

# Add repo to path for render_animode imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Import reusable functions from render_animode
from render_animode import (
    Joint, parse_urdf, build_kinematic_tree,
    forward_kinematics, compute_joint_local_transform,
    compute_trajectory_value, compute_frame_joint_values,
    clear_scene, setup_render_engine, setup_render_settings,
    setup_envmap, load_scene_parts, apply_upright_rotation_blender,
    normalize_parts, build_normalize_matrices,
    get_realistic_materials, resolve_views, sample_views_for_animode,
    clear_animation, _apply_materials_for_pass,
    _render_views, _set_fast_render,
    mat_translate,
    ENVMAP_DIR,
)


# ======================================================================
# 6 Joint Mutators
# ======================================================================

NEGATIVE_TYPES = [
    "wrong_joint_type",
    "wrong_axis",
    "wrong_direction",
    "over_motion",
    "wrong_parts_moving",
    "jitter",
]


def get_active_joint_names(split_info, joints_by_name):
    """Get joint names that are active or passive in this animode."""
    active = set()
    for jname, jcls in split_info.get("joint_classification", {}).items():
        if jcls in ("active", "passive"):
            active.add(jname)
    return active


def mutate_wrong_joint_type(joints, active_names):
    """Swap revolute/continuous <-> prismatic for active joints."""
    for j in joints:
        if j.name not in active_names or j.jtype == "fixed":
            continue
        if j.jtype in ("revolute", "continuous"):
            j.jtype = "prismatic"
            j.lower *= 0.5
            j.upper *= 0.5
        elif j.jtype == "prismatic":
            j.jtype = "revolute"
            j.lower *= 10.0
            j.upper *= 10.0


def mutate_wrong_axis(joints, active_names):
    """Cyclic-permute the dominant joint axis (X->Y, Y->Z, Z->X)."""
    for j in joints:
        if j.name not in active_names or j.jtype == "fixed":
            continue
        ax = list(j.axis)
        dominant = max(range(3), key=lambda i: abs(ax[i]))
        new_dominant = (dominant + 1) % 3
        sign = 1.0 if ax[dominant] > 0 else -1.0
        new_axis = [0.0, 0.0, 0.0]
        new_axis[new_dominant] = sign
        j.axis = Vector(new_axis).normalized()


def mutate_wrong_direction(joints, active_names, offset_deg=45.0):
    """Rotate joint axis by offset_deg around a perpendicular direction."""
    offset_rad = math.radians(offset_deg)
    for j in joints:
        if j.name not in active_names or j.jtype == "fixed":
            continue
        axis = Vector(j.axis).normalized()
        candidates = [Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))]
        perp_seed = min(candidates, key=lambda v: abs(axis.dot(v)))
        perp = axis.cross(perp_seed).normalized()
        rot = Matrix.Rotation(offset_rad, 3, perp)
        new_axis = rot @ axis
        j.axis = new_axis.normalized()


def mutate_over_motion(joints, active_names):
    """Scale joint limits way beyond URDF bounds."""
    for j in joints:
        if j.name not in active_names or j.jtype == "fixed":
            continue
        if j.jtype in ("revolute", "continuous"):
            j.lower = -2 * math.pi
            j.upper = 2 * math.pi
        elif j.jtype == "prismatic":
            j.lower *= 10.0
            j.upper *= 10.0


def compute_jittery_frame_values(joints_by_name, split_info, t, rng, jitter_std=0.3):
    """Same as compute_frame_joint_values but with per-frame Gaussian noise."""
    base_values = compute_frame_joint_values(joints_by_name, split_info, t)
    for jname, q in base_values.items():
        joint = joints_by_name.get(jname)
        if joint and joint.jtype != "fixed" and abs(q) > 1e-10:
            far_q = joint.upper if abs(joint.upper) >= abs(joint.lower) else joint.lower
            noise = rng.gauss(0, abs(far_q) * jitter_std)
            base_values[jname] = q + noise
    return base_values


# ======================================================================
# Negative Animation: compute frame joint values WITHOUT collision detection
# ======================================================================

def compute_negative_frame_values(joints, joints_by_name, split_info, num_frames,
                                  neg_type, rng, jitter_std=0.3):
    """Compute per-frame joint values for a negative type.

    Returns dict {frame: {jname: q_value}}.
    NO collision detection — negative samples intentionally interpenetrate.
    """
    frame_values = {}

    for frame in range(1, num_frames + 1):
        t = (frame - 1) / max(num_frames - 1, 1)

        if neg_type == "jitter":
            jv = compute_jittery_frame_values(joints_by_name, split_info, t,
                                              rng, jitter_std)
        else:
            # For mutated joints, compute values using the normal trajectory
            # but with the mutated joint parameters
            jv = compute_frame_joint_values(joints_by_name, split_info, t)

        frame_values[frame] = jv

    return frame_values


def animate_negative(parts, joints, children_map, root_link,
                     frame_joint_values, num_frames, center, scale,
                     factory_name="", split_info=None):
    """Animate parts with given frame joint values in normalized space.

    Same as animate_scene_normalized but with no collision avoidance.
    Handles lid_flip trajectory if present.
    """
    S, S_inv = build_normalize_matrices(center, scale, factory_name)

    rest_values = {j.name: 0.0 for j in joints}
    T_rest_world = forward_kinematics(joints, children_map, root_link, rest_values)
    T_rest_norm = {k: S @ v @ S_inv for k, v in T_rest_world.items()}
    T_rest_norm_inv = {k: v.inverted() for k, v in T_rest_norm.items()}

    # lid_flip handling (same as render_animode)
    traj_type = split_info.get("trajectory_type", "") if split_info else ""
    lid_flip_links = set()
    flip_start_frac = 0.5
    flip_angle_total = math.pi
    flip_axis_vec = Vector((1.0, 0.0, 0.0))
    lid_rest_centroids = {}
    if traj_type == "lid_flip":
        custom_params = split_info.get("custom_lid_params", {})
        lid_flip_links = set(custom_params.get("lid_links", []))
        flip_start_frac = custom_params.get("flip_start_frac", 0.5)
        flip_angle_total = custom_params.get("flip_angle", math.pi)
        flip_axis_vec = Vector(custom_params.get("flip_axis", [1.0, 0.0, 0.0])).normalized()
        for lname in lid_flip_links:
            obj = parts.get(lname)
            if obj and obj.data.vertices:
                verts = [v.co.copy() for v in obj.data.vertices]
                centroid = sum(verts, Vector()) / len(verts)
                lid_rest_centroids[lname] = centroid

    for frame in range(1, num_frames + 1):
        joint_values = frame_joint_values[frame]
        t_norm = (frame - 1) / max(num_frames - 1, 1)

        T_frame_world = forward_kinematics(joints, children_map, root_link, joint_values)
        T_frame_norm = {k: S @ v @ S_inv for k, v in T_frame_world.items()}

        for link_name, obj in parts.items():
            if link_name not in T_frame_norm or link_name not in T_rest_norm_inv:
                continue

            delta = T_frame_norm[link_name] @ T_rest_norm_inv[link_name]

            if traj_type == "lid_flip" and link_name in lid_flip_links and t_norm > flip_start_frac:
                flip_t = (t_norm - flip_start_frac) / max(1.0 - flip_start_frac, 1e-6)
                current_angle = flip_angle_total * 0.5 * (1 - math.cos(math.pi * flip_t))
                c_rest = lid_rest_centroids.get(link_name, Vector((0.0, 0.0, 0.0)))
                c_lifted = delta @ c_rest
                R_flip = Matrix.Rotation(current_angle, 4, flip_axis_vec)
                T_to_c = mat_translate(-c_lifted.x, -c_lifted.y, -c_lifted.z)
                T_from_c = mat_translate(c_lifted.x, c_lifted.y, c_lifted.z)
                delta = T_from_c @ R_flip @ T_to_c @ delta

            obj.matrix_world = delta
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    for link_name, obj in parts.items():
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'


def swap_active_links(split_info, links, parts):
    """For wrong_parts_moving: swap active <-> fixed joints.

    Makes the currently-fixed joints active and vice versa, so the WRONG
    parts animate. Only meaningful when there are both active AND fixed joints.
    Returns None if swap would result in no active joints (e.g. single-joint objects).
    """
    modified = copy.deepcopy(split_info)

    jc = modified.get("joint_classification", {})
    has_active = any(c == "active" for c in jc.values())
    has_fixed = any(c == "fixed" for c in jc.values())

    if not (has_active and has_fixed):
        # Single-joint or no fixed joints — swap is meaningless
        return None

    new_jc = {}
    for jname, cls in jc.items():
        if cls == "active":
            new_jc[jname] = "fixed"
        elif cls == "fixed":
            new_jc[jname] = "active"
        else:
            new_jc[jname] = cls
    modified["joint_classification"] = new_jc

    # Update active_joints / fixed_joints lists
    modified["active_joints"] = [j for j, c in new_jc.items() if c == "active"]
    modified["fixed_joints"] = [j for j, c in new_jc.items() if c == "fixed"]

    return modified


# ======================================================================
# Main
# ======================================================================

def main():
    with open(args.metadata) as f:
        metadata = json.load(f)

    print(f"\n{'='*60}")
    print(f"NEGATIVE Rendering: {metadata['output_name']}/{metadata['identifier']}")
    print(f"  Types: {args.neg_types}")
    print(f"{'='*60}")

    metadata_dir = os.path.dirname(os.path.realpath(args.metadata))
    # Output goes to separate directory: {output_dir}/{output_name}/{identifier}/
    nega_base = os.path.join(
        args.output_dir,
        metadata["output_name"],
        str(metadata["identifier"]),
    )
    os.makedirs(nega_base, exist_ok=True)

    urdf_path = metadata["urdf_path"]
    scene_dir = metadata["scene_dir"]
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.normpath(os.path.join(metadata_dir, urdf_path))
    if not os.path.isabs(scene_dir):
        scene_dir = os.path.normpath(os.path.join(metadata_dir, scene_dir))
    metadata["urdf_path"] = urdf_path
    metadata["scene_dir"] = scene_dir
    center = metadata["normalize"]["center"]
    scale = metadata["normalize"]["scale"]

    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found: {urdf_path}")
        sys.exit(1)

    # Parse URDF
    links, joints, root_link = parse_urdf(urdf_path)
    parent_map, children_map = build_kinematic_tree(joints)
    joints_by_name = {j.name: j for j in joints}

    # Resolve animodes
    all_splits = metadata.get("splits", {})
    if args.animode == "all":
        animodes_to_render = sorted(all_splits.keys())
    else:
        animodes_to_render = [a.strip() for a in args.animode.split(",")]
        for a in animodes_to_render:
            if a not in all_splits:
                print(f"ERROR: animode '{a}' not found. Available: {list(all_splits.keys())}")
                sys.exit(1)

    # Resolve views
    static_views, moving_views, is_sample_mode = resolve_views(args.views)
    num_frames = int(args.fps * args.duration)

    # Select envmap
    envmaps = [os.path.join(ENVMAP_DIR, f) for f in os.listdir(ENVMAP_DIR)
                if f.endswith(('.hdr', '.exr'))]
    envmap_path = random.choice(envmaps) if envmaps else None

    factory = metadata.get("factory", "")
    cam_center = [0.0, 0.0, 0.0]
    cam_distance = args.cam_distance

    if args.color_mode == "both":
        color_passes = ["realistic", "group"]
    else:
        color_passes = [args.color_mode]

    # ── One-time Blender setup ──
    clear_scene()
    setup_render_engine()
    setup_render_settings(args.resolution, args.fps, num_frames, args.samples)
    if envmap_path:
        setup_envmap(envmap_path)

    parts = load_scene_parts(metadata, links, joints, root_link)
    if not parts:
        print("ERROR: No parts loaded")
        sys.exit(1)

    apply_upright_rotation_blender(parts, factory)
    normalize_parts(parts, center, scale)

    realistic_mats = None
    if any(p == "realistic" for p in color_passes):
        realistic_mats = get_realistic_materials(metadata, links)

    if "realistic" in color_passes:
        _apply_materials_for_pass("realistic", parts, links,
                                  all_splits[animodes_to_render[0]],
                                  realistic_mats, metadata)
    current_mat_pass = "realistic" if "realistic" in color_passes else None

    rng = random.Random(int(metadata.get("identifier", 0)))
    neg_metadata = {}

    # ── Process each animode × neg_type ──
    for animode_name in animodes_to_render:
        split_info = all_splits[animode_name]
        active_names = get_active_joint_names(split_info, joints_by_name)

        if not active_names:
            print(f"  SKIP {animode_name}: no active joints")
            continue

        for neg_type in args.neg_types:
            if neg_type not in NEGATIVE_TYPES:
                print(f"  WARNING: unknown neg type '{neg_type}', skipping")
                continue

            print(f"\n{'─'*50}")
            print(f"  {animode_name} / {neg_type}")
            print(f"{'─'*50}")

            animode_dir = os.path.join(nega_base, animode_name)
            neg_out_dir = os.path.join(animode_dir, "negatives", neg_type)
            os.makedirs(neg_out_dir, exist_ok=True)

            # Check skip
            if args.skip_existing:
                existing = [f for f in os.listdir(neg_out_dir) if f.endswith(".mp4")]
                if existing:
                    print(f"    SKIP (existing: {len(existing)} videos)")
                    continue

            # Deep copy joints for mutation
            mut_joints = copy.deepcopy(joints)
            mut_joints_by_name = {j.name: j for j in mut_joints}
            use_split_info = split_info
            meta_entry = {"neg_type": neg_type, "animode": animode_name}

            if neg_type == "wrong_joint_type":
                mutate_wrong_joint_type(mut_joints, active_names)
                meta_entry["description"] = "Joint types swapped (revolute<->prismatic)"

            elif neg_type == "wrong_axis":
                mutate_wrong_axis(mut_joints, active_names)
                meta_entry["description"] = "Joint axes permuted (X->Y->Z->X)"

            elif neg_type == "wrong_direction":
                mutate_wrong_direction(mut_joints, active_names,
                                       args.neg_direction_offset_deg)
                meta_entry["description"] = f"Axis offset by {args.neg_direction_offset_deg}°"

            elif neg_type == "over_motion":
                mutate_over_motion(mut_joints, active_names)
                meta_entry["description"] = "Joint limits scaled to extreme range"

            elif neg_type == "wrong_parts_moving":
                swapped = swap_active_links(split_info, links, parts)
                if swapped is None:
                    print("    SKIP: cannot invert part assignment")
                    continue
                use_split_info = swapped
                # Use original joints (no mutation), just different parts move
                mut_joints = copy.deepcopy(joints)
                mut_joints_by_name = {j.name: j for j in mut_joints}
                meta_entry["description"] = "Wrong parts animated (inverted coloring)"

            elif neg_type == "jitter":
                # No joint mutation — jitter is applied per-frame
                meta_entry["description"] = f"Gaussian jitter noise (std={args.neg_jitter_std})"

            # Rebuild kinematic tree with mutated joints
            _, mut_children_map = build_kinematic_tree(mut_joints)

            # Compute frame joint values (NO collision detection)
            frame_jv = compute_negative_frame_values(
                mut_joints, mut_joints_by_name, use_split_info, num_frames,
                neg_type, rng, args.neg_jitter_std)

            # Animate
            clear_animation(parts)
            animate_negative(parts, mut_joints, mut_children_map, root_link,
                             frame_jv, num_frames, center, scale,
                             factory_name=factory, split_info=use_split_info)

            # Render all color passes for this neg_type
            for color_pass in color_passes:
                suffix_map = {"group": "_group", "part": "_part", "realistic": "_nobg"}
                vid_suffix = suffix_map.get(color_pass, "_nobg")
                is_fast = color_pass in ("group", "part")

                if is_fast:
                    _set_fast_render(True, original_samples=args.samples)

                if color_pass == "realistic":
                    if args.bg_mode == "both":
                        render_nobg, render_withbg = True, True
                    elif args.bg_mode == "withbg":
                        render_nobg, render_withbg = False, True
                    else:
                        render_nobg, render_withbg = True, False
                else:
                    render_nobg, render_withbg = True, False

                # Apply materials for this pass
                if current_mat_pass != color_pass:
                    _apply_materials_for_pass(color_pass, parts, links,
                                              use_split_info, realistic_mats, metadata)
                    current_mat_pass = color_pass

                if is_sample_mode:
                    object_id = f"{metadata.get('output_name', '')}_{metadata.get('identifier', '')}"
                    sv, mv = sample_views_for_animode(object_id, animode_name)
                else:
                    sv, mv = static_views, moving_views

                _render_views(sv, mv, neg_out_dir, vid_suffix,
                              render_nobg, render_withbg,
                              cam_center, cam_distance, num_frames, args.fps,
                              args.skip_existing)

                if is_fast:
                    _set_fast_render(False, original_samples=args.samples)

            neg_metadata[f"{animode_name}/{neg_type}"] = meta_entry

    # Save negative metadata
    neg_meta_path = os.path.join(nega_base, "negative_metadata.json")
    with open(neg_meta_path, "w") as f:
        json.dump(neg_metadata, f, indent=2)
    print(f"\nNegative metadata saved to {neg_meta_path}")

    print(f"\n{'='*60}")
    print(f"DONE! {len(neg_metadata)} negative sample sets rendered.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
