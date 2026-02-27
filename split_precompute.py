#!/usr/bin/env python3
"""
Unified dual-part split precompute for all factory types (IM, PhysXNet, PhysX_mobility).

For each object, generates normalized part0.obj (body) + part1.obj (moving) splits,
one per valid animode. Normalization is done on the whole object first, then split.

Animode → split mapping:
  0 = revolute joints only     → moving = revolute groups
  1 = prismatic joints only    → moving = prismatic groups
  2 = continuous joints only   → moving = continuous groups
  3 = all joints (default)     → moving = all movable groups
  10+N = Nth joint only        → moving = joint N's group

Output structure:
  {output_dir}/{Factory}/{identifier}/
      part0.obj, part1.obj           # default split (all movable → part1)
      anim0/part0.obj, part1.obj     # revolute-only (if applicable)
      anim1/part0.obj, part1.obj     # prismatic-only
      anim10/part0.obj, part1.obj    # per-joint
      ...
      metadata.json

Identity:
  - IM factories: identifier = seed (each seed = unique geometry)
  - PhysXNet/PhysX_mobility: identifier = object_id (seeds share geometry)

Usage:
  python split_precompute.py --factory ElectronicsPhysXNetFactory --seed 10005
  python split_precompute.py --factory DishwasherFactory --seed 0 --base /mnt/cpfs/yurh/Infinite-Mobility
  python split_precompute.py --all --output_dir ./precompute
"""

import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import trimesh


# ======================================================================
# URDF parsing
# ======================================================================

def _rotation_from_rpy(rpy):
    """Build 3x3 rotation matrix from roll-pitch-yaw (XYZ Euler)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr            ],
    ])


def parse_urdf_joints(urdf_path):
    """Parse URDF to get per-joint info with descendant group indices + axis geometry.

    Returns:
        joints: list of (joint_name, joint_type, frozenset_of_group_indices,
                         motion_range, axis_origin_3d, axis_direction_3d)
            axis_origin/direction are in the object's world frame (pre-normalization).
            Only non-fixed joints with significant motion are returned.
        static_gidxs: set of group indices not under any movable joint.
        all_gidxs: set of all group indices found in the URDF.
    """
    # Min motion thresholds to filter negligible joints
    MIN_PRISMATIC = 0.01   # 1cm
    MIN_ROTARY = 0.1       # ~6 degrees

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # ── Parse all joints with full transform info ──
    # joint_info[child_link] = (parent_link, jtype, jname, lower, upper, xyz, rpy, axis_local)
    joint_info = {}
    children_map = {}  # parent_link → [child_link, ...]
    for joint in root.findall("joint"):
        jtype = joint.get("type", "fixed")
        jname = joint.get("name", "")
        parent = joint.find("parent").get("link")
        child = joint.find("child").get("link")
        lower, upper = 0.0, 0.0
        lim = joint.find("limit")
        if lim is not None:
            lower = float(lim.get("lower", 0))
            upper = float(lim.get("upper", 0))
        # Origin transform (parent → joint frame)
        orig = joint.find("origin")
        xyz = np.array([0.0, 0.0, 0.0])
        rpy = np.array([0.0, 0.0, 0.0])
        if orig is not None:
            if orig.get("xyz"):
                xyz = np.array([float(v) for v in orig.get("xyz").split()])
            if orig.get("rpy"):
                rpy = np.array([float(v) for v in orig.get("rpy").split()])
        # Axis in joint frame (default X)
        ax = joint.find("axis")
        axis_local = np.array([1.0, 0.0, 0.0])
        if ax is not None and ax.get("xyz"):
            axis_local = np.array([float(v) for v in ax.get("xyz").split()])
            n = np.linalg.norm(axis_local)
            if n > 1e-8:
                axis_local = axis_local / n

        joint_info[child] = (parent, jtype, jname, lower, upper, xyz, rpy, axis_local)
        children_map.setdefault(parent, []).append(child)

    # ── Compute world-frame transforms for each link via BFS from root ──
    # Find root: a link that is never a child
    all_children = set(joint_info.keys())
    all_parents = set(j[0] for j in joint_info.values())
    roots = all_parents - all_children
    if not roots:
        # Fallback: first link
        roots = {root.find("link").get("name")}
    root_link = sorted(roots)[0]

    # link_world[link_name] = (world_position_3, world_rotation_3x3)
    link_world = {root_link: (np.zeros(3), np.eye(3))}
    queue = [root_link]
    while queue:
        parent = queue.pop(0)
        p_pos, p_rot = link_world[parent]
        for child in children_map.get(parent, []):
            if child in link_world:
                continue
            _, _, _, _, _, xyz, rpy, _ = joint_info[child]
            # Child world position = parent_world + parent_rotation @ local_xyz
            c_rot_local = _rotation_from_rpy(rpy)
            c_pos = p_pos + p_rot @ xyz
            c_rot = p_rot @ c_rot_local
            link_world[child] = (c_pos, c_rot)
            queue.append(child)

    # Collect all group indices from link names
    all_gidxs = set()
    for link in root.findall("link"):
        m = re.match(r"l_(\d+)", link.get("name", ""))
        if m:
            all_gidxs.add(int(m.group(1)))

    def bfs_descendant_gidxs(start_link):
        """BFS from a link, collecting all descendant l_N indices."""
        bfs_queue = [start_link]
        visited = set()
        gidxs = set()
        while bfs_queue:
            link = bfs_queue.pop(0)
            if link in visited:
                continue
            visited.add(link)
            m = re.match(r"l_(\d+)", link)
            if m:
                gidxs.add(int(m.group(1)))
            for child in children_map.get(link, []):
                bfs_queue.append(child)
        return gidxs

    # ── Find all movable joints ──
    joints = []
    all_moving_gidxs = set()

    for child_link, (parent_link, jtype, jname, lower, upper, xyz, rpy, axis_local) in joint_info.items():
        if jtype not in ("revolute", "prismatic", "continuous"):
            continue
        motion_range = abs(upper - lower)
        if jtype == "prismatic" and motion_range < MIN_PRISMATIC:
            continue
        if jtype in ("revolute", "continuous") and motion_range < MIN_ROTARY:
            continue
        desc_gidxs = bfs_descendant_gidxs(child_link)
        if not desc_gidxs:
            continue

        # Compute world-frame axis origin and direction
        p_pos, p_rot = link_world.get(parent_link, (np.zeros(3), np.eye(3)))
        axis_origin_world = p_pos + p_rot @ xyz
        axis_dir_world = p_rot @ axis_local
        n = np.linalg.norm(axis_dir_world)
        if n > 1e-8:
            axis_dir_world = axis_dir_world / n

        joints.append((jname, jtype, frozenset(desc_gidxs), motion_range,
                        axis_origin_world.copy(), axis_dir_world.copy()))
        all_moving_gidxs.update(desc_gidxs)

    # Sort joints deterministically by name
    joints.sort(key=lambda x: x[0])

    static_gidxs = all_gidxs - all_moving_gidxs
    return joints, static_gidxs, all_gidxs


# ======================================================================
# Mesh loading
# ======================================================================

def find_objs_dir(scene_dir, factory=None, identifier=None):
    """Find the objs directory, handling both IM nested paths and PhysXNet flat paths."""
    # PhysXNet/PhysX_mobility: outputs/{Factory}/{id}/outputs/{Factory}/{id}/objs/
    if factory and identifier:
        nested = os.path.join(scene_dir, "outputs", factory, str(identifier), "objs")
        if os.path.isdir(nested):
            return nested

    # IM: outputs/{Factory}/{seed}/outputs/{Factory}/{seed}/objs/
    # Try to auto-discover nested path
    for candidate in glob.glob(os.path.join(scene_dir, "outputs", "*", "*", "objs")):
        if os.path.isdir(candidate):
            return candidate

    # Flat fallback
    flat = os.path.join(scene_dir, "objs")
    if os.path.isdir(flat):
        return flat

    return None


def load_group_meshes(scene_dir, factory=None, identifier=None):
    """Load all per-group OBJ meshes and assemble in world coordinates.

    Each OBJ is centroid-subtracted. origins.json provides the world-space
    centroid for each group. We add the origin offset to each group's vertices
    so all groups are in a shared world frame before normalization.

    Returns: {gidx: trimesh.Trimesh}  (vertices in world coordinates)
    """
    objs_dir = find_objs_dir(scene_dir, factory, identifier)
    if objs_dir is None:
        print(f"  ERROR: No objs directory found in {scene_dir}")
        return {}

    # Load origins.json for world-space assembly
    origins = {}
    origins_path = os.path.join(scene_dir, "origins.json")
    if os.path.exists(origins_path):
        with open(origins_path) as f:
            origins = json.load(f)

    meshes = {}
    for entry in sorted(os.listdir(objs_dir)):
        entry_path = os.path.join(objs_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        try:
            gidx = int(entry)
        except ValueError:
            continue

        obj_path = os.path.join(entry_path, f"{gidx}.obj")
        if not os.path.exists(obj_path):
            # Try any .obj in the directory
            obj_files = glob.glob(os.path.join(entry_path, "*.obj"))
            if obj_files:
                obj_path = obj_files[0]
            else:
                continue

        # Resolve symlinks
        real_path = os.path.realpath(obj_path)
        if not os.path.exists(real_path):
            print(f"  WARNING: Broken symlink {obj_path} -> {real_path}")
            continue

        try:
            mesh = trimesh.load(real_path, force='mesh', process=False)
            if mesh.vertices.shape[0] > 0:
                # Apply origin offset to move from local to world coordinates
                origin = origins.get(str(gidx))
                if origin is not None:
                    mesh.vertices += np.array(origin, dtype=mesh.vertices.dtype)
                meshes[gidx] = mesh
        except Exception as e:
            print(f"  WARNING: Failed to load {obj_path}: {e}")

    return meshes


# ======================================================================
# Normalization
# ======================================================================

def normalize_meshes_inplace(meshes_by_gidx, bound=0.95):
    """Box-normalize all meshes jointly to [-bound, bound].

    Modifies vertices in-place. Returns (center, scale).
    """
    all_verts = np.concatenate([m.vertices for m in meshes_by_gidx.values()], axis=0)
    bmin = all_verts.min(axis=0)
    bmax = all_verts.max(axis=0)
    center = (bmax + bmin) / 2.0
    extent = (bmax - bmin).max()

    if extent < 1e-8:
        return center.tolist(), 1.0

    scale = 2.0 * bound / extent

    for mesh in meshes_by_gidx.values():
        mesh.vertices = (mesh.vertices - center) * scale

    return center.tolist(), float(scale)


# ======================================================================
# Split logic
# ======================================================================

def build_splits(joints, static_gidxs, all_gidxs):
    """Build animode → (body_gidxs, moving_gidxs, active_joint_indices) mapping.

    Each split records which joints are "active" — i.e., whose moving groups
    are in part1 for that split. This determines which joint params to export.

    Returns: dict of {animode_key: (frozenset_body, frozenset_moving, list_of_joint_indices)}
    Only includes unique splits (deduplicates).
    """
    if not joints:
        return {}

    # Default: all movable → part1, ALL joints active
    all_moving = set()
    for jname, jtype, gidxs, *_ in joints:
        all_moving.update(gidxs)
    default_body = frozenset(all_gidxs - all_moving)
    default_moving = frozenset(all_moving)
    all_joint_indices = list(range(len(joints)))

    splits = {"default": (default_body, default_moving, all_joint_indices)}
    seen_moving = {default_moving: "default"}

    # Per-type splits: only joints of that type are active
    type_to_animode = {"revolute": 0, "prismatic": 1, "continuous": 2}
    type_groups = defaultdict(set)
    type_joint_indices = defaultdict(list)
    for i, (jname, jtype, gidxs, *_) in enumerate(joints):
        type_groups[jtype].update(gidxs)
        type_joint_indices[jtype].append(i)

    for jtype, animode in type_to_animode.items():
        if jtype not in type_groups:
            continue
        moving = frozenset(type_groups[jtype])
        if moving in seen_moving:
            continue  # Same as an existing split
        body = frozenset(all_gidxs - moving)
        key = f"anim{animode}"
        splits[key] = (body, moving, type_joint_indices[jtype])
        seen_moving[moving] = key

    # Per-joint splits (only if >1 joint): single joint active
    if len(joints) > 1:
        for i, (jname, jtype, gidxs, *rest) in enumerate(joints):
            moving = frozenset(gidxs)
            if moving in seen_moving:
                continue  # Same as an existing split
            body = frozenset(all_gidxs - moving)
            key = f"anim{10 + i}"
            splits[key] = (body, moving, [i])
            seen_moving[moving] = key

    return splits


def _skew_matrix(v):
    """Skew-symmetric matrix for cross product: [v]_x."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def compute_joint_transform(joint_tuple, q, center, scale):
    """Compute 4x4 delta transform for a joint at value q in normalized space.

    For prismatic: pure translation along axis_dir * q * scale.
    For revolute/continuous: rotation around normalized axis_origin by q radians.
    """
    jname, jtype, gidxs, mrange, axis_origin, axis_dir = joint_tuple
    norm_origin = (np.array(axis_origin) - np.array(center)) * scale
    axis = np.array(axis_dir)
    n = np.linalg.norm(axis)
    if n > 1e-8:
        axis = axis / n

    T = np.eye(4)
    if jtype == "prismatic":
        T[:3, 3] = axis * q * scale
    else:  # revolute, continuous
        c, s = np.cos(q), np.sin(q)
        K = _skew_matrix(axis)
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        T[:3, :3] = R
        T[:3, 3] = norm_origin - R @ norm_origin  # rotation around point
    return T


def _subsample_verts(verts, max_pts=5000):
    """Subsample vertices if too many, for efficient KDTree queries."""
    if len(verts) <= max_pts:
        return verts
    step = max(1, len(verts) // max_pts)
    return verts[::step]


def detect_collision_promotions(joints, meshes_by_gidx, body_gidxs, moving_gidxs,
                                active_joint_indices, center, scale,
                                threshold=0.01, surface_samples=10000):
    """Detect passive articulated parts that would be collided by animated parts.

    Uses surface sampling (not just vertices) to detect collisions between mesh
    surfaces that intersect even when vertices are far apart (e.g., rack with
    large triangles passing through a thin door panel).

    For each active joint, moves its groups to target pose, samples points on
    the transformed mesh surface, then checks whether any passive joint's
    surface points are within `threshold` distance.

    Args:
        joints: list of (name, type, frozenset_groups, motion_range, axis_origin, axis_dir)
        meshes_by_gidx: {gidx: trimesh.Trimesh} with normalized vertices
        body_gidxs: frozenset of current body group indices
        moving_gidxs: frozenset of current moving group indices
        active_joint_indices: list of joint indices currently active
        center: normalization center [x, y, z]
        scale: normalization scale factor
        threshold: collision distance threshold in normalized space
        surface_samples: number of surface sample points per mesh group

    Returns:
        promoted_gidxs: set of group indices to promote from body→moving
        promoted_jidxs: set of joint indices to add to active list
    """
    from scipy.spatial import cKDTree

    active_set = set(active_joint_indices)

    # Step 1: Find passive articulated parts (body parts owned by a non-active joint)
    passive_joint_to_gidxs = {}  # ji → set of gidxs in body
    for ji, (jname, jtype, gidxs, *_) in enumerate(joints):
        if ji in active_set:
            continue
        body_owned = gidxs & body_gidxs
        if body_owned:
            passive_joint_to_gidxs[ji] = body_owned

    if not passive_joint_to_gidxs:
        return set(), set()

    # Step 2: Sample animated mesh surfaces at trajectory points
    animated_samples_list = []
    for ji in active_joint_indices:
        jname, jtype, gidxs, mrange, ax_orig, ax_dir = joints[ji]
        target_q = mrange  # positive direction for most furniture joints

        for q_frac in [0.5, 0.75, 1.0]:
            q = target_q * q_frac
            T = compute_joint_transform(joints[ji], q, center, scale)
            R = T[:3, :3]
            t = T[:3, 3]

            for gidx in gidxs:
                if gidx not in meshes_by_gidx:
                    continue
                mesh = meshes_by_gidx[gidx]
                if len(mesh.faces) == 0:
                    continue
                # Sample points on mesh surface (captures large triangles)
                n_samples = min(surface_samples, max(len(mesh.faces), 1000))
                samples, _ = trimesh.sample.sample_surface(mesh, n_samples)
                transformed = (R @ samples.T).T + t
                animated_samples_list.append(transformed)

    if not animated_samples_list:
        return set(), set()

    all_animated = np.vstack(animated_samples_list)
    tree = cKDTree(all_animated)

    # Step 3: Query each passive joint's surface samples
    promoted_gidxs = set()
    promoted_jidxs = set()

    for ji, body_owned in passive_joint_to_gidxs.items():
        collision_found = False
        for gidx in body_owned:
            if gidx not in meshes_by_gidx:
                continue
            mesh = meshes_by_gidx[gidx]
            if len(mesh.faces) == 0:
                continue
            # Sample passive mesh surface too
            n_samples = min(surface_samples, max(len(mesh.faces), 1000))
            samples, _ = trimesh.sample.sample_surface(mesh, n_samples)
            dists, _ = tree.query(samples)
            if dists.min() < threshold:
                collision_found = True
                break

        if collision_found:
            promoted_jidxs.add(ji)
            # Promote ALL groups under this joint that are in body
            for g in joints[ji][2]:  # gidxs field
                if g in body_gidxs:
                    promoted_gidxs.add(g)

    return promoted_gidxs, promoted_jidxs


def apply_collision_promotions(splits, joints, meshes_by_gidx, center, scale):
    """Post-process splits: promote collision-affected passive parts to moving.

    For each non-default split, runs collision detection between animated parts
    at target pose and passive articulated parts at rest. If collision detected,
    the passive parts move from body→moving and their joints are activated.

    Returns updated splits dict with same structure.
    """
    updated = {}
    for key, (body, moving, active_ji) in splits.items():
        if key == "default":
            # Default has ALL joints active → no passive articulated parts
            updated[key] = (body, moving, active_ji)
            continue

        promoted_gidxs, promoted_jidxs = detect_collision_promotions(
            joints, meshes_by_gidx, body, moving, active_ji, center, scale)

        if promoted_gidxs:
            new_moving = moving | frozenset(promoted_gidxs)
            new_body = body - frozenset(promoted_gidxs)
            new_active = sorted(set(active_ji) | promoted_jidxs)
            updated[key] = (frozenset(new_body), frozenset(new_moving), new_active)
            promoted_joint_names = [joints[ji][0] for ji in promoted_jidxs]
            print(f"    {key}: collision promoted {sorted(promoted_gidxs)} groups, "
                  f"+joints {promoted_joint_names} → {len(new_active)} total active")
        else:
            updated[key] = (body, moving, active_ji)

    return updated


def build_joint_param_tensor(joints, active_joint_indices, center, scale, n_max=8):
    """Build [N_max, 12] joint parameter tensor for a specific split.

    Per-joint row (12-dim):
        [0:3]  axis_origin (normalized coords)
        [3:6]  axis_direction (unit vector)
        [6:9]  joint type one-hot [revolute, prismatic, continuous]
        [9:11] motion range [min=0, max]
        [11]   exists flag (1.0 for real, 0.0 for padding)

    Only joints in active_joint_indices are included; rest are zero-padded.
    """
    TYPE_MAP = {"revolute": 0, "prismatic": 1, "continuous": 2}
    center = np.array(center)
    tensor = np.zeros((n_max, 12), dtype=np.float32)

    for slot, ji in enumerate(active_joint_indices):
        if slot >= n_max:
            break
        jname, jtype, gidxs, mrange, axis_origin, axis_dir = joints[ji]
        # Normalize axis origin with same transform as mesh vertices
        norm_origin = (axis_origin - center) * scale
        tensor[slot, 0:3] = norm_origin
        tensor[slot, 3:6] = axis_dir
        # One-hot type
        type_idx = TYPE_MAP.get(jtype, 0)
        tensor[slot, 6 + type_idx] = 1.0
        # Range: [0, motion_range] (lower is always 0 in our URDFs)
        tensor[slot, 9] = 0.0
        tensor[slot, 10] = mrange
        # Exists flag
        tensor[slot, 11] = 1.0

    return tensor


def export_split(meshes_by_gidx, body_gidxs, moving_gidxs, out_dir,
                 joints=None, active_joint_indices=None, center=None, scale=None):
    """Merge and export part0.obj (body), part1.obj (moving), and joints.npy."""
    os.makedirs(out_dir, exist_ok=True)

    body_meshes = [meshes_by_gidx[g] for g in sorted(body_gidxs) if g in meshes_by_gidx]
    moving_meshes = [meshes_by_gidx[g] for g in sorted(moving_gidxs) if g in meshes_by_gidx]

    if not moving_meshes:
        print(f"  WARNING: Empty split - body={len(body_meshes)}, moving=0")
        return False

    if body_meshes:
        part0 = trimesh.util.concatenate(body_meshes) if len(body_meshes) > 1 else body_meshes[0].copy()
    else:
        # All parts are moving (e.g. LiteDoorFactory) — create empty placeholder
        part0 = trimesh.Trimesh()
    part1 = trimesh.util.concatenate(moving_meshes) if len(moving_meshes) > 1 else moving_meshes[0].copy()

    # Strip visual data for clean export
    part0.visual = trimesh.visual.ColorVisuals()
    part1.visual = trimesh.visual.ColorVisuals()

    part0.export(os.path.join(out_dir, "part0.obj"))
    part1.export(os.path.join(out_dir, "part1.obj"))

    # Export joint param tensor if axis data available
    if joints is not None and active_joint_indices is not None and center is not None:
        tensor = build_joint_param_tensor(joints, active_joint_indices, center, scale)
        np.save(os.path.join(out_dir, "joints.npy"), tensor)

    return True


# ======================================================================
# Source detection
# ======================================================================

def detect_source(factory_name):
    """Detect factory source: 'im', 'physxnet', or 'physxmob'."""
    try:
        from physxnet_factory_rules import factory_dataset
        ds = factory_dataset(factory_name)
        if ds == "physxnet":
            return "physxnet"
        elif ds == "physxmob":
            return "physxmob"
    except ImportError:
        pass
    # Heuristic fallback for ad-hoc factory names
    if "PhysXNet" in factory_name:
        return "physxnet"
    if "PhysXMobility" in factory_name or "PhysXMob" in factory_name:
        return "physxmob"
    return "im"


def resolve_identifier(factory_name, seed, source):
    """Resolve the identifier for the scene directory.

    For all sources, the identifier is whatever was used as the directory name
    under outputs/{Factory}/. This matches how setup_physxnet_scene.py and
    render_articulation.py create scene directories.

    For PhysXNet, if the scene was set up with --id X, the directory is X.
    For IM, the directory is the seed number.
    """
    return str(seed), False


def get_scene_dir(factory_name, identifier, source, base_dir):
    """Get the scene directory for a given factory/identifier."""
    return os.path.join(base_dir, "outputs", factory_name, identifier)


def load_json_data(factory_name, identifier, source):
    """Load the full JSON data for a PhysXNet/PhysX_mobility object.

    Returns: dict or None.
    """
    try:
        if source == "physxnet":
            from physxnet_loader import load_physxnet_json
            return load_physxnet_json(identifier)
        elif source == "physxmob":
            from physxnet_loader import load_physxmob_json
            return load_physxmob_json(identifier)
    except ImportError:
        pass
    return None


def extract_part_info(json_data):
    """Extract part names and movement semantics from JSON data.

    Returns: {label_int: {"name": str, "movement": str, "is_moving": bool}}
    """
    if not json_data:
        return {}
    parts = json_data.get("parts", [])
    result = {}
    MOVE_KEYWORDS = ["rotat", "slid", "swing", "pivot", "open", "fold", "moves",
                     "translat", "travel", "reciprocat"]
    # Words that negate movement even if a keyword appears
    ANCHOR_KEYWORDS = ["anchor", "stationary", "does not move", "no movement",
                       "no independent movement", "remains fixed", "fixed in place"]
    for p in parts:
        label = p.get("label")
        if label is None:
            continue
        name = p.get("name", f"part_{label}")
        move_desc = p.get("Movement_description", "")
        desc_lower = move_desc.lower()
        is_anchor = any(kw in desc_lower for kw in ANCHOR_KEYWORDS)
        has_move = any(kw in desc_lower for kw in MOVE_KEYWORDS)
        is_moving = has_move and not is_anchor
        result[int(label)] = {
            "name": name,
            "movement": move_desc,
            "is_moving": is_moving,
        }
    return result


def validate_splits(splits, part_info, factory_name, identifier):
    """Cross-validate URDF-derived splits against JSON movement descriptions.

    Prints warnings for mismatches.
    """
    if not part_info:
        return
    default_split = splits.get("default")
    if not default_split:
        return
    body_gidxs, moving_gidxs = default_split[0], default_split[1]

    warnings = []
    for gidx in moving_gidxs:
        info = part_info.get(gidx)
        if info and not info["is_moving"]:
            warnings.append(
                f"    WARN: group {gidx} \"{info['name']}\" is URDF-moving "
                f"but JSON says: \"{info['movement'][:60]}\"")

    for gidx in body_gidxs:
        info = part_info.get(gidx)
        if info and info["is_moving"]:
            warnings.append(
                f"    WARN: group {gidx} \"{info['name']}\" is URDF-static "
                f"but JSON says: \"{info['movement'][:60]}\"")

    if warnings:
        print(f"  Cross-validation ({factory_name}/{identifier}):")
        for w in warnings:
            print(w)


# ======================================================================
# Main processing
# ======================================================================

def process_object(factory_name, seed, source, base_dir, output_dir, force=False):
    """Process a single object: load, normalize, split, export.

    Returns True if splits were generated.
    """
    identifier = None

    identifier, _ = resolve_identifier(factory_name, seed, source)
    if identifier is None:
        print(f"  ERROR: Cannot resolve identifier for {factory_name} seed={seed}")
        return False

    out_base = os.path.join(output_dir, factory_name, identifier)

    # Dedup check: skip if already processed
    if not force and os.path.exists(os.path.join(out_base, "metadata.json")):
        print(f"  SKIP {factory_name}/{identifier} (already exists)")
        return True

    # Clean old output if force-regenerating
    if force and os.path.isdir(out_base):
        import shutil
        shutil.rmtree(out_base)

    scene_dir = get_scene_dir(factory_name, identifier, source, base_dir)
    urdf_path = os.path.join(scene_dir, "scene.urdf")

    if not os.path.exists(urdf_path):
        print(f"  ERROR: No URDF at {urdf_path}")
        return False

    # 1. Load JSON semantic data (PhysXNet/PhysX_mobility only)
    json_data = load_json_data(factory_name, identifier, source)
    part_info = extract_part_info(json_data)

    # 2. Parse URDF
    joints, static_gidxs, all_gidxs = parse_urdf_joints(urdf_path)
    if not joints:
        print(f"  SKIP {factory_name}/{identifier}: no movable joints")
        return False

    print(f"  {factory_name}/{identifier}: {len(all_gidxs)} groups, "
          f"{len(joints)} joints, static={sorted(static_gidxs)}")
    if part_info:
        for gidx in sorted(all_gidxs):
            info = part_info.get(gidx, {})
            name = info.get("name", "?")
            is_moving = info.get("is_moving", False)
            tag = "MOVE" if is_moving else "STATIC"
            print(f"    group {gidx}: \"{name}\" [{tag}]")

    # 2. Load meshes
    meshes = load_group_meshes(scene_dir, factory_name, identifier)
    if not meshes:
        print(f"  ERROR: No meshes loaded from {scene_dir}")
        return False

    # Verify all URDF groups have meshes
    missing = all_gidxs - set(meshes.keys())
    if missing:
        print(f"  WARNING: Missing meshes for groups {sorted(missing)}")

    # 3. Normalize whole object
    center, scale = normalize_meshes_inplace(meshes, bound=0.95)

    # Verify range
    all_v = np.concatenate([m.vertices for m in meshes.values()])
    print(f"  Normalized range: [{all_v.min():.3f}, {all_v.max():.3f}]")

    # 4. Post-filter: remove prismatic joints with negligible normalized motion
    #    In normalized space, prismatic motion = motion_range * scale.
    #    Threshold 0.1 ≈ 5% of object size — below this it's invisible at 256px.
    MIN_NORMALIZED_PRISMATIC = 0.1
    filtered_joints = []
    for jname, jtype, gidxs, mrange, ax_orig, ax_dir in joints:
        if jtype == "prismatic":
            norm_motion = mrange * scale
            if norm_motion < MIN_NORMALIZED_PRISMATIC:
                print(f"  FILTER joint {jname}: prismatic {mrange:.4f}m "
                      f"= {norm_motion:.4f} normalized (< {MIN_NORMALIZED_PRISMATIC})")
                continue
        filtered_joints.append((jname, jtype, gidxs, mrange, ax_orig, ax_dir))

    if not filtered_joints:
        print(f"  SKIP {factory_name}/{identifier}: no joints with significant motion")
        return False

    if len(filtered_joints) < len(joints):
        # Recompute static groups with filtered joints
        filtered_moving = set()
        for _, _, gidxs, *_ in filtered_joints:
            filtered_moving.update(gidxs)
        static_gidxs = all_gidxs - filtered_moving
        joints = filtered_joints
        print(f"  After filtering: {len(joints)} joints, static={sorted(static_gidxs)}")

    # Build splits (now includes active_joint_indices per split)
    splits = build_splits(joints, static_gidxs, all_gidxs)
    if not splits:
        print(f"  ERROR: No valid splits generated")
        return False

    # 4b. Collision-aware promotion: detect passive parts pushed by animated parts
    original_active = {k: set(v[2]) for k, v in splits.items()}
    splits = apply_collision_promotions(splits, joints, meshes, center, scale)
    # Track which joints were promoted per split
    promoted_joints_per_split = {}
    for k, (_, _, active_ji) in splits.items():
        promoted = set(active_ji) - original_active.get(k, set())
        if promoted:
            promoted_joints_per_split[k] = [joints[ji][0] for ji in sorted(promoted)]

    # Cross-validate URDF splits vs JSON movement descriptions
    validate_splits(splits, part_info, factory_name, identifier)

    # 5. Export each split (part0.obj + part1.obj + joints.npy)
    n_exported = 0
    split_info = {}

    for key, (body, moving, active_ji) in sorted(splits.items()):
        # All splits go into subdirs (default/ included for co-located videos)
        split_dir = os.path.join(out_base, key)

        ok = export_split(meshes, body, moving, split_dir,
                          joints=joints, active_joint_indices=active_ji,
                          center=center, scale=scale)
        if ok:
            n_exported += 1
            # Per-split active joint names for metadata
            active_joint_names = [joints[ji][0] for ji in active_ji]
            split_info[key] = {
                "body": sorted(body),
                "moving": sorted(moving),
                "active_joints": active_joint_names,
                "n_joints": len(active_ji),
            }
            # Record collision-promoted joints if any
            if key in promoted_joints_per_split:
                split_info[key]["promoted_joints"] = promoted_joints_per_split[key]
            # Add joint type info for type-based splits
            if key.startswith("anim") and not key.startswith("anim1") or key == "anim1":
                animode_num = key.replace("anim", "")
                if animode_num.isdigit():
                    num = int(animode_num)
                    if num < 10:
                        type_map = {0: "revolute", 1: "prismatic", 2: "continuous"}
                        if num in type_map:
                            split_info[key]["type"] = type_map[num]

    # 6. Write metadata (with axis info per joint)
    category = json_data.get("category", factory_name) if json_data else factory_name
    center_arr = np.array(center)
    metadata = {
        "factory": factory_name,
        "identifier": identifier,
        "source": source,
        "category": category,
        "normalize": {"center": center, "scale": scale},
        "parts": {
            str(gidx): {
                "name": part_info[gidx]["name"],
                "is_moving": part_info[gidx]["is_moving"],
                "movement_desc": part_info[gidx]["movement"],
            }
            for gidx in sorted(all_gidxs) if gidx in part_info
        } if part_info else {},
        "joints": [
            {
                "name": jname, "type": jtype, "groups": sorted(gidxs),
                "motion_range": round(mrange, 6),
                "axis_origin": ((ax_orig - center_arr) * scale).round(6).tolist(),
                "axis_direction": ax_dir.round(6).tolist(),
            }
            for jname, jtype, gidxs, mrange, ax_orig, ax_dir in joints
        ],
        "splits": split_info,
    }

    meta_path = os.path.join(out_base, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Exported {n_exported} splits to {out_base}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Unified dual-part split precompute")
    parser.add_argument("--factory", type=str, help="Factory name")
    parser.add_argument("--seed", type=str, help="Seed (IM) or object_id (PhysXNet)")
    parser.add_argument("--output_dir", type=str, default="./precompute",
                        help="Output root directory")
    parser.add_argument("--base", type=str, default=None,
                        help="Base directory with outputs/ (default: auto-detect)")
    parser.add_argument("--source", choices=["auto", "im", "physxnet", "physxmob"],
                        default="auto", help="Data source type")
    parser.add_argument("--all", action="store_true",
                        help="Process all objects in batch_test_physxnet.py OBJECTS list")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if output exists")
    parser.add_argument("--max_seeds", type=int, default=0,
                        help="Max seeds per factory (0=unlimited)")
    parser.add_argument("--shard", type=str, default=None,
                        help="Process shard K/N (e.g. 0/4 = first quarter)")
    args = parser.parse_args()

    # Default base directories
    INFINIGEN_SIM_BASE = "/mnt/data/yurh/Infinigen-Sim"
    IM_BASE = "/mnt/cpfs/yurh/Infinite-Mobility"

    if args.all:
        # Process all objects from batch_test_physxnet.py OBJECTS list
        # Plus IM factories if available
        objects = []

        # PhysXNet/PhysX_mobility objects
        physxnet_objects = [
            ("FurniturePhysXNetFactory", "46088", "physxnet"),
            ("ChairPhysXNetFactory", "39882", "physxnet"),
            ("LightingPhysXNetFactory", "17394", "physxnet"),
            ("ElectronicsPhysXNetFactory", "6723", "physxnet"),
            ("AppliancePhysXNetFactory", "10459", "physxnet"),
            ("BagPhysXNetFactory", "8871", "physxnet"),
            ("DoorPhysXNetFactory", "9288", "physxnet"),
            ("ContainerPhysXNetFactory", "47219", "physxnet"),
            ("AgriPhysXMobilityFactory", "101064", "physxmob"),
            ("ArchPhysXMobilityFactory", "103242", "physxmob"),
            ("BathPhysXMobilityFactory", "101517", "physxmob"),
            ("BuildPhysXMobilityFactory", "102903", "physxmob"),
        ]

        # IM factories (check what's available)
        im_factories_seeds = []
        im_outputs = os.path.join(IM_BASE, "outputs")
        if os.path.isdir(im_outputs):
            for fname in sorted(os.listdir(im_outputs)):
                fdir = os.path.join(im_outputs, fname)
                if not os.path.isdir(fdir):
                    continue
                # Skip PhysXNet factories
                if "PhysX" in fname:
                    continue
                for seed_name in sorted(os.listdir(fdir)):
                    seed_dir = os.path.join(fdir, seed_name)
                    if os.path.isdir(seed_dir) and seed_name.isdigit():
                        urdf = os.path.join(seed_dir, "scene.urdf")
                        if os.path.exists(urdf):
                            im_factories_seeds.append((fname, seed_name, "im"))

        # Apply --max_seeds: limit seeds per factory
        if args.max_seeds > 0:
            from collections import OrderedDict
            factory_seeds = OrderedDict()
            for fname, seed, src in im_factories_seeds:
                factory_seeds.setdefault(fname, []).append((fname, seed, src))
            im_factories_seeds = []
            for fname, entries in factory_seeds.items():
                im_factories_seeds.extend(entries[:args.max_seeds])

        all_objects = [(f, s, src, args.base or INFINIGEN_SIM_BASE)
                       for f, s, src in physxnet_objects]
        all_objects += [(f, s, src, IM_BASE)
                        for f, s, src in im_factories_seeds]

        print(f"PhysXNet/PhysX_mobility: {len(physxnet_objects)} objects")
        print(f"IM factories: {len(im_factories_seeds)} seeds")
        print(f"Total: {len(all_objects)} objects")

        # Apply --shard for parallel runs
        if args.shard:
            k, n = map(int, args.shard.split("/"))
            shard_size = len(all_objects) // n + (1 if len(all_objects) % n else 0)
            all_objects = all_objects[k * shard_size:(k + 1) * shard_size]
            print(f"Shard {k}/{n}: processing {len(all_objects)} objects")

        success = 0
        total = 0
        for factory, seed, source, base in all_objects:
            total += 1
            ok = process_object(factory, seed, source, base, args.output_dir, args.force)
            if ok:
                success += 1

        print(f"\nDone: {success}/{total} objects processed")
        print(f"Output: {os.path.abspath(args.output_dir)}")

    elif args.factory and args.seed:
        source = args.source
        if source == "auto":
            source = detect_source(args.factory)

        if args.base:
            base = args.base
        elif source == "im":
            base = IM_BASE
        else:
            base = INFINIGEN_SIM_BASE

        ok = process_object(args.factory, args.seed, source, base, args.output_dir, args.force)
        if not ok:
            sys.exit(1)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python split_precompute.py --factory ElectronicsPhysXNetFactory --seed 10005")
        print("  python split_precompute.py --factory DishwasherFactory --seed 0 --base /mnt/cpfs/yurh/Infinite-Mobility")
        print("  python split_precompute.py --all --output_dir ./precompute")
        sys.exit(1)


if __name__ == "__main__":
    main()
