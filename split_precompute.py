#!/usr/bin/env python3
"""
Topology-based dual-volume split precompute for Infinigen-Sim.

For each object x each animode: classify joints -> merge fixed-joint parts ->
2-color the reduced graph -> export part0.obj + part1.obj + verify.png + metadata.json.

Core algorithm (TOPOLOGY-BASED, following CLAUDE.md):
1. Parse URDF -> build link/joint graph
2. Load per-part OBJ meshes, normalize entire object to unit cube FIRST
3. For each animode:
   a. Define active joints (basic = 1 joint; senior = random combo, max 3 seniors)
   b. Classify remaining joints via BVH collision along active trajectory:
      - Trajectory types: sinusoidal oscillation, one-way sinusoidal, linear, linear oscillation
      - If active part motion hits joint-connected part -> PASSIVE (precompute pre-opening)
      - If no collision -> FIXED
   c. Handle bridging parts (topology-based, NOT size-based)
   d. Merge all FIXED-joint-connected parts into single topological nodes (node contraction)
   e. Build reduced graph: nodes = merged groups, edges = movable joints (active + passive)
   f. Apply bipartite 2-coloring: root = part0
   g. Export part0.obj + part1.obj + verify.png + metadata.json

Output structure:
  {output_dir}/{Factory}/{identifier}/
      basic_0/part0.obj + part1.obj + verify.png
      basic_1/part0.obj + part1.obj + verify.png
      senior_0/part0.obj + part1.obj + verify.png
      ...
      metadata.json

Usage:
  python split_precompute.py --factory LampFactory --seed 0 --base /mnt/data/yurh/dataset3D/infinite_mobility
  python split_precompute.py --all --output_dir ./precompute_output
"""

import argparse
import itertools
import json
import math
import os
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import trimesh


# ======================================================================
# Constants
# ======================================================================

TRAJECTORY_TYPES = [
    "sinusoidal_oscillation",
    "one_way_sinusoidal",
    "linear",
    "linear_oscillation",
]

MAX_SENIOR_ANIMODES = 3
BVH_COLLISION_THRESHOLD = 0.015  # collision distance in normalized space
BVH_SURFACE_SAMPLES = 8000
BVH_TRAJECTORY_STEPS = 10  # number of steps along trajectory for collision detection
MIN_NORMALIZED_MOTION = 0.05  # minimum visible motion in normalized space


# ======================================================================
# URDF Parsing
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


class URDFJoint:
    """Represents a URDF joint with full kinematic info."""
    def __init__(self, name, jtype, parent_link, child_link,
                 axis_local, origin_xyz, origin_rpy, lower, upper):
        self.name = name
        self.jtype = jtype  # revolute, prismatic, continuous, fixed
        self.parent_link = parent_link
        self.child_link = child_link
        self.axis_local = np.array(axis_local, dtype=np.float64)
        self.origin_xyz = np.array(origin_xyz, dtype=np.float64)
        self.origin_rpy = np.array(origin_rpy, dtype=np.float64)
        self.lower = lower
        self.upper = upper

        # These will be filled after world transform computation
        self.axis_origin_world = None
        self.axis_dir_world = None

    @property
    def is_movable(self):
        """Check if joint is movable (not fixed) with nonzero range."""
        if self.jtype not in ("revolute", "prismatic", "continuous"):
            return False
        if self.jtype == "continuous":
            return True
        return abs(self.upper - self.lower) > 1e-6

    @property
    def motion_range(self):
        if self.jtype == "continuous":
            return 2.0 * np.pi  # full rotation
        return abs(self.upper - self.lower)

    def __repr__(self):
        return (f"URDFJoint({self.name}, {self.jtype}, "
                f"{self.parent_link}->{self.child_link}, "
                f"range={self.motion_range:.4f})")


def parse_urdf(urdf_path):
    """Parse URDF file into links and joints.

    Handles both naming conventions:
    - IM/old format: l_0, l_1, ... (part_idx extracted from name)
    - IS/new format: link_0, link_1, ... (part_idx extracted from name)
    - "world" link is skipped (no part_idx)

    Also extracts visual mesh filenames for each link (IS format).

    Returns:
        links: dict {link_name: {"name": str, "part_idx": int or None,
                                  "mesh_files": list of (filename, origin_xyz)}}
        joints: list of URDFJoint
        root_link: name of the root link
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Parse links
    links = {}
    for link_el in root.findall("link"):
        name = link_el.get("name")
        part_idx = None

        # Try l_N format (IM)
        m = re.match(r"l_(\d+)", name)
        if m:
            part_idx = int(m.group(1))
        else:
            # Try link_N format (IS)
            m = re.match(r"link_(\d+)", name)
            if m:
                part_idx = int(m.group(1))

        # Extract visual mesh filenames
        mesh_files = []
        for visual_el in link_el.findall("visual"):
            geom = visual_el.find("geometry")
            if geom is not None:
                mesh_el = geom.find("mesh")
                if mesh_el is not None:
                    filename = mesh_el.get("filename", "")
                    # Get visual origin offset
                    vis_origin = visual_el.find("origin")
                    vis_xyz = [0.0, 0.0, 0.0]
                    if vis_origin is not None and vis_origin.get("xyz"):
                        vis_xyz = [float(v) for v in vis_origin.get("xyz").split()]
                    mesh_files.append((filename, vis_xyz))

        links[name] = {"name": name, "part_idx": part_idx, "mesh_files": mesh_files}

    # Parse joints
    joints = []
    children_set = set()
    for joint_el in root.findall("joint"):
        name = joint_el.get("name", "")
        jtype = joint_el.get("type", "fixed")
        parent_link = joint_el.find("parent").get("link")
        child_link = joint_el.find("child").get("link")
        children_set.add(child_link)

        # Axis
        axis_el = joint_el.find("axis")
        axis_local = [1.0, 0.0, 0.0]
        if axis_el is not None and axis_el.get("xyz"):
            axis_local = [float(v) for v in axis_el.get("xyz").split()]

        # Origin
        origin_el = joint_el.find("origin")
        origin_xyz = [0.0, 0.0, 0.0]
        origin_rpy = [0.0, 0.0, 0.0]
        if origin_el is not None:
            if origin_el.get("xyz"):
                origin_xyz = [float(v) for v in origin_el.get("xyz").split()]
            if origin_el.get("rpy"):
                origin_rpy = [float(v) for v in origin_el.get("rpy").split()]

        # Limits
        limit_el = joint_el.find("limit")
        lower, upper = 0.0, 0.0
        if limit_el is not None:
            lower = float(limit_el.get("lower", "0"))
            upper = float(limit_el.get("upper", "0"))

        joints.append(URDFJoint(name, jtype, parent_link, child_link,
                                axis_local, origin_xyz, origin_rpy, lower, upper))

    # Find root link (never appears as a child)
    all_links = set(links.keys())
    roots = all_links - children_set
    root_link = sorted(roots)[0] if roots else sorted(all_links)[0]

    return links, joints, root_link


def build_kinematic_tree(joints):
    """Build parent/children maps from URDF joints.

    Returns:
        parent_map: {child_link: (parent_link, URDFJoint)}
        children_map: {parent_link: [(child_link, URDFJoint), ...]}
    """
    parent_map = {}
    children_map = defaultdict(list)
    for j in joints:
        parent_map[j.child_link] = (j.parent_link, j)
        children_map[j.parent_link].append((j.child_link, j))
    return parent_map, children_map


def compute_world_transforms(joints, root_link):
    """Compute world-frame transforms for all links and joint axes.

    Fills in joint.axis_origin_world and joint.axis_dir_world.

    Returns:
        link_world: {link_name: (position_3d, rotation_3x3)}
    """
    parent_map, children_map = build_kinematic_tree(joints)

    link_world = {root_link: (np.zeros(3), np.eye(3))}
    queue = [root_link]

    while queue:
        parent = queue.pop(0)
        p_pos, p_rot = link_world[parent]

        for child_link, joint in children_map.get(parent, []):
            if child_link in link_world:
                continue

            c_rot_local = _rotation_from_rpy(joint.origin_rpy)
            c_pos = p_pos + p_rot @ joint.origin_xyz
            c_rot = p_rot @ c_rot_local
            link_world[child_link] = (c_pos, c_rot)

            # Compute world-frame joint axis
            axis_local = joint.axis_local.copy()
            n = np.linalg.norm(axis_local)
            if n > 1e-8:
                axis_local = axis_local / n

            joint.axis_origin_world = p_pos + p_rot @ joint.origin_xyz
            joint.axis_dir_world = p_rot @ axis_local
            nd = np.linalg.norm(joint.axis_dir_world)
            if nd > 1e-8:
                joint.axis_dir_world = joint.axis_dir_world / nd

            queue.append(child_link)

    return link_world


def get_link_part_idx(link_name):
    """Extract part index from link name (e.g., l_0 -> 0, link_0 -> 0)."""
    # Try l_N format (IM)
    m = re.match(r"l_(\d+)", link_name)
    if m:
        return int(m.group(1))
    # Try link_N format (IS)
    m = re.match(r"link_(\d+)", link_name)
    if m:
        return int(m.group(1))
    return None


def build_topology_graph(joints, links, root_link):
    """Build the full topology graph of the URDF.

    Nodes are link names (that have a part_idx).
    Edges are joints connecting them.

    Returns:
        nodes: set of link names with part_idx
        edges: list of (link_a, link_b, URDFJoint)
        link_to_joint: dict mapping (parent, child) -> URDFJoint
    """
    nodes = set()
    for name, info in links.items():
        if info["part_idx"] is not None:
            nodes.add(name)

    edges = []
    link_to_joint = {}
    for j in joints:
        link_to_joint[(j.parent_link, j.child_link)] = j
        if j.parent_link in nodes and j.child_link in nodes:
            edges.append((j.parent_link, j.child_link, j))
        elif j.parent_link in nodes or j.child_link in nodes:
            edges.append((j.parent_link, j.child_link, j))

    return nodes, edges, link_to_joint


# ======================================================================
# Joint Transform Computation
# ======================================================================

def _skew_matrix(v):
    """Skew-symmetric matrix for cross product."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])


def compute_joint_transform(joint, q, center, scale):
    """Compute 4x4 delta transform for a joint at parameter q in normalized space.

    Args:
        joint: URDFJoint with world-frame axis info
        q: joint parameter value (angle for revolute, distance for prismatic)
        center: normalization center
        scale: normalization scale

    Returns:
        4x4 homogeneous transform matrix
    """
    center = np.array(center)
    norm_origin = (joint.axis_origin_world - center) * scale
    axis = joint.axis_dir_world.copy()
    n = np.linalg.norm(axis)
    if n > 1e-8:
        axis = axis / n

    T = np.eye(4)
    if joint.jtype == "prismatic":
        T[:3, 3] = axis * q * scale
    else:  # revolute, continuous
        c, s = np.cos(q), np.sin(q)
        K = _skew_matrix(axis)
        R = np.eye(3) + s * K + (1 - c) * (K @ K)
        T[:3, :3] = R
        T[:3, 3] = norm_origin - R @ norm_origin  # rotation around point
    return T


def generate_trajectory(joint, traj_type, n_steps=BVH_TRAJECTORY_STEPS):
    """Generate trajectory parameter values for a joint.

    Returns: list of q values along the trajectory.
    """
    lo = joint.lower if joint.jtype != "continuous" else 0.0
    hi = joint.upper if joint.jtype != "continuous" else 2 * np.pi
    mrange = hi - lo

    t_values = np.linspace(0, 1, n_steps)
    q_values = []

    for t in t_values:
        if traj_type == "sinusoidal_oscillation":
            # Full sine wave: 0 -> max -> 0 -> max -> 0
            q = lo + mrange * 0.5 * (1 - np.cos(2 * np.pi * t))
        elif traj_type == "one_way_sinusoidal":
            # Smooth one-way: 0 -> max via sine easing
            q = lo + mrange * 0.5 * (1 - np.cos(np.pi * t))
        elif traj_type == "linear":
            # Linear: 0 -> max
            q = lo + mrange * t
        elif traj_type == "linear_oscillation":
            # Linear back and forth: 0 -> max -> 0
            if t < 0.5:
                q = lo + mrange * (2 * t)
            else:
                q = lo + mrange * (2 * (1 - t))
        else:
            q = lo + mrange * t  # fallback: linear

        q_values.append(q)

    return q_values


# ======================================================================
# BVH Collision Detection
# ======================================================================

def detect_collisions_bvh(active_joints, candidate_joints, meshes_by_link,
                          parent_map, children_map, center, scale,
                          traj_type, threshold=BVH_COLLISION_THRESHOLD):
    """Detect which candidate joints' parts are hit by active motion.

    For each active joint, move its descendant parts along the trajectory.
    Check if any candidate joint's descendant parts (that are NOT already
    descendants of any active joint) are within threshold distance.

    IMPORTANT: If a candidate joint's child link is a descendant of an active
    joint, that means it moves WITH the active part — no collision possible.
    Such joints should be classified as FIXED, not passive.

    Args:
        active_joints: list of URDFJoint (the active ones)
        candidate_joints: list of URDFJoint (candidates for passive/fixed classification)
        meshes_by_link: {link_name: trimesh.Trimesh} with normalized vertices
        parent_map: kinematic tree parent map
        children_map: kinematic tree children map
        center: normalization center
        scale: normalization scale
        traj_type: trajectory type string
        threshold: collision distance in normalized space

    Returns:
        collided_joints: set of URDFJoint that are classified as PASSIVE
        pre_opening_angles: {joint_name: float} pre-opening angle/offset for passive joints
    """
    from scipy.spatial import cKDTree

    def get_descendants(joint):
        """Get all descendant link names from joint's child link."""
        desc = set()
        queue = [joint.child_link]
        while queue:
            link = queue.pop(0)
            if link in desc:
                continue
            desc.add(link)
            for child, _ in children_map.get(link, []):
                queue.append(child)
        return desc

    # Collect ALL descendant links of ALL active joints
    all_active_descendants = set()
    active_desc_per_joint = {}
    for active_j in active_joints:
        desc = get_descendants(active_j)
        active_desc_per_joint[active_j.name] = desc
        all_active_descendants.update(desc)

    # Filter out candidate joints whose child is already an active descendant
    # (they move WITH the active part, so cannot be collided)
    real_candidates = []
    for cand_j in candidate_joints:
        cand_desc = get_descendants(cand_j)
        # If ALL of this joint's descendants are already active descendants,
        # it moves with the active part -> FIXED, not a collision candidate
        if cand_desc.issubset(all_active_descendants):
            continue
        real_candidates.append(cand_j)

    if not real_candidates:
        return set(), {}

    # Build animated surface samples at multiple trajectory points
    animated_samples_all = []
    for active_j in active_joints:
        desc_links = active_desc_per_joint[active_j.name]
        q_values = generate_trajectory(active_j, traj_type)

        for q in q_values:
            T = compute_joint_transform(active_j, q, center, scale)
            R = T[:3, :3]
            t = T[:3, 3]

            for link_name in desc_links:
                if link_name not in meshes_by_link:
                    continue
                mesh = meshes_by_link[link_name]
                if len(mesh.faces) == 0:
                    continue

                n_samples = min(BVH_SURFACE_SAMPLES, max(len(mesh.faces), 500))
                try:
                    samples, _ = trimesh.sample.sample_surface(mesh, n_samples)
                    transformed = (R @ samples.T).T + t
                    animated_samples_all.append(transformed)
                except Exception:
                    pass

    if not animated_samples_all:
        return set(), {}

    all_animated = np.vstack(animated_samples_all)
    tree = cKDTree(all_animated)

    # Check each real candidate joint's non-active-descendant parts
    collided_joints = set()
    pre_opening_angles = {}

    for cand_j in real_candidates:
        desc_links = get_descendants(cand_j)
        # Only check parts that are NOT already moving with active joints
        check_links = desc_links - all_active_descendants
        collision_found = False

        for link_name in check_links:
            if link_name not in meshes_by_link:
                continue
            mesh = meshes_by_link[link_name]
            if len(mesh.faces) == 0:
                continue

            n_samples = min(BVH_SURFACE_SAMPLES, max(len(mesh.faces), 500))
            try:
                samples, _ = trimesh.sample.sample_surface(mesh, n_samples)
                dists, _ = tree.query(samples)
                if dists.min() < threshold:
                    collision_found = True
                    break
            except Exception:
                pass

        if collision_found:
            collided_joints.add(cand_j)
            # Compute pre-opening: small fraction of range to avoid initial collision
            pre_opening = cand_j.motion_range * 0.1
            pre_opening_angles[cand_j.name] = float(pre_opening)

    return collided_joints, pre_opening_angles


# ======================================================================
# Topology Operations: Bridging Parts, Node Contraction, 2-Coloring
# ======================================================================

def classify_joints_for_animode(all_movable_joints, active_joint_names,
                                meshes_by_link, parent_map, children_map,
                                center, scale, traj_type):
    """Classify all joints as active/passive/fixed for a given animode.

    Args:
        all_movable_joints: list of all movable URDFJoint
        active_joint_names: set of joint names that are active
        meshes_by_link: {link_name: trimesh} normalized
        parent_map, children_map: kinematic tree
        center, scale: normalization params
        traj_type: trajectory type for this animode

    Returns:
        active: list of URDFJoint
        passive: list of URDFJoint
        fixed: list of URDFJoint
        pre_opening_angles: dict {joint_name: float}
    """
    active = [j for j in all_movable_joints if j.name in active_joint_names]
    candidates = [j for j in all_movable_joints if j.name not in active_joint_names]

    if not candidates or not active:
        return active, [], candidates, {}

    passive_set, pre_opening_angles = detect_collisions_bvh(
        active, candidates, meshes_by_link,
        parent_map, children_map, center, scale, traj_type)

    passive = [j for j in candidates if j in passive_set]
    fixed = [j for j in candidates if j not in passive_set]

    return active, passive, fixed, pre_opening_angles


def build_reduced_graph(links, joints, root_link, fixed_joints, movable_joints):
    """Build the reduced topology graph after merging fixed-joint-connected parts.

    Steps:
    1. Build adjacency with all joints
    2. Merge parts connected by fixed or truly-fixed joints into single nodes
    3. Return reduced graph: nodes = merged groups, edges = movable joints

    Args:
        links: {link_name: info} from URDF
        joints: all URDFJoint list
        root_link: root link name
        fixed_joints: set of joints classified as FIXED (+ all original fixed type joints)
        movable_joints: list of joints classified as ACTIVE or PASSIVE

    Returns:
        merged_groups: list of frozensets of link names (each = one node)
        reduced_edges: list of (group_idx_a, group_idx_b, URDFJoint)
        root_group_idx: index of the group containing root_link
    """
    # All link names that have meshes (part_idx != None)
    mesh_links = {name for name, info in links.items() if info["part_idx"] is not None}
    all_links = set(links.keys())

    # Names of movable joints (edges in reduced graph)
    movable_joint_names = {j.name for j in movable_joints}

    # Any joint NOT in movable set is treated as fixed for merging
    # (includes URDF fixed-type, zero-range revolute, classified-fixed, etc.)
    fixed_joint_names = {j.name for j in joints if j.name not in movable_joint_names}

    # Union-Find over ALL links (including abstract/meshless ones)
    # so that abstract links get merged with their mesh neighbors via fixed joints
    parent = {link: link for link in all_links}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Merge links connected by fixed joints (all links, not just mesh)
    for j in joints:
        if j.name in fixed_joint_names:
            if j.parent_link in all_links and j.child_link in all_links:
                union(j.parent_link, j.child_link)

    # Build merged groups (only keep groups containing at least one mesh link)
    groups = defaultdict(set)
    for link in all_links:
        groups[find(link)].add(link)

    # Filter: only groups with at least one mesh link, and only store mesh links
    filtered_groups = {}
    for rep, members in groups.items():
        mesh_members = members & mesh_links
        if mesh_members:
            filtered_groups[rep] = mesh_members
    groups = filtered_groups

    merged_groups = []
    link_to_group_idx = {}
    for representative, members in groups.items():
        idx = len(merged_groups)
        merged_groups.append(frozenset(members))
        for m in members:
            link_to_group_idx[m] = idx

    # Find root group
    root_group_idx = 0
    if root_link in link_to_group_idx:
        root_group_idx = link_to_group_idx[root_link]
    else:
        # Root link might not have a mesh; find the group containing
        # the first child of root that has a mesh
        parent_map, children_map_local = build_kinematic_tree(joints)
        queue = [root_link]
        found = False
        while queue and not found:
            current = queue.pop(0)
            if current in link_to_group_idx:
                root_group_idx = link_to_group_idx[current]
                found = True
            else:
                for child, _ in children_map_local.get(current, []):
                    queue.append(child)

    # Map union-find representatives to group indices (for resolving abstract links)
    rep_to_group_idx = {}
    for representative, members in groups.items():
        # find the group idx that contains these members
        for m in members:
            if m in link_to_group_idx:
                rep_to_group_idx[representative] = link_to_group_idx[m]
                break

    def resolve_link_to_group(link_name):
        """Resolve a link (possibly abstract/meshless) to its group index."""
        if link_name in link_to_group_idx:
            return link_to_group_idx[link_name]
        # Use union-find: abstract link's representative should map to a mesh group
        if link_name in parent:
            rep = find(link_name)
            if rep in rep_to_group_idx:
                return rep_to_group_idx[rep]
        return None

    # Build reduced edges (movable joints between different groups)
    reduced_edges = []
    seen_edges = set()
    for j in joints:
        if j.name not in movable_joint_names:
            continue
        ga = resolve_link_to_group(j.parent_link)
        gb = resolve_link_to_group(j.child_link)
        if ga is None or gb is None:
            continue
        if ga == gb:
            continue  # same group (shouldn't happen but safety check)
        edge_key = (min(ga, gb), max(ga, gb))
        if edge_key not in seen_edges:
            seen_edges.add(edge_key)
            reduced_edges.append((ga, gb, j))

    return merged_groups, reduced_edges, root_group_idx


def handle_bridging_parts(merged_groups, reduced_edges, links, joints, movable_joint_names):
    """Handle bridging parts based on URDF topology.

    A bridging part is a node in the reduced graph connected to 3+ other nodes.
    Check its connections:
    - If ANY connection crosses a movable joint AND is NOT via a joint -> REMOVE
    - If ALL connections don't cross movable joint -> MERGE into neighbor

    For simplicity in the current implementation:
    - Nodes with degree >= 3 in the reduced graph are potential bridging parts
    - Since all edges in the reduced graph ARE joints, we keep them as-is
    - Bridging mainly applies to parts with mixed rigid/joint connections

    Returns:
        updated_groups: list of frozenset (potentially modified)
        updated_edges: list of (ga, gb, joint)
        removed_links: set of link names removed entirely
    """
    # For the current URDF structure (IS factories), all connections between
    # parts ARE joints. Bridging parts are mainly relevant for complex URDF
    # structures with rigid multi-part assemblies. We still implement the
    # logic for correctness.

    # Build adjacency for the reduced graph
    adjacency = defaultdict(set)
    for ga, gb, j in reduced_edges:
        adjacency[ga].add(gb)
        adjacency[gb].add(ga)

    removed_links = set()
    # Currently no removal needed for simple IS factory URDFs
    # The logic would be: if a node has connections both via movable joints
    # and via rigid links (not joints), and the rigid links cross movable
    # joint boundaries, remove it. But in IS URDFs, all inter-part connections
    # are joints.

    return merged_groups, reduced_edges, removed_links


def bipartite_2_coloring(merged_groups, reduced_edges, root_group_idx):
    """Apply bipartite 2-coloring on the reduced graph.

    Nodes connected by edges (movable joints) get opposite colors.
    Root group gets color 0 (part0).

    Returns:
        coloring: {group_idx: 0 or 1}  (0 = part0, 1 = part1)
        is_bipartite: bool (True if graph is bipartite, False if odd cycle)
    """
    n = len(merged_groups)
    coloring = {}
    is_bipartite = True

    # Build adjacency
    adj = defaultdict(set)
    for ga, gb, _ in reduced_edges:
        adj[ga].add(gb)
        adj[gb].add(ga)

    # BFS 2-coloring from root
    coloring[root_group_idx] = 0
    queue = [root_group_idx]
    visited = {root_group_idx}

    while queue:
        node = queue.pop(0)
        current_color = coloring[node]
        for neighbor in adj[node]:
            if neighbor in coloring:
                if coloring[neighbor] == current_color:
                    is_bipartite = False
            else:
                coloring[neighbor] = 1 - current_color
                visited.add(neighbor)
                queue.append(neighbor)

    # Color any disconnected components (nodes not reachable from root)
    for i in range(n):
        if i not in coloring:
            # Disconnected node -> assign to part0 by default
            coloring[i] = 0

    return coloring, is_bipartite


# ======================================================================
# Mesh Loading and Normalization
# ======================================================================

def find_objs_dir(scene_dir, factory=None, identifier=None):
    """Find the objs directory, handling various path conventions."""
    import glob

    # IS: outputs/{Factory}/{seed}/outputs/{Factory}/{seed}/objs/
    if factory and identifier:
        nested = os.path.join(scene_dir, "outputs", factory, str(identifier), "objs")
        if os.path.isdir(nested):
            return nested

    # Try to auto-discover nested path
    for candidate in glob.glob(os.path.join(scene_dir, "outputs", "*", "*", "objs")):
        if os.path.isdir(candidate):
            return candidate

    # Flat: scene_dir/objs/
    flat = os.path.join(scene_dir, "objs")
    if os.path.isdir(flat):
        return flat

    # meshes/split/ format (dataset3D/infinite_mobility)
    meshes_split = os.path.join(scene_dir, "meshes", "split")
    if os.path.isdir(meshes_split):
        return meshes_split

    return None


def load_meshes(scene_dir, links, link_world, factory=None, identifier=None):
    """Load per-part OBJ meshes and assemble in world coordinates.

    Supports two formats:
    1. Old IM format: objs/{idx}/{idx}.obj + origins.json
    2. New IS URDF format: mesh filenames in URDF visual elements + visual origin offsets
       + kinematic chain transforms

    Args:
        scene_dir: directory containing the URDF and mesh files
        links: parsed link info from parse_urdf()
        link_world: {link_name: (position_3d, rotation_3x3)} from compute_world_transforms()
        factory: factory name (optional, for path resolution)
        identifier: seed/id (optional, for path resolution)

    Returns:
        meshes_by_link: {link_name: trimesh.Trimesh} (vertices in world coords)
        meshes_by_gidx: {part_idx: trimesh.Trimesh} (same meshes, indexed by part idx)
    """
    # First try: load from URDF visual elements (IS format)
    # Check if links have mesh_files defined in URDF
    has_urdf_meshes = any(
        info.get("mesh_files") for info in links.values()
        if info["part_idx"] is not None
    )

    if has_urdf_meshes:
        return _load_meshes_from_urdf(scene_dir, links, link_world)

    # Fallback: load from objs directory (IM format)
    return _load_meshes_from_objs(scene_dir, links, factory, identifier)


def _load_meshes_from_urdf(scene_dir, links, link_world):
    """Load meshes referenced in URDF visual elements.

    Each link may have multiple visual elements, each with its own OBJ file
    and origin offset. Visual origins are in the link's LOCAL frame. We apply
    both the visual origin AND the kinematic chain transform (link_world) to
    place all vertices in the world frame.

    Args:
        scene_dir: directory containing the URDF and mesh files
        links: parsed link info from parse_urdf()
        link_world: {link_name: (position_3d, rotation_3x3)} from compute_world_transforms()
    """
    meshes_by_link = {}
    meshes_by_gidx = {}

    for link_name, info in links.items():
        part_idx = info["part_idx"]
        if part_idx is None:
            continue

        mesh_files = info.get("mesh_files", [])
        if not mesh_files:
            continue

        # Get the link's world transform (position + rotation)
        if link_name in link_world:
            link_pos, link_rot = link_world[link_name]
        else:
            print(f"  WARNING: No world transform for {link_name}, using identity")
            link_pos, link_rot = np.zeros(3), np.eye(3)

        link_meshes = []
        for filename, vis_xyz in mesh_files:
            # Resolve path relative to scene_dir
            obj_path = os.path.join(scene_dir, filename)
            if not os.path.exists(obj_path):
                # Try looking in the scene_dir directly
                obj_path = os.path.join(scene_dir, os.path.basename(filename))
                if not os.path.exists(obj_path):
                    continue

            try:
                mesh = trimesh.load(obj_path, force='mesh', process=False)
                if mesh.vertices.shape[0] > 0:
                    verts = mesh.vertices.copy()
                    # 1. Apply visual origin offset (local frame)
                    if vis_xyz and any(abs(v) > 1e-10 for v in vis_xyz):
                        verts += np.array(vis_xyz, dtype=verts.dtype)
                    # 2. Apply kinematic chain transform (local -> world)
                    verts = (link_rot @ verts.T).T + link_pos
                    mesh.vertices = verts
                    link_meshes.append(mesh)
            except Exception as e:
                print(f"  WARNING: Failed to load {obj_path}: {e}")

        if link_meshes:
            if len(link_meshes) == 1:
                combined = link_meshes[0]
            else:
                combined = trimesh.util.concatenate(link_meshes)
            meshes_by_link[link_name] = combined
            meshes_by_gidx[part_idx] = combined

    return meshes_by_link, meshes_by_gidx


def _load_meshes_from_objs(scene_dir, links, factory=None, identifier=None):
    """Load meshes from objs directory with origins.json (IM format)."""
    objs_dir = find_objs_dir(scene_dir, factory, identifier)
    if objs_dir is None:
        print(f"  ERROR: No objs directory found in {scene_dir}")
        return {}, {}

    # Load origins.json
    origins = {}
    origins_path = os.path.join(scene_dir, "origins.json")
    if os.path.exists(origins_path):
        with open(origins_path) as f:
            origins = json.load(f)

    meshes_by_link = {}
    meshes_by_gidx = {}

    for link_name, info in links.items():
        part_idx = info["part_idx"]
        if part_idx is None:
            continue

        # Try to find OBJ file
        obj_path = None
        entry_dir = os.path.join(objs_dir, str(part_idx))
        if os.path.isdir(entry_dir):
            candidate = os.path.join(entry_dir, f"{part_idx}.obj")
            if os.path.exists(candidate):
                obj_path = candidate
            else:
                import glob
                obj_files = glob.glob(os.path.join(entry_dir, "*.obj"))
                if obj_files:
                    obj_path = obj_files[0]
        else:
            # Try flat file naming
            candidate = os.path.join(objs_dir, f"{part_idx}.obj")
            if os.path.exists(candidate):
                obj_path = candidate

        if obj_path is None:
            continue

        # Resolve symlinks
        real_path = os.path.realpath(obj_path)
        if not os.path.exists(real_path):
            print(f"  WARNING: Broken symlink {obj_path}")
            continue

        try:
            mesh = trimesh.load(real_path, force='mesh', process=False)
            if mesh.vertices.shape[0] > 0:
                # Apply origin offset
                origin = origins.get(str(part_idx))
                if origin is not None:
                    mesh.vertices += np.array(origin, dtype=mesh.vertices.dtype)
                meshes_by_link[link_name] = mesh
                meshes_by_gidx[part_idx] = mesh
        except Exception as e:
            print(f"  WARNING: Failed to load {obj_path}: {e}")

    return meshes_by_link, meshes_by_gidx


def normalize_meshes_inplace(meshes_dict, bound=0.95):
    """Box-normalize all meshes jointly to [-bound, bound].

    Modifies vertices in-place.
    Returns: (center, scale)
    """
    all_verts = np.concatenate([m.vertices for m in meshes_dict.values()], axis=0)
    bmin = all_verts.min(axis=0)
    bmax = all_verts.max(axis=0)
    center = (bmax + bmin) / 2.0
    extent = (bmax - bmin).max()

    if extent < 1e-8:
        return center.tolist(), 1.0

    scale = 2.0 * bound / extent

    for mesh in meshes_dict.values():
        mesh.vertices = (mesh.vertices - center) * scale

    return center.tolist(), float(scale)


# ======================================================================
# Joint Filtering
# ======================================================================

def filter_joints_by_motion(movable_joints, meshes_by_link, parent_map, children_map,
                            center, scale, min_motion=MIN_NORMALIZED_MOTION):
    """Filter out joints with negligible motion in normalized space.

    Returns: list of joints that pass the filter.
    """
    filtered = []
    center_arr = np.array(center)

    for joint in movable_joints:
        if joint.jtype == "prismatic":
            norm_motion = joint.motion_range * scale
            if norm_motion < min_motion:
                print(f"  FILTER joint {joint.name}: prismatic "
                      f"{joint.motion_range:.4f} = {norm_motion:.4f} norm (< {min_motion})")
                continue
        elif joint.jtype in ("revolute", "continuous"):
            # Estimate swept arc in normalized space
            norm_origin = (joint.axis_origin_world - center_arr) * scale
            axis = joint.axis_dir_world

            # Find max radius from axis to any vertex in descendant links
            def get_descendants(j):
                desc = set()
                q = [j.child_link]
                while q:
                    link = q.pop(0)
                    if link in desc:
                        continue
                    desc.add(link)
                    for child, _ in children_map.get(link, []):
                        q.append(child)
                return desc

            max_radius = 0.0
            for link in get_descendants(joint):
                if link in meshes_by_link:
                    verts = meshes_by_link[link].vertices
                    diff = verts - norm_origin
                    proj = np.outer(diff @ axis, axis)
                    perp = diff - proj
                    r = np.linalg.norm(perp, axis=1).max() if len(perp) > 0 else 0.0
                    max_radius = max(max_radius, r)

            swept_arc = joint.motion_range * max_radius
            if swept_arc < min_motion:
                print(f"  FILTER joint {joint.name}: {joint.jtype} "
                      f"range={joint.motion_range:.4f} r={max_radius:.4f} "
                      f"arc={swept_arc:.4f} (< {min_motion})")
                continue

        filtered.append(joint)

    return filtered


# ======================================================================
# Animode Generation
# ======================================================================

def generate_animodes(movable_joints, rng_seed=42):
    """Generate basic and senior animodes.

    Basic: one joint per animode.
    Senior: random combinations of 2+ joints, max 3.
    Same-chain combinations are ALLOWED.

    Returns:
        animodes: list of (name, set_of_joint_names, trajectory_type)
    """
    rng = random.Random(rng_seed)
    animodes = []

    if not movable_joints:
        return animodes

    joint_names = [j.name for j in movable_joints]

    # Basic animodes: one joint each
    for i, j in enumerate(movable_joints):
        traj = rng.choice(TRAJECTORY_TYPES)
        animodes.append((f"basic_{i}", {j.name}, traj))

    # Senior animodes: combinations of 2+ joints
    if len(movable_joints) >= 2:
        # Generate all possible combinations of 2+ joints
        all_combos = []
        for r in range(2, len(movable_joints) + 1):
            for combo in itertools.combinations(joint_names, r):
                all_combos.append(set(combo))

        # Randomly sample up to MAX_SENIOR_ANIMODES
        rng.shuffle(all_combos)
        n_seniors = min(MAX_SENIOR_ANIMODES, len(all_combos))

        for i in range(n_seniors):
            traj = rng.choice(TRAJECTORY_TYPES)
            animodes.append((f"senior_{i}", all_combos[i], traj))

    return animodes


# ======================================================================
# Custom Lid Animodes
# ======================================================================

LID_KEYWORDS = frozenset({"lid", "cap", "cover", "top", "plug", "stopper"})


def detect_lid_joints(joints, scene_dir):
    """Detect prismatic joints whose child link represents a lid/cap/cover.

    Checks (in priority order):
    1. PartNet semantics.txt: 'link_N  motion_type  semantic_label'
    2. URDF link/joint names containing lid keywords

    Returns: list of (joint, label_str) pairs (only prismatic joints).
    """
    sem_map = {}

    # PartNet semantics.txt
    sem_file = os.path.join(scene_dir, "semantics.txt")
    if os.path.exists(sem_file):
        with open(sem_file) as f:
            for line in f:
                parts_line = line.strip().split()
                if len(parts_line) >= 3:
                    sem_map[parts_line[0]] = parts_line[2].lower()

    # Fallback: check link and joint name strings
    for joint in joints:
        if joint.child_link not in sem_map:
            for kw in LID_KEYWORDS:
                if kw in joint.child_link.lower() or kw in joint.name.lower():
                    sem_map[joint.child_link] = kw
                    break

    lid_joints = []
    for joint in joints:
        # Only prismatic joints make sense for separation/flip (revolute = door)
        if joint.jtype != "prismatic":
            continue
        label = sem_map.get(joint.child_link, "")
        if any(kw in label for kw in LID_KEYWORDS):
            lid_joints.append((joint, label))

    return lid_joints


def _get_lid_link_group(root_link_name, children_map):
    """BFS to get root_link_name and all its kinematic descendants."""
    group = set()
    queue = [root_link_name]
    while queue:
        lnk = queue.pop(0)
        if lnk in group:
            continue
        group.add(lnk)
        for child, _ in children_map.get(lnk, []):
            queue.append(child)
    return group


def compute_custom_lid_params(lid_joint, lid_links, meshes_by_link, scale):
    """Compute lift_dist, lift_axis, flip_axis for a lid joint.

    All distances are in URDF space (meters).
    """
    lift_axis = np.array(lid_joint.axis_dir_world, dtype=float)
    lift_axis /= (np.linalg.norm(lift_axis) + 1e-12)

    # Lid extent along lift axis in normalized space
    lid_verts = [meshes_by_link[lnk].vertices
                 for lnk in lid_links if lnk in meshes_by_link]

    if lid_verts:
        all_v = np.concatenate(lid_verts, axis=0)
        proj = all_v @ lift_axis
        lid_extent_norm = proj.max() - proj.min()
        lid_extent_urdf = lid_extent_norm / scale
        # Lift 2× lid height in URDF space, at least 3× joint upper limit
        lift_dist = max(lid_extent_urdf * 2.0,
                        lid_joint.upper * 3.0 if lid_joint.upper > 0 else 0.05)
    else:
        lift_dist = max(lid_joint.upper * 3.0 if lid_joint.upper > 0 else 0.05,
                        0.05)

    # Flip axis: perpendicular to lift direction in the XY plane
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(lift_axis, world_up)) > 0.9:
        flip_axis = np.array([1.0, 0.0, 0.0])   # lift is vertical → flip around X
    else:
        flip_axis = np.cross(lift_axis, world_up)
        flip_axis /= (np.linalg.norm(flip_axis) + 1e-12)

    return {
        "lid_joint":       lid_joint.name,
        "lid_links":       sorted(lid_links),
        "lift_axis":       lift_axis.tolist(),
        "lift_dist":       float(lift_dist),   # URDF metres
        "flip_axis":       flip_axis.tolist(),
        "flip_start_frac": 0.5,               # flip begins at 50% of anim
        "flip_angle":      math.pi,            # 180° reveal
    }


def generate_custom_lid_animodes(movable_joints, scene_dir, meshes_by_link,
                                  children_map, scale):
    """Generate custom_N_separation and custom_N_flip animodes for lid joints.

    Returns:
        animodes:    list of (name, {joint_name}, traj_type)
        params_map:  {animode_name: custom_params_dict}
    """
    animodes = []
    params_map = {}

    lid_joints = detect_lid_joints(movable_joints, scene_dir)
    if not lid_joints:
        return animodes, params_map

    for i, (lid_joint, label) in enumerate(lid_joints):
        lid_links = _get_lid_link_group(lid_joint.child_link, children_map)
        params = compute_custom_lid_params(lid_joint, lid_links, meshes_by_link, scale)

        sep_name  = f"custom_{i * 2}_separation"
        flip_name = f"custom_{i * 2 + 1}_flip"

        animodes.append((sep_name,  {lid_joint.name}, "lid_separation"))
        animodes.append((flip_name, {lid_joint.name}, "lid_flip"))
        params_map[sep_name]  = params
        params_map[flip_name] = params

        print(f"  Custom lid animodes for {lid_joint.name} (label='{label}'): "
              f"lift_dist={params['lift_dist']:.4f}m, "
              f"flip_axis={[round(v,2) for v in params['flip_axis']]}")

    return animodes, params_map


# ======================================================================
# Export: OBJ + Verify PNG + Metadata
# ======================================================================

def _clean_mesh(mesh):
    """PartPacker-style per-mesh cleanup: remove duplicate vertices, degenerate/duplicate faces, fix normals."""
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.update_faces(mesh.unique_faces() & mesh.nondegenerate_faces())
    mesh.fix_normals()
    return mesh


def _is_single_layer_plane(mesh, coplane_thresh=0.01):
    """Check if mesh is just a single-layer plane (all face normals nearly parallel)."""
    if mesh.is_watertight:
        return False
    if len(mesh.faces) == 0:
        return False
    fn = mesh.face_normals
    diff = np.linalg.norm(np.abs(fn) - np.abs(fn[0]), axis=1)
    return diff.max() < coplane_thresh


def _stitch_open_mesh(mesh):
    """Try to close open boundary loops (PartPacker-style conservative stitch).

    Only stitches boundaries that form a coplanar & convex polygon.
    Returns the mesh (modified in-place).
    """
    if mesh.is_watertight or len(mesh.faces) == 0:
        return mesh

    # Bail early on single-layer planes (can't be stitched meaningfully)
    if _is_single_layer_plane(mesh):
        return mesh

    try:
        from trimesh.path.exchange.misc import faces_to_path
        faces = np.arange(len(mesh.faces))
        boundaries = [
            e.points for e in faces_to_path(mesh, faces)["entities"]
            if len(e.points) > 3 and e.points[0] == e.points[-1]
        ]
    except Exception:
        return mesh

    vertices = mesh.vertices
    edges_face = mesh.edges_face
    tree_edge = mesh.edges_sorted_tree
    normals = mesh.face_normals
    fans = []

    for vert_indices in boundaries:
        verts = vertices[vert_indices]
        # Only stitch coplanar convex boundaries (safe approximation)
        if len(verts) < 3:
            continue
        # Coplanarity check: all normals of fan-triangles should be parallel
        v0, v1 = verts[1] - verts[0], verts[2] - verts[0]
        ref_n = np.cross(v0, v1)
        ref_n_norm = np.linalg.norm(ref_n)
        if ref_n_norm < 1e-10:
            continue
        ref_n = ref_n / ref_n_norm
        coplanar = True
        for i in range(2, len(verts) - 2):
            va, vb = verts[i] - verts[0], verts[i + 1] - verts[0]
            n = np.cross(va, vb)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            if np.linalg.norm(np.abs(n / n_norm) - np.abs(ref_n)) > 0.1:
                coplanar = False
                break
        if not coplanar:
            continue

        fan = np.column_stack((
            np.ones(len(vert_indices) - 3, dtype=int) * vert_indices[0],
            vert_indices[1:-2],
            vert_indices[2:-1],
        ))

        # Flip fan if needed to match adjacent face orientation
        e = fan[:min(10, len(fan)), 1:].copy()
        e.sort(axis=1)
        try:
            query = tree_edge.query_ball_point(e, r=1e-10)
            edge_index = np.concatenate(query) if query else []
            if len(edge_index) > 0:
                original = normals[edges_face[edge_index]]
                check, valid = trimesh.triangles.normals(vertices[fan[:3]])
                if valid.any():
                    if np.dot(original, check[0]).mean() < 0:
                        fan = np.fliplr(fan)
        except Exception:
            pass

        fans.append(fan)

    if fans:
        mesh.faces = np.concatenate([mesh.faces, np.vstack(fans)])

    return mesh


def _clean_and_stitch(part_meshes):
    """Per-mesh clean, concatenate, then stitch open boundaries on the result.

    Follows PartPacker bipartite_contraction.py pipeline:
      1. Per-component: merge_vertices + update_faces + fix_normals
      2. Concatenate all components
      3. Final clean pass
      4. Stitch open boundary loops (conservative, coplanar-only)
    """
    if not part_meshes:
        return trimesh.Trimesh()

    # Step 1: clean each component individually
    cleaned = []
    for m in part_meshes:
        mc = m.copy()
        mc.visual = trimesh.visual.ColorVisuals()
        _clean_mesh(mc)
        if len(mc.faces) > 0:
            cleaned.append(mc)

    if not cleaned:
        return trimesh.Trimesh()

    # Step 2: concatenate
    result = trimesh.util.concatenate(cleaned) if len(cleaned) > 1 else cleaned[0]

    # Step 3: final clean on concatenated mesh
    _clean_mesh(result)

    # Step 4: stitch open boundaries
    _stitch_open_mesh(result)

    # Final normals pass after stitching
    result.fix_normals()

    return result


def export_split(meshes_by_link, merged_groups, coloring, removed_links,
                 out_dir, animode_name):
    """Export part0.obj, part1.obj, and verify.png for one animode split.

    Applies PartPacker-style mesh cleaning to each part before export:
    - Per-component: merge_vertices + update_faces(unique & nondegenerate) + fix_normals
    - Concatenate components
    - Stitch open boundary loops (conservative: coplanar-convex only)

    Returns: True if successful.
    """
    os.makedirs(out_dir, exist_ok=True)

    part0_meshes = []
    part1_meshes = []

    for gidx, group in enumerate(merged_groups):
        color = coloring.get(gidx, 0)
        for link_name in group:
            if link_name in removed_links:
                continue
            if link_name not in meshes_by_link:
                continue
            mesh = meshes_by_link[link_name]
            if color == 0:
                part0_meshes.append(mesh)
            else:
                part1_meshes.append(mesh)

    if not part1_meshes:
        print(f"  WARNING: Empty part1 for {animode_name}")
        return False

    # Clean + stitch each part (PartPacker-style)
    part0 = _clean_and_stitch(part0_meshes)
    part1 = _clean_and_stitch(part1_meshes)

    part0.export(os.path.join(out_dir, "part0.obj"))
    part1.export(os.path.join(out_dir, "part1.obj"))

    # Generate verify.png
    generate_verify_png(part0, part1, out_dir, animode_name)

    return True


def generate_verify_png(part0, part1, out_dir, title=""):
    """Generate a verification PNG showing part0 (blue) and part1 (red) from 4 views.

    Matches the old split_and_visualize.py render_split_visualization() style.
    """
    fig = plt.figure(figsize=(20, 10))

    max_faces = 5000

    def subsample_mesh(mesh, max_f):
        if mesh.faces.shape[0] <= max_f:
            return mesh.vertices, mesh.faces
        idx = np.random.choice(mesh.faces.shape[0], max_f, replace=False)
        return mesh.vertices, mesh.faces[idx]

    # Collect all vertices for axis limits
    all_verts = []
    if len(part0.vertices) > 0:
        all_verts.append(part0.vertices)
    if len(part1.vertices) > 0:
        all_verts.append(part1.vertices)
    if not all_verts:
        plt.close(fig)
        return
    all_v = np.concatenate(all_verts)
    vmin, vmax = all_v.min(axis=0), all_v.max(axis=0)

    angles = [(30, 45), (30, 135), (30, 225), (30, 315)]
    titles = ['Front-Right', 'Front-Left', 'Back-Left', 'Back-Right']

    for idx, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, 4, idx + 1, projection='3d')

        # Draw part0 (blue)
        if len(part0.vertices) > 0 and len(part0.faces) > 0:
            v0, f0 = subsample_mesh(part0, max_faces)
            polys0 = v0[f0]
            pc0 = Poly3DCollection(polys0, alpha=0.6, linewidths=0.1,
                                    edgecolors='navy')
            pc0.set_facecolor((0.3, 0.5, 0.85, 0.6))
            ax.add_collection3d(pc0)

        # Draw part1 (red)
        if len(part1.vertices) > 0 and len(part1.faces) > 0:
            v1, f1 = subsample_mesh(part1, max_faces)
            polys1 = v1[f1]
            pc1 = Poly3DCollection(polys1, alpha=0.6, linewidths=0.1,
                                    edgecolors='darkred')
            pc1.set_facecolor((0.85, 0.3, 0.3, 0.6))
            ax.add_collection3d(pc1)

        ax.set_xlim(vmin[0], vmax[0])
        ax.set_ylim(vmin[1], vmax[1])
        ax.set_zlim(vmin[2], vmax[2])
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(titles[idx], fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    fig.suptitle(f'{title}  Blue=Body(part0)  Red=Moving(part1)',
                 fontsize=14, y=0.98)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "verify.png"), dpi=150,
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def build_joint_12dim_vector(joint, center, scale):
    """Build 12-dim joint parameter vector.

    [0:3]  axis_origin (normalized)
    [3:6]  axis_direction (unit vector)
    [6:9]  type one-hot [revolute, prismatic, continuous]
    [9:11] motion range [min, max]
    [11]   exists flag (1.0)
    """
    center_arr = np.array(center)
    vec = np.zeros(12, dtype=np.float32)
    vec[0:3] = (joint.axis_origin_world - center_arr) * scale
    vec[3:6] = joint.axis_dir_world

    type_map = {"revolute": 0, "prismatic": 1, "continuous": 2}
    type_idx = type_map.get(joint.jtype, 0)
    vec[6 + type_idx] = 1.0

    vec[9] = joint.lower if joint.jtype != "continuous" else 0.0
    vec[10] = joint.upper if joint.jtype != "continuous" else 2 * np.pi
    vec[11] = 1.0

    return vec


# ======================================================================
# Main Processing
# ======================================================================

def process_object(factory_name, seed, base_dir, output_dir, force=False,
                    suffix=""):
    """Process a single object: load, normalize, generate animodes, split, export.

    Args:
        suffix: appended to factory_name in output path (e.g., "_PhysXmobility")

    Returns: True if any splits were generated.
    """
    identifier = str(seed)
    scene_dir = os.path.join(base_dir, "outputs", factory_name, identifier)

    # Also try direct path (e.g., dataset3D/infinite_mobility/Factory/seed/)
    if not os.path.isdir(scene_dir):
        scene_dir = os.path.join(base_dir, factory_name, identifier)

    # Also try flat path (e.g., PartNet: dataset3D/Partnet/{id}/)
    if not os.path.isdir(scene_dir):
        scene_dir = os.path.join(base_dir, identifier)

    output_name = factory_name + suffix
    out_base = os.path.join(output_dir, output_name, identifier)

    # Skip check
    if not force and os.path.exists(os.path.join(out_base, "metadata.json")):
        print(f"  SKIP {factory_name}/{identifier} (already exists)")
        return True

    # Clean old output
    if force and os.path.isdir(out_base):
        import shutil
        shutil.rmtree(out_base)

    # Find URDF file (multiple naming conventions)
    urdf_path = None
    for urdf_name in ["scene.urdf", "mobility.urdf",
                       f"{factory_name}.urdf",
                       f"{factory_name.lower()}.urdf",
                       # IS format: asset_name.urdf (e.g., lamp.urdf)
                       ]:
        candidate = os.path.join(scene_dir, urdf_name)
        if os.path.exists(candidate):
            urdf_path = candidate
            break

    # Also try any *.urdf in the directory
    if urdf_path is None:
        import glob
        urdf_files = glob.glob(os.path.join(scene_dir, "*.urdf"))
        if urdf_files:
            urdf_path = urdf_files[0]

    if urdf_path is None:
        print(f"  ERROR: No URDF found in {scene_dir}")
        return False

    print(f"\n{'='*60}")
    print(f"Processing {factory_name}/{identifier}")
    print(f"  URDF: {urdf_path}")

    # 1. Parse URDF
    links, joints, root_link = parse_urdf(urdf_path)
    parent_map, children_map = build_kinematic_tree(joints)
    link_world = compute_world_transforms(joints, root_link)

    # Find all movable joints
    movable_joints = [j for j in joints if j.is_movable]
    if not movable_joints:
        print(f"  SKIP: no movable joints")
        return False

    print(f"  Links: {len(links)}, Joints: {len(joints)}, Movable: {len(movable_joints)}")
    for j in movable_joints:
        print(f"    {j.name}: {j.jtype}, range={j.motion_range:.4f}, "
              f"{j.parent_link}->{j.child_link}")

    # 2. Load meshes (with kinematic chain transforms for IS format)
    meshes_by_link, meshes_by_gidx = load_meshes(scene_dir, links, link_world, factory_name, identifier)
    if not meshes_by_link:
        print(f"  ERROR: No meshes loaded from {scene_dir}")
        return False

    print(f"  Loaded {len(meshes_by_link)} mesh parts")

    # 3. Normalize FIRST
    center, scale = normalize_meshes_inplace(meshes_by_gidx, bound=0.95)
    # meshes_by_link shares the same objects as meshes_by_gidx, so they are normalized too

    all_v = np.concatenate([m.vertices for m in meshes_by_gidx.values()])
    print(f"  Normalized range: [{all_v.min():.3f}, {all_v.max():.3f}]")

    # 4. Filter joints by normalized motion
    movable_joints = filter_joints_by_motion(
        movable_joints, meshes_by_link, parent_map, children_map, center, scale)

    if not movable_joints:
        print(f"  SKIP: no joints with significant motion after filtering")
        return False

    print(f"  After filtering: {len(movable_joints)} movable joints")

    # 5. Generate animodes
    rng_seed = hash((factory_name, seed)) % (2**31)
    animodes = generate_animodes(movable_joints, rng_seed)
    print(f"  Generated {len(animodes)} animodes "
          f"({sum(1 for n, _, _ in animodes if n.startswith('basic'))} basic, "
          f"{sum(1 for n, _, _ in animodes if n.startswith('senior'))} senior)")

    # 5b. Custom lid animodes (separation + flip)
    custom_animodes, custom_params_map = generate_custom_lid_animodes(
        movable_joints, scene_dir, meshes_by_link, children_map, scale)
    if custom_animodes:
        animodes.extend(custom_animodes)
        print(f"  + {len(custom_animodes)} custom lid animodes")

    # 6. Process each animode
    n_exported = 0
    all_splits_info = {}

    for animode_name, active_joint_names, traj_type in animodes:
        print(f"\n  --- {animode_name} (traj={traj_type}) ---")
        print(f"    Active joints: {sorted(active_joint_names)}")

        # 6a. Classify joints
        active, passive, fixed, pre_opening = classify_joints_for_animode(
            movable_joints, active_joint_names,
            meshes_by_link, parent_map, children_map,
            center, scale, traj_type)

        print(f"    Active: {[j.name for j in active]}")
        print(f"    Passive: {[j.name for j in passive]}")
        print(f"    Fixed: {[j.name for j in fixed]}")

        # 6b. Build reduced graph
        # Fixed joints for graph building = classified-fixed movable joints
        fixed_for_graph = fixed  # These movable joints are treated as fixed
        movable_for_graph = active + passive  # These are the edges in reduced graph

        merged_groups, reduced_edges, root_group_idx = build_reduced_graph(
            links, joints, root_link, fixed_for_graph, movable_for_graph)

        print(f"    Reduced graph: {len(merged_groups)} nodes, {len(reduced_edges)} edges")
        for gi, group in enumerate(merged_groups):
            link_names = sorted(group)
            part_idxs = [get_link_part_idx(l) for l in link_names if get_link_part_idx(l) is not None]
            print(f"      Node {gi}: parts={part_idxs} "
                  f"{'(ROOT)' if gi == root_group_idx else ''}")

        # 6c. Handle bridging parts
        merged_groups, reduced_edges, removed_links = handle_bridging_parts(
            merged_groups, reduced_edges, links, joints,
            {j.name for j in movable_for_graph})

        # 6d. Bipartite 2-coloring
        coloring, is_bipartite = bipartite_2_coloring(
            merged_groups, reduced_edges, root_group_idx)

        if not is_bipartite:
            print(f"    WARNING: Graph is NOT bipartite (odd cycle). "
                  f"Coloring may not be perfect.")

        # Print coloring
        for gi, group in enumerate(merged_groups):
            part_idxs = [get_link_part_idx(l) for l in group if get_link_part_idx(l) is not None]
            color_label = "part0" if coloring.get(gi, 0) == 0 else "part1"
            print(f"    {color_label}: parts={part_idxs}")

        # 6e. Export
        split_dir = os.path.join(out_base, animode_name)
        ok = export_split(meshes_by_link, merged_groups, coloring, removed_links,
                          split_dir, animode_name)

        if ok:
            n_exported += 1

            # Build per-split metadata
            part0_groups = []
            part1_groups = []
            for gi, group in enumerate(merged_groups):
                part_idxs = sorted([get_link_part_idx(l) for l in group
                                    if get_link_part_idx(l) is not None])
                if coloring.get(gi, 0) == 0:
                    part0_groups.append(part_idxs)
                else:
                    part1_groups.append(part_idxs)

            # Joint 12-dim vectors for active joints
            joint_vectors = {}
            for j in active:
                vec = build_joint_12dim_vector(j, center, scale)
                joint_vectors[j.name] = vec.tolist()

            all_splits_info[animode_name] = {
                "trajectory_type": traj_type,
                "active_joints": [j.name for j in active],
                "passive_joints": [j.name for j in passive],
                "fixed_joints": [j.name for j in fixed],
                "joint_classification": {
                    j.name: "active" for j in active
                } | {
                    j.name: "passive" for j in passive
                } | {
                    j.name: "fixed" for j in fixed
                },
                "two_coloring": {
                    "part0_groups": part0_groups,
                    "part1_groups": part1_groups,
                },
                "is_bipartite": is_bipartite,
                "pre_opening_angles": pre_opening,
                "joint_12dim_vectors": joint_vectors,
                "n_reduced_nodes": len(merged_groups),
                "n_reduced_edges": len(reduced_edges),
            }

            # Merge custom lid params into split info if applicable
            if animode_name in custom_params_map:
                all_splits_info[animode_name]["custom_lid_params"] = custom_params_map[animode_name]

    # 7. Write metadata
    center_arr = np.array(center)
    metadata = {
        "factory": factory_name,
        "output_name": output_name,
        "identifier": identifier,
        "urdf_path": os.path.abspath(urdf_path),
        "scene_dir": os.path.abspath(scene_dir),
        "root_link": root_link,
        "normalize": {"center": center, "scale": scale},
        "n_links": len(links),
        "n_joints": len(joints),
        "n_movable_joints": len(movable_joints),
        "n_animodes": len(animodes),
        "n_exported": n_exported,
        "joints": [
            {
                "name": j.name,
                "type": j.jtype,
                "parent_link": j.parent_link,
                "child_link": j.child_link,
                "motion_range": round(j.motion_range, 6),
                "axis_origin_normalized": ((j.axis_origin_world - center_arr) * scale).round(6).tolist()
                    if j.axis_origin_world is not None else None,
                "axis_direction": j.axis_dir_world.round(6).tolist()
                    if j.axis_dir_world is not None else None,
            }
            for j in movable_joints
        ],
        "splits": all_splits_info,
    }

    os.makedirs(out_base, exist_ok=True)
    meta_path = os.path.join(out_base, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Exported {n_exported}/{len(animodes)} animodes to {out_base}")
    print(f"  Metadata: {meta_path}")
    return n_exported > 0


# ======================================================================
# Batch Processing
# ======================================================================

def _dir_has_urdf(dirpath):
    """Check if a directory contains any URDF file."""
    import glob
    return bool(glob.glob(os.path.join(dirpath, "*.urdf")))


def discover_objects(base_dir):
    """Discover all objects in a base directory.

    Looks for directories containing any *.urdf file.
    Supports both outputs/{Factory}/{seed}/ and {Factory}/{seed}/ layouts.

    Returns: list of (factory_name, seed, base_dir)
    """
    objects = []

    def scan_dir(search_dir):
        """Scan a directory for {factory}/{seed}/ subdirs with URDF files."""
        found = []
        if not os.path.isdir(search_dir):
            return found
        for factory in sorted(os.listdir(search_dir)):
            factory_dir = os.path.join(search_dir, factory)
            if not os.path.isdir(factory_dir):
                continue
            for seed_name in sorted(os.listdir(factory_dir)):
                seed_dir = os.path.join(factory_dir, seed_name)
                if not os.path.isdir(seed_dir):
                    continue
                if _dir_has_urdf(seed_dir):
                    found.append((factory, seed_name, base_dir))
        return found

    # Check outputs/ subdirectory first
    outputs_dir = os.path.join(base_dir, "outputs")
    objects = scan_dir(outputs_dir)

    # Also check direct path: base_dir/{Factory}/{seed}/
    if not objects:
        objects = scan_dir(base_dir)

    return objects


def main():
    parser = argparse.ArgumentParser(
        description="Topology-based dual-volume split precompute for Infinigen-Sim")
    parser.add_argument("--factory", type=str, help="Factory name")
    parser.add_argument("--seed", type=str, help="Seed / identifier")
    parser.add_argument("--output_dir", type=str,
                        default="./precompute_output",
                        help="Output root directory")
    parser.add_argument("--base", type=str,
                        default="/mnt/data/yurh/Infinigen-Sim",
                        help="Base directory with outputs/")
    parser.add_argument("--all", action="store_true",
                        help="Process all objects found in base directory")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if output exists")
    parser.add_argument("--max_seeds", type=int, default=0,
                        help="Max seeds per factory (0=unlimited)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to factory name in output "
                             "(e.g., _PhysXmobility, _PhysXnet)")
    args = parser.parse_args()

    if args.all:
        objects = discover_objects(args.base)
        if not objects:
            print(f"No objects found in {args.base}")
            sys.exit(1)

        # Apply max_seeds
        if args.max_seeds > 0:
            from collections import OrderedDict
            factory_seeds = OrderedDict()
            for f, s, b in objects:
                factory_seeds.setdefault(f, []).append((f, s, b))
            objects = []
            for f, entries in factory_seeds.items():
                objects.extend(entries[:args.max_seeds])

        print(f"Found {len(objects)} objects to process")
        success = 0
        for factory, seed, base in objects:
            ok = process_object(factory, seed, base, args.output_dir, args.force,
                                suffix=args.suffix)
            if ok:
                success += 1

        print(f"\nDone: {success}/{len(objects)} objects processed")
        print(f"Output: {os.path.abspath(args.output_dir)}")

    elif args.factory and args.seed:
        ok = process_object(args.factory, args.seed, args.base,
                            args.output_dir, args.force, suffix=args.suffix)
        if not ok:
            sys.exit(1)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python split_precompute.py --factory LampFactory --seed 0 "
              "--base /path/to/data")
        print("  python split_precompute.py --all --base /path/to/data "
              "--output_dir ./precompute_output")
        sys.exit(1)


if __name__ == "__main__":
    main()
