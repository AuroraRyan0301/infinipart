#!/usr/bin/env python3
"""
Blender render script for Infinigen-Sim precomputed animodes.

Reads metadata.json from precompute pipeline, loads URDF + OBJs,
applies forward kinematics per frame, renders 32 camera views.

Run with:
  CUDA_VISIBLE_DEVICES=2 /mnt/data/yurh/blender-3.6.0-linux-x64/blender \
      --background --python render_animode.py -- \
      --metadata ./output/dishwasher/1001/metadata.json \
      --animode basic_0 --views hemi --resolution 512 --duration 4
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

import bpy
from mathutils import Matrix, Vector, Euler

# ======================================================================
# Constants
# ======================================================================

ENVMAP_DIR = "/mnt/data/yurh/dataset3D/envmap/indoor"
FFMPEG_BIN = "/usr/local/bin/ffmpeg"

# 16 fixed hemisphere views: 4 elevations x 4 azimuths
_HEMI_ELEVS = [5, 25, 45, 65]
_HEMI_AZIMS = [-67.5, -22.5, 22.5, 67.5]
HEMI_VIEWS = {}
for _i, _elev in enumerate(_HEMI_ELEVS):
    for _j, _azim in enumerate(_HEMI_AZIMS):
        HEMI_VIEWS[f"hemi_{_i*4+_j:02d}"] = (_elev, _azim)

# 8 orbit views (moving cameras, 180 degree arcs)
ORBIT_VIEWS = {
    "orbit_00": (10, 180, 10, 0),
    "orbit_01": (10, -180, 10, 0),
    "orbit_02": (30, 180, 30, 0),
    "orbit_03": (30, -180, 30, 0),
    "orbit_04": (50, 150, 15, 0),
    "orbit_05": (50, -150, 15, 0),
    "orbit_06": (15, 170, 50, 0),
    "orbit_07": (15, -170, 50, 0),
}

# 8 sweep views (pans, tilts, diagonal sweeps)
SWEEP_VIEWS = {
    "sweep_00": (15, -80, 15, 80),
    "sweep_01": (35, -75, 35, 75),
    "sweep_02": (55, -60, 55, 60),
    "sweep_03": (25, 80, 25, -80),
    "sweep_04": (5, 0, 70, 0),
    "sweep_05": (5, -45, 65, -45),
    "sweep_06": (5, 45, 65, 45),
    "sweep_07": (10, -60, 55, 60),
}

# Named convenience views (static)
NAMED_VIEWS = {
    "front": (25, 0),
    "side": (25, 90),
    "back": (25, 180),
    "threequarter": (25, 45),
}


# ======================================================================
# Argument Parsing
# ======================================================================

def parse_args():
    # Blender passes args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render animode videos from precompute metadata")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--animode", default="all", help="Animode name or 'all'")
    parser.add_argument("--views", nargs="+", default=["hemi"],
                        help="View groups: hemi, orbit, sweep, all, front, side, etc.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--envmap", default=None, help="Path to envmap HDR/EXR")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--color_mode", default="both", choices=["part", "group", "both"],
                        help="part: binary part0/part1 coloring; "
                             "group: each reduced graph node gets unique color; "
                             "both: render both passes (default)")
    return parser.parse_args(argv)


# ======================================================================
# URDF Parsing (Blender context, uses mathutils)
# ======================================================================

class Joint:
    """URDF joint info for forward kinematics."""
    def __init__(self, name, jtype, parent_link, child_link,
                 axis, origin_xyz, origin_rpy, lower, upper):
        self.name = name
        self.jtype = jtype
        self.parent_link = parent_link
        self.child_link = child_link
        self.axis = Vector(axis).normalized()
        self.origin_xyz = Vector(origin_xyz)
        self.origin_rpy = origin_rpy  # (r, p, y)
        self.lower = lower
        self.upper = upper

    @property
    def motion_range(self):
        if self.jtype == "continuous":
            return 2.0 * math.pi
        return abs(self.upper - self.lower)


def parse_urdf(urdf_path):
    """Parse URDF into links and joints.

    Returns:
        links: dict {name: {"part_idx": int|None, "mesh_files": [(path, vis_xyz)]}}
        joints: list of Joint
        root_link: str
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = os.path.dirname(urdf_path)

    links = {}
    for link_el in root.findall("link"):
        name = link_el.get("name")
        part_idx = None
        m = re.match(r"l_(\d+)", name)
        if m:
            part_idx = int(m.group(1))
        else:
            m = re.match(r"link_(\d+)", name)
            if m:
                part_idx = int(m.group(1))

        mesh_files = []
        for visual_el in link_el.findall("visual"):
            geom = visual_el.find("geometry")
            if geom is not None:
                mesh_el = geom.find("mesh")
                if mesh_el is not None:
                    filename = mesh_el.get("filename", "")
                    # Resolve relative path
                    if not os.path.isabs(filename):
                        filename = os.path.join(urdf_dir, filename)
                    vis_origin = visual_el.find("origin")
                    vis_xyz = [0.0, 0.0, 0.0]
                    if vis_origin is not None and vis_origin.get("xyz"):
                        vis_xyz = [float(v) for v in vis_origin.get("xyz").split()]
                    mesh_files.append((filename, vis_xyz))

        links[name] = {"name": name, "part_idx": part_idx, "mesh_files": mesh_files}

    joints = []
    children_set = set()
    for joint_el in root.findall("joint"):
        name = joint_el.get("name", "")
        jtype = joint_el.get("type", "fixed")
        parent_link = joint_el.find("parent").get("link")
        child_link = joint_el.find("child").get("link")
        children_set.add(child_link)

        axis_el = joint_el.find("axis")
        axis = [1.0, 0.0, 0.0]
        if axis_el is not None and axis_el.get("xyz"):
            axis = [float(v) for v in axis_el.get("xyz").split()]

        origin_el = joint_el.find("origin")
        origin_xyz = [0.0, 0.0, 0.0]
        origin_rpy = [0.0, 0.0, 0.0]
        if origin_el is not None:
            if origin_el.get("xyz"):
                origin_xyz = [float(v) for v in origin_el.get("xyz").split()]
            if origin_el.get("rpy"):
                origin_rpy = [float(v) for v in origin_el.get("rpy").split()]

        limit_el = joint_el.find("limit")
        lower, upper = 0.0, 0.0
        if limit_el is not None:
            lower = float(limit_el.get("lower", "0"))
            upper = float(limit_el.get("upper", "0"))

        joints.append(Joint(name, jtype, parent_link, child_link,
                            axis, origin_xyz, origin_rpy, lower, upper))

    all_links = set(links.keys())
    roots = all_links - children_set
    root_link = sorted(roots)[0] if roots else sorted(all_links)[0]

    return links, joints, root_link


def build_kinematic_tree(joints):
    """Build parent/children maps."""
    parent_map = {}
    children_map = defaultdict(list)
    for j in joints:
        parent_map[j.child_link] = (j.parent_link, j)
        children_map[j.parent_link].append((j.child_link, j))
    return parent_map, children_map


# ======================================================================
# Transform Math (mathutils)
# ======================================================================

def mat_translate(x, y, z):
    m = Matrix.Identity(4)
    m[0][3] = x
    m[1][3] = y
    m[2][3] = z
    return m


def mat_rotate_rpy(r, p, y):
    """Rotation matrix from roll-pitch-yaw (XYZ extrinsic = ZYX intrinsic)."""
    ex = Euler((r, p, y), 'XYZ')
    return ex.to_matrix().to_4x4()


def mat_rotate_axis_angle(axis, angle):
    """4x4 rotation around axis by angle (radians)."""
    ax = axis.normalized()
    return Matrix.Rotation(angle, 4, ax)


def compute_joint_local_transform(joint, q_value):
    """T = T_origin(xyz, rpy) x T_joint(q)."""
    ox, oy, oz = joint.origin_xyz
    T_origin = mat_translate(ox, oy, oz) @ mat_rotate_rpy(*joint.origin_rpy)

    if joint.jtype == "fixed":
        T_joint = Matrix.Identity(4)
    elif joint.jtype in ("revolute", "continuous"):
        T_joint = mat_rotate_axis_angle(joint.axis, q_value)
    elif joint.jtype == "prismatic":
        ax = joint.axis.normalized()
        T_joint = mat_translate(ax.x * q_value, ax.y * q_value, ax.z * q_value)
    else:
        T_joint = Matrix.Identity(4)

    return T_origin @ T_joint


# ======================================================================
# Trajectory Generation
# ======================================================================

def compute_trajectory_value(joint, traj_type, t):
    """Compute joint parameter q at normalized time t in [0, 1]."""
    lo = joint.lower if joint.jtype != "continuous" else 0.0
    hi = joint.upper if joint.jtype != "continuous" else 2 * math.pi
    mrange = hi - lo

    if traj_type == "sinusoidal_oscillation":
        q = lo + mrange * 0.5 * (1 - math.cos(2 * math.pi * t))
    elif traj_type == "one_way_sinusoidal":
        q = lo + mrange * 0.5 * (1 - math.cos(math.pi * t))
    elif traj_type == "linear":
        q = lo + mrange * t
    elif traj_type == "linear_oscillation":
        if t < 0.5:
            q = lo + mrange * (2 * t)
        else:
            q = lo + mrange * (2 * (1 - t))
    else:
        q = lo + mrange * t
    return q


# ======================================================================
# Forward Kinematics
# ======================================================================

def forward_kinematics(joints, children_map, root_link, joint_values):
    """Compute world transforms for all links given joint parameter values.

    Args:
        joint_values: {joint_name: q_value}

    Returns:
        link_transforms: {link_name: 4x4 Matrix}
    """
    link_transforms = {root_link: Matrix.Identity(4)}
    queue = deque([root_link])
    visited = set()

    while queue:
        link_name = queue.popleft()
        if link_name in visited:
            continue
        visited.add(link_name)

        parent_T = link_transforms.get(link_name, Matrix.Identity(4))
        for child_link, joint in children_map.get(link_name, []):
            q = joint_values.get(joint.name, 0.0)
            T_local = compute_joint_local_transform(joint, q)
            link_transforms[child_link] = parent_T @ T_local
            queue.append(child_link)

    return link_transforms


def compute_frame_joint_values(joints_by_name, split_info, t):
    """Compute joint values at normalized time t for a given animode split.

    Active joints: follow trajectory
    Passive joints: constant pre-opening angle
    Fixed/other joints: 0
    """
    traj_type = split_info["trajectory_type"]
    classification = split_info["joint_classification"]
    pre_opening = split_info.get("pre_opening_angles", {})

    values = {}
    for jname, cls in classification.items():
        joint = joints_by_name.get(jname)
        if joint is None:
            continue
        if cls == "active":
            values[jname] = compute_trajectory_value(joint, traj_type, t)
        elif cls == "passive":
            # Animate passive joints from 0 to pre_opening angle
            # using the same trajectory shape as active joints
            target = pre_opening.get(jname, 0.0)
            if abs(target) > 1e-8:
                # Scale t through trajectory shape to get smooth 0->target motion
                # Use one_way_sinusoidal for smooth ease-in/out
                values[jname] = target * 0.5 * (1 - math.cos(math.pi * t))
            else:
                values[jname] = 0.0
        else:  # fixed
            values[jname] = 0.0

    return values


# ======================================================================
# Part Loading
# ======================================================================

def clear_scene():
    """Remove all objects, meshes, cameras, lights from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block)
    for block in bpy.data.lights:
        bpy.data.lights.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)


def import_obj(filepath, name=None):
    """Import an OBJ file and return the imported object."""
    before = set(bpy.data.objects.keys())

    # Use legacy importer for Blender 3.x
    try:
        bpy.ops.import_scene.obj(filepath=filepath, axis_forward='-Y', axis_up='Z')
    except Exception:
        bpy.ops.wm.obj_import(filepath=filepath, forward_axis='NEGATIVE_Y', up_axis='Z')

    after = set(bpy.data.objects.keys())
    new_names = after - before
    if not new_names:
        return None

    # CRITICAL: reset rotation_euler to (0,0,0) after import
    # Blender OBJ importer applies axis conversion as object-level rotation
    # but our OBJs are already Z-up
    imported_objs = [bpy.data.objects[n] for n in new_names]
    for obj in imported_objs:
        obj.rotation_euler = (0, 0, 0)

    # If multiple objects imported, join them
    if len(imported_objs) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in imported_objs:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = imported_objs[0]
        bpy.ops.object.join()
        result = bpy.context.active_object
    else:
        result = imported_objs[0]

    if name:
        result.name = name
    return result


def load_parts_im_format(scene_dir, links, link_to_part_idx):
    """Load parts for IM/PhysXNet format: objs/{idx}/{idx}.obj + origins.json."""
    origins_path = os.path.join(scene_dir, "origins.json")
    with open(origins_path) as f:
        origins = json.load(f)

    parts = {}  # link_name -> blender object

    # Find objs/ directory (may be nested: outputs/{Factory}/{id}/objs/)
    objs_dir = os.path.join(scene_dir, "objs")
    if not os.path.isdir(objs_dir):
        # Search for nested objs/ directory
        for root, dirs, files in os.walk(scene_dir):
            if "objs" in dirs:
                objs_dir = os.path.join(root, "objs")
                break

    for link_name, info in links.items():
        idx = info["part_idx"]
        if idx is None:
            continue

        obj_path = os.path.join(objs_dir, str(idx), f"{idx}.obj")
        if not os.path.exists(obj_path):
            continue

        obj = import_obj(obj_path, name=f"part_{link_name}")
        if obj is None:
            continue

        # Apply origin offset
        origin_key = str(idx)
        if origin_key in origins:
            ox, oy, oz = origins[origin_key]
            obj.location = Vector((ox, oy, oz))
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.transform_apply(location=True)
            obj.select_set(False)

        parts[link_name] = obj

    return parts


def load_parts_urdf_format(scene_dir, links, joints, root_link):
    """Load parts for IS/PartNet format: mesh files referenced in URDF.

    Each link may have multiple visual meshes, each with a visual origin offset.
    We also need to apply kinematic chain transforms (joint origins) to get
    each part into the correct world position.
    """
    # Compute world transforms via BFS (same as split_precompute)
    children_map = defaultdict(list)
    for j in joints:
        children_map[j.parent_link].append((j.child_link, j))

    # BFS to compute link world transforms
    link_T = {root_link: Matrix.Identity(4)}
    queue = deque([root_link])
    visited = set()

    while queue:
        parent = queue.popleft()
        if parent in visited:
            continue
        visited.add(parent)

        parent_T = link_T[parent]
        for child_link, joint in children_map.get(parent, []):
            T_local = compute_joint_local_transform(joint, 0.0)
            link_T[child_link] = parent_T @ T_local
            queue.append(child_link)

    parts = {}
    for link_name, info in links.items():
        if info["part_idx"] is None:
            continue
        if not info["mesh_files"]:
            continue

        link_objs = []
        for mesh_path, vis_xyz in info["mesh_files"]:
            if not os.path.exists(mesh_path):
                print(f"  WARNING: mesh not found: {mesh_path}")
                continue

            obj = import_obj(mesh_path, name=f"part_{link_name}_{len(link_objs)}")
            if obj is None:
                continue

            # Apply visual origin offset
            if any(abs(v) > 1e-8 for v in vis_xyz):
                obj.location = Vector(vis_xyz)
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.transform_apply(location=True)
                obj.select_set(False)

            link_objs.append(obj)

        if not link_objs:
            continue

        # Join multiple visuals into one object
        if len(link_objs) > 1:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in link_objs:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = link_objs[0]
            bpy.ops.object.join()
            result = bpy.context.active_object
        else:
            result = link_objs[0]

        result.name = f"part_{link_name}"

        # Apply kinematic chain transform to get world position
        if link_name in link_T:
            T_world = link_T[link_name]
            # Transform all vertices by T_world
            mesh = result.data
            for vert in mesh.vertices:
                v = T_world @ Vector((*vert.co, 1.0))
                vert.co = v.xyz

        parts[link_name] = result

    return parts


def load_scene_parts(metadata, links, joints, root_link):
    """Auto-detect format and load parts."""
    scene_dir = metadata["scene_dir"]
    origins_path = os.path.join(scene_dir, "origins.json")

    if os.path.exists(origins_path):
        print(f"  Loading parts (IM/PhysXNet format) from {scene_dir}")
        return load_parts_im_format(scene_dir, links, None)
    else:
        print(f"  Loading parts (URDF-direct format) from {scene_dir}")
        return load_parts_urdf_format(scene_dir, links, joints, root_link)


# ======================================================================
# Normalization
# ======================================================================

def normalize_parts(parts, center, scale):
    """Apply precompute normalization to all loaded parts."""
    center_v = Vector(center)
    for link_name, obj in parts.items():
        mesh = obj.data
        for vert in mesh.vertices:
            v = Vector(vert.co)
            vert.co = ((v - center_v) * scale)[:]
        mesh.update()


# ======================================================================
# Materials
# ======================================================================

def assign_material(obj, color, metallic=0.3, roughness=0.5, name=None):
    """Assign a simple PBR material to an object."""
    mat_name = name or f"mat_{obj.name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)


# ======================================================================
# Camera
# ======================================================================

def create_camera(name, center, distance, elev_deg, azim_deg, lens=50):
    """Create a static camera at spherical coordinates."""
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
    """Create a camera that moves from start to end in spherical coords."""
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

    # Linear interpolation
    if cam.animation_data and cam.animation_data.action:
        for fc in cam.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'

    return cam


def remove_camera(cam):
    """Remove a camera object and its data."""
    cam_data = cam.data
    bpy.data.objects.remove(cam, do_unlink=True)
    bpy.data.cameras.remove(cam_data)


# ======================================================================
# Render Setup
# ======================================================================

def setup_render_engine():
    """Set up Cycles with GPU."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences

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


def setup_render_settings(resolution, fps, num_frames, samples):
    """Configure render settings."""
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = num_frames

    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    try:
        scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    except TypeError:
        pass
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01

    scene.render.use_persistent_data = True
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    try:
        scene.view_settings.view_transform = 'Filmic'
    except TypeError:
        scene.view_settings.view_transform = 'AgX'
    try:
        scene.view_settings.look = 'Medium Contrast'
    except TypeError:
        scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def setup_envmap(envmap_path, strength=1.0):
    """Set up environment map lighting."""
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    node_links = world.node_tree.links

    nodes.clear()
    bg_node = nodes.new('ShaderNodeBackground')
    bg_node.inputs['Strength'].default_value = strength
    env_node = nodes.new('ShaderNodeTexEnvironment')
    env_node.image = bpy.data.images.load(envmap_path)
    output_node = nodes.new('ShaderNodeOutputWorld')

    node_links.new(env_node.outputs['Color'], bg_node.inputs['Color'])
    node_links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    print(f"  Envmap: {os.path.basename(envmap_path)}")
    return envmap_path


# ======================================================================
# Animation (per-frame FK update)
# ======================================================================

def animate_scene(parts, joints, children_map, root_link,
                  joints_by_name, split_info, num_frames):
    """Set keyframes for all parts across all frames using forward kinematics.

    For each frame, compute joint values, run FK, set part transforms.
    """
    # We need to store the initial (rest pose, normalized) vertex positions
    # and update them per frame. Since Blender keyframes work on object
    # transforms (not vertices), we use matrix_world approach:
    #
    # Strategy: store normalized rest vertices. Per frame, compute FK,
    # then compute delta from rest to animated and apply as matrix_world.
    #
    # Actually simpler: we work at the link level.
    # Each link has a rest transform (at q=0 for all joints).
    # Per frame, we compute the animated transform, then the delta is:
    #   T_animated = FK(q_values)
    #   T_rest = FK(q=0 for all)
    #   delta = T_animated @ T_rest.inv()
    # We apply delta to the part's vertices (already at rest position).
    #
    # But vertex-level animation per frame is expensive. Better approach:
    # store rest vertices, set object matrix_world = delta each frame.
    #
    # Even better: use Blender shape keys or just object transform keyframes.
    #
    # The cleanest approach for articulated rendering:
    # 1. Load parts at rest pose (q=0, already normalized)
    # 2. Per frame: compute FK, get link transforms T_frame[link]
    # 3. Compute rest transforms T_rest[link] (FK at q=0)
    # 4. Delta = T_frame[link] @ T_rest[link].inv()
    # 5. Set part.matrix_world = delta (since vertices are in rest-normalized space)

    # Compute rest transforms (q=0 for all joints)
    rest_values = {j.name: 0.0 for j in joints}
    T_rest = forward_kinematics(joints, children_map, root_link, rest_values)

    # Invert rest transforms per link
    T_rest_inv = {}
    for link_name, T in T_rest.items():
        T_rest_inv[link_name] = T.inverted()

    # For each frame, compute animated transforms and set keyframes
    for frame in range(1, num_frames + 1):
        t = (frame - 1) / max(num_frames - 1, 1)
        joint_values = compute_frame_joint_values(joints_by_name, split_info, t)

        # Run FK
        T_frame = forward_kinematics(joints, children_map, root_link, joint_values)

        # Apply to parts
        for link_name, obj in parts.items():
            if link_name not in T_frame:
                continue
            if link_name not in T_rest_inv:
                continue

            # Delta transform: from rest to animated
            delta = T_frame[link_name] @ T_rest_inv[link_name]

            # Apply normalization: the delta should be computed in normalized space
            # Our parts are already in normalized space. The FK transforms are in
            # original URDF space. We need to convert:
            # delta_norm = S @ delta @ S_inv
            # where S scales by 'scale' and translates by -center.
            # But since center/scale are uniform, and FK origin offsets are small,
            # the rotation part of delta is the same. Only translation needs scaling.
            #
            # Actually, for the delta approach to work correctly, we need to
            # compute FK in normalized space. Let's do that instead.

            obj.matrix_world = delta
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Set linear interpolation for all keyframes
    for link_name, obj in parts.items():
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'


def animate_scene_normalized(parts, joints, children_map, root_link,
                             joints_by_name, split_info, num_frames,
                             center, scale):
    """Animate using normalized-space FK.

    Compute FK in URDF space, then convert to normalized space for delta.
    The normalization is: p_norm = (p_world - center) * scale

    For a 4x4 transform T in world space, the normalized version is:
      T_norm = S @ T @ S_inv
    where S = scale_matrix(scale) @ translate(-center)
    """
    # Build S and S_inv as 4x4 matrices
    cx, cy, cz = center
    S = Matrix.Identity(4)
    S[0][0] = scale
    S[1][1] = scale
    S[2][2] = scale
    S[0][3] = -cx * scale
    S[1][3] = -cy * scale
    S[2][3] = -cz * scale

    S_inv = Matrix.Identity(4)
    S_inv[0][0] = 1.0 / scale
    S_inv[1][1] = 1.0 / scale
    S_inv[2][2] = 1.0 / scale
    S_inv[0][3] = cx
    S_inv[1][3] = cy
    S_inv[2][3] = cz

    # Rest transforms in normalized space
    rest_values = {j.name: 0.0 for j in joints}
    T_rest_world = forward_kinematics(joints, children_map, root_link, rest_values)
    T_rest_norm = {k: S @ v @ S_inv for k, v in T_rest_world.items()}
    T_rest_norm_inv = {k: v.inverted() for k, v in T_rest_norm.items()}

    for frame in range(1, num_frames + 1):
        t = (frame - 1) / max(num_frames - 1, 1)
        joint_values = compute_frame_joint_values(joints_by_name, split_info, t)

        T_frame_world = forward_kinematics(joints, children_map, root_link, joint_values)
        T_frame_norm = {k: S @ v @ S_inv for k, v in T_frame_world.items()}

        for link_name, obj in parts.items():
            if link_name not in T_frame_norm or link_name not in T_rest_norm_inv:
                continue

            delta = T_frame_norm[link_name] @ T_rest_norm_inv[link_name]
            obj.matrix_world = delta
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Linear interpolation
    for link_name, obj in parts.items():
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'


# ======================================================================
# Video Encoding
# ======================================================================

def frames_to_video(frame_dir, output_mp4, fps):
    """Encode PNG sequence to MP4 with ffmpeg."""
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
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr[:500]}")
            return False
        return True
    except Exception as e:
        print(f"  ffmpeg failed: {e}")
        return False


# ======================================================================
# View Resolution
# ======================================================================

def resolve_views(view_args):
    """Resolve view group names to dict of {name: config}.

    Returns:
        static_views: {name: (elev, azim)}
        moving_views: {name: (start_elev, start_azim, end_elev, end_azim)}
    """
    static = {}
    moving = {}

    for vg in view_args:
        vg_lower = vg.lower()
        if vg_lower == "all":
            static.update(HEMI_VIEWS)
            moving.update(ORBIT_VIEWS)
            moving.update(SWEEP_VIEWS)
        elif vg_lower == "hemi":
            static.update(HEMI_VIEWS)
        elif vg_lower == "orbit":
            moving.update(ORBIT_VIEWS)
        elif vg_lower == "sweep":
            moving.update(SWEEP_VIEWS)
        elif vg_lower in NAMED_VIEWS:
            static[vg_lower] = NAMED_VIEWS[vg_lower]
        elif vg_lower in HEMI_VIEWS:
            static[vg_lower] = HEMI_VIEWS[vg_lower]
        elif vg_lower in ORBIT_VIEWS:
            moving[vg_lower] = ORBIT_VIEWS[vg_lower]
        elif vg_lower in SWEEP_VIEWS:
            moving[vg_lower] = SWEEP_VIEWS[vg_lower]
        else:
            print(f"  WARNING: unknown view '{vg}', skipping")

    return static, moving


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    # Load metadata
    with open(args.metadata) as f:
        metadata = json.load(f)

    print(f"\n{'='*60}")
    print(f"Rendering {metadata['output_name']}/{metadata['identifier']}")
    print(f"{'='*60}")

    # Check required fields
    if "urdf_path" not in metadata or "scene_dir" not in metadata:
        print("ERROR: metadata.json missing urdf_path/scene_dir. Re-run precompute.")
        sys.exit(1)

    urdf_path = metadata["urdf_path"]
    scene_dir = metadata["scene_dir"]
    center = metadata["normalize"]["center"]
    scale = metadata["normalize"]["scale"]

    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found: {urdf_path}")
        sys.exit(1)

    # Parse URDF
    links, joints, root_link = parse_urdf(urdf_path)
    parent_map, children_map = build_kinematic_tree(joints)
    joints_by_name = {j.name: j for j in joints}

    # Resolve animodes to render
    all_splits = metadata.get("splits", {})
    if args.animode == "all":
        animodes_to_render = sorted(all_splits.keys())
    else:
        if args.animode not in all_splits:
            print(f"ERROR: animode '{args.animode}' not found. Available: {list(all_splits.keys())}")
            sys.exit(1)
        animodes_to_render = [args.animode]

    # Resolve views
    static_views, moving_views = resolve_views(args.views)
    total_views = len(static_views) + len(moving_views)
    print(f"  Animodes: {animodes_to_render}")
    print(f"  Views: {len(static_views)} static + {len(moving_views)} moving = {total_views}")

    num_frames = int(args.fps * args.duration)
    print(f"  Frames: {num_frames} ({args.duration}s @ {args.fps}fps)")

    # Select envmap
    if args.envmap:
        envmap_path = args.envmap
    else:
        envmaps = [os.path.join(ENVMAP_DIR, f) for f in os.listdir(ENVMAP_DIR)
                    if f.endswith(('.hdr', '.exr'))]
        envmap_path = random.choice(envmaps) if envmaps else None

    # Output directory is the parent of metadata.json
    meta_dir = os.path.dirname(os.path.abspath(args.metadata))

    # Render each animode
    for animode_name in animodes_to_render:
        print(f"\n--- Animode: {animode_name} ---")
        split_info = all_splits[animode_name]
        animode_dir = os.path.join(meta_dir, animode_name)

        # Set up Blender scene
        clear_scene()
        setup_render_engine()
        setup_render_settings(args.resolution, args.fps, num_frames, args.samples)
        if envmap_path:
            setup_envmap(envmap_path)

        # Load parts
        parts = load_scene_parts(metadata, links, joints, root_link)
        if not parts:
            print(f"  ERROR: No parts loaded, skipping {animode_name}")
            continue

        print(f"  Loaded {len(parts)} parts: {sorted(parts.keys())}")

        # Normalize
        normalize_parts(parts, center, scale)

        # Animate (same for all color modes — do once)
        animate_scene_normalized(parts, joints, children_map, root_link,
                                 joints_by_name, split_info, num_frames,
                                 center, scale)

        # Camera setup
        cam_center = [0.0, 0.0, 0.0]
        cam_distance = 3.6  # ~2.0 * 1.8

        # Determine which color passes to render
        two_coloring = split_info.get("two_coloring", {})
        if args.color_mode == "both":
            color_passes = ["part", "group"]
        else:
            color_passes = [args.color_mode]

        GROUP_COLORS = [
            (0.22, 0.46, 0.72),  # blue
            (0.89, 0.35, 0.13),  # orange
            (0.17, 0.63, 0.17),  # green
            (0.84, 0.15, 0.16),  # red
            (0.58, 0.40, 0.74),  # purple
            (0.55, 0.34, 0.29),  # brown
            (0.89, 0.47, 0.76),  # pink
            (0.50, 0.50, 0.50),  # gray
            (0.74, 0.74, 0.13),  # olive
            (0.09, 0.75, 0.81),  # cyan
            (0.98, 0.60, 0.01),  # amber
            (0.40, 0.76, 0.65),  # teal
        ]

        for color_pass in color_passes:
            vid_suffix = "_group" if color_pass == "group" else "_nobg"
            print(f"  --- Color pass: {color_pass} (suffix: {vid_suffix}) ---")

            # Assign materials for this pass
            if color_pass == "group":
                all_groups = []
                for group in two_coloring.get("part0_groups", []):
                    all_groups.append(group)
                for group in two_coloring.get("part1_groups", []):
                    all_groups.append(group)
                idx_to_group = {}
                for gi, group in enumerate(all_groups):
                    for link_idx in group:
                        idx_to_group[link_idx] = gi
                for link_name, obj in parts.items():
                    idx = links[link_name]["part_idx"]
                    gi = idx_to_group.get(idx, 0)
                    color = GROUP_COLORS[gi % len(GROUP_COLORS)]
                    assign_material(obj, color, metallic=0.25, roughness=0.5)
            else:
                part0_link_idxs = set()
                for group in two_coloring.get("part0_groups", []):
                    part0_link_idxs.update(group)
                part1_link_idxs = set()
                for group in two_coloring.get("part1_groups", []):
                    part1_link_idxs.update(group)
                for link_name, obj in parts.items():
                    idx = links[link_name]["part_idx"]
                    if idx in part0_link_idxs:
                        assign_material(obj, (0.45, 0.55, 0.65), metallic=0.2, roughness=0.6)
                    elif idx in part1_link_idxs:
                        assign_material(obj, (0.85, 0.55, 0.25), metallic=0.3, roughness=0.4)
                    else:
                        assign_material(obj, (0.6, 0.6, 0.6), metallic=0.1, roughness=0.7)

            # Render static views
            for view_name, (elev, azim) in sorted(static_views.items()):
                mp4_path = os.path.join(animode_dir, f"{view_name}{vid_suffix}.mp4")
                if args.skip_existing and os.path.exists(mp4_path):
                    print(f"  SKIP {view_name}{vid_suffix} (exists)")
                    continue

                frame_dir = os.path.join(animode_dir, f"{view_name}{vid_suffix}")
                os.makedirs(frame_dir, exist_ok=True)

                cam = create_camera(view_name, cam_center, cam_distance, elev, azim)
                bpy.context.scene.camera = cam
                bpy.context.scene.render.filepath = os.path.join(frame_dir, "frame_")
                bpy.ops.render.render(animation=True)

                ok = frames_to_video(frame_dir, mp4_path, args.fps)
                if ok:
                    print(f"  {view_name}{vid_suffix}: OK")
                else:
                    print(f"  {view_name}{vid_suffix}: FAIL (ffmpeg)")

                remove_camera(cam)

            # Render moving views
            for view_name, (se, sa, ee, ea) in sorted(moving_views.items()):
                mp4_path = os.path.join(animode_dir, f"{view_name}{vid_suffix}.mp4")
                if args.skip_existing and os.path.exists(mp4_path):
                    print(f"  SKIP {view_name}{vid_suffix} (exists)")
                    continue

                frame_dir = os.path.join(animode_dir, f"{view_name}{vid_suffix}")
                os.makedirs(frame_dir, exist_ok=True)

                cam = create_animated_camera(view_name, cam_center, cam_distance,
                                             se, sa, ee, ea, num_frames)
                bpy.context.scene.camera = cam
                bpy.context.scene.render.filepath = os.path.join(frame_dir, "frame_")
                bpy.ops.render.render(animation=True)

                ok = frames_to_video(frame_dir, mp4_path, args.fps)
                if ok:
                    print(f"  {view_name}{vid_suffix}: OK")
                else:
                    print(f"  {view_name}{vid_suffix}: FAIL (ffmpeg)")

                remove_camera(cam)

    # Update metadata with envmap info
    if envmap_path:
        metadata["envmap"] = os.path.basename(envmap_path)
        meta_path = os.path.abspath(args.metadata)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\nDone rendering {metadata['output_name']}/{metadata['identifier']}")


if __name__ == "__main__":
    main()
