#!/usr/bin/env python3
"""
Verify precomputed splits by reassembling part0/part1 + joints.npy into
a colored Blender scene with axis annotations, then render a single image.

Usage (Blender):
  blender --background --python verify_split.py -- \
      --split_dir precompute/DishwasherFactory/0/default

  # Batch: all splits for one object
  blender --background --python verify_split.py -- \
      --object_dir precompute/DishwasherFactory/0

  # Also export URDF + GLB
  blender --background --python verify_split.py -- \
      --split_dir precompute/DishwasherFactory/0/default --export_urdf --export_glb

Output:
  {split_dir}/verify.png          — colored render with axis annotations
  {split_dir}/verify.urdf         — reconstructed URDF (if --export_urdf)
  {split_dir}/verify.glb          — colored GLB (if --export_glb)
"""

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom

import bpy
import bmesh
import numpy as np
from mathutils import Vector, Matrix, Euler

# Detect Blender version
IS_BLENDER_4X = bpy.app.version[0] >= 4


# ══════════════════════════════════════════════════════════════
# Color palette
# ══════════════════════════════════════════════════════════════

# Part colors (RGBA linear)
COLOR_BODY = (0.15, 0.35, 0.65, 1.0)     # steel blue
COLOR_MOVING = (0.85, 0.35, 0.10, 1.0)   # orange

# Axis colors — one per joint, cycling
AXIS_COLORS = [
    (0.1, 0.85, 0.2, 1.0),    # green
    (0.9, 0.9, 0.1, 1.0),     # yellow
    (0.9, 0.1, 0.9, 1.0),     # magenta
    (0.1, 0.9, 0.9, 1.0),     # cyan
    (1.0, 0.5, 0.0, 1.0),     # orange
    (0.5, 0.1, 1.0, 1.0),     # purple
    (1.0, 0.2, 0.2, 1.0),     # red
    (0.2, 1.0, 0.5, 1.0),     # mint
]

# Joint type label colors for cone tips
CONE_COLORS = {
    "revolute":   (1.0, 0.3, 0.3, 1.0),   # red
    "prismatic":  (0.3, 0.3, 1.0, 1.0),   # blue
    "continuous": (0.3, 1.0, 0.3, 1.0),   # green
}

JOINT_TYPES = ["revolute", "prismatic", "continuous"]


# ══════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════

def clear_scene():
    """Remove all objects, materials, and data."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)


def make_solid_material(name, color):
    """Create a solid-color Principled BSDF material."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf is None:
        for n in nodes:
            if n.type == 'BSDF_PRINCIPLED':
                bsdf = n
                break
    if bsdf:
        bsdf.inputs['Base Color'].default_value = color
        bsdf.inputs['Roughness'].default_value = 0.5
        bsdf.inputs['Metallic'].default_value = 0.1
    return mat


def import_obj(filepath):
    """Import OBJ, reset rotation, return the imported object."""
    before = set(bpy.data.objects)
    if IS_BLENDER_4X:
        bpy.ops.wm.obj_import(filepath=filepath,
                               forward_axis='NEGATIVE_Y', up_axis='Z')
    else:
        bpy.ops.import_scene.obj(filepath=filepath,
                                  use_edges=False, use_smooth_groups=True,
                                  axis_forward='-Y', axis_up='Z')
    after = set(bpy.data.objects)
    new_objs = list(after - before)

    if not new_objs:
        return None

    # Join if multiple objects imported
    if len(new_objs) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in new_objs:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = new_objs[0]
        bpy.ops.object.join()
        result = bpy.context.active_object
    else:
        result = new_objs[0]

    # Reset rotation (critical OBJ import fix)
    result.rotation_euler = (0, 0, 0)
    bpy.ops.object.select_all(action='DESELECT')
    result.select_set(True)
    bpy.context.view_layer.objects.active = result
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Smooth shading
    if IS_BLENDER_4X:
        bpy.ops.object.shade_smooth()
    else:
        for poly in result.data.polygons:
            poly.use_smooth = True

    return result


def assign_material(obj, mat):
    """Replace all materials on an object with a single material."""
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def create_cylinder_along_axis(origin, direction, length=3.0, radius=0.02, name="axis"):
    """Create a thin cylinder mesh aligned along a given axis direction.

    The cylinder is centered at `origin` and extends ±length/2 along `direction`.
    """
    # Create cylinder along Z, then rotate to match direction
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=length, location=(0, 0, 0),
        vertices=16, enter_editmode=False,
    )
    cyl = bpy.context.active_object
    cyl.name = name

    # Compute rotation from Z-axis to target direction
    z_axis = Vector((0, 0, 1))
    dir_vec = Vector(direction).normalized()
    if dir_vec.length < 1e-6:
        dir_vec = z_axis

    rot = z_axis.rotation_difference(dir_vec)
    cyl.rotation_mode = 'QUATERNION'
    cyl.rotation_quaternion = rot
    cyl.location = Vector(origin)

    # Apply transform
    bpy.ops.object.select_all(action='DESELECT')
    cyl.select_set(True)
    bpy.context.view_layer.objects.active = cyl
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    return cyl


def create_cone_tip(origin, direction, length=0.12, radius=0.05, name="cone"):
    """Create a cone at the positive end of the axis to show direction."""
    dir_vec = Vector(direction).normalized()
    tip_pos = Vector(origin) + dir_vec * 1.2  # Place at the end of the cylinder

    bpy.ops.mesh.primitive_cone_add(
        radius1=radius, radius2=0, depth=length,
        location=(0, 0, 0), vertices=16,
    )
    cone = bpy.context.active_object
    cone.name = name

    z_axis = Vector((0, 0, 1))
    rot = z_axis.rotation_difference(dir_vec)
    cone.rotation_mode = 'QUATERNION'
    cone.rotation_quaternion = rot
    cone.location = tip_pos

    bpy.ops.object.select_all(action='DESELECT')
    cone.select_set(True)
    bpy.context.view_layer.objects.active = cone
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    return cone


def create_origin_sphere(origin, radius=0.05, name="origin"):
    """Create a small sphere at the joint axis origin."""
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, location=Vector(origin),
        segments=12, ring_count=6,
    )
    sphere = bpy.context.active_object
    sphere.name = name
    return sphere


# ══════════════════════════════════════════════════════════════
# Camera & Render
# ══════════════════════════════════════════════════════════════

def setup_camera(center, distance, elev_deg=35, azim_deg=30):
    """Create camera at spherical coords looking at center."""
    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    x = distance * math.cos(elev) * math.cos(azim) + center[0]
    y = distance * math.cos(elev) * math.sin(azim) + center[1]
    z = distance * math.sin(elev) + center[2]

    cam_data = bpy.data.cameras.new(name="VerifyCamera")
    cam_data.lens = 50
    cam_data.clip_start = 0.001
    cam_data.clip_end = 100
    cam = bpy.data.objects.new("VerifyCamera", cam_data)
    bpy.context.collection.objects.link(cam)

    cam.location = Vector((x, y, z))
    direction = Vector(center) - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    bpy.context.scene.camera = cam
    return cam


def setup_lighting():
    """Simple 3-point lighting for verification render."""
    # Key light
    key_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    key_data.energy = 200
    key_data.size = 2.0
    key = bpy.data.objects.new("KeyLight", key_data)
    bpy.context.collection.objects.link(key)
    key.location = (2.0, -1.5, 3.0)
    key.rotation_euler = Euler((math.radians(45), 0, math.radians(30)))

    # Fill light
    fill_data = bpy.data.lights.new(name="FillLight", type='AREA')
    fill_data.energy = 80
    fill_data.size = 3.0
    fill = bpy.data.objects.new("FillLight", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (-2.0, 1.0, 2.0)
    fill.rotation_euler = Euler((math.radians(50), 0, math.radians(-150)))

    # Rim light
    rim_data = bpy.data.lights.new(name="RimLight", type='AREA')
    rim_data.energy = 120
    rim_data.size = 1.5
    rim = bpy.data.objects.new("RimLight", rim_data)
    bpy.context.collection.objects.link(rim)
    rim.location = (-0.5, 2.5, 1.5)
    rim.rotation_euler = Euler((math.radians(30), 0, math.radians(-90)))

    # Light gray background
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs['Color'].default_value = (0.85, 0.85, 0.88, 1.0)
        bg.inputs['Strength'].default_value = 0.3


def setup_render(resolution=1024, samples=64):
    """Configure Cycles render settings."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    # Color management
    view_settings = scene.view_settings
    try:
        view_settings.view_transform = 'Filmic'
    except TypeError:
        view_settings.view_transform = 'AgX'


# ══════════════════════════════════════════════════════════════
# URDF generation
# ══════════════════════════════════════════════════════════════

def generate_urdf(split_dir, joints_tensor, part0_path, part1_path):
    """Generate a simple URDF from part0 (body) + part1 (moving) + joint params."""
    root = ET.Element("robot", name="verified_object")

    # Base link (part0)
    base_link = ET.SubElement(root, "link", name="base_link")
    vis = ET.SubElement(base_link, "visual")
    geom = ET.SubElement(vis, "geometry")
    ET.SubElement(geom, "mesh", filename=os.path.basename(part0_path))

    n_joints = int(joints_tensor[:, 11].sum())

    for i in range(n_joints):
        row = joints_tensor[i]
        origin = row[0:3]
        axis_dir = row[3:6]
        type_idx = int(row[6:9].argmax())
        jtype = JOINT_TYPES[type_idx]
        range_min, range_max = float(row[9]), float(row[10])

        child_name = f"moving_link_{i}"

        # Child link (all point to part1 mesh)
        child_link = ET.SubElement(root, "link", name=child_name)
        vis_c = ET.SubElement(child_link, "visual")
        geom_c = ET.SubElement(vis_c, "geometry")
        ET.SubElement(geom_c, "mesh", filename=os.path.basename(part1_path))

        # Joint
        joint = ET.SubElement(root, "joint",
                              name=f"joint_{jtype}_{i}", type=jtype)
        ET.SubElement(joint, "parent", link="base_link")
        ET.SubElement(joint, "child", link=child_name)
        ET.SubElement(joint, "origin",
                      xyz=f"{origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}")
        ET.SubElement(joint, "axis",
                      xyz=f"{axis_dir[0]:.6f} {axis_dir[1]:.6f} {axis_dir[2]:.6f}")
        if jtype != "continuous":
            ET.SubElement(joint, "limit",
                          lower=f"{range_min:.6f}", upper=f"{range_max:.6f}")

    # Pretty print
    rough = ET.tostring(root, encoding='unicode')
    parsed = minidom.parseString(rough)
    urdf_str = parsed.toprettyxml(indent="  ")
    # Remove extra XML declaration
    lines = urdf_str.split('\n')
    if lines[0].startswith('<?xml'):
        lines = lines[1:]

    urdf_path = os.path.join(split_dir, "verify.urdf")
    with open(urdf_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  URDF: {urdf_path}")
    return urdf_path


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def process_split(split_dir, resolution=1024, samples=64,
                  export_urdf=False, export_glb=False):
    """Process a single split directory: render verify.png + optional exports."""
    part0_path = os.path.join(split_dir, "part0.obj")
    part1_path = os.path.join(split_dir, "part1.obj")
    joints_path = os.path.join(split_dir, "joints.npy")

    if not os.path.exists(part0_path) or not os.path.exists(part1_path):
        print(f"  SKIP: Missing part0.obj or part1.obj in {split_dir}")
        return False

    joints_tensor = None
    if os.path.exists(joints_path):
        joints_tensor = np.load(joints_path)
    else:
        print(f"  WARNING: No joints.npy in {split_dir}, rendering parts only")

    print(f"\n{'='*60}")
    print(f"  Verifying: {split_dir}")
    print(f"{'='*60}")

    clear_scene()

    # ── Import parts ──
    part0 = import_obj(part0_path)
    if part0:
        part0.name = "part0_body"
        mat_body = make_solid_material("mat_body", COLOR_BODY)
        assign_material(part0, mat_body)
        print(f"  Part0 (body):   {len(part0.data.vertices)} verts")

    part1 = import_obj(part1_path)
    if part1:
        part1.name = "part1_moving"
        mat_moving = make_solid_material("mat_moving", COLOR_MOVING)
        assign_material(part1, mat_moving)
        print(f"  Part1 (moving): {len(part1.data.vertices)} verts")

    # ── Create axis annotations from joints.npy ──
    n_joints = 0
    if joints_tensor is not None:
        n_joints = int(joints_tensor[:, 11].sum())
        print(f"  Joints: {n_joints}")

        for i in range(n_joints):
            row = joints_tensor[i]
            origin = row[0:3].tolist()
            axis_dir = row[3:6].tolist()
            type_idx = int(row[6:9].argmax())
            jtype = JOINT_TYPES[type_idx]
            mrange = row[10]

            color = AXIS_COLORS[i % len(AXIS_COLORS)]
            mat_axis = make_solid_material(f"mat_axis_{i}", color)
            # Emission for axis lines so they're always visible
            bsdf = mat_axis.node_tree.nodes.get("Principled BSDF")
            if bsdf is None:
                for n in mat_axis.node_tree.nodes:
                    if n.type == 'BSDF_PRINCIPLED':
                        bsdf = n
                        break
            if bsdf:
                try:
                    bsdf.inputs['Emission Color'].default_value = color
                    bsdf.inputs['Emission Strength'].default_value = 2.0
                except KeyError:
                    try:
                        bsdf.inputs['Emission'].default_value = color
                    except KeyError:
                        pass

            # Cylinder along axis (long enough to extend well beyond [-0.95, 0.95] box)
            cyl = create_cylinder_along_axis(origin, axis_dir,
                                             length=3.0, radius=0.02,
                                             name=f"axis_{i}_{jtype}")
            assign_material(cyl, mat_axis)

            # Cone tip showing direction
            cone = create_cone_tip(origin, axis_dir,
                                   name=f"cone_{i}_{jtype}")
            cone_color = CONE_COLORS.get(jtype, color)
            mat_cone = make_solid_material(f"mat_cone_{i}", cone_color)
            assign_material(cone, mat_cone)

            # Origin sphere (larger for visibility)
            sphere = create_origin_sphere(origin, radius=0.05,
                                          name=f"origin_{i}")
            mat_sphere = make_solid_material(f"mat_origin_{i}",
                                             (1.0, 1.0, 1.0, 1.0))
            assign_material(sphere, mat_sphere)

            print(f"    [{i}] {jtype}: origin={[f'{v:.3f}' for v in origin]}, "
                  f"dir={[f'{v:.3f}' for v in axis_dir]}, range={mrange:.3f}")

    # ── Camera & lighting ──
    # Compute bounding box center and distance
    all_verts = []
    for obj in [part0, part1]:
        if obj:
            all_verts.extend([list(v.co) for v in obj.data.vertices])
    if all_verts:
        verts_arr = np.array(all_verts)
        center = ((verts_arr.max(axis=0) + verts_arr.min(axis=0)) / 2).tolist()
        extent = (verts_arr.max(axis=0) - verts_arr.min(axis=0)).max()
        distance = extent * 1.8
    else:
        center = [0, 0, 0]
        distance = 3.0

    setup_camera(center, distance, elev_deg=35, azim_deg=30)
    setup_lighting()
    setup_render(resolution=resolution, samples=samples)

    # ── Render ──
    out_path = os.path.join(split_dir, "verify.png")
    bpy.context.scene.render.filepath = out_path
    bpy.context.scene.frame_set(1)
    bpy.ops.render.render(write_still=True)
    print(f"  Rendered: {out_path}")

    # ── Optional exports ──
    if export_urdf and joints_tensor is not None:
        generate_urdf(split_dir, joints_tensor, part0_path, part1_path)

    if export_glb:
        glb_path = os.path.join(split_dir, "verify.glb")
        bpy.ops.export_scene.gltf(
            filepath=glb_path,
            export_format='GLB',
            use_selection=False,
        )
        print(f"  GLB: {glb_path}")

    return True


def main():
    # Parse args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Verify precomputed splits")
    parser.add_argument("--split_dir", type=str, default=None,
                        help="Single split directory to verify")
    parser.add_argument("--object_dir", type=str, default=None,
                        help="Object directory (verify all splits)")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--export_urdf", action="store_true")
    parser.add_argument("--export_glb", action="store_true")
    args = parser.parse_args(argv)

    if args.split_dir:
        process_split(args.split_dir, args.resolution, args.samples,
                      args.export_urdf, args.export_glb)
    elif args.object_dir:
        # Process all split subdirs
        for entry in sorted(os.listdir(args.object_dir)):
            subdir = os.path.join(args.object_dir, entry)
            if os.path.isdir(subdir) and os.path.exists(os.path.join(subdir, "part0.obj")):
                process_split(subdir, args.resolution, args.samples,
                              args.export_urdf, args.export_glb)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  blender --background --python verify_split.py -- \\")
        print("      --split_dir precompute/DishwasherFactory/0/default")
        print("  blender --background --python verify_split.py -- \\")
        print("      --object_dir precompute/DishwasherFactory/0 --export_urdf --export_glb")


if __name__ == "__main__":
    main()
