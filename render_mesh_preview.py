"""
Blender script: render OBJ mesh from multiple viewpoints with solid shading.

Usage (called as subprocess):
  blender --background --python render_mesh_preview.py -- \
    --obj path/to/mesh.obj --color 0.35,0.55,0.85 \
    --output path/to/output.png --views 3 --resolution 512
"""

import sys
import os
import math
import argparse

import bpy
import mathutils


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--color", default="0.35,0.55,0.85")
    parser.add_argument("--output", required=True)
    parser.add_argument("--views", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--bg_color", default="0.04,0.04,0.06")
    return parser.parse_args(argv)


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for c in bpy.data.collections:
        if c.name != "Collection":
            bpy.data.collections.remove(c)


def import_obj(path):
    bpy.ops.wm.obj_import(filepath=path)
    imported = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not imported:
        imported = [o for o in bpy.data.objects if o.type == 'MESH']
    return imported


def setup_material(objects, color_rgb):
    mat = bpy.data.materials.new(name="MeshColor")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()

    # Principled BSDF
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.inputs['Base Color'].default_value = (*color_rgb, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.5
    bsdf.inputs['Metallic'].default_value = 0.1

    output = nodes.new('ShaderNodeOutputMaterial')
    mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    for obj in objects:
        obj.data.materials.clear()
        obj.data.materials.append(mat)


def get_bounds(objects):
    """Get combined bounding box of all objects."""
    all_coords = []
    for obj in objects:
        bbox = [obj.matrix_world @ mathutils.Vector(c) for c in obj.bound_box]
        all_coords.extend(bbox)
    if not all_coords:
        return mathutils.Vector((0, 0, 0)), 1.0
    xs = [c.x for c in all_coords]
    ys = [c.y for c in all_coords]
    zs = [c.z for c in all_coords]
    center = mathutils.Vector((
        (min(xs) + max(xs)) / 2,
        (min(ys) + max(ys)) / 2,
        (min(zs) + max(zs)) / 2,
    ))
    extent = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return center, extent


def setup_camera_and_lights(center, extent, elev_deg=25, azim_deg=45, resolution=512):
    """Create camera and lights pointing at center."""
    dist = extent * 1.6

    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)

    cam_x = center.x + dist * math.cos(elev) * math.sin(azim)
    cam_y = center.y - dist * math.cos(elev) * math.cos(azim)
    cam_z = center.z + dist * math.sin(elev)

    cam_data = bpy.data.cameras.new(name="Camera")
    cam_data.type = 'PERSP'
    cam_data.lens = 50
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    cam_obj.location = (cam_x, cam_y, cam_z)

    direction = center - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    bpy.context.scene.camera = cam_obj

    # Key light
    key_data = bpy.data.lights.new(name="KeyLight", type='SUN')
    key_data.energy = 3.0
    key_obj = bpy.data.objects.new("KeyLight", key_data)
    bpy.context.collection.objects.link(key_obj)
    key_obj.rotation_euler = (math.radians(50), math.radians(10), math.radians(30))

    # Fill light
    fill_data = bpy.data.lights.new(name="FillLight", type='SUN')
    fill_data.energy = 1.5
    fill_obj = bpy.data.objects.new("FillLight", fill_data)
    bpy.context.collection.objects.link(fill_obj)
    fill_obj.rotation_euler = (math.radians(40), math.radians(-30), math.radians(-60))

    # Rim light
    rim_data = bpy.data.lights.new(name="RimLight", type='SUN')
    rim_data.energy = 1.0
    rim_obj = bpy.data.objects.new("RimLight", rim_data)
    bpy.context.collection.objects.link(rim_obj)
    rim_obj.rotation_euler = (math.radians(-20), math.radians(0), math.radians(180))

    return cam_obj


def setup_render(resolution, bg_color):
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True

    # Enable GPU
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGB'

    # Background color
    scene.world = bpy.data.worlds.new(name="World")
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get("Background")
    if bg_node:
        bg_node.inputs['Color'].default_value = (*bg_color, 1.0)
        bg_node.inputs['Strength'].default_value = 0.5


def render_view(output_path):
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def main():
    args = parse_args()

    color_rgb = tuple(float(x) for x in args.color.split(","))
    bg_color = tuple(float(x) for x in args.bg_color.split(","))

    if not os.path.exists(args.obj):
        print(f"ERROR: OBJ not found: {args.obj}")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Render each view, save as separate files, then combine with PIL
    view_paths = []
    azimuths = [30 + i * (120 / max(args.views - 1, 1)) for i in range(args.views)]

    for vi, azim in enumerate(azimuths):
        clear_scene()
        objects = import_obj(args.obj)
        if not objects:
            print(f"ERROR: No mesh objects imported from {args.obj}")
            sys.exit(1)

        setup_material(objects, color_rgb)
        center, extent = get_bounds(objects)
        setup_camera_and_lights(center, extent, elev_deg=25, azim_deg=azim,
                                resolution=args.resolution)
        setup_render(args.resolution, bg_color)

        view_path = args.output.replace(".png", f"_v{vi}.png")
        render_view(view_path)
        view_paths.append(view_path)
        print(f"  Rendered view {vi}: azim={azim:.0f}° -> {view_path}")

    # Write view paths to a sidecar file for the caller
    manifest = args.output.replace(".png", "_views.txt")
    with open(manifest, "w") as f:
        for p in view_paths:
            f.write(p + "\n")

    print(f"Done: {len(view_paths)} views rendered")


if __name__ == "__main__":
    main()
