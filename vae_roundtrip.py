#!/usr/bin/env python3
"""PartPacker VAE roundtrip: OBJ -> encode -> decode -> mesh + Blender comparison render."""
import argparse
import importlib
import os
import subprocess
import sys
import numpy as np
import torch
import trimesh

PARTPACKER_ROOT = "/mnt/cpfs/yurh/PartPacker"
VAE_CKPT = os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt")
VAE_CONFIG = "vae.configs.part_woenc"
BLENDER = "/mnt/cpfs/yurh/blender-4.2.18-linux-x64/blender"

sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

import fpsample
import meshiki
from vae.model import Model


def load_vae(device):
    ckpt = torch.load(VAE_CKPT, weights_only=True)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    config = importlib.import_module(VAE_CONFIG).make_config()
    model = Model(config).eval().to(device).to(torch.bfloat16)
    model.load_state_dict(ckpt, strict=True)
    del ckpt
    print(f"Loaded VAE from {VAE_CKPT}")
    return model


def prepare_input(vertices, faces, num_fps=2048, num_fps_salient=2048):
    mesh = meshiki.Mesh(vertices, faces)
    uniform_pts = mesh.uniform_point_sample(200000)
    uniform_pts = meshiki.fps(uniform_pts, 32768)
    salient_pts = mesh.salient_point_sample(16384, thresh_bihedral=15)

    sample = {}
    sample["pointcloud"] = torch.from_numpy(uniform_pts)
    fps_idx = fpsample.bucket_fps_kdline_sampling(uniform_pts, num_fps, h=5, start_idx=0)
    sample["fps_indices"] = torch.from_numpy(fps_idx).long()
    sample["pointcloud_dorases"] = torch.from_numpy(salient_pts)
    fps_idx_s = fpsample.bucket_fps_kdline_sampling(salient_pts, num_fps_salient, h=5, start_idx=0)
    sample["fps_indices_dorases"] = torch.from_numpy(fps_idx_s).long()
    return sample


def roundtrip(obj_path, vae_model, device, resolution=384):
    """OBJ -> encode -> decode -> mesh vertices/faces."""
    mesh = trimesh.load(obj_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()

    sample = prepare_input(mesh.vertices.astype(np.float32), mesh.faces)
    for k in sample:
        sample[k] = sample[k].unsqueeze(0).to(device)

    with torch.inference_mode():
        result = vae_model(sample, resolution=resolution)

    vertices, faces = result["meshes"][0]
    # Filter NaN vertices from hierarchical query
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    valid_mask = ~np.isnan(mesh.vertices).any(axis=1)
    if not valid_mask.all():
        # Remove faces that reference NaN vertices, then clean
        face_valid = valid_mask[mesh.faces].all(axis=1)
        mesh = trimesh.Trimesh(mesh.vertices, mesh.faces[face_valid], process=True)
    return mesh.vertices, mesh.faces


BLENDER_RENDER_SCRIPT = '''
import bpy
import sys
import os
import math

argv = sys.argv[sys.argv.index("--") + 1:]
p0_obj = argv[0]
p1_obj = argv[1]
output_png = argv[2]

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
for c in bpy.data.collections:
    bpy.data.collections.remove(c)

def import_obj(path, color, name):
    bpy.ops.wm.obj_import(filepath=path)
    obj = bpy.context.selected_objects[0]
    obj.name = name
    mat = bpy.data.materials.new(name=f"mat_{name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Metallic"].default_value = 0.1
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    return obj

BLUE = (0.15, 0.35, 0.85, 1.0)
ORANGE = (0.95, 0.55, 0.1, 1.0)

o0 = import_obj(p0_obj, BLUE, "part0")
o1 = import_obj(p1_obj, ORANGE, "part1")

import mathutils
all_verts = []
for obj in [o0, o1]:
    for v in obj.data.vertices:
        all_verts.append(obj.matrix_world @ v.co)
if all_verts:
    bbox_min = mathutils.Vector((min(v.x for v in all_verts), min(v.y for v in all_verts), min(v.z for v in all_verts)))
    bbox_max = mathutils.Vector((max(v.x for v in all_verts), max(v.y for v in all_verts), max(v.z for v in all_verts)))
    center = (bbox_min + bbox_max) / 2
    size = max((bbox_max - bbox_min).x, (bbox_max - bbox_min).y, (bbox_max - bbox_min).z)
    scale = 2.0 / size if size > 0 else 1.0
    for obj in [o0, o1]:
        obj.location -= center
        obj.scale *= scale

cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 50
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam
cam.location = (2.5, -2.5, 2.0)
cam.rotation_euler = (math.radians(60), 0, math.radians(45))

light_data = bpy.data.lights.new("Light", type='SUN')
light_data.energy = 3
light = bpy.data.objects.new("Light", light_data)
bpy.context.scene.collection.objects.link(light)
light.location = (3, -3, 5)
light.rotation_euler = (math.radians(30), math.radians(15), 0)

world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.95, 0.95, 0.95, 1.0)
bg.inputs["Strength"].default_value = 0.5

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.samples = 64
scene.render.resolution_x = 800
scene.render.resolution_y = 800
scene.render.film_transparent = False
scene.render.filepath = output_png
scene.render.image_settings.file_format = 'PNG'
bpy.ops.render.render(write_still=True)
'''


def render_blender(p0_obj, p1_obj, output_png):
    script_path = "/tmp/vae_roundtrip_render.py"
    with open(script_path, 'w') as f:
        f.write(BLENDER_RENDER_SCRIPT)
    cmd = [BLENDER, "--background", "--python", script_path, "--", p0_obj, p1_obj, output_png]
    subprocess.run(cmd, capture_output=True, timeout=120)


def make_comparison(images, labels, output_path):
    from PIL import Image, ImageDraw
    imgs = [Image.open(p) for p in images if os.path.exists(p)]
    if not imgs:
        return
    w, h = imgs[0].size
    margin = 40
    canvas = Image.new('RGB', (w * len(imgs), h + margin), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    for i, (img, label) in enumerate(zip(imgs, labels)):
        canvas.paste(img, (i * w, margin))
        tw = draw.textlength(label)
        draw.text(((i * w + w // 2 - tw // 2), 5), label, fill=(0, 0, 0))
    canvas.save(output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part0", required=True)
    parser.add_argument("--part1", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    vae_model = load_vae(device)

    obj_paths = {}
    for part_name, obj_path in [("part0", args.part0), ("part1", args.part1)]:
        print(f"\nRoundtrip {part_name}: {obj_path}")
        orig = trimesh.load(obj_path, process=False, force="mesh")
        print(f"  Original: {len(orig.vertices)}v {len(orig.faces)}f")

        verts, faces = roundtrip(obj_path, vae_model, device, args.resolution)
        recon = trimesh.Trimesh(verts, faces, process=False)
        bodies = len(recon.split(only_watertight=False))
        print(f"  Reconstructed: {len(recon.vertices)}v {len(recon.faces)}f | "
              f"watertight={recon.is_watertight} | bodies={bodies}")

        orig_out = os.path.join(args.output_dir, f"{part_name}_original.obj")
        recon_out = os.path.join(args.output_dir, f"{part_name}_recon.obj")
        orig.export(orig_out)
        recon.export(recon_out)
        obj_paths[f"{part_name}_orig"] = orig_out
        obj_paths[f"{part_name}_recon"] = recon_out
        print(f"  Saved: {recon_out}")

    if not args.no_render:
        print("\nRendering comparison...")
        render_paths = []
        render_labels = []

        # Original GT
        png = os.path.join(args.output_dir, "1_original.png")
        render_blender(obj_paths["part0_orig"], obj_paths["part1_orig"], png)
        render_paths.append(png)
        render_labels.append("Original GT")

        # VAE Roundtrip
        png = os.path.join(args.output_dir, "2_vae_roundtrip.png")
        render_blender(obj_paths["part0_recon"], obj_paths["part1_recon"], png)
        render_paths.append(png)
        render_labels.append("PartPacker VAE Roundtrip")

        # Comparison
        comp = os.path.join(args.output_dir, "compare.png")
        make_comparison(render_paths, render_labels, comp)
        print(f"  Comparison: {comp}")

    print(f"\nDone! Results in {args.output_dir}")


if __name__ == "__main__":
    main()
