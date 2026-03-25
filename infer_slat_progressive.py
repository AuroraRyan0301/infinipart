#!/usr/bin/env python3
"""
Inference with progressive DualPartSLatModel (512 only, single stage).

Uses GT SLat coords as input (overfit/eval setting):
  1. Load GT SLat coords from slat_gt cache
  2. Flow matching: noise → SLat latent at GT coords
  3. SLat decoder → mesh
  4. Blender render: GT vs Pred comparison (blue=part0, orange=part1)

Usage:
  CUDA_VISIBLE_DEVICES=1 python infer_slat_progressive.py \
    --ckpt /mnt/data_ssd/infinigen-sim/slat_progressive_full/step_20000.pt \
    --output_dir ./output/slat_progressive_infer \
    --max_samples 10
"""
import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import trimesh

TRELLIS_ROOT = "/mnt/cpfs/yurh/TRELLIS.2"
sys.path.insert(0, TRELLIS_ROOT)
sys.path.insert(0, os.path.join(TRELLIS_ROOT, "o-voxel"))
sys.path.insert(0, "/mnt/cpfs/yurh/Infinigen-Sim")


import cumesh
from trellis2.modules.sparse import SparseTensor

SLAT_NORM_MEAN = torch.tensor([
    0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
    -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
    0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
    -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665
])
SLAT_NORM_STD = torch.tensor([
    5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
    5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
    4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
    5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691
])
SLAT_IN_CHANNELS = 32


def denormalize_slat(feats, device):
    return feats * SLAT_NORM_STD.to(device) + SLAT_NORM_MEAN.to(device)


def flow_sampling(model, x0_st, x1_st, vjepa_feats, num_steps=50, rescale_t=3.0):
    """Flow matching: noise → SLat latent (Euler, t: 1→0)."""
    device = x0_st.feats.device
    x0 = x0_st.replace(torch.randn(x0_st.feats.shape[0], SLAT_IN_CHANNELS, device=device))
    x1 = x1_st.replace(torch.randn(x1_st.feats.shape[0], SLAT_IN_CHANNELS, device=device))

    t_seq = np.linspace(1, 0, num_steps + 1)
    t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
    t_pairs = [(t_seq[i], t_seq[i + 1]) for i in range(num_steps)]

    with torch.inference_mode():
        for t_val, t_prev in t_pairs:
            t = torch.tensor([1000 * t_val], device=device, dtype=torch.float32)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                v0, v1 = model(x0, x1, t, vjepa_feats)
            dt = t_val - t_prev
            x0 = x0.replace(x0.feats - dt * v0.feats.float())
            x1 = x1.replace(x1.feats - dt * v1.feats.float())

    return x0, x1


def postprocess_mesh_cumesh(vertices, faces, grid_size=512, decimation_target=100000,
                            max_hole_perimeter=3e-2, min_component_size=1e-4,
                            remesh_band=1.0, remesh_project=0.9):
    """TRELLIS 2 exact post-processing pipeline (postprocess.py).
    Uses remesh branch: DC remeshing to rebuild topology from fragmented mesh."""
    vertices = vertices.cuda()
    faces = faces.cuda()

    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)

    # Step 0: Initial fill holes
    mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
    vertices_clean, faces_clean = mesh.read()

    # Build BVH on cleaned mesh
    bvh = cumesh.cuBVH(vertices_clean, faces_clean)

    # Remesh: Dual Contouring to rebuild topology (TRELLIS 2 remesh branch)
    aabb = torch.tensor([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]], device='cuda', dtype=torch.float32)
    center = aabb.mean(dim=0)
    scale = (aabb[1] - aabb[0]).max().item()
    resolution = grid_size

    mesh.init(*cumesh.remeshing.remesh_narrow_band_dc(
        vertices_clean, faces_clean,
        center=center,
        scale=(resolution + 3 * remesh_band) / resolution * scale,
        resolution=resolution,
        band=remesh_band,
        project_back=remesh_project,
        verbose=False,
        bvh=bvh,
    ))

    # Simplify
    mesh.simplify(decimation_target)

    # Final cleanup
    mesh.remove_duplicate_faces()
    mesh.repair_non_manifold_edges()
    mesh.remove_small_connected_components(min_component_size)
    mesh.fill_holes(max_hole_perimeter=max_hole_perimeter)
    mesh.unify_face_orientations()
    return mesh.read()


def decode_slat_to_mesh(decoder, slat_st, resolution=512):
    """Decode SLat SparseTensor → trimesh, with cumesh post-processing."""
    decoder.set_resolution(resolution)
    with torch.inference_mode():
        results = decoder(slat_st)
    if isinstance(results, list) and len(results) > 0:
        mesh_obj = results[0]
        v, f = postprocess_mesh_cumesh(mesh_obj.vertices, mesh_obj.faces)
        return trimesh.Trimesh(v.cpu().numpy(), f.cpu().numpy(), process=False)
    return None


BLENDER_RENDER_SCRIPT = '''
import bpy
import sys
import os
import math

argv = sys.argv[sys.argv.index("--") + 1:]
p0_obj = argv[0]
p1_obj = argv[1]
output_png = argv[2]
title = argv[3] if len(argv) > 3 else ""

# Clear scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
for c in bpy.data.collections:
    bpy.data.collections.remove(c)

# Import meshes
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

# Auto-center and scale
import mathutils
all_verts = []
for obj in [o0, o1]:
    mesh = obj.data
    for v in mesh.vertices:
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

# Camera
cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 50
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam
cam.location = (2.5, -2.5, 2.0)
cam.rotation_euler = (math.radians(60), 0, math.radians(45))

# Light
light_data = bpy.data.lights.new("Light", type='SUN')
light_data.energy = 3
light = bpy.data.objects.new("Light", light_data)
bpy.context.scene.collection.objects.link(light)
light.location = (3, -3, 5)
light.rotation_euler = (math.radians(30), math.radians(15), 0)

# World
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.95, 0.95, 0.95, 1.0)
bg.inputs["Strength"].default_value = 0.5

# Render settings
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


def render_blender(p0_obj, p1_obj, output_png, title=""):
    """Render part0 (blue) + part1 (orange) via Blender."""
    blender = "/mnt/cpfs/yurh/blender-4.2.18-linux-x64/blender"
    script_path = "/tmp/slat_render_tmp.py"
    with open(script_path, 'w') as f:
        f.write(BLENDER_RENDER_SCRIPT)
    cmd = [blender, "--background", "--python", script_path, "--",
           p0_obj, p1_obj, output_png, title]
    subprocess.run(cmd, capture_output=True, timeout=120)


def make_comparison_image(images, labels, output_path):
    """Stitch multiple PNGs into a comparison grid with labels."""
    from PIL import Image, ImageDraw, ImageFont
    imgs = [Image.open(p) for p in images if os.path.exists(p)]
    if not imgs:
        return

    w, h = imgs[0].size
    margin = 40
    total_w = w * len(imgs)
    total_h = h + margin

    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for i, (img, label) in enumerate(zip(imgs, labels)):
        canvas.paste(img, (i * w, margin))
        tw = draw.textlength(label)
        draw.text(((i * w + w // 2 - tw // 2), 5), label, fill=(0, 0, 0))

    canvas.save(output_path)


def discover_samples(data_root, slat_gt_root, precompute_root, max_samples=10,
                     vae_coords_dir=None, one_per_cat=False):
    """Find samples that have SLat GT + VJEPA + original OBJs."""
    samples = []
    seen_cats = set()
    for cat in sorted(os.listdir(slat_gt_root)):
        cat_gt = os.path.join(slat_gt_root, cat)
        cat_data = os.path.join(data_root, cat)
        if not os.path.isdir(cat_gt) or not os.path.isdir(cat_data):
            continue
        cat_samples = []
        for f in sorted(os.listdir(cat_gt)):
            if not f.endswith('.pt'):
                continue
            mid = f[:-3]
            parts = mid.split("_", 1)
            if len(parts) < 2:
                continue
            seed, animode = parts[0], parts[1]

            # Check VJEPA
            views_dir = os.path.join(cat_data, mid, "views")
            if not os.path.isdir(views_dir):
                continue
            jepa_files = sorted([os.path.join(views_dir, j) for j in os.listdir(views_dir)
                                  if j.endswith("_nobg_jepa.pt")])
            if not jepa_files:
                continue

            # If using VAE coords, check they exist
            vae_coords_path = None
            if vae_coords_dir:
                vae_coords_path = os.path.join(vae_coords_dir, cat, f)
                if not os.path.exists(vae_coords_path):
                    continue

            # Check original OBJs for GT comparison
            p0_gt = os.path.join(precompute_root, cat, seed, animode, "part0.obj")
            p1_gt = os.path.join(precompute_root, cat, seed, animode, "part1.obj")

            cat_samples.append({
                "id": f"{cat}/{mid}",
                "gt_slat": os.path.join(cat_gt, f),
                "jepa": jepa_files[0],
                "p0_gt_obj": p0_gt if os.path.exists(p0_gt) else None,
                "p1_gt_obj": p1_gt if os.path.exists(p1_gt) else None,
                "vae_coords": vae_coords_path,
            })

        if cat_samples:
            if one_per_cat:
                samples.append(random.choice(cat_samples))
            else:
                samples.extend(cat_samples)

    random.shuffle(samples)
    return samples[:max_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default="/mnt/data_ssd/infinigen-sim-data/encoded")
    parser.add_argument("--slat_gt_root", default="/mnt/data_ssd/infinigen-sim-data/slat_gt")
    parser.add_argument("--precompute_root", default="/mnt/data_ssd/infinigen-sim-data/precompute")
    parser.add_argument("--output_dir", default="./output/slat_progressive_infer")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=20000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_render", action="store_true", help="Skip Blender rendering")
    parser.add_argument("--vae_coords_dir", type=str, default=None,
                        help="Use VAE occ coords instead of GT coords (path to precomputed vae_coords/)")
    parser.add_argument("--one_per_cat", action="store_true", help="Only 1 sample per category")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model
    from dual_part_slat import build_dual_part_model, load_slat_decoder
    ckpt = torch.load(args.ckpt, weights_only=False, map_location=device)
    cross_attn_start = ckpt.get("cross_attn_start_block", 0)
    print(f"Loading DualPartSLatModel (512, cross_attn from block {cross_attn_start})...")
    model = build_dual_part_model(device=device, resolution="512",
                                   cross_attn_start_block=cross_attn_start)
    model.vjepa_proj.load_state_dict(ckpt["vjepa_proj"])
    model.part_cross_attns.load_state_dict(ckpt["part_cross_attns"])
    model.eval()
    print(f"  Loaded checkpoint: step={ckpt.get('step', '?')}")

    # Load SLat decoder
    print("Loading SLat Decoder...")
    decoder = load_slat_decoder(device=device)

    # Discover samples
    samples = discover_samples(args.data_root, args.slat_gt_root, args.precompute_root,
                                max_samples=args.max_samples,
                                vae_coords_dir=args.vae_coords_dir,
                                one_per_cat=args.one_per_cat)
    coord_source = "VAE occ" if args.vae_coords_dir else "GT SLat"
    print(f"\nInference on {len(samples)} samples (coords: {coord_source})\n")

    for i, s in enumerate(samples):
        safe = s["id"].replace("/", "_")
        try:
            gt = torch.load(s["gt_slat"], weights_only=False, map_location=device)
            p0_gt_data = gt["p0_lr"]
            p1_gt_data = gt["p1_lr"]

            # Choose coords source
            if s.get("vae_coords") and os.path.exists(s["vae_coords"]):
                vae_c = torch.load(s["vae_coords"], weights_only=False, map_location=device)
                p0_coords = vae_c["p0_coords"].to(device)
                p1_coords = vae_c["p1_coords"].to(device)
                coord_tag = "VAE"
            else:
                p0_coords = p0_gt_data["coords"].to(device)
                p1_coords = p1_gt_data["coords"].to(device)
                coord_tag = "GT"

            total_tokens = p0_coords.shape[0] + p1_coords.shape[0]
            if total_tokens > args.max_tokens:
                print(f"  [{i+1}] SKIP {s['id']}: {total_tokens} tokens > {args.max_tokens}")
                continue

            jepa = torch.load(s["jepa"], weights_only=False, map_location=device)
            if jepa.dim() == 2:
                jepa = jepa.unsqueeze(0)

            print(f"  [{i+1}/{len(samples)}] {s['id']} [{coord_tag}] "
                  f"(p0={p0_coords.shape[0]}, p1={p1_coords.shape[0]} tokens)")

            # Flow matching sampling
            t0 = time.time()
            x0_st = SparseTensor(feats=torch.zeros(p0_coords.shape[0], SLAT_IN_CHANNELS, device=device),
                                  coords=p0_coords)
            x1_st = SparseTensor(feats=torch.zeros(p1_coords.shape[0], SLAT_IN_CHANNELS, device=device),
                                  coords=p1_coords)
            pred0, pred1 = flow_sampling(model, x0_st, x1_st, jepa, num_steps=args.num_steps)

            # Denormalize
            pred0 = pred0.replace(denormalize_slat(pred0.feats.float(), device))
            pred1 = pred1.replace(denormalize_slat(pred1.feats.float(), device))
            dt = time.time() - t0
            print(f"    Flow sampling: {dt:.1f}s")

            # Decode to mesh
            mesh0 = decode_slat_to_mesh(decoder, pred0, resolution=512)
            mesh1 = decode_slat_to_mesh(decoder, pred1, resolution=512)

            # Also decode GT SLat for comparison (only when using GT coords — same coord space)
            gt_mesh0, gt_mesh1 = None, None
            if not s.get("vae_coords"):
                gt0_st = SparseTensor(feats=p0_gt_data["feats"].to(device), coords=p0_coords)
                gt1_st = SparseTensor(feats=p1_gt_data["feats"].to(device), coords=p1_coords)
                gt_mesh0 = decode_slat_to_mesh(decoder, gt0_st, resolution=512)
                gt_mesh1 = decode_slat_to_mesh(decoder, gt1_st, resolution=512)

            # Save OBJs
            obj_paths = {}
            for name, mesh in [("pred_p0", mesh0), ("pred_p1", mesh1),
                                ("slat_gt_p0", gt_mesh0), ("slat_gt_p1", gt_mesh1)]:
                if mesh is not None:
                    path = os.path.join(args.output_dir, f"{safe}_{name}.obj")
                    mesh.export(path)
                    obj_paths[name] = path

            # Copy original GT OBJs if available
            for part, key in [("p0", "p0_gt_obj"), ("p1", "p1_gt_obj")]:
                if s[key] and os.path.exists(s[key]):
                    dst = os.path.join(args.output_dir, f"{safe}_orig_gt_{part}.obj")
                    import shutil
                    shutil.copy2(s[key], dst)
                    obj_paths[f"orig_gt_{part}"] = dst

            print(f"    Meshes: pred_p0={mesh0.vertices.shape[0] if mesh0 else 0}v, "
                  f"pred_p1={mesh1.vertices.shape[0] if mesh1 else 0}v")

            # Blender render
            if not args.no_render:
                render_paths = []
                render_labels = []

                # 1. Original GT (from precompute OBJs)
                if "orig_gt_p0" in obj_paths and "orig_gt_p1" in obj_paths:
                    png = os.path.join(args.output_dir, f"{safe}_1_orig_gt.png")
                    render_blender(obj_paths["orig_gt_p0"], obj_paths["orig_gt_p1"], png)
                    render_paths.append(png)
                    render_labels.append("Original GT")

                # 2. SLat GT roundtrip (encode→decode)
                if "slat_gt_p0" in obj_paths and "slat_gt_p1" in obj_paths:
                    png = os.path.join(args.output_dir, f"{safe}_2_slat_gt.png")
                    render_blender(obj_paths["slat_gt_p0"], obj_paths["slat_gt_p1"], png)
                    render_paths.append(png)
                    render_labels.append("SLat GT Roundtrip")

                # 3. Our prediction
                if "pred_p0" in obj_paths and "pred_p1" in obj_paths:
                    png = os.path.join(args.output_dir, f"{safe}_3_pred.png")
                    render_blender(obj_paths["pred_p0"], obj_paths["pred_p1"], png)
                    render_paths.append(png)
                    render_labels.append("Ours (Progressive)")

                # Comparison panel
                if len(render_paths) >= 2:
                    comp = os.path.join(args.output_dir, f"{safe}_compare.png")
                    make_comparison_image(render_paths, render_labels, comp)
                    print(f"    Comparison: {comp}")

        except Exception as e:
            print(f"  [{i+1}] ERROR {s['id']}: {e}")
            import traceback
            traceback.print_exc()

        torch.cuda.empty_cache()

    print(f"\nDone! Results in {args.output_dir}")


if __name__ == "__main__":
    main()
