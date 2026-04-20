#!/usr/bin/env python3
"""
Quick inference: pick diverse samples, flow sample, decode, render comparison with GT.
No manifest needed — discovers from encoded_solidified + diff_jepa_filtered.
"""
import argparse, glob, importlib, os, random, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn

PARTPACKER_ROOT = "/mnt/cpfs/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

BLENDER = "/mnt/cpfs/yurh/blender-4.2.18-linux-x64/blender"
DIT_DIM = 1536


def load_flow_model(ckpt_path, device):
    from flow.modules.dit import DiT
    dit = DiT(
        hidden_dim=1536, num_heads=16, num_layers=24,
        latent_size=4096, latent_dim=64,
        qknorm=True, qknorm_type="RMSNorm",
        use_pos_embed=False, use_parts=True, part_embed_mode="part2_only",
    ).to(device, dtype=torch.bfloat16)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Detect projector type from state_dict keys
    proj_keys = list(ckpt["proj"].keys())
    is_selfattn = any("layers." in k or "mlp.0.weight" == k for k in proj_keys)

    if is_selfattn:
        # Import SelfAttnProjector from training script
        sys.path.insert(0, PARTPACKER_ROOT)
        from train_partnet_vjepa_ddp import SelfAttnProjector
        cond_dim = ckpt["proj"]["mlp.0.weight"].shape[1]
        proj = SelfAttnProjector(cond_dim, DIT_DIM, n_layers=2, n_heads=8).to(device).bfloat16()
        print(f"[Flow] Projector: SelfAttnProjector (cond_dim={cond_dim})")
    else:
        cond_dim = ckpt["proj"]["weight"].shape[1]
        proj = nn.Linear(cond_dim, DIT_DIM).to(device).bfloat16()
        print(f"[Flow] Projector: Linear (cond_dim={cond_dim})")

    dit.load_state_dict(ckpt["dit"], strict=True)
    proj.load_state_dict(ckpt["proj"], strict=True)
    dit.eval()
    proj.eval()
    step = ckpt.get("step", 0)
    del ckpt
    print(f"[Flow] Loaded step {step} from {ckpt_path}")
    return dit, proj, step


def load_vae(device):
    from vae.model import Model as VAEModel
    config = importlib.import_module("vae.configs.part_woenc").make_config()
    vae = VAEModel(config).to(device).bfloat16()
    vae_ckpt = torch.load(os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt"),
                           map_location=device, weights_only=True)
    if "model" in vae_ckpt:
        vae_ckpt = vae_ckpt["model"]
    vae.load_state_dict(vae_ckpt, strict=True)
    vae.eval()
    print("[VAE] Loaded")
    return vae


@torch.inference_mode()
def flow_sample(dit, proj, jepa_feat, num_steps=50, cfg_scale=5.0, device="cuda"):
    cond = proj(jepa_feat.unsqueeze(0).to(device))
    cond_null = torch.zeros_like(cond)
    cond_input = torch.cat([cond, cond_null], dim=0)
    x = torch.randn(1, 8192, 64, device=device)
    sigmas = np.linspace(1, 0, num_steps + 1)
    sigmas = 3.0 * sigmas / (1 + (3.0 - 1) * sigmas)
    for i in range(num_steps):
        sigma, sigma_prev = sigmas[i], sigmas[i + 1]
        timesteps = torch.tensor([1000 * sigma] * 2, device=device, dtype=torch.float32)
        x_input = torch.cat([x, x], dim=0).to(dtype=torch.bfloat16)
        pred = dit(x_input, cond_input, timesteps).float()
        cond_v, uncond_v = pred.chunk(2, dim=0)
        pred_v = uncond_v + (cond_v - uncond_v) * cfg_scale
        x = x - (sigma - sigma_prev) * pred_v
    return x


@torch.inference_mode()
def decode_latent_to_obj(vae, latent, output_path, resolution=384):
    import trimesh
    output = vae({"latent": latent}, resolution=resolution)
    meshes = output.get("meshes", [])
    if meshes and len(meshes) > 0:
        vertices, faces = meshes[0]
        if vertices is not None and faces is not None:
            v = vertices.cpu().numpy() if torch.is_tensor(vertices) else np.asarray(vertices)
            f = faces.cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
            valid = ~np.isnan(v).any(axis=1)
            if valid.sum() < 3:
                return False
            new_idx = np.full(len(v), -1, dtype=int)
            new_idx[valid] = np.arange(valid.sum())
            v_clean = v[valid]
            # GLB coords -> OBJ Y-up
            TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
            v_clean = v_clean @ TRIMESH_GLB_EXPORT.T
            face_valid = valid[f].all(axis=1)
            f_clean = new_idx[f[face_valid]]
            mesh = trimesh.Trimesh(v_clean, f_clean, process=False)
            mesh.export(output_path)
            return True
    return False


BLENDER_RENDER_SCRIPT = '''
import bpy, sys, os, math, mathutils

argv = sys.argv[sys.argv.index("--") + 1:]
p0_obj, p1_obj, output_png = argv[0], argv[1], argv[2]
title = argv[3] if len(argv) > 3 else ""

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


def render_blender(p0_obj, p1_obj, output_png, title=""):
    script_path = "/tmp/pp_render.py"
    with open(script_path, "w") as f:
        f.write(BLENDER_RENDER_SCRIPT)
    cmd = [BLENDER, "--background", "--python", script_path, "--",
           p0_obj, p1_obj, output_png, title]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)


def discover_samples(encoded_root, jepa_root, precompute_root):
    """Find samples with gt_latent + diff_jepa + GT mesh."""
    samples = []
    for cat in os.listdir(encoded_root):
        cat_dir = os.path.join(encoded_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for mid in os.listdir(cat_dir):
            gt = os.path.join(cat_dir, mid, "gt_latent.pt")
            jepa_views = sorted(glob.glob(os.path.join(jepa_root, cat, mid, "views", "*_nobg_jepa.pt")))
            if not os.path.isfile(gt) or not jepa_views:
                continue
            parts = mid.split("_", 1)
            if len(parts) != 2:
                continue
            seed, animode = parts
            gt_p0 = os.path.join(precompute_root, cat, seed, animode, "part0.obj")
            gt_p1 = os.path.join(precompute_root, cat, seed, animode, "part1.obj")
            if os.path.isfile(gt_p0) and os.path.isfile(gt_p1):
                samples.append({
                    "cat": cat, "mid": mid, "seed": seed, "animode": animode,
                    "gt_latent": gt, "jepa": jepa_views[0],
                    "gt_p0": gt_p0, "gt_p1": gt_p1,
                })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--encoded_root", default="/mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_solidified")
    parser.add_argument("--jepa_root", default="/mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_diff_jepa_filtered")
    parser.add_argument("--precompute_root", default="/mnt/data/yurh/Infinigen-Sim/data_ssd/precompute_solidified")
    parser.add_argument("--output_dir", default="./output/infer_quick")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--no_render", action="store_true", help="Skip Blender rendering")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda")

    print("Discovering samples...")
    all_samples = discover_samples(args.encoded_root, args.jepa_root, args.precompute_root)
    print(f"Found {len(all_samples)} matched samples")

    # Pick diverse (one per category)
    by_cat = {}
    for s in all_samples:
        by_cat.setdefault(s["cat"], []).append(s)
    picked = []
    for cat in sorted(by_cat):
        picked.append(random.choice(by_cat[cat]))
    random.shuffle(picked)
    # Fill remaining from PhysXMobility (largest pool)
    if len(picked) < args.n_samples and "PhysXMobility" in by_cat:
        extra = [s for s in by_cat["PhysXMobility"] if s not in picked]
        random.shuffle(extra)
        picked.extend(extra[:args.n_samples - len(picked)])
    picked = picked[:args.n_samples]

    dit, proj, step = load_flow_model(args.ckpt, device)
    vae = load_vae(device)

    out_dir = os.path.join(args.output_dir, f"step_{step}")
    os.makedirs(out_dir, exist_ok=True)

    for i, s in enumerate(picked):
        tag = f"{s['cat']}_{s['mid']}"
        print(f"\n[{i+1}/{len(picked)}] {tag}")

        jepa = torch.load(s["jepa"], map_location=device, weights_only=False).to(torch.bfloat16)

        t0 = time.time()
        pred_latent = flow_sample(dit, proj, jepa,
                                   num_steps=args.num_steps, cfg_scale=args.cfg_scale,
                                   device=device)
        print(f"  Flow: {time.time()-t0:.1f}s")

        pred_p0 = os.path.join(out_dir, f"{tag}_pred_p0.obj")
        pred_p1 = os.path.join(out_dir, f"{tag}_pred_p1.obj")

        t0 = time.time()
        ok0 = decode_latent_to_obj(vae, pred_latent[:, :4096, :], pred_p0, args.resolution)
        ok1 = decode_latent_to_obj(vae, pred_latent[:, 4096:, :], pred_p1, args.resolution)
        print(f"  Decode: {time.time()-t0:.1f}s (p0={'OK' if ok0 else 'FAIL'}, p1={'OK' if ok1 else 'FAIL'})")

        if not args.no_render and ok0 and ok1:
            # Render pred
            pred_png = os.path.join(out_dir, f"{tag}_pred.png")
            render_blender(pred_p0, pred_p1, pred_png, f"Pred {tag}")
            # Render GT
            gt_png = os.path.join(out_dir, f"{tag}_gt.png")
            render_blender(s["gt_p0"], s["gt_p1"], gt_png, f"GT {tag}")
            # Hstack comparison
            compare_png = os.path.join(out_dir, f"{tag}_compare.png")
            if os.path.isfile(pred_png) and os.path.isfile(gt_png):
                subprocess.run([
                    "ffmpeg", "-y", "-i", pred_png, "-i", gt_png,
                    "-filter_complex", "hstack=inputs=2", compare_png
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  Rendered: {compare_png}")

        torch.cuda.empty_cache()

    print(f"\nDone! Results in: {out_dir}")


if __name__ == "__main__":
    main()
