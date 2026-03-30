#!/usr/bin/env python3
"""
Inference for PartPacker Flow DiT: decode predicted latents to mesh,
render blue/orange comparison with GT.

Usage:
  CUDA_VISIBLE_DEVICES=0 python infer_partpacker.py \
    --ckpt /path/to/step_XXXX.pt \
    --manifest /path/to/manifest.json \
    --output_dir ./output/infer_partpacker \
    --n_train 5 --n_test_view 5 --n_test_obj 5
"""
import argparse, importlib, json, os, random, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn

PARTPACKER_ROOT = "/mnt/cpfs/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

VJEPA_DIM = 1408
DIT_DIM = 1536
SIGMA_MIN = 1e-5

# ================================================================
# Model loading
# ================================================================

def load_flow_model(ckpt_path, device):
    """Load DiT + proj from checkpoint."""
    from flow.modules.dit import DiT

    dit = DiT(
        hidden_dim=1536, num_heads=16, num_layers=24,
        latent_size=4096, latent_dim=64,
        qknorm=True, qknorm_type="RMSNorm",
        use_pos_embed=False, use_parts=True, part_embed_mode="part2_only",
    ).to(device, dtype=torch.bfloat16)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cond_dim = ckpt["proj"]["weight"].shape[1]  # infer from saved weight
    proj = nn.Linear(cond_dim, DIT_DIM).to(device).bfloat16()
    dit.load_state_dict(ckpt["dit"], strict=True)
    proj.load_state_dict(ckpt["proj"], strict=True)
    dit.eval()
    proj.eval()
    step = ckpt.get("step", 0)
    print(f"Loaded DiT+proj from {ckpt_path} (step {step})")
    return dit, proj, step


def load_vae(device):
    """Load PartPacker VAE decoder."""
    from vae.model import Model as VAEModel
    config = importlib.import_module("vae.configs.part_woenc").make_config()
    vae = VAEModel(config).to(device).bfloat16()
    vae_ckpt = torch.load(os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt"),
                           map_location=device, weights_only=True)
    if "model" in vae_ckpt:
        vae_ckpt = vae_ckpt["model"]
    vae.load_state_dict(vae_ckpt, strict=True)
    vae.eval()
    print("VAE loaded")
    return vae


# ================================================================
# Flow sampling (Euler ODE)
# ================================================================

@torch.inference_mode()
def flow_sample(dit, proj, jepa_feat, num_steps=50, cfg_scale=5.0, device="cuda"):
    """Sample from flow model with shifted sigma schedule + CFG (matches training script)."""
    cond = proj(jepa_feat.unsqueeze(0).to(device))  # [1, 10240, 1536]
    cond_null = torch.zeros_like(cond)
    cond_input = torch.cat([cond, cond_null], dim=0)  # [2, 10240, 1536]

    x = torch.randn(1, 8192, 64, device=device)
    # Shifted sigma schedule (shift=3.0)
    sigmas = np.linspace(1, 0, num_steps + 1)
    sigmas = 3.0 * sigmas / (1 + (3.0 - 1) * sigmas)
    sigmas_pair = [(sigmas[i], sigmas[i + 1]) for i in range(num_steps)]

    for sigma, sigma_prev in sigmas_pair:
        timesteps = torch.tensor([1000 * sigma] * 2, device=device, dtype=torch.float32)
        x_input = torch.cat([x, x], dim=0).to(dtype=torch.bfloat16)
        pred = dit(x_input, cond_input, timesteps).float()
        cond_v, uncond_v = pred.chunk(2, dim=0)
        pred_v = uncond_v + (cond_v - uncond_v) * cfg_scale
        x = x - (sigma - sigma_prev) * pred_v
    return x  # [1, 8192, 64]


# ================================================================
# VAE decode latent -> mesh OBJ
# ================================================================

@torch.inference_mode()
def decode_latent_to_obj(vae, latent, output_path, resolution=384):
    """Decode [1, 4096, 64] latent to OBJ mesh."""
    import trimesh
    output = vae({"latent": latent}, resolution=resolution)
    meshes = output.get("meshes", [])
    if meshes and len(meshes) > 0:
        vertices, faces = meshes[0]
        if vertices is not None and faces is not None:
            v = vertices.cpu().numpy() if torch.is_tensor(vertices) else np.asarray(vertices)
            f = faces.cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
            # Filter NaN vertices
            valid = ~np.isnan(v).any(axis=1)
            if valid.sum() < 3:
                return False
            # Remap faces to valid vertices
            new_idx = np.full(len(v), -1, dtype=int)
            new_idx[valid] = np.arange(valid.sum())
            v_clean = v[valid]
            face_valid = valid[f].all(axis=1)
            f_clean = new_idx[f[face_valid]]
            mesh = trimesh.Trimesh(v_clean, f_clean, process=False)
            mesh.export(output_path)
            return True
    return False


# ================================================================
# Blender render (same script as infer_slat_progressive.py)
# ================================================================

BLENDER = "/mnt/cpfs/yurh/blender-4.2.18-linux-x64/blender"
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


def render_comparison(pred_p0, pred_p1, gt_p0, gt_p1, output_png, title=""):
    pred_png = output_png.replace(".png", "_pred.png")
    gt_png = output_png.replace(".png", "_gt.png")

    render_blender(pred_p0, pred_p1, pred_png, f"Pred: {title}")
    render_blender(gt_p0, gt_p1, gt_png, f"GT: {title}")

    if os.path.isfile(pred_png) and os.path.isfile(gt_png):
        subprocess.run([
            "ffmpeg", "-y", "-i", pred_png, "-i", gt_png,
            "-filter_complex", "hstack=inputs=2", output_png
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(pred_png)
        os.remove(gt_png)


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", default="./output/infer_partpacker")
    parser.add_argument("--n_train", type=int, default=5)
    parser.add_argument("--n_test_view", type=int, default=5)
    parser.add_argument("--n_test_obj", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    dit, proj, step = load_flow_model(args.ckpt, device)
    vae = load_vae(device)

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Select samples from each split (one per unique animode)
    def pick_samples(samples, n):
        """Pick n unique animodes, one view each."""
        by_animode = {}
        for s in samples:
            mid = s["model_id"]
            if mid not in by_animode:
                by_animode[mid] = s
        pool = list(by_animode.values())
        random.shuffle(pool)
        return pool[:n]

    selections = {
        "train": pick_samples(manifest["train"], args.n_train),
        "test_view": pick_samples(manifest["test_view"], args.n_test_view),
        "test_obj": pick_samples(manifest["test_obj"], args.n_test_obj),
    }

    # Precompute solidified dir for GT mesh
    SOLIDIFIED = "/mnt/data_ssd/infinigen-sim-data/precompute_solidified"

    for split_name, samples in selections.items():
        split_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        print(f"\n=== {split_name} ({len(samples)} samples) ===")

        for i, s in enumerate(samples):
            model_id = s["model_id"]
            cat = s.get("category", "?")
            obj_name = s.get("object_name", "?")
            seed_id = s.get("seed", model_id.split("_")[0])
            animode = "_".join(model_id.split("_")[1:])
            tag = f"{cat}_{model_id}"

            print(f"  [{i+1}/{len(samples)}] {tag} ({obj_name})", flush=True)

            # Load JEPA features
            jepa = torch.load(s["jepa_path"], map_location=device, weights_only=False)
            if jepa.dim() == 2:
                jepa = jepa  # [10240, 1408]

            # Load GT latent
            gt_latent = torch.load(s["gt_path"], map_location=device, weights_only=False)
            if gt_latent.dim() == 2:
                gt_latent = gt_latent.unsqueeze(0)  # [1, 8192, 64]

            # Flow sample
            t0 = time.time()
            pred_latent = flow_sample(dit, proj, jepa,
                                      num_steps=args.num_steps, device=device)
            print(f"    Flow: {time.time()-t0:.1f}s", flush=True)

            # Decode pred
            pred_p0_path = os.path.join(split_dir, f"{tag}_pred_p0.obj")
            pred_p1_path = os.path.join(split_dir, f"{tag}_pred_p1.obj")
            pred_p0_latent = pred_latent[:, :4096, :]
            pred_p1_latent = pred_latent[:, 4096:, :]

            t0 = time.time()
            decode_latent_to_obj(vae, pred_p0_latent, pred_p0_path, args.resolution)
            decode_latent_to_obj(vae, pred_p1_latent, pred_p1_path, args.resolution)
            print(f"    Decode: {time.time()-t0:.1f}s", flush=True)

            # GT mesh paths (from solidified precompute)
            gt_p0 = os.path.join(SOLIDIFIED, "PhysXMobility", seed_id, animode, "part0.obj")
            gt_p1 = os.path.join(SOLIDIFIED, "PhysXMobility", seed_id, animode, "part1.obj")

            # Render comparison
            compare_png = os.path.join(split_dir, f"{tag}_compare.png")
            render_comparison(pred_p0_path, pred_p1_path, gt_p0, gt_p1, compare_png, title=tag)
            print(f"    Render: {compare_png}", flush=True)

            torch.cuda.empty_cache()

    print(f"\nDone! Results: {args.output_dir}")


if __name__ == "__main__":
    main()
