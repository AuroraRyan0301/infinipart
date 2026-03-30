#!/usr/bin/env python3
"""
Evaluate Ours (JEPA-conditioned PartPacker) vs Baseline (DINOv2 single-image PartPacker).

Ours: step_20000 checkpoint, JEPA video features as condition
Baseline: pretrained PartPacker, 4 frames from video → DINOv2 → pick best/avg
GT: solidified precompute meshes

Metrics: Chamfer Distance (dCD), Volume IoU (vIoU), centroid distance (dcDist)
"""
import argparse, importlib, json, os, sys, time, random
import numpy as np
import torch
import torch.nn as nn
import subprocess
import trimesh

PARTPACKER_ROOT = "/mnt/cpfs/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

VJEPA_DIM = 1408
DIT_DIM = 1536
SOLIDIFIED = "/mnt/data_ssd/infinigen-sim-data/precompute_solidified"
BLENDER = "/mnt/cpfs/yurh/blender-4.2.18-linux-x64/blender"

# ================================================================
# Metrics
# ================================================================

from scipy.spatial import cKDTree


def chamfer_distance(pts_a, pts_b):
    """Symmetric Chamfer Distance (2048 pts, SINGAPO/PAct standard)."""
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    d_a, _ = tree_b.query(pts_a)
    d_b, _ = tree_a.query(pts_b)
    return (d_a.mean() + d_b.mean()) / 2


def centroid_distance(mesh_a, mesh_b):
    return np.linalg.norm(mesh_a.centroid - mesh_b.centroid)


def generalized_iou_3d(box_a, box_b):
    """1 - gIoU for 3D AABB (lower = better)."""
    min_a, max_a = box_a
    min_b, max_b = box_b
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_vol = np.prod(np.maximum(inter_max - inter_min, 0))
    vol_a = np.prod(max_a - min_a)
    vol_b = np.prod(max_b - min_b)
    union_vol = vol_a + vol_b - inter_vol
    enc_min = np.minimum(min_a, min_b)
    enc_max = np.maximum(max_a, max_b)
    enc_vol = np.prod(enc_max - enc_min)
    if enc_vol < 1e-10:
        return 1.0
    iou = inter_vol / max(union_vol, 1e-10)
    giou = iou - (enc_vol - union_vol) / enc_vol
    return 1.0 - giou


def volumetric_iou(mesh_a, mesh_b, resolution=64):
    try:
        bounds = np.array([
            np.minimum(mesh_a.bounds[0], mesh_b.bounds[0]),
            np.maximum(mesh_a.bounds[1], mesh_b.bounds[1]),
        ])
        pitch = (bounds[1] - bounds[0]).max() / resolution
        vox_a = mesh_a.voxelized(pitch)
        vox_b = mesh_b.voxelized(pitch)
        mat_a, mat_b = vox_a.matrix, vox_b.matrix
        shape = np.maximum(mat_a.shape, mat_b.shape)
        a = np.zeros(shape, dtype=bool)
        b = np.zeros(shape, dtype=bool)
        a[:mat_a.shape[0], :mat_a.shape[1], :mat_a.shape[2]] = mat_a
        b[:mat_b.shape[0], :mat_b.shape[1], :mat_b.shape[2]] = mat_b
        inter = (a & b).sum()
        union = (a | b).sum()
        return float(inter / max(union, 1))
    except:
        return -1.0


def average_overlapping_ratio(mesh_a, mesh_b, n_samples=10000):
    """AOR: fraction of points from mesh_a inside mesh_b (inter-part collision)."""
    try:
        pts_a = mesh_a.sample(n_samples)
        if mesh_b.is_watertight:
            inside = mesh_b.contains(pts_a)
            return float(inside.sum() / len(inside))
        else:
            dists = trimesh.proximity.signed_distance(mesh_b, pts_a)
            return float((dists > 0).sum() / len(dists))
    except:
        return -1.0


def mesh_connected_components(mesh):
    try:
        return len(mesh.split(only_watertight=False))
    except:
        return -1


def compute_metrics(pred_p0_path, pred_p1_path, gt_p0_path, gt_p1_path, gt_latent=None, pred_latent=None):
    """Compute all 7 metrics for a sample."""
    results = {}
    pred_meshes = {}
    gt_meshes = {}

    for part in ["p0", "p1"]:
        pred_path = pred_p0_path if part == "p0" else pred_p1_path
        gt_path = gt_p0_path if part == "p0" else gt_p1_path
        if not os.path.isfile(pred_path) or not os.path.isfile(gt_path):
            for m in ["dCD", "dcDist", "dgIoU", "vIoU", "CC"]:
                results[f"{part}_{m}"] = float("nan")
            continue
        try:
            pred = trimesh.load(pred_path, process=True, force="mesh")
            gt = trimesh.load(gt_path, process=True, force="mesh")
            if len(pred.faces) < 3 or len(gt.faces) < 3:
                for m in ["dCD", "dcDist", "dgIoU", "vIoU", "CC"]:
                    results[f"{part}_{m}"] = float("nan")
                continue
            pred_meshes[part] = pred
            gt_meshes[part] = gt
            # dCD
            results[f"{part}_dCD"] = chamfer_distance(pred.sample(2048), gt.sample(2048))
            # dcDist
            results[f"{part}_dcDist"] = centroid_distance(pred, gt)
            # dgIoU
            results[f"{part}_dgIoU"] = generalized_iou_3d(
                (pred.bounds[0], pred.bounds[1]), (gt.bounds[0], gt.bounds[1]))
            # vIoU
            results[f"{part}_vIoU"] = volumetric_iou(pred, gt)
            # CC
            results[f"{part}_CC"] = mesh_connected_components(pred)
        except Exception as e:
            for m in ["dCD", "dcDist", "dgIoU", "vIoU", "CC"]:
                results[f"{part}_{m}"] = float("nan")

    # AOR (inter-part collision)
    if "p0" in pred_meshes and "p1" in pred_meshes:
        results["AOR"] = average_overlapping_ratio(pred_meshes["p0"], pred_meshes["p1"])
    else:
        results["AOR"] = float("nan")

    # latent MSE
    if gt_latent is not None and pred_latent is not None:
        results["latent_mse"] = float(torch.nn.functional.mse_loss(
            pred_latent.float(), gt_latent.float()).item())
    else:
        results["latent_mse"] = float("nan")

    # Averages
    for m in ["dCD", "dcDist", "dgIoU", "vIoU", "CC"]:
        vals = [results.get(f"p0_{m}", float("nan")), results.get(f"p1_{m}", float("nan"))]
        vals = [v for v in vals if not np.isnan(v)]
        results[f"avg_{m}"] = np.mean(vals) if vals else float("nan")

    return results


# ================================================================
# Model loading
# ================================================================

def load_ours(ckpt_path, device):
    """Load our JEPA-conditioned DiT + proj."""
    from flow.modules.dit import DiT
    dit = DiT(
        hidden_dim=1536, num_heads=16, num_layers=24,
        latent_size=4096, latent_dim=64,
        qknorm=True, qknorm_type="RMSNorm",
        use_pos_embed=False, use_parts=True, part_embed_mode="part2_only",
    ).to(device, dtype=torch.bfloat16)
    proj = nn.Linear(VJEPA_DIM, DIT_DIM).to(device).bfloat16()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dit.load_state_dict(ckpt["dit"], strict=True)
    proj.load_state_dict(ckpt["proj"], strict=True)
    dit.eval(); proj.eval()
    return dit, proj


def load_baseline(device):
    """Load original PartPacker (DINOv2-conditioned)."""
    from flow.model import Model as FlowModel, ModelConfig
    config = ModelConfig(
        vae_conf="vae.configs.part_woenc",
        vae_ckpt_path=os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt"),
        qknorm=True, qknorm_type="RMSNorm",
        use_pos_embed=False, dino_model="dinov2_vitg14",
        hidden_dim=1536, flow_shift=3.0,
        logitnorm_mean=1.0, logitnorm_std=1.0,
        latent_size=4096, use_parts=True, part_embed_mode="part2_only",
    )
    model = FlowModel(config).to(device).bfloat16()
    flow_ckpt = torch.load(os.path.join(PARTPACKER_ROOT, "pretrained", "flow.pt"),
                            map_location=device, weights_only=False)
    if "model" in flow_ckpt:
        flow_ckpt = flow_ckpt["model"]
    model.load_state_dict(flow_ckpt, strict=False)
    model.eval()
    return model


def load_vae(device):
    from vae.model import Model as VAEModel
    config = importlib.import_module("vae.configs.part_woenc").make_config()
    vae = VAEModel(config).to(device).bfloat16()
    vae_ckpt = torch.load(os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt"),
                           map_location=device, weights_only=True)
    if "model" in vae_ckpt: vae_ckpt = vae_ckpt["model"]
    vae.load_state_dict(vae_ckpt, strict=True)
    vae.eval()
    return vae


# ================================================================
# Flow sampling
# ================================================================

@torch.inference_mode()
def sample_ours(dit, proj, jepa_feat, num_steps=50, cfg_scale=5.0, device="cuda"):
    cond = proj(jepa_feat.unsqueeze(0).to(device))
    cond_null = torch.zeros_like(cond)
    cond_input = torch.cat([cond, cond_null], dim=0)
    x = torch.randn(1, 8192, 64, device=device)
    sigmas = np.linspace(1, 0, num_steps + 1)
    sigmas = 3.0 * sigmas / (1 + (3.0 - 1) * sigmas)
    for i in range(num_steps):
        s, sp = sigmas[i], sigmas[i+1]
        t = torch.tensor([1000*s]*2, device=device, dtype=torch.float32)
        xi = torch.cat([x, x], dim=0).to(dtype=torch.bfloat16)
        pred = dit(xi, cond_input, t).float()
        cv, uv = pred.chunk(2, dim=0)
        pv = uv + (cv - uv) * cfg_scale
        x = x - (s - sp) * pv
    return x


@torch.inference_mode()
def sample_baseline(model, image_tensor, num_steps=50, cfg_scale=7.0):
    data = {"cond_images": image_tensor}
    results = model(data, num_steps=num_steps, cfg_scale=cfg_scale)
    return results["latent"]


# ================================================================
# Decode + export
# ================================================================

@torch.inference_mode()
def decode_to_obj(vae, latent, output_path, resolution=384):
    output = vae({"latent": latent}, resolution=resolution)
    meshes = output.get("meshes", [])
    if meshes:
        vertices, faces = meshes[0]
        v = vertices.cpu().numpy() if torch.is_tensor(vertices) else np.asarray(vertices)
        f = faces.cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
        valid = ~np.isnan(v).any(axis=1)
        if valid.sum() < 3: return False
        new_idx = np.full(len(v), -1, dtype=int)
        new_idx[valid] = np.arange(valid.sum())
        mesh = trimesh.Trimesh(v[valid], new_idx[f[valid[f].all(axis=1)]], process=False)
        mesh.export(output_path)
        return True
    return False


# ================================================================
# Blender render (blue/orange comparison)
# ================================================================

BLENDER_SCRIPT = '''
import bpy, sys, os, math, mathutils
argv = sys.argv[sys.argv.index("--") + 1:]
p0_obj, p1_obj, output_png = argv[0], argv[1], argv[2]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
for c in bpy.data.collections: bpy.data.collections.remove(c)
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
    if obj.data.materials: obj.data.materials[0] = mat
    else: obj.data.materials.append(mat)
    return obj
o0 = import_obj(p0_obj, (0.15, 0.35, 0.85, 1.0), "part0")
o1 = import_obj(p1_obj, (0.95, 0.55, 0.1, 1.0), "part1")
all_verts = []
for obj in [o0, o1]:
    for v in obj.data.vertices: all_verts.append(obj.matrix_world @ v.co)
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
scene.render.filepath = output_png
scene.render.image_settings.file_format = 'PNG'
bpy.ops.render.render(write_still=True)
'''


def render_blender(p0, p1, output_png):
    script = "/tmp/eval_render.py"
    with open(script, "w") as f:
        f.write(BLENDER_SCRIPT)
    subprocess.run([BLENDER, "--background", "--python", script, "--", p0, p1, output_png],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)


def render_3way(ours_p0, ours_p1, base_p0, base_p1, gt_p0, gt_p1, output_png, label=""):
    """Render 3-way comparison: Ours | Baseline | GT"""
    tmp_ours = output_png.replace(".png", "_ours.png")
    tmp_base = output_png.replace(".png", "_base.png")
    tmp_gt = output_png.replace(".png", "_gt.png")
    render_blender(ours_p0, ours_p1, tmp_ours)
    render_blender(base_p0, base_p1, tmp_base)
    render_blender(gt_p0, gt_p1, tmp_gt)
    # hstack 3
    if all(os.path.isfile(f) for f in [tmp_ours, tmp_base, tmp_gt]):
        subprocess.run([
            "ffmpeg", "-y", "-i", tmp_ours, "-i", tmp_base, "-i", tmp_gt,
            "-filter_complex", "hstack=inputs=3", output_png
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for f in [tmp_ours, tmp_base, tmp_gt]:
            os.remove(f)


# ================================================================
# Extract frames from video for baseline
# ================================================================

def extract_frames(video_path, n_frames=4):
    """Extract n equally-spaced frames from video, return as [N, 3, 518, 518] tensor."""
    from decord import VideoReader, cpu
    import torch.nn.functional as F
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # [N, H, W, 3]
    # Convert to RGBA (white bg, add alpha=1)
    processed = []
    for frame in frames:
        img = frame.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        img = F.interpolate(img, size=(518, 518), mode="bilinear", align_corners=False)
        processed.append(img.squeeze(0))
    return torch.stack(processed)  # [N, 3, 518, 518]


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Ours checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", default="./output/eval_ours_vs_baseline")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_filter", type=str, default=None, help="Only run this split (train/test_view/test_obj)")
    parser.add_argument("--no_render", action="store_true", help="Skip Blender rendering")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--n_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--all", action="store_true", help="Run ALL animodes, no sampling")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device("cuda")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("Loading models...", flush=True)
    dit, proj = load_ours(args.ckpt, device)
    vae = load_vae(device)
    print("Loading baseline PartPacker...", flush=True)
    baseline_model = load_baseline(device)
    print("All models loaded.", flush=True)

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Pick samples (one per unique animode)
    def pick_unique(samples, n):
        by_anim = {}
        for s in samples:
            mid = s["model_id"]
            if mid not in by_anim:
                by_anim[mid] = s
        pool = list(by_anim.values())
        random.shuffle(pool)
        return pool[:n]

    PRECOMPUTE = "/mnt/data_ssd/infinigen-sim-data/precompute/PhysXMobility"
    JEPA_ROOT = "/mnt/data_ssd/infinigen-sim-data/encoded/PhysXMobility"

    all_results = {"ours": [], "baseline": []}

    splits = ["train", "test_view", "test_obj"]
    if args.split_filter:
        splits = [args.split_filter]
    for split_name in splits:
        samples = manifest.get(split_name, [])
        if not samples:
            continue
        if args.all:
            selected = pick_unique(samples, 99999)  # all unique animodes
        else:
            selected = pick_unique(samples, args.n_samples)
        # Shard
        if args.n_shards > 1:
            selected = [s for i, s in enumerate(selected) if i % args.n_shards == args.shard]
        split_dir = os.path.join(args.output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        print(f"\n=== {split_name} ({len(selected)} samples) ===", flush=True)

        for i, s in enumerate(selected):
            model_id = s["model_id"]
            seed_id = s.get("seed", model_id.split("_")[0])
            animode = "_".join(model_id.split("_")[1:])
            cat = s.get("category", "?")
            obj_name = s.get("object_name", "?")
            tag = f"{cat}_{model_id}"
            print(f"  [{i+1}/{len(selected)}] {tag} ({obj_name})", flush=True)

            # GT paths
            gt_p0 = os.path.join(SOLIDIFIED, "PhysXMobility", seed_id, animode, "part0.obj")
            gt_p1 = os.path.join(SOLIDIFIED, "PhysXMobility", seed_id, animode, "part1.obj")
            if not os.path.isfile(gt_p0) or not os.path.isfile(gt_p1):
                print(f"    GT missing, skip", flush=True)
                continue

            # === OURS ===
            jepa = torch.load(s["jepa_path"], map_location=device, weights_only=False)
            t0 = time.time()
            pred_latent = sample_ours(dit, proj, jepa, num_steps=args.num_steps, device=device)
            ours_p0 = os.path.join(split_dir, f"{tag}_ours_p0.obj")
            ours_p1 = os.path.join(split_dir, f"{tag}_ours_p1.obj")
            decode_to_obj(vae, pred_latent[:, :4096, :], ours_p0, args.resolution)
            decode_to_obj(vae, pred_latent[:, 4096:, :], ours_p1, args.resolution)
            print(f"    Ours: {time.time()-t0:.1f}s", flush=True)

            # === BASELINE (4 frames from video) ===
            # Find a nobg video for this animode
            anim_dir = os.path.join(PRECOMPUTE, seed_id, animode)
            videos = sorted([f for f in os.listdir(anim_dir) if f.endswith("_nobg.mp4")])
            base_p0 = os.path.join(split_dir, f"{tag}_base_p0.obj")
            base_p1 = os.path.join(split_dir, f"{tag}_base_p1.obj")
            if videos:
                vid_path = os.path.join(anim_dir, videos[0])
                frames = extract_frames(vid_path, n_frames=4)  # [4, 3, 518, 518]
                # Use middle frame as single-image condition
                mid_frame = frames[len(frames)//2].unsqueeze(0).to(device)  # [1, 3, 518, 518]
                t0 = time.time()
                base_latent = sample_baseline(baseline_model, mid_frame,
                                              num_steps=args.num_steps, cfg_scale=7.0)
                decode_to_obj(vae, base_latent[:, :4096, :], base_p0, args.resolution)
                decode_to_obj(vae, base_latent[:, 4096:, :], base_p1, args.resolution)
                print(f"    Baseline: {time.time()-t0:.1f}s", flush=True)
            else:
                print(f"    No video for baseline, skip", flush=True)
                continue

            # === Metrics ===
            gt_latent = torch.load(s["gt_path"], map_location="cpu", weights_only=False)
            if gt_latent.dim() == 2:
                gt_latent = gt_latent.unsqueeze(0)
            ours_metrics = compute_metrics(ours_p0, ours_p1, gt_p0, gt_p1,
                                           gt_latent=gt_latent, pred_latent=pred_latent.cpu())
            base_metrics = compute_metrics(base_p0, base_p1, gt_p0, gt_p1,
                                           gt_latent=gt_latent, pred_latent=base_latent.cpu())
            ours_metrics["split"] = split_name
            ours_metrics["tag"] = tag
            base_metrics["split"] = split_name
            base_metrics["tag"] = tag
            all_results["ours"].append(ours_metrics)
            all_results["baseline"].append(base_metrics)

            print(f"    Ours   dCD={ours_metrics['avg_dCD']:.4f} dcDist={ours_metrics['avg_dcDist']:.4f} dgIoU={ours_metrics['avg_dgIoU']:.4f} vIoU={ours_metrics['avg_vIoU']:.4f} AOR={ours_metrics['AOR']:.4f} CC={ours_metrics['avg_CC']:.0f} latMSE={ours_metrics['latent_mse']:.4f}")
            print(f"    Base   dCD={base_metrics['avg_dCD']:.4f} dcDist={base_metrics['avg_dcDist']:.4f} dgIoU={base_metrics['avg_dgIoU']:.4f} vIoU={base_metrics['avg_vIoU']:.4f} AOR={base_metrics['AOR']:.4f} CC={base_metrics['avg_CC']:.0f} latMSE={base_metrics['latent_mse']:.4f}")

            # === Render 3-way comparison ===
            if not args.no_render:
                compare_png = os.path.join(split_dir, f"{tag}_compare.png")
                render_3way(ours_p0, ours_p1, base_p0, base_p1, gt_p0, gt_p1, compare_png, tag)
                print(f"    Render: {compare_png}", flush=True)

            torch.cuda.empty_cache()

    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY (Ours vs Baseline)")
    print("=" * 80)
    metrics_list = ["dCD", "dcDist", "dgIoU", "vIoU", "AOR", "CC", "latent_mse"]
    header = f"{'Method':>10s} | {'Split':>10s} | " + " | ".join(f"{m:>10s}" for m in metrics_list) + " | n"
    print(header)
    print("-" * len(header))
    for method in ["ours", "baseline"]:
        res = all_results[method]
        if not res:
            continue
        for split in ["train", "test_view", "test_obj"]:
            split_res = [r for r in res if r["split"] == split]
            if not split_res:
                continue
            vals = {}
            for m in metrics_list:
                key = f"avg_{m}" if m not in ["AOR", "latent_mse"] else m
                v = [r[key] for r in split_res if not np.isnan(r.get(key, float("nan")))]
                vals[m] = np.mean(v) if v else float("nan")
            row = f"{method:>10s} | {split:>10s} | " + " | ".join(f"{vals[m]:10.4f}" for m in metrics_list) + f" | {len(split_res)}"
            print(row)

    # Save results
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None if np.isnan(x) else x)
    print(f"\nResults: {results_path}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
