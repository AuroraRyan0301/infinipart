#!/usr/bin/env python3
"""
Evaluation script for PartPacker video-conditioned dual volume prediction.

Standard metrics following SINGAPO / PAct / Articulate-Anything:
  - Chamfer Distance (dCD): surface-level mesh accuracy, 2048 sampled points
  - Centroid Distance (dcDist): part center distance, sensitive to small parts
  - Generalized IoU (dgIoU): bounding box level part accuracy (1 - gIoU)
  - Average Overlapping Ratio (AOR): inter-part collision/penetration
  - Volumetric IoU (vIoU): occupancy overlap between pred and GT parts
  - Connected Components (CC): mesh fragmentation (ideal = 1 per part)

Supports two test modes:
  - test_view: held-out views from training objects
  - test_obj: completely unseen (OOD) objects

Usage:
  CUDA_VISIBLE_DEVICES=0 python eval_partpacker.py \
    --ckpt /mnt/data_ssd/infinigen-sim-data/checkpoints/overfit_physxmob_v2/step_10000.pt \
    --manifest /mnt/data_ssd/infinigen-sim-data/checkpoints/overfit_physxmob_v2/manifest.json \
    --output_dir ./output/eval_overfit_v2 \
    --decode_resolution 384 \
    --num_samples 50
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

PARTPACKER_ROOT = "/mnt/cpfs/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

LATENT_SIZE = 2048
LATENT_DIM = 64
TOTAL_LATENT = LATENT_SIZE * 2
VJEPA_DIM = 1408
DIT_DIM = 1536


# ================================================================
# Metrics
# ================================================================

def chamfer_distance(pts_a, pts_b):
    """Symmetric Chamfer Distance between two point sets [N,3] and [M,3]."""
    from scipy.spatial import cKDTree
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    dist_a, _ = tree_b.query(pts_a)
    dist_b, _ = tree_a.query(pts_b)
    return (dist_a.mean() + dist_b.mean()) / 2


def centroid_distance(mesh_a, mesh_b):
    """Euclidean distance between mesh centroids."""
    c_a = mesh_a.centroid
    c_b = mesh_b.centroid
    return np.linalg.norm(c_a - c_b)


def generalized_iou_3d(box_a, box_b):
    """Generalized IoU for 3D axis-aligned bounding boxes.
    box: (min_xyz, max_xyz) each [3,].
    Returns 1 - gIoU (lower = better)."""
    min_a, max_a = box_a
    min_b, max_b = box_b

    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_vol = np.prod(np.maximum(inter_max - inter_min, 0))

    vol_a = np.prod(max_a - min_a)
    vol_b = np.prod(max_b - min_b)
    union_vol = vol_a + vol_b - inter_vol

    enclosing_min = np.minimum(min_a, min_b)
    enclosing_max = np.maximum(max_a, max_b)
    enclosing_vol = np.prod(enclosing_max - enclosing_min)

    if enclosing_vol < 1e-10:
        return 1.0

    iou = inter_vol / max(union_vol, 1e-10)
    giou = iou - (enclosing_vol - union_vol) / enclosing_vol
    return 1.0 - giou


def average_overlapping_ratio(mesh_a, mesh_b, n_samples=10000):
    """AOR: fraction of sampled points from mesh_a that are inside mesh_b.
    Measures unrealistic inter-part collision."""
    try:
        if not mesh_a.is_watertight or not mesh_b.is_watertight:
            # Fallback: use proximity instead of containment
            pts_a = mesh_a.sample(n_samples)
            dists = trimesh.proximity.signed_distance(mesh_b, pts_a)
            return float((dists > 0).sum() / len(dists))
        pts_a = mesh_a.sample(n_samples)
        inside = mesh_b.contains(pts_a)
        return float(inside.sum() / len(inside))
    except Exception:
        return -1.0


def volumetric_iou(mesh_a, mesh_b, resolution=64):
    """Volumetric IoU: voxelize both meshes and compute IoU."""
    try:
        bounds = np.array([
            np.minimum(mesh_a.bounds[0], mesh_b.bounds[0]),
            np.maximum(mesh_a.bounds[1], mesh_b.bounds[1]),
        ])
        pitch = (bounds[1] - bounds[0]).max() / resolution
        vox_a = mesh_a.voxelized(pitch)
        vox_b = mesh_b.voxelized(pitch)
        # Align to same grid
        mat_a = vox_a.matrix
        mat_b = vox_b.matrix
        # Pad to same size
        shape = np.maximum(mat_a.shape, mat_b.shape)
        a = np.zeros(shape, dtype=bool)
        b = np.zeros(shape, dtype=bool)
        a[:mat_a.shape[0], :mat_a.shape[1], :mat_a.shape[2]] = mat_a
        b[:mat_b.shape[0], :mat_b.shape[1], :mat_b.shape[2]] = mat_b
        inter = (a & b).sum()
        union = (a | b).sum()
        return float(inter / max(union, 1)) if union > 0 else 0.0
    except Exception:
        return -1.0


def mesh_connected_components(mesh):
    """Count connected components."""
    try:
        return len(mesh.split(only_watertight=False))
    except:
        return -1


def compute_all_metrics(pred_mesh, gt_mesh, n_sample_pts=2048):
    """Compute all metrics for a single part pair."""
    results = {}

    # Sample points
    try:
        pred_pts = pred_mesh.sample(n_sample_pts)
        gt_pts = gt_mesh.sample(n_sample_pts)
    except Exception:
        return {"error": "sampling_failed"}

    # Chamfer Distance
    results["dCD"] = chamfer_distance(pred_pts, gt_pts)

    # Centroid Distance
    results["dcDist"] = centroid_distance(pred_mesh, gt_mesh)

    # Generalized IoU (bbox level)
    pred_box = (pred_mesh.bounds[0], pred_mesh.bounds[1])
    gt_box = (gt_mesh.bounds[0], gt_mesh.bounds[1])
    results["dgIoU"] = generalized_iou_3d(pred_box, gt_box)

    # Volumetric IoU
    results["vIoU"] = volumetric_iou(pred_mesh, gt_mesh)

    # Connected components
    results["CC"] = mesh_connected_components(pred_mesh)
    results["CC_gt"] = mesh_connected_components(gt_mesh)

    # Watertight
    results["watertight"] = pred_mesh.is_watertight

    return results


# ================================================================
# Model loading + inference
# ================================================================

def load_dit(device, ckpt_path):
    from flow.modules.dit import DiT
    dit = DiT(
        hidden_dim=DIT_DIM, num_heads=16, num_layers=24,
        latent_size=LATENT_SIZE, latent_dim=LATENT_DIM,
        qknorm=True, qknorm_type="RMSNorm",
        use_pos_embed=False, use_parts=True, part_embed_mode="part2_only",
    ).to(device, dtype=torch.bfloat16)
    proj = torch.nn.Linear(VJEPA_DIM, DIT_DIM).to(device, dtype=torch.bfloat16)

    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    dit.load_state_dict(ckpt["dit"])
    if "proj" in ckpt:
        proj.load_state_dict(ckpt["proj"])
    step = ckpt.get("step", "?")
    print(f"Loaded DiT from {ckpt_path} (step {step})")
    dit.eval()
    proj.eval()
    return dit, proj, step


def load_vae(device):
    from vae.model import Model
    vae_ckpt = os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt")
    ckpt = torch.load(vae_ckpt, weights_only=True)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    config = importlib.import_module("vae.configs.part_woenc").make_config()
    vae = Model(config).eval().to(device, dtype=torch.bfloat16)
    vae.load_state_dict(ckpt, strict=True)
    del ckpt
    print("VAE loaded")
    return vae


def flow_inference(dit, cond, device, num_steps=50, cfg_scale=5.0):
    cond_null = torch.zeros_like(cond)
    cond_input = torch.cat([cond, cond_null], dim=0)
    x = torch.randn(1, TOTAL_LATENT, LATENT_DIM, device=device)
    sigmas = np.linspace(1, 0, num_steps + 1)
    sigmas = 3.0 * sigmas / (1 + (3.0 - 1) * sigmas)
    with torch.inference_mode():
        for i in range(num_steps):
            sigma, sigma_prev = sigmas[i], sigmas[i + 1]
            t = torch.tensor([1000 * sigma] * 2, device=device, dtype=torch.float32)
            x_in = torch.cat([x, x], dim=0).to(dtype=torch.bfloat16)
            pred = dit(x_in, cond_input, t).float()
            cond_v, uncond_v = pred.chunk(2, dim=0)
            v = uncond_v + (cond_v - uncond_v) * cfg_scale
            x = x - (sigma - sigma_prev) * v
    return x


def decode_latent(vae, latent, device, resolution=384):
    """Decode [1, 4096, 64] -> (mesh_p0, mesh_p1)."""
    lat0 = latent[:, :LATENT_SIZE, :]
    lat1 = latent[:, LATENT_SIZE:, :]
    meshes = []
    for lat in [lat0, lat1]:
        data = {"latent": lat.to(device, dtype=torch.bfloat16)}
        with torch.inference_mode():
            results = vae(data, resolution=resolution)
        if "meshes" in results and results["meshes"]:
            verts, faces = results["meshes"][0]
            # Filter NaN
            v = verts if isinstance(verts, np.ndarray) else verts.cpu().numpy() if hasattr(verts, 'cpu') else np.array(verts)
            f = faces if isinstance(faces, np.ndarray) else faces.cpu().numpy() if hasattr(faces, 'cpu') else np.array(faces)
            valid = ~np.isnan(v).any(axis=1)
            if not valid.all():
                face_valid = valid[f].all(axis=1)
                mesh = trimesh.Trimesh(v, f[face_valid], process=True)
            else:
                mesh = trimesh.Trimesh(v, f, process=False)
            meshes.append(mesh)
        else:
            meshes.append(None)
    return meshes[0], meshes[1]


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", default="./output/eval")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Max samples per test split")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--decode_resolution", type=int, default=384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save_meshes", action="store_true",
                        help="Save predicted + GT meshes as OBJ")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load manifest
    with open(args.manifest) as f:
        manifest = json.load(f)
    test_view = manifest.get("test_view", [])
    test_obj = manifest.get("test_obj", [])
    print(f"Manifest: {len(test_view)} test_view, {len(test_obj)} test_obj samples")

    # Load models
    dit, proj, train_step = load_dit(device, args.ckpt)
    vae = load_vae(device)

    # Deduplicate: pick one view per model_id
    def dedup_samples(samples, max_n):
        seen = set()
        out = []
        for s in samples:
            mid = s["model_id"]
            if mid in seen:
                continue
            seen.add(mid)
            out.append(s)
            if len(out) >= max_n:
                break
        return out

    all_results = {}

    for split_name, samples in [("test_view", test_view), ("test_obj", test_obj)]:
        if not samples:
            continue
        samples = dedup_samples(samples, args.num_samples)
        print(f"\n{'='*60}")
        print(f"Evaluating {split_name}: {len(samples)} samples")
        print(f"{'='*60}")

        split_results = []
        for i, s in enumerate(samples):
            sample_id = s["id"]
            safe_id = sample_id.replace("/", "_")
            try:
                # Load inputs
                jepa = torch.load(s["jepa_path"], weights_only=False,
                                  map_location=device).to(dtype=torch.bfloat16)
                if jepa.dim() == 2:
                    jepa = jepa.unsqueeze(0)
                gt_latent = torch.load(s["gt_path"], weights_only=False,
                                        map_location=device).float()
                if gt_latent.dim() == 2:
                    gt_latent = gt_latent.unsqueeze(0)

                # Inference
                with torch.inference_mode():
                    cond = proj(jepa)
                pred_latent = flow_inference(dit, cond, device,
                                             args.num_steps, args.cfg_scale)

                # Latent MSE
                latent_mse = F.mse_loss(pred_latent.float(), gt_latent).item()

                # Decode to mesh
                pred_p0, pred_p1 = decode_latent(vae, pred_latent, device,
                                                  args.decode_resolution)
                gt_p0, gt_p1 = decode_latent(vae, gt_latent, device,
                                              args.decode_resolution)

                if pred_p0 is None or pred_p1 is None or gt_p0 is None or gt_p1 is None:
                    print(f"  [{i+1}] {sample_id}: decode failed")
                    continue

                # Compute metrics per part
                m0 = compute_all_metrics(pred_p0, gt_p0)
                m1 = compute_all_metrics(pred_p1, gt_p1)

                # AOR: inter-part penetration
                aor_pred = average_overlapping_ratio(pred_p0, pred_p1)
                aor_gt = average_overlapping_ratio(gt_p0, gt_p1)

                result = {
                    "id": sample_id,
                    "category": s.get("category", "?"),
                    "object_name": s.get("object_name", "?"),
                    "latent_mse": latent_mse,
                    "part0": m0,
                    "part1": m1,
                    "aor_pred": aor_pred,
                    "aor_gt": aor_gt,
                    # Averages
                    "dCD_avg": (m0.get("dCD", 0) + m1.get("dCD", 0)) / 2,
                    "dcDist_avg": (m0.get("dcDist", 0) + m1.get("dcDist", 0)) / 2,
                    "dgIoU_avg": (m0.get("dgIoU", 0) + m1.get("dgIoU", 0)) / 2,
                    "vIoU_avg": (m0.get("vIoU", 0) + m1.get("vIoU", 0)) / 2,
                }
                split_results.append(result)

                print(f"  [{i+1}/{len(samples)}] {sample_id}: "
                      f"dCD={result['dCD_avg']:.4f} dcDist={result['dcDist_avg']:.4f} "
                      f"dgIoU={result['dgIoU_avg']:.4f} vIoU={result['vIoU_avg']:.4f} "
                      f"AOR={aor_pred:.4f} mse={latent_mse:.4f}")

                # Save meshes
                if args.save_meshes:
                    mesh_dir = os.path.join(args.output_dir, "meshes", safe_id)
                    os.makedirs(mesh_dir, exist_ok=True)
                    pred_p0.export(os.path.join(mesh_dir, "pred_p0.obj"))
                    pred_p1.export(os.path.join(mesh_dir, "pred_p1.obj"))
                    gt_p0.export(os.path.join(mesh_dir, "gt_p0.obj"))
                    gt_p1.export(os.path.join(mesh_dir, "gt_p1.obj"))

                del pred_latent, gt_latent, jepa, cond
                del pred_p0, pred_p1, gt_p0, gt_p1
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [{i+1}] {sample_id}: ERROR {e}")
                import traceback
                traceback.print_exc()
                continue

        all_results[split_name] = split_results

        # Summary
        if split_results:
            print(f"\n--- {split_name} Summary ({len(split_results)} samples) ---")
            for metric in ["dCD_avg", "dcDist_avg", "dgIoU_avg", "vIoU_avg",
                           "aor_pred", "latent_mse"]:
                vals = [r[metric] for r in split_results if r.get(metric, -1) >= 0]
                if vals:
                    print(f"  {metric:>12s}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # Save results
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "ckpt": args.ckpt,
            "train_step": train_step,
            "num_steps": args.num_steps,
            "cfg_scale": args.cfg_scale,
            "decode_resolution": args.decode_resolution,
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved: {results_path}")
    print(f"Meshes: {args.output_dir}/meshes/" if args.save_meshes else "")


if __name__ == "__main__":
    main()
