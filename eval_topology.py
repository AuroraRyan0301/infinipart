#!/usr/bin/env python3
"""
Topology evaluation: measure connected component correctness of predicted meshes.

Tests two models:
  1. Our video model (VJEPA2 -> DiT, step_22000 checkpoint)
  2. Original PartPacker (DINOv2 -> DiT, pretrained)

For each sample:
  - Decodes predicted latent via VAE -> mesh -> counts connected components
  - Loads raw OBJ from precompute as ground truth topology (no VAE roundtrip)
  - Also decodes GT latent via VAE for VAE-roundtrip CC reference
  - Evaluates with part0/part1 swapped (due to assignment symmetry)

Three CC reference levels:
  - obj_cc: raw part0.obj/part1.obj from precompute (true topology)
  - gt_vae_cc: GT latent -> VAE decode (VAE roundtrip noise)
  - pred_cc: model prediction -> VAE decode

Usage:
  CUDA_VISIBLE_DEVICES=2 python eval_topology.py \
    --mode ours --ckpt /mnt/data_ssd/infinigen-sim/train_output/step_22000.pt \
    --output eval_topo_ours.json

  CUDA_VISIBLE_DEVICES=3 python eval_topology.py \
    --mode original --output eval_topo_original.json
"""

import argparse
import glob
import importlib
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

PARTPACKER_ROOT = "/mnt/data/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

LATENT_SIZE = 4096
LATENT_DIM = 64
TOTAL_LATENT = LATENT_SIZE * 2  # 8192
VJEPA_DIM = 1408
DIT_DIM = 1536
DATA_ROOT = "/mnt/data_ssd/infinigen-sim"

# ================================================================
# Data sampling
# ================================================================

def discover_eval_samples(data_root, max_animodes=200, views_per_animode=3,
                          seed=42):
    """Sample animodes spread across objects, pick 3 views each."""
    rng = random.Random(seed)

    # Group animodes by object (category/seed)
    obj_animodes = {}  # (cat, seed) -> list of animode_dirs
    for cat_name in sorted(os.listdir(data_root)):
        cat_dir = os.path.join(data_root, cat_name)
        if not os.path.isdir(cat_dir) or cat_name.startswith("."):
            continue
        if cat_name in ("train_output", "train_output_v2", ".errors"):
            continue
        for model_id in sorted(os.listdir(cat_dir)):
            model_dir = os.path.join(cat_dir, model_id)
            if not os.path.isdir(model_dir):
                continue
            gt_path = os.path.join(model_dir, "gt_latent.pt")
            if not os.path.exists(gt_path):
                continue
            views_dir = os.path.join(model_dir, "views")
            if not os.path.isdir(views_dir):
                continue
            jepa_files = sorted(glob.glob(
                os.path.join(views_dir, "v*_nobg_jepa.pt")))
            if len(jepa_files) < views_per_animode:
                continue

            # Parse object key: e.g. "8_senior_3" -> seed="8", animode="senior_3"
            parts = model_id.split("_", 1)
            obj_seed = parts[0]
            obj_key = (cat_name, obj_seed)
            if obj_key not in obj_animodes:
                obj_animodes[obj_key] = []
            obj_animodes[obj_key].append({
                "category": cat_name,
                "model_id": model_id,
                "gt_path": gt_path,
                "jepa_files": jepa_files,
            })

    # Sample: spread across objects
    all_objects = list(obj_animodes.keys())
    rng.shuffle(all_objects)

    samples = []
    for obj_key in all_objects:
        if len(samples) >= max_animodes:
            break
        animodes = obj_animodes[obj_key]
        # Take up to 2 animodes per object to spread coverage
        chosen = rng.sample(animodes, min(2, len(animodes)))
        for anim in chosen:
            if len(samples) >= max_animodes:
                break
            # Pick N views
            view_picks = rng.sample(anim["jepa_files"],
                                     min(views_per_animode, len(anim["jepa_files"])))
            for vp in view_picks:
                samples.append({
                    "id": f"{anim['category']}/{anim['model_id']}",
                    "category": anim["category"],
                    "model_id": anim["model_id"],
                    "gt_path": anim["gt_path"],
                    "jepa_path": vp,
                    "view": os.path.basename(vp),
                })

    print(f"Sampled {len(samples)} inference tasks "
          f"({len(set(s['id'] for s in samples))} animodes, "
          f"{len(set((s['category'], s['model_id'].split('_')[0]) for s in samples))} objects)")
    return samples


PRECOMPUTE_ROOT = "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output"


def find_obj_paths(sample):
    """Find raw part0.obj/part1.obj from precompute for a sample.
    Returns (part0_path, part1_path) or (None, None).
    """
    cat = sample["category"]
    model_id = sample["model_id"]
    parts = model_id.split("_", 1)
    if len(parts) < 2:
        return None, None
    seed, animode = parts[0], parts[1]
    animode_dir = os.path.join(PRECOMPUTE_ROOT, cat, seed, animode)
    p0 = os.path.join(animode_dir, "part0.obj")
    p1 = os.path.join(animode_dir, "part1.obj")
    if os.path.exists(p0) and os.path.exists(p1):
        return p0, p1
    return None, None


def count_obj_components(obj_path, threshold_ratio=0.01, min_faces=10):
    """Count connected components in a raw OBJ mesh file.
    Filters tiny fragments (< 1% of total faces or < 10 faces).
    Returns n_components or -1 on error.
    """
    try:
        mesh = trimesh.load(obj_path, force='mesh', process=False)
        if len(mesh.faces) == 0:
            return 0
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            total_faces = sum(len(c.faces) for c in components)
            threshold = max(total_faces * threshold_ratio, min_faces)
            components = [c for c in components if len(c.faces) >= threshold]
        return len(components)
    except Exception:
        return -1


def find_video_for_sample(sample):
    """Find a nobg.mp4 video for a sample (for original PartPacker first-frame input).

    data_ssd layout:  {category}/{seed}_{animode}/views/vXX_nobg_jepa.pt
    precompute layout: {category}/{seed}/{animode}/hemi_XX_nobg.mp4
    """
    cat = sample["category"]
    model_id = sample["model_id"]  # e.g. "8_senior_3"

    # Parse seed and animode from model_id
    # model_id format: "{seed}_{animode}" where animode can be "basic_0", "senior_3", etc.
    parts = model_id.split("_", 1)
    if len(parts) < 2:
        return None
    seed = parts[0]
    animode = parts[1]

    # Look in precompute_output
    animode_dir = os.path.join(PRECOMPUTE_ROOT, cat, seed, animode)
    if os.path.isdir(animode_dir):
        for f in sorted(os.listdir(animode_dir)):
            if f.endswith("_nobg.mp4"):
                return os.path.join(animode_dir, f)

    # Also check data_ssd animode dir directly
    views_dir = os.path.dirname(sample["jepa_path"])
    animode_ssd_dir = os.path.dirname(views_dir)
    for d in [animode_ssd_dir, views_dir]:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.endswith("_nobg.mp4"):
                    return os.path.join(d, f)

    return None


# ================================================================
# Model builders
# ================================================================

def build_our_model(device, ckpt_path):
    """Build our VJEPA2-conditioned model."""
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
    print(f"Our model loaded: step={step}, loss={ckpt.get('loss', 0):.6f}")
    dit.eval()
    proj.eval()
    return dit, proj


def build_original_model(device):
    """Build original PartPacker (DINOv2 -> DiT)."""
    from flow.model import Model
    prev_cwd = os.getcwd()
    os.chdir(PARTPACKER_ROOT)
    ckpt_path = os.path.join(PARTPACKER_ROOT, "pretrained", "flow.pt")
    ckpt_dict = torch.load(ckpt_path, weights_only=True)
    if "model" in ckpt_dict:
        ckpt_dict = ckpt_dict["model"]
    config = importlib.import_module(
        "flow.configs.big_parts_strict_pvae").make_config()
    model = Model(config).eval().to(device).to(torch.bfloat16)
    model.load_state_dict(ckpt_dict, strict=True)
    os.chdir(prev_cwd)
    del ckpt_dict
    print("Original PartPacker loaded (DINOv2-giant + DiT + VAE)")
    return model


def load_vae(device):
    """Load VAE for mesh decoding."""
    from vae.model import Model as VAE
    vae_config = importlib.import_module("vae.configs.part_woenc").make_config()
    vae = VAE(vae_config).eval().to(device, dtype=torch.bfloat16)
    vae_ckpt_path = os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt")
    ckpt = torch.load(vae_ckpt_path, weights_only=True, map_location=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    vae.load_state_dict(ckpt, strict=True)
    del ckpt
    print("VAE loaded")
    return vae


# ================================================================
# Inference
# ================================================================

def flow_inference(dit, cond, device, num_steps=50, cfg_scale=5.0):
    """Flow matching sampling for our model."""
    cond_null = torch.zeros_like(cond)
    cond_input = torch.cat([cond, cond_null], dim=0)
    x = torch.randn(1, TOTAL_LATENT, LATENT_DIM, device=device)
    sigmas = np.linspace(1, 0, num_steps + 1)
    sigmas = 3.0 * sigmas / (1 + (3.0 - 1) * sigmas)
    with torch.inference_mode():
        for i in range(num_steps):
            sigma, sigma_prev = sigmas[i], sigmas[i + 1]
            timesteps = torch.tensor(
                [1000 * sigma] * 2, device=device, dtype=torch.float32)
            x_input = torch.cat([x, x], dim=0).to(dtype=torch.bfloat16)
            pred = dit(x_input, cond_input, timesteps).float()
            cond_v, uncond_v = pred.chunk(2, dim=0)
            pred_v = uncond_v + (cond_v - uncond_v) * cfg_scale
            x = x - (sigma - sigma_prev) * pred_v
    return x


def infer_ours(dit, proj, sample, device, num_steps=50, cfg_scale=5.0):
    """Run our model inference. Returns predicted latent [1, 8192, 64]."""
    vj = torch.load(sample["jepa_path"], weights_only=False,
                     map_location=device).to(dtype=torch.bfloat16)
    if vj.dim() == 2:
        vj = vj.unsqueeze(0)
    with torch.no_grad():
        cond = proj(vj)
    pred = flow_inference(dit, cond, device, num_steps, cfg_scale)
    del vj, cond
    return pred


def preprocess_image_for_partpacker(image_rgb):
    """Preprocess RGB image for original PartPacker."""
    import rembg
    from flow.utils import recenter_foreground
    rgba = rembg.remove(image_rgb)
    mask = rgba[..., -1] > 0
    image = recenter_foreground(rgba, mask, border_ratio=0.1)
    image = cv2.resize(image, (518, 518), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = image[..., :3] * image[..., 3:4] + (1 - image[..., 3:4])
    return image


def extract_first_frame(video_path):
    """Extract first frame from video as RGB numpy array."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def infer_original(model, sample, device, num_steps=50, cfg_scale=7.0):
    """Run original PartPacker on first frame of video. Returns latent [1, 8192, 64]."""
    video_path = find_video_for_sample(sample)

    if video_path is None or not os.path.exists(video_path):
        return None

    frame = extract_first_frame(video_path)
    if frame is None:
        return None

    image_pp = preprocess_image_for_partpacker(frame)
    image_t = torch.from_numpy(image_pp).permute(2, 0, 1).contiguous()
    image_t = image_t.unsqueeze(0).float().to(device)
    data = {"cond_images": image_t}
    with torch.inference_mode():
        results = model(data, num_steps=num_steps, cfg_scale=cfg_scale,
                        verbose=False)
    return results["latent"]


# ================================================================
# Topology metrics
# ================================================================

def count_mesh_components(vae, latent_part, device, resolution=64):
    """Decode part latent -> mesh -> count connected components.
    Returns (n_components, n_faces) or (-1, 0) on error.
    """
    try:
        data = {"latent": latent_part.to(device, dtype=torch.bfloat16)}
        with torch.inference_mode():
            results = vae(data, resolution=resolution)
        if "meshes" not in results or len(results["meshes"]) == 0:
            return 0, 0
        vertices, faces = results["meshes"][0]
        if len(vertices) == 0 or len(faces) == 0:
            return 0, 0
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            total_faces = sum(len(c.faces) for c in components)
            threshold = max(total_faces * 0.01, 10)
            components = [c for c in components if len(c.faces) >= threshold]
        return len(components), len(faces)
    except Exception as e:
        return -1, 0


def compute_latent_mse(pred, gt):
    """Compute MSE between predicted and GT latents."""
    return F.mse_loss(pred.float(), gt.float()).item()


def _topo_metrics(pred_p0_cc, pred_p1_cc, ref_p0_cc, ref_p1_cc):
    """Compute topo metrics between pred and a reference (swap-aware)."""
    match_orig = (pred_p0_cc == ref_p0_cc and pred_p1_cc == ref_p1_cc)
    match_swap = (pred_p0_cc == ref_p1_cc and pred_p1_cc == ref_p0_cc)
    topo_correct = match_orig or match_swap

    len_orig = (abs(pred_p0_cc - ref_p0_cc) <= 1 and abs(pred_p1_cc - ref_p1_cc) <= 1)
    len_swap = (abs(pred_p0_cc - ref_p1_cc) <= 1 and abs(pred_p1_cc - ref_p0_cc) <= 1)
    topo_lenient = len_orig or len_swap

    err_orig = abs(pred_p0_cc - ref_p0_cc) + abs(pred_p1_cc - ref_p1_cc)
    err_swap = abs(pred_p0_cc - ref_p1_cc) + abs(pred_p1_cc - ref_p0_cc)
    cc_error = min(err_orig, err_swap)

    return topo_correct, topo_lenient, cc_error


def evaluate_one(pred_latent, gt_latent, vae, device, sample, resolution=64):
    """Evaluate one prediction: topology + MSE, with and without swap.

    Computes three levels of CC reference:
      - obj_cc: raw part0.obj/part1.obj (true topology, no VAE)
      - gt_vae_cc: GT latent -> VAE decode (VAE roundtrip)
      - pred_cc: model prediction -> VAE decode

    Returns dict with all metrics for this sample.
    """
    result = {}

    # Latent MSE (original assignment)
    result["mse_orig"] = compute_latent_mse(pred_latent, gt_latent)

    # Latent MSE (swapped: pred_p0->gt_p1, pred_p1->gt_p0)
    pred_swapped = torch.cat([pred_latent[:, LATENT_SIZE:, :],
                               pred_latent[:, :LATENT_SIZE, :]], dim=1)
    result["mse_swap"] = compute_latent_mse(pred_swapped, gt_latent)
    result["mse_best"] = min(result["mse_orig"], result["mse_swap"])

    # --- Raw OBJ CC (true topology) ---
    p0_obj, p1_obj = find_obj_paths(sample)
    if p0_obj is not None:
        result["obj_p0_cc"] = count_obj_components(p0_obj)
        result["obj_p1_cc"] = count_obj_components(p1_obj)
    else:
        result["obj_p0_cc"] = -1
        result["obj_p1_cc"] = -1

    # --- GT VAE CC (VAE roundtrip) ---
    gt_p0_cc, gt_p0_nf = count_mesh_components(
        vae, gt_latent[:, :LATENT_SIZE, :], device, resolution)
    gt_p1_cc, gt_p1_nf = count_mesh_components(
        vae, gt_latent[:, LATENT_SIZE:, :], device, resolution)

    result["gt_vae_p0_cc"] = gt_p0_cc
    result["gt_vae_p1_cc"] = gt_p1_cc
    result["gt_vae_p0_faces"] = gt_p0_nf
    result["gt_vae_p1_faces"] = gt_p1_nf

    # --- Pred CC ---
    pred_p0_cc, pred_p0_nf = count_mesh_components(
        vae, pred_latent[:, :LATENT_SIZE, :], device, resolution)
    pred_p1_cc, pred_p1_nf = count_mesh_components(
        vae, pred_latent[:, LATENT_SIZE:, :], device, resolution)

    result["pred_p0_cc"] = pred_p0_cc
    result["pred_p1_cc"] = pred_p1_cc
    result["pred_p0_faces"] = pred_p0_nf
    result["pred_p1_faces"] = pred_p1_nf

    # --- Topo metrics vs OBJ (true topology) ---
    if result["obj_p0_cc"] >= 0 and result["obj_p1_cc"] >= 0:
        tc, tl, ce = _topo_metrics(pred_p0_cc, pred_p1_cc,
                                    result["obj_p0_cc"], result["obj_p1_cc"])
        result["topo_vs_obj"] = tc
        result["topo_vs_obj_lenient"] = tl
        result["cc_error_vs_obj"] = ce
        # Also: VAE roundtrip distortion (gt_vae vs obj)
        vtc, vtl, vce = _topo_metrics(gt_p0_cc, gt_p1_cc,
                                       result["obj_p0_cc"], result["obj_p1_cc"])
        result["vae_topo_vs_obj"] = vtc
        result["vae_topo_vs_obj_lenient"] = vtl
        result["vae_cc_error_vs_obj"] = vce
    else:
        result["topo_vs_obj"] = None
        result["topo_vs_obj_lenient"] = None
        result["cc_error_vs_obj"] = None
        result["vae_topo_vs_obj"] = None
        result["vae_topo_vs_obj_lenient"] = None
        result["vae_cc_error_vs_obj"] = None

    # --- Topo metrics vs GT VAE (backward compat) ---
    tc, tl, ce = _topo_metrics(pred_p0_cc, pred_p1_cc, gt_p0_cc, gt_p1_cc)
    result["topo_vs_vae"] = tc
    result["topo_vs_vae_lenient"] = tl
    result["cc_error_vs_vae"] = ce

    # Legacy keys (keep for backward compat with analysis scripts)
    result["gt_p0_cc"] = gt_p0_cc
    result["gt_p1_cc"] = gt_p1_cc
    result["topo_correct"] = tc
    result["topo_correct_lenient"] = tl
    result["cc_error"] = ce

    return result


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Topology evaluation")
    parser.add_argument("--mode", choices=["ours", "original"], required=True)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint for our model")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--max_animodes", type=int, default=200)
    parser.add_argument("--views_per_animode", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=50,
                        help="Flow sampling steps")
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                        help="CFG scale (ours=5.0, original=7.0)")
    parser.add_argument("--vae_resolution", type=int, default=64,
                        help="VAE decode resolution for CC counting")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output is None:
        args.output = f"eval_topo_{args.mode}.json"

    if args.mode == "ours" and args.ckpt is None:
        args.ckpt = "/mnt/data_ssd/infinigen-sim/train_output/step_22000.pt"

    if args.mode == "original" and args.cfg_scale == 5.0:
        args.cfg_scale = 7.0  # original PartPacker default

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Discover samples
    samples = discover_eval_samples(
        args.data_root,
        max_animodes=args.max_animodes,
        views_per_animode=args.views_per_animode,
        seed=args.seed,
    )
    print(f"Total inference tasks: {len(samples)}")

    # Load models
    print(f"\nLoading models on {device}...")
    if args.mode == "ours":
        dit, proj = build_our_model(device, args.ckpt)
        orig_model = None
    else:
        dit, proj = None, None
        orig_model = build_original_model(device)

    vae = load_vae(device)
    torch.cuda.empty_cache()

    # Run evaluation
    results = []
    skipped = 0
    t0 = time.time()

    for i, sample in enumerate(samples):
        try:
            # Load GT
            gt = torch.load(sample["gt_path"], weights_only=False,
                            map_location=device).float()
            if gt.dim() == 2:
                gt = gt.unsqueeze(0)

            # Run inference
            if args.mode == "ours":
                pred = infer_ours(dit, proj, sample, device,
                                   args.num_steps, args.cfg_scale)
            else:
                pred = infer_original(orig_model, sample, device,
                                       args.num_steps, args.cfg_scale)
                if pred is None:
                    skipped += 1
                    continue

            # Evaluate
            metrics = evaluate_one(pred, gt, vae, device, sample, args.vae_resolution)
            metrics["id"] = sample["id"]
            metrics["view"] = sample["view"]
            metrics["category"] = sample["category"]
            results.append(metrics)

            del pred, gt
            torch.cuda.empty_cache()

            # Progress
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(samples) - i - 1) / rate if rate > 0 else 0

                recent = results[-10:]
                avg_mse = np.mean([r["mse_best"] for r in recent])
                topo_ok = np.mean([r["topo_correct"] for r in recent])
                topo_ok_l = np.mean([r["topo_correct_lenient"] for r in recent])
                avg_p0cc = np.mean([r["pred_p0_cc"] for r in recent
                                     if r["pred_p0_cc"] >= 0])
                avg_p1cc = np.mean([r["pred_p1_cc"] for r in recent
                                     if r["pred_p1_cc"] >= 0])

                print(f"  [{i+1}/{len(samples)}] "
                      f"mse_best={avg_mse:.4f} "
                      f"topo={topo_ok:.0%} topo_l={topo_ok_l:.0%} "
                      f"p0cc={avg_p0cc:.1f} p1cc={avg_p1cc:.1f} "
                      f"| {rate:.1f} it/s, ETA {eta/60:.0f}m "
                      f"| skip={skipped}")

        except Exception as e:
            print(f"  [{i+1}] ERROR {sample['id']}: {e}")
            skipped += 1
            continue

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}m ({len(results)} results, {skipped} skipped)")

    # Aggregate
    if results:
        def _safe_mean(vals):
            vals = [v for v in vals if v is not None and v >= 0]
            return float(np.mean(vals)) if vals else -1.0

        # Results with valid OBJ CC
        has_obj = [r for r in results
                   if r.get("obj_p0_cc", -1) >= 0 and r.get("obj_p1_cc", -1) >= 0]

        agg = {
            "mode": args.mode,
            "ckpt": args.ckpt,
            "num_steps": args.num_steps,
            "cfg_scale": args.cfg_scale,
            "vae_resolution": args.vae_resolution,
            "n_total": len(results),
            "n_skipped": skipped,
            "n_with_obj": len(has_obj),
            "n_animodes": len(set(r["id"] for r in results)),
            "n_categories": len(set(r["category"] for r in results)),
            "elapsed_min": elapsed / 60,

            # Latent MSE
            "mse_orig_mean": float(np.mean([r["mse_orig"] for r in results])),
            "mse_swap_mean": float(np.mean([r["mse_swap"] for r in results])),
            "mse_best_mean": float(np.mean([r["mse_best"] for r in results])),

            # CC means
            "pred_p0_cc_mean": _safe_mean([r["pred_p0_cc"] for r in results]),
            "pred_p1_cc_mean": _safe_mean([r["pred_p1_cc"] for r in results]),
            "gt_vae_p0_cc_mean": _safe_mean([r["gt_vae_p0_cc"] for r in results]),
            "gt_vae_p1_cc_mean": _safe_mean([r["gt_vae_p1_cc"] for r in results]),
            "obj_p0_cc_mean": _safe_mean([r["obj_p0_cc"] for r in has_obj]),
            "obj_p1_cc_mean": _safe_mean([r["obj_p1_cc"] for r in has_obj]),

            # Pred vs OBJ (true topology)
            "topo_vs_obj_rate": _safe_mean([float(r["topo_vs_obj"]) for r in has_obj]),
            "topo_vs_obj_lenient_rate": _safe_mean([float(r["topo_vs_obj_lenient"]) for r in has_obj]),
            "cc_error_vs_obj_mean": _safe_mean([r["cc_error_vs_obj"] for r in has_obj]),

            # VAE roundtrip distortion (gt_vae vs OBJ)
            "vae_topo_vs_obj_rate": _safe_mean([float(r["vae_topo_vs_obj"]) for r in has_obj]),
            "vae_topo_vs_obj_lenient_rate": _safe_mean([float(r["vae_topo_vs_obj_lenient"]) for r in has_obj]),
            "vae_cc_error_vs_obj_mean": _safe_mean([r["vae_cc_error_vs_obj"] for r in has_obj]),

            # Pred vs GT VAE
            "topo_vs_vae_rate": float(np.mean([r["topo_vs_vae"] for r in results])),
            "topo_vs_vae_lenient_rate": float(np.mean([r["topo_vs_vae_lenient"] for r in results])),
            "cc_error_vs_vae_mean": float(np.mean([r["cc_error_vs_vae"] for r in results])),

            # Per-category breakdown
            "per_category": {},

            # Raw results
            "results": results,
        }

        # Per-category
        from collections import defaultdict
        cat_results = defaultdict(list)
        for r in results:
            cat_results[r["category"]].append(r)

        for cat, cat_rs in sorted(cat_results.items()):
            cat_obj = [r for r in cat_rs
                       if r.get("obj_p0_cc", -1) >= 0 and r.get("obj_p1_cc", -1) >= 0]
            agg["per_category"][cat] = {
                "n": len(cat_rs),
                "n_with_obj": len(cat_obj),
                "mse_best_mean": float(np.mean([r["mse_best"] for r in cat_rs])),
                "pred_p0_cc_mean": _safe_mean([r["pred_p0_cc"] for r in cat_rs]),
                "pred_p1_cc_mean": _safe_mean([r["pred_p1_cc"] for r in cat_rs]),
                "obj_p0_cc_mean": _safe_mean([r["obj_p0_cc"] for r in cat_obj]) if cat_obj else -1,
                "obj_p1_cc_mean": _safe_mean([r["obj_p1_cc"] for r in cat_obj]) if cat_obj else -1,
                "topo_vs_obj_rate": _safe_mean([float(r["topo_vs_obj"]) for r in cat_obj]) if cat_obj else -1,
                "cc_error_vs_obj_mean": _safe_mean([r["cc_error_vs_obj"] for r in cat_obj]) if cat_obj else -1,
                "topo_vs_vae_rate": float(np.mean([r["topo_vs_vae"] for r in cat_rs])),
                "cc_error_vs_vae_mean": float(np.mean([r["cc_error_vs_vae"] for r in cat_rs])),
            }

        # Print summary
        print(f"\n{'='*70}")
        print(f"  Mode: {args.mode}")
        print(f"  Samples: {agg['n_total']} ({agg['n_with_obj']} with OBJ, "
              f"{agg['n_animodes']} animodes, {agg['n_categories']} categories)")
        print(f"  MSE best: {agg['mse_best_mean']:.4f}")
        print(f"\n  --- CC Means ---")
        print(f"  {'':>12} {'p0':>8} {'p1':>8}")
        print(f"  {'OBJ (true)':>12} {agg['obj_p0_cc_mean']:>8.2f} {agg['obj_p1_cc_mean']:>8.2f}")
        print(f"  {'GT VAE':>12} {agg['gt_vae_p0_cc_mean']:>8.2f} {agg['gt_vae_p1_cc_mean']:>8.2f}")
        print(f"  {'Pred':>12} {agg['pred_p0_cc_mean']:>8.2f} {agg['pred_p1_cc_mean']:>8.2f}")
        print(f"\n  --- Topo Accuracy (Pred vs Reference) ---")
        print(f"  {'Reference':>12} {'Exact':>8} {'±1':>8} {'CC err':>8}")
        print(f"  {'vs OBJ':>12} {agg['topo_vs_obj_rate']:>7.1%} "
              f"{agg['topo_vs_obj_lenient_rate']:>7.1%} "
              f"{agg['cc_error_vs_obj_mean']:>8.2f}")
        print(f"  {'vs GT VAE':>12} {agg['topo_vs_vae_rate']:>7.1%} "
              f"{agg['topo_vs_vae_lenient_rate']:>7.1%} "
              f"{agg['cc_error_vs_vae_mean']:>8.2f}")
        print(f"\n  --- VAE Roundtrip Distortion (GT VAE vs OBJ) ---")
        print(f"  {'Exact':>8} {'±1':>8} {'CC err':>8}")
        print(f"  {agg['vae_topo_vs_obj_rate']:>7.1%} "
              f"{agg['vae_topo_vs_obj_lenient_rate']:>7.1%} "
              f"{agg['vae_cc_error_vs_obj_mean']:>8.2f}")
        print(f"\n  Per-category:")
        print(f"  {'Category':<22} {'N':>3} {'MSE':>7} {'vsOBJ%':>7} {'vsVAE%':>7} "
              f"{'OBJ_cc':>8} {'Pred_cc':>9}")
        for cat, ci in sorted(agg["per_category"].items()):
            obj_cc_str = (f"{ci['obj_p0_cc_mean']:.1f}/{ci['obj_p1_cc_mean']:.1f}"
                          if ci.get("obj_p0_cc_mean", -1) >= 0 else "n/a")
            pred_cc_str = f"{ci['pred_p0_cc_mean']:.1f}/{ci['pred_p1_cc_mean']:.1f}"
            tobj = f"{ci['topo_vs_obj_rate']:.0%}" if ci.get("topo_vs_obj_rate", -1) >= 0 else "n/a"
            print(f"  {cat:<22} {ci['n']:>3} {ci['mse_best_mean']:>7.4f} "
                  f"{tobj:>7} {ci['topo_vs_vae_rate']:>6.0%} "
                  f"{obj_cc_str:>8} {pred_cc_str:>9}")
        print(f"{'='*70}")

        # Save
        with open(args.output, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
