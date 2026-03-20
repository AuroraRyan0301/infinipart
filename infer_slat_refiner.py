#!/usr/bin/env python3
"""
Inference with DualPartSLatModel: generate refined mesh from VJEPA + VAE coords.

For evaluation: use GT SLat coords (from preprocess_slat_gt.py) and GT VJEPA features,
run flow matching sampling, decode to mesh, compare with GT.

Usage:
  CUDA_VISIBLE_DEVICES=3 python infer_slat_refiner.py \
    --ckpt /mnt/data_ssd/infinigen-sim/slat_refiner/slat_refiner_latest.pt \
    --output_dir ./output/slat_refiner_infer
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import trimesh

TRELLIS_ROOT = "/mnt/cpfs/yurh/TRELLIS.2"
sys.path.insert(0, TRELLIS_ROOT)
sys.path.insert(0, os.path.join(TRELLIS_ROOT, "o-voxel"))
sys.path.insert(0, "/mnt/cpfs/yurh/Infinigen-Sim")

from trellis2.modules.sparse import SparseTensor
from dual_part_slat import build_dual_part_model, load_slat_decoder


def flow_sampling(model, x0_st, x1_st, vjepa_feats, num_steps=50, cfg_scale=1.0):
    """Flow matching sampling: noise → SLat latent."""
    device = x0_st.feats.device

    # Initialize from noise
    noise0 = SparseTensor(
        feats=torch.randn_like(x0_st.feats),
        coords=x0_st.coords)
    noise1 = SparseTensor(
        feats=torch.randn_like(x1_st.feats),
        coords=x1_st.coords)

    sigmas = np.linspace(0, 1, num_steps + 1)

    x0 = noise0
    x1 = noise1

    with torch.inference_mode():
        for i in range(num_steps):
            t_val = sigmas[i]
            t = torch.tensor([1000 * t_val], device=device, dtype=torch.float32)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                v0, v1 = model(x0, x1, t, vjepa_feats)

            dt = sigmas[i + 1] - sigmas[i]
            x0 = x0.replace(x0.feats + dt * v0.feats.float())
            x1 = x1.replace(x1.feats + dt * v1.feats.float())

    return x0, x1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--slat_cache", default="/mnt/data_ssd/infinigen-sim/slat_cache")
    parser.add_argument("--data_root", default="/mnt/data_ssd/infinigen-sim")
    parser.add_argument("--precompute_root", default="/mnt/cpfs/yurh/Infinigen-Sim/precompute_output")
    parser.add_argument("--output_dir", default="./output/slat_refiner_infer")
    parser.add_argument("--max_samples", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    random.seed(args.seed)

    # Load model
    print("Loading DualPartSLatModel...")
    model = build_dual_part_model(device=device)

    # Load trained weights
    ckpt = torch.load(args.ckpt, weights_only=False, map_location=device)
    model.vjepa_proj.load_state_dict(ckpt["vjepa_proj"])
    model.part_cross_attns.load_state_dict(ckpt["part_cross_attns"])
    print(f"Loaded checkpoint: step={ckpt.get('step', '?')}")
    model.eval()

    # Load SLat decoder
    print("Loading SLat Decoder...")
    decoder = load_slat_decoder(device=device)

    # Find samples
    samples = []
    for cat in sorted(os.listdir(args.data_root)):
        cat_dir = os.path.join(args.data_root, cat)
        if not os.path.isdir(cat_dir) or cat.startswith(".") or cat in ("train_output", "train_output_v2", ".errors"):
            continue
        for mid in sorted(os.listdir(cat_dir)):
            md = os.path.join(cat_dir, mid)
            if not os.path.isdir(md) or not os.path.exists(os.path.join(md, "gt_latent.pt")):
                continue
            parts = mid.split("_", 1)
            if len(parts) < 2:
                continue
            seed_val, animode = parts[0], parts[1]

            p0_obj = os.path.join(args.precompute_root, cat, seed_val, animode, "part0.obj")
            p1_obj = os.path.join(args.precompute_root, cat, seed_val, animode, "part1.obj")
            h0 = hashlib.md5(p0_obj.encode()).hexdigest()[:12]
            h1 = hashlib.md5(p1_obj.encode()).hexdigest()[:12]
            p0_slat = os.path.join(args.slat_cache, f"{h0}.pt")
            p1_slat = os.path.join(args.slat_cache, f"{h1}.pt")

            views_dir = os.path.join(md, "views")
            if not os.path.isdir(views_dir):
                continue
            jepa_files = [os.path.join(views_dir, f) for f in sorted(os.listdir(views_dir)) if f.endswith("_nobg_jepa.pt")]

            if os.path.exists(p0_slat) and os.path.exists(p1_slat) and jepa_files:
                samples.append({
                    "id": f"{cat}/{mid}",
                    "p0_slat": p0_slat, "p1_slat": p1_slat,
                    "p0_obj": p0_obj, "p1_obj": p1_obj,
                    "jepa": jepa_files[0],
                })

    random.shuffle(samples)
    samples = samples[:args.max_samples]
    print(f"Evaluating {len(samples)} samples")

    for i, s in enumerate(samples):
        try:
            # Load GT SLat (for coords)
            p0_data = torch.load(s["p0_slat"], weights_only=False)
            p1_data = torch.load(s["p1_slat"], weights_only=False)
            jepa = torch.load(s["jepa"], weights_only=False)
            if jepa.dim() == 2:
                jepa = jepa.unsqueeze(0)

            # Build input SparseTensors (noise will be added in sampling)
            x0_st = SparseTensor(
                feats=p0_data["slat_feats"].to(device),
                coords=p0_data["slat_coords"].to(device))
            x1_st = SparseTensor(
                feats=p1_data["slat_feats"].to(device),
                coords=p1_data["slat_coords"].to(device))

            # Flow sampling
            t0 = time.time()
            pred0, pred1 = flow_sampling(
                model, x0_st, x1_st, jepa.to(device),
                num_steps=args.num_steps)
            dt = time.time() - t0

            # Decode to mesh
            for part_name, pred, data in [("part0", pred0, p0_data), ("part1", pred1, p1_data)]:
                gs = data["grid_size"]
                decoder.set_resolution(gs)
                with torch.inference_mode():
                    result = decoder(pred)
                if isinstance(result, list) and len(result) > 0:
                    mesh_obj = result[0]
                    v, f = mesh_obj.vertices, mesh_obj.faces
                    m = trimesh.Trimesh(v.cpu().numpy(), f.cpu().numpy(), process=False)
                    safe = s["id"].replace("/", "_")
                    m.export(os.path.join(args.output_dir, f"{safe}_{part_name}_pred.obj"))

            print(f"  [{i+1}/{len(samples)}] {s['id']}: {dt:.1f}s")

        except Exception as e:
            print(f"  [{i+1}] ERROR {s['id']}: {e}")
            import traceback
            traceback.print_exc()

        torch.cuda.empty_cache()

    print(f"\nDone! Results in {args.output_dir}")


if __name__ == "__main__":
    main()
