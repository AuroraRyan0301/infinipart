#!/usr/bin/env python3
"""
Step 1: Precompute PartPacker VAE occupancy coords for SLat inference.

For each sample with gt_latent.pt:
  gt_latent [1, 8192, 64] → split part0 [1,4096,64] + part1 [1,4096,64]
  → VAE decode → query 32³ grid → occ > 0 → coords [N, 4] (batch_idx, x, y, z)
  → save to vae_coords.pt

Run in partpacker_wan env:
  CUDA_VISIBLE_DEVICES=1 python precompute_vae_coords.py --output_dir ./output/vae_coords
"""
import argparse
import importlib
import os
import sys
import time

import torch
import numpy as np

PARTPACKER_ROOT = "/mnt/cpfs/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)


def load_vae(device):
    from vae.model import Model
    ckpt_path = os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt")
    ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    config = importlib.import_module("vae.configs.part_woenc").make_config()
    model = Model(config).eval().to(device).bfloat16()
    model.load_state_dict(ckpt)
    print(f"Loaded PartPacker VAE from {ckpt_path}")
    return model


def vae_to_coords(vae, latent, device, res=32):
    """VAE latent [1, 4096, 64] → coords [N, 4] in range [0, res-1]."""
    from vae.utils import construct_grid_points
    hidden = vae.decode(latent.to(device).to(vae.precision))
    if hasattr(vae, 'norm_query_context') and vae.config.use_flash_query:
        hidden = vae.norm_query_context(hidden)

    grid_points = construct_grid_points(res).to(device).reshape(-1, 3)
    occ_parts = []
    for i in range(0, grid_points.shape[0], 65536):
        chunk = grid_points[i:i+65536].unsqueeze(0)
        pred = vae.query(chunk, hidden).squeeze(-1).float()
        occ_parts.append(pred)
    vertex_occ = torch.cat(occ_parts, dim=1).squeeze(0).reshape(res+1, res+1, res+1)

    # Trilinear: vertex → voxel center
    voxel_occ = (vertex_occ[:-1,:-1,:-1] + vertex_occ[1:,:-1,:-1] +
                 vertex_occ[:-1,1:,:-1] + vertex_occ[1:,1:,:-1] +
                 vertex_occ[:-1,:-1,1:] + vertex_occ[1:,:-1,1:] +
                 vertex_occ[:-1,1:,1:] + vertex_occ[1:,1:,1:]) / 8

    coords_3d = torch.argwhere(voxel_occ > 0)  # [N, 3]
    if coords_3d.shape[0] == 0:
        return None
    batch_coords = torch.cat([
        torch.zeros(coords_3d.shape[0], 1, dtype=torch.int32, device=device),
        coords_3d.int()
    ], dim=1)
    return batch_coords


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/data_ssd/infinigen-sim")
    parser.add_argument("--slat_gt_root", default="/mnt/data_ssd/infinigen-sim/slat_gt")
    parser.add_argument("--output_dir", default="./output/vae_coords")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    vae = load_vae(device)

    # Discover all samples with slat_gt
    count = 0
    for cat in sorted(os.listdir(args.slat_gt_root)):
        cat_gt = os.path.join(args.slat_gt_root, cat)
        cat_data = os.path.join(args.data_root, cat)
        if not os.path.isdir(cat_gt) or not os.path.isdir(cat_data):
            continue

        cat_out = os.path.join(args.output_dir, cat)
        os.makedirs(cat_out, exist_ok=True)

        for f in sorted(os.listdir(cat_gt)):
            if not f.endswith('.pt'):
                continue
            mid = f[:-3]
            out_path = os.path.join(cat_out, f)
            if os.path.exists(out_path):
                continue

            # Need gt_latent.pt for VAE input
            gt_latent_path = os.path.join(cat_data, mid, "gt_latent.pt")
            if not os.path.exists(gt_latent_path):
                continue

            try:
                gt_latent = torch.load(gt_latent_path, weights_only=False, map_location=device)
                if gt_latent.dim() == 2:
                    gt_latent = gt_latent.unsqueeze(0)
            except (EOFError, RuntimeError):
                continue

            p0_lat = gt_latent[:, :4096, :]
            p1_lat = gt_latent[:, 4096:, :]

            with torch.no_grad():
                p0_coords = vae_to_coords(vae, p0_lat, device)
                p1_coords = vae_to_coords(vae, p1_lat, device)

            if p0_coords is None or p1_coords is None:
                continue

            torch.save({
                "p0_coords": p0_coords.cpu(),
                "p1_coords": p1_coords.cpu(),
            }, out_path)

            count += 1
            if count % 50 == 0:
                print(f"  [{count}] {cat}/{mid}: p0={p0_coords.shape[0]} p1={p1_coords.shape[0]}")

    print(f"\nDone! {count} samples saved to {args.output_dir}")


if __name__ == "__main__":
    main()
