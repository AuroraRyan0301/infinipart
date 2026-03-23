#!/usr/bin/env python3
"""
Encode all part0/part1 OBJs → GT SLat latent at LR (512) and HR (1024).

Normalization: verts / 2 → [-0.5, 0.5] (parts are already in [-1, 1] from split_precompute).
Uses ORIGINAL OBJs only (no remesh), skips empty parts.

Output per pair:
  {slat_gt_dir}/{category}/{seed}_{animode}/gt_slat.pt
    Contains: {
      "p0_lr": {"feats": [N, 32], "coords": [N, 4]},
      "p1_lr": {"feats": [M, 32], "coords": [M, 4]},
      "p0_hr": {"feats": [N', 32], "coords": [N', 4]},
      "p1_hr": {"feats": [M', 32], "coords": [M', 4]},
    }

Usage (2 GPUs):
  CUDA_VISIBLE_DEVICES=2 python encode_slat_gt.py --gpu_id 0 --num_gpus 2 &
  CUDA_VISIBLE_DEVICES=3 python encode_slat_gt.py --gpu_id 1 --num_gpus 2 &
  wait
"""

import argparse
import os
import sys
import time

import torch
import trimesh

TRELLIS_ROOT = "/mnt/cpfs/yurh/TRELLIS.2"
sys.path.insert(0, TRELLIS_ROOT)
sys.path.insert(0, os.path.join(TRELLIS_ROOT, "o-voxel"))
sys.path.insert(0, "/mnt/cpfs/yurh/Infinigen-Sim")

from trellis2.modules.sparse import SparseTensor
from o_voxel.convert.flexible_dual_grid import mesh_to_flexible_dual_grid
from dual_part_slat import load_slat_encoder

LR_GRID = 512   # → latent coords [0, 31]
HR_GRID = 1024  # → latent coords [0, 63]


def encode_one_obj(obj_path, encoder, device, grid_size):
    """OBJ → verts/2 → [-0.5, 0.5] → SLat encoder → {feats, coords}."""
    mesh = trimesh.load(obj_path, force='mesh', process=True)
    mesh.merge_vertices()
    if len(mesh.vertices) < 3 or len(mesh.faces) < 1:
        return None

    verts = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)
    verts = verts / 2.0  # [-1, 1] → [-0.5, 0.5]

    try:
        coords, dual_verts, intersected = mesh_to_flexible_dual_grid(
            verts, faces, grid_size=grid_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2)
    except Exception:
        return None

    if coords.shape[0] == 0:
        return None

    relative_verts = dual_verts * grid_size - coords.float()
    batch_coords = torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1)
    vertices_st = SparseTensor(feats=relative_verts, coords=batch_coords).to(device)
    intersected_st = vertices_st.replace(intersected).to(device)

    with torch.inference_mode():
        slat = encoder(vertices_st, intersected_st)

    return {
        "feats": slat.feats.cpu().float(),
        "coords": slat.coords.cpu().int(),
    }


def discover_all_pairs(data_root, precompute_root):
    """Find all animode pairs with gt_latent.pt + part0/part1 OBJs."""
    skip_dirs = {"train_output", "train_output_v2", ".errors",
                 "slat_cache", "slat_refiner", "slat_refiner_v2", "slat_refiner_of4",
                 "vae_finetune", "vae_finetune_v2", "vae_remesh", "manifest.json"}
    pairs = []
    for cat in sorted(os.listdir(data_root)):
        cat_dir = os.path.join(data_root, cat)
        if not os.path.isdir(cat_dir) or cat.startswith(".") or cat in skip_dirs:
            continue
        for mid in sorted(os.listdir(cat_dir)):
            md = os.path.join(cat_dir, mid)
            if not os.path.isdir(md) or not os.path.exists(os.path.join(md, "gt_latent.pt")):
                continue
            parts = mid.split("_", 1)
            if len(parts) < 2:
                continue
            seed, animode = parts[0], parts[1]
            p0 = os.path.join(precompute_root, cat, seed, animode, "part0.obj")
            p1 = os.path.join(precompute_root, cat, seed, animode, "part1.obj")
            if os.path.exists(p0) and os.path.exists(p1):
                pairs.append({
                    "cat": cat, "mid": mid,
                    "p0": p0, "p1": p1,
                })
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/data_ssd/infinigen-sim")
    parser.add_argument("--precompute_root", default="/mnt/cpfs/yurh/Infinigen-Sim/precompute_output")
    parser.add_argument("--output_dir", default="/mnt/data_ssd/infinigen-sim/slat_gt")
    parser.add_argument("--gpu_id", type=int, default=0, help="Shard index for multi-GPU")
    parser.add_argument("--num_gpus", type=int, default=1, help="Total shards")
    args = parser.parse_args()

    device = torch.device("cuda:0")
    print(f"[GPU shard {args.gpu_id}/{args.num_gpus}] Loading encoder...")
    encoder = load_slat_encoder(device=device)

    pairs = discover_all_pairs(args.data_root, args.precompute_root)
    # Shard
    pairs = [p for i, p in enumerate(pairs) if i % args.num_gpus == args.gpu_id]
    print(f"[GPU {args.gpu_id}] {len(pairs)} pairs to encode")

    os.makedirs(args.output_dir, exist_ok=True)
    skip_log_path = os.path.join(args.output_dir, f"skipped_gpu{args.gpu_id}.txt")
    skip_log = open(skip_log_path, "w")
    done = 0
    skipped = 0
    t0 = time.time()

    for pi, pair in enumerate(pairs):
        out_dir = os.path.join(args.output_dir, pair["cat"])
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{pair['mid']}.pt")

        if os.path.exists(out_path):
            done += 1
            continue

        result = {}
        fail_reason = None
        for part_name, obj_path in [("p0", pair["p0"]), ("p1", pair["p1"])]:
            for label, gs in [("lr", LR_GRID), ("hr", HR_GRID)]:
                key = f"{part_name}_{label}"
                enc = encode_one_obj(obj_path, encoder, device, gs)
                if enc is None:
                    fail_reason = f"{key} empty/failed: {obj_path}"
                    break
                result[key] = enc
            if fail_reason:
                break

        if fail_reason:
            skipped += 1
            msg = f"SKIP {pair['cat']}/{pair['mid']}: {fail_reason}"
            skip_log.write(msg + "\n")
            skip_log.flush()
            print(f"  [{pi+1}/{len(pairs)}] {msg}")
            continue

        torch.save(result, out_path)
        done += 1

        if done % 20 == 0 or pi == 0:
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            n0_lr = result["p0_lr"]["feats"].shape[0]
            n1_lr = result["p1_lr"]["feats"].shape[0]
            n0_hr = result["p0_hr"]["feats"].shape[0]
            n1_hr = result["p1_hr"]["feats"].shape[0]
            print(f"  [{pi+1}/{len(pairs)}] {pair['cat']}/{pair['mid']}: "
                  f"LR({n0_lr}+{n1_lr}) HR({n0_hr}+{n1_hr}) | "
                  f"{done} done, {skipped} skip, {rate:.1f}/s")

        torch.cuda.empty_cache()

    skip_log.close()
    elapsed = time.time() - t0
    print(f"\n[GPU {args.gpu_id}] Done: {done} encoded, {skipped} skipped in {elapsed/60:.1f}m")
    print(f"Output: {args.output_dir}")
    if skipped > 0:
        print(f"Skipped parts logged to: {skip_log_path}")


if __name__ == "__main__":
    main()
