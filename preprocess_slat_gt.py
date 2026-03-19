#!/usr/bin/env python3
"""
Preprocess GT SLat: encode all part OBJs to SLat latent using TRELLIS 2 SLat Encoder.

For each part OBJ:
  OBJ → normalize → mesh_to_flexible_dual_grid → SLat Encoder → latent [N, 32]
  Save as .pt with coords + latent + metadata

Usage:
  CUDA_VISIBLE_DEVICES=2 python preprocess_slat_gt.py \
    --data_root /mnt/cpfs/yurh/Infinigen-Sim/precompute_output \
    --training_data /mnt/data_ssd/infinigen-sim \
    --output_dir /mnt/data_ssd/infinigen-sim/slat_cache
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import trimesh

TRELLIS_ROOT = "/mnt/cpfs/yurh/TRELLIS.2"
sys.path.insert(0, TRELLIS_ROOT)
sys.path.insert(0, os.path.join(TRELLIS_ROOT, "o-voxel"))

from o_voxel.convert.flexible_dual_grid import mesh_to_flexible_dual_grid
from trellis2.modules.sparse import SparseTensor
from safetensors.torch import load_file

CKPT_DIR = "/mnt/data/yurh/TRELLIS.2-4B/ckpts"
REMESH_DIR = "/mnt/data_ssd/infinigen-sim/vae_remesh"


def load_slat_encoder(device="cuda:0"):
    from trellis2.models.sc_vaes.fdg_vae import FlexiDualGridVaeEncoder
    with open(f"{CKPT_DIR}/shape_enc_next_dc_f16c32_fp16.json") as f:
        cfg = json.load(f)
    encoder = FlexiDualGridVaeEncoder(**cfg["args"]).to(device).eval()
    encoder.load_state_dict(load_file(f"{CKPT_DIR}/shape_enc_next_dc_f16c32_fp16.safetensors"))
    return encoder


def get_mesh_path(obj_path):
    """Return remeshed path if exists, else original."""
    h = hashlib.md5(obj_path.encode()).hexdigest()[:12]
    remeshed = os.path.join(REMESH_DIR, f"{h}.obj")
    if os.path.exists(remeshed):
        return remeshed
    return obj_path


def encode_one(obj_path, encoder, device, max_voxels=50000):
    """Encode one OBJ → SLat latent. Returns dict or None."""
    mesh_path = get_mesh_path(obj_path)
    mesh = trimesh.load(mesh_path, force='mesh', process=True)
    mesh.merge_vertices()
    if len(mesh.vertices) < 3 or len(mesh.faces) < 1:
        return None

    verts = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.long)

    # Normalize to [-0.45, 0.45]
    center = (verts.max(0).values + verts.min(0).values) / 2
    scale = (verts.max(0).values - verts.min(0).values).max()
    if scale < 1e-6:
        return None
    verts = (verts - center) / scale * 0.9

    # Adaptive grid size
    grid_size = None
    for gs in [512, 256, 128, 64]:
        coords, dual_verts, intersected = mesh_to_flexible_dual_grid(
            verts, faces, grid_size=gs,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2)
        if coords.shape[0] <= max_voxels:
            grid_size = gs
            break

    if grid_size is None:
        return None

    n_voxels = coords.shape[0]
    relative_verts = dual_verts * grid_size - coords.float()
    batch_coords = torch.cat([torch.zeros_like(coords[:, 0:1]), coords], dim=-1)
    vertices_st = SparseTensor(feats=relative_verts, coords=batch_coords).to(device)
    intersected_st = vertices_st.replace(intersected).to(device)

    with torch.inference_mode():
        slat = encoder(vertices_st, intersected_st)

    return {
        "slat_feats": slat.feats.cpu().float(),
        "slat_coords": slat.coords.cpu().int(),
        "input_coords": coords.cpu().int(),
        "grid_size": grid_size,
        "n_voxels": n_voxels,
        "n_latent": slat.feats.shape[0],
        "obj_path": obj_path,
        "mesh_path": mesh_path,
        "normalize_center": center.tolist(),
        "normalize_scale": float(scale),
    }


def discover_objs(data_root, training_data_root):
    """Find part OBJs that have training data."""
    precompute_root = os.path.abspath(data_root)
    training_data_root = os.path.abspath(training_data_root)
    objs = []
    for cat in sorted(os.listdir(training_data_root)):
        cat_dir = os.path.join(training_data_root, cat)
        if not os.path.isdir(cat_dir) or cat.startswith("."):
            continue
        if cat in ("train_output", "train_output_v2", ".errors"):
            continue
        for mid in sorted(os.listdir(cat_dir)):
            md = os.path.join(cat_dir, mid)
            if not os.path.isdir(md) or not os.path.exists(os.path.join(md, "gt_latent.pt")):
                continue
            parts = mid.split("_", 1)
            if len(parts) < 2:
                continue
            seed, animode = parts[0], parts[1]
            for pname in ["part0.obj", "part1.obj"]:
                obj_path = os.path.join(precompute_root, cat, seed, animode, pname)
                if os.path.exists(obj_path):
                    objs.append(obj_path)
    return objs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/cpfs/yurh/Infinigen-Sim/precompute_output")
    parser.add_argument("--training_data", default="/mnt/data_ssd/infinigen-sim")
    parser.add_argument("--output_dir", default="/mnt/data_ssd/infinigen-sim/slat_cache")
    parser.add_argument("--max_voxels", type=int, default=50000)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    objs = discover_objs(args.data_root, args.training_data)
    print(f"Found {len(objs)} OBJ files")

    existing = len([f for f in os.listdir(args.output_dir) if f.endswith(".pt")])
    print(f"Already cached: {existing}")

    encoder = load_slat_encoder(args.device)
    print("SLat Encoder loaded")

    t0 = time.time()
    ok, skip, err = 0, 0, 0

    for i, obj_path in enumerate(objs):
        h = hashlib.md5(obj_path.encode()).hexdigest()[:12]
        out_path = os.path.join(args.output_dir, f"{h}.pt")

        if os.path.exists(out_path):
            skip += 1
            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                print(f"  [{i+1}/{len(objs)}] ok={ok} skip={skip} err={err} | {rate:.1f}/s")
            continue

        try:
            result = encode_one(obj_path, encoder, args.device, args.max_voxels)
            if result is not None:
                torch.save(result, out_path)
                ok += 1
            else:
                err += 1
        except Exception as e:
            err += 1
            if err <= 5:
                print(f"  ERROR [{i+1}] {os.path.basename(obj_path)}: {e}")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(objs) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(objs)}] ok={ok} skip={skip} err={err} | "
                  f"{rate:.1f}/s ETA {eta/60:.0f}m")

        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    total = len([f for f in os.listdir(args.output_dir) if f.endswith(".pt")])
    print(f"\nDone in {elapsed/60:.1f}m: ok={ok} skip={skip} err={err} total_cached={total}")


if __name__ == "__main__":
    main()
