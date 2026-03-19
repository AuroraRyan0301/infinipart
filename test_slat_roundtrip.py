#!/usr/bin/env python3
"""
Test SLat roundtrip: GT OBJ → mesh_to_flexible_dual_grid → SLat Encoder → SLat Decoder → mesh

Verifies that TRELLIS 2's SLat VAE can faithfully reconstruct our part meshes.
"""
import os
import sys
import time

import torch
import trimesh
import numpy as np

TRELLIS_ROOT = "/mnt/cpfs/yurh/TRELLIS.2"
sys.path.insert(0, TRELLIS_ROOT)
sys.path.insert(0, os.path.join(TRELLIS_ROOT, "o-voxel"))

from o_voxel.convert.flexible_dual_grid import mesh_to_flexible_dual_grid, flexible_dual_grid_to_mesh
from trellis2.modules.sparse import SparseTensor
from dual_part_slat import load_slat_encoder, load_slat_decoder

PRECOMPUTE_ROOT = "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output"
REMESH_DIR = "/mnt/data_ssd/infinigen-sim/vae_remesh"
OUTPUT_DIR = "/mnt/cpfs/yurh/Infinigen-Sim/output/slat_roundtrip"

SAMPLES = [
    "cabinet/3/basic_1/part0.obj",
    "cabinet/3/basic_1/part1.obj",
    "drawer/0/basic_0/part0.obj",
    "faucet/0/basic_0/part0.obj",
    "PhysXNet_PhysXnet/10000/basic_0/part0.obj",
    "PhysXMobility/100015/basic_0/part0.obj",
    "toaster/0/basic_0/part0.obj",
    "microwave/0/basic_0/part0.obj",
]


def get_mesh_path(rel_path):
    """Get remeshed path if exists, else original."""
    import hashlib
    full = os.path.join(PRECOMPUTE_ROOT, rel_path)
    h = hashlib.md5(full.encode()).hexdigest()[:12]
    remeshed = os.path.join(REMESH_DIR, f"{h}.obj")
    if os.path.exists(remeshed):
        return remeshed, True
    return full, False


def mesh_to_slat_input(mesh_path, grid_size=512, device="cuda:0"):
    """Convert mesh to FlexiDualGrid format for SLat Encoder."""
    mesh = trimesh.load(mesh_path, force='mesh', process=True)
    mesh.merge_vertices()
    verts = torch.tensor(mesh.vertices, dtype=torch.float32).cpu()
    faces = torch.tensor(mesh.faces, dtype=torch.long).cpu()

    # mesh_to_flexible_dual_grid: CPU tensors + aabb as list (following TRELLIS 2 convention)
    # Normalize mesh to [-0.5, 0.5] first
    center = (verts.max(0).values + verts.min(0).values) / 2
    scale = (verts.max(0).values - verts.min(0).values).max()
    if scale > 0:
        verts = (verts - center) / scale * 0.9  # fit in [-0.45, 0.45]

    coords, dual_verts, intersected = mesh_to_flexible_dual_grid(
        verts.cpu(), faces.cpu(),
        grid_size=grid_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        face_weight=1.0,
        boundary_weight=0.2,
        regularization_weight=1e-2,
    )

    # CRITICAL: encoder expects relative vertex offset, not absolute position
    # Following TRELLIS 2 convention: feats = dual_vertices * resolution - voxel_indices
    relative_verts = dual_verts * grid_size - coords.float()

    batch_coords = torch.cat([
        torch.zeros(coords.shape[0], 1, dtype=torch.int32),
        coords.int()
    ], dim=1)

    vertices_st = SparseTensor(
        feats=relative_verts.to(device).half(),
        coords=batch_coords.to(device),
    )
    intersected_st = vertices_st.replace(intersected.to(device).half())

    return vertices_st, intersected_st, mesh


def slat_to_mesh(slat, decoder, resolution):
    """Decode SLat to mesh via SLat Decoder."""
    decoder.set_resolution(resolution)
    with torch.inference_mode():
        result = decoder(slat, return_subs=False)
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda:0"

    print("Loading SLat Encoder + Decoder...")
    encoder = load_slat_encoder(device=device)
    decoder = load_slat_decoder(device=device)

    for rel_path in SAMPLES:
        mesh_path, is_remeshed = get_mesh_path(rel_path)
        name = rel_path.replace("/", "_").replace(".obj", "")

        if not os.path.exists(mesh_path):
            print(f"SKIP: {rel_path}")
            continue

        print(f"\n=== {rel_path} (remeshed={is_remeshed}) ===")

        try:
            # Step 1: Mesh → FlexiDualGrid
            t0 = time.time()
            vertices_st, intersected_st, orig_mesh = mesh_to_slat_input(
                mesh_path, grid_size=256, device=device)
            n_voxels = vertices_st.feats.shape[0]
            print(f"  FDG: {n_voxels} voxels, {time.time()-t0:.1f}s")

            # Reduce grid until voxels < max_tokens
            max_tokens = 50000
            for try_grid in [256, 128, 64]:
                if n_voxels <= max_tokens:
                    break
                print(f"  Too many voxels ({n_voxels}), trying grid_size={try_grid}")
                vertices_st, intersected_st, orig_mesh = mesh_to_slat_input(
                    mesh_path, grid_size=try_grid, device=device)
                n_voxels = vertices_st.feats.shape[0]
            print(f"  Final: {n_voxels} voxels (max={max_tokens})")

            # Step 2: Encode → SLat
            t0 = time.time()
            with torch.inference_mode():
                slat = encoder(vertices_st, intersected_st)
            print(f"  Encode: latent [{slat.feats.shape}], {time.time()-t0:.1f}s")

            # Step 3: Decode → Mesh (resolution must match grid_size used in encoding)
            actual_grid = vertices_st.coords[:, 1:].max().item() + 1
            t0 = time.time()
            result = slat_to_mesh(slat, decoder, resolution=actual_grid)
            dt = time.time() - t0

            print(f"  Decode: {dt:.1f}s")

            # Export original
            orig_mesh.export(os.path.join(OUTPUT_DIR, f"{name}_gt.obj"))

            # Extract mesh from decoder output (List[Mesh])
            if isinstance(result, list) and len(result) > 0:
                mesh_obj = result[0]
                if hasattr(mesh_obj, 'vertices') and hasattr(mesh_obj, 'faces'):
                    decoded_mesh = trimesh.Trimesh(
                        mesh_obj.vertices.cpu().numpy(),
                        mesh_obj.faces.cpu().numpy(),
                        process=False)
                    decoded_mesh.export(os.path.join(OUTPUT_DIR, f"{name}_slat.obj"))
                    comps = decoded_mesh.split(only_watertight=False)
                    total_f = sum(len(c.faces) for c in comps)
                    cc = len([c for c in comps if len(c.faces) >= max(total_f*0.01, 10)])
                    print(f"  SLat: v={len(decoded_mesh.vertices)} f={len(decoded_mesh.faces)} CC={cc}")
            else:
                print(f"  Unexpected result: {type(result)}")

            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone! Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
