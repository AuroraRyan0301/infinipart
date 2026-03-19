#!/usr/bin/env python3
"""
Preprocess OBJ meshes -> VAE training tensors (.pt files).

Two-phase pipeline:
  Phase 1: Blender Voxel Remesh non-watertight OBJs (ensures correct occupancy GT)
  Phase 2: meshiki point sampling + FPS + occupancy queries -> .pt

Usage:
  python preprocess_vae_data.py --data_root ./precompute_output \
    --training_data /mnt/data_ssd/infinigen-sim \
    --output_dir /mnt/data_ssd/infinigen-sim/vae_cache \
    --workers 16
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
import trimesh

BLENDER_BIN = "/mnt/data/yurh/blender-4.2.18-linux-x64/blender"


# ================================================================
# Phase 1: Blender Voxel Remesh for non-watertight OBJs
# ================================================================

BLENDER_REMESH_SCRIPT = '''
import bpy, json, os, sys

argv = sys.argv[sys.argv.index("--") + 1:]
manifest = json.loads(argv[0])

def clear():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for b in bpy.data.meshes:
        if not b.users: bpy.data.meshes.remove(b)

for item in manifest:
    src, dst, voxel_size = item["src"], item["dst"], item["voxel_size"]
    try:
        clear()
        before = set(bpy.data.objects.keys())
        try:
            bpy.ops.wm.obj_import(filepath=src, forward_axis='NEGATIVE_Y', up_axis='Z')
        except:
            bpy.ops.import_scene.obj(filepath=src, axis_forward='-Y', axis_up='Z')
        after = set(bpy.data.objects.keys())
        new = [bpy.data.objects[n] for n in after - before]
        for o in new:
            o.rotation_euler = (0, 0, 0)
        if len(new) > 1:
            bpy.ops.object.select_all(action='DESELECT')
            for o in new: o.select_set(True)
            bpy.context.view_layer.objects.active = new[0]
            bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active or new[0]
        bpy.context.view_layer.objects.active = obj

        # Clean
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=0.0001)
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')

        # Voxel Remesh
        mod = obj.modifiers.new(name="Remesh", type='REMESH')
        mod.mode = 'VOXEL'
        mod.voxel_size = voxel_size
        mod.use_smooth_shade = False
        bpy.ops.object.modifier_apply(modifier="Remesh")

        # Export
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.ops.wm.obj_export(filepath=dst, forward_axis='NEGATIVE_Y', up_axis='Z',
                               export_selected_objects=True)
        nv = len(obj.data.vertices)
        nf = len(obj.data.polygons)
        print(f"OK {os.path.basename(src)}: v={nv} f={nf}")
    except Exception as e:
        print(f"FAIL {os.path.basename(src)}: {e}")

print("BLENDER_DONE")
'''


def remesh_non_watertight(obj_paths, remesh_dir, voxel_size=0.01, batch_size=50):
    """Batch remesh non-watertight OBJs via Blender. Returns dict of src->remeshed_path."""
    os.makedirs(remesh_dir, exist_ok=True)

    # Check which need remeshing
    to_remesh = []
    already_wt = 0
    already_done = 0
    for p in obj_paths:
        h = hashlib.md5(p.encode()).hexdigest()[:12]
        dst = os.path.join(remesh_dir, f"{h}.obj")
        if os.path.exists(dst):
            already_done += 1
            continue
        try:
            m = trimesh.load(p, force='mesh', process=False)
            if m.is_watertight:
                already_wt += 1
                continue
        except:
            pass
        to_remesh.append({"src": p, "dst": dst, "voxel_size": voxel_size})

    print(f"  Remesh: {len(to_remesh)} non-watertight, "
          f"{already_wt} already watertight, {already_done} already remeshed")

    if not to_remesh:
        return

    # Process in batches (Blender startup is expensive)
    for i in range(0, len(to_remesh), batch_size):
        batch = to_remesh[i:i + batch_size]
        manifest_json = json.dumps(batch)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(BLENDER_REMESH_SCRIPT)
            script_path = f.name

        try:
            result = subprocess.run(
                [BLENDER_BIN, "--background", "--python", script_path, "--", manifest_json],
                capture_output=True, text=True, timeout=600)
            ok = result.stdout.count("OK ")
            fail = result.stdout.count("FAIL ")
            print(f"  Batch {i//batch_size + 1}: {ok} OK, {fail} FAIL")
            if fail > 0:
                for line in result.stdout.split("\n"):
                    if "FAIL" in line:
                        print(f"    {line}")
        except subprocess.TimeoutExpired:
            print(f"  Batch {i//batch_size + 1}: TIMEOUT")
        finally:
            os.unlink(script_path)


def get_mesh_path(obj_path, remesh_dir):
    """Return remeshed path if exists, otherwise original."""
    h = hashlib.md5(obj_path.encode()).hexdigest()[:12]
    remeshed = os.path.join(remesh_dir, f"{h}.obj")
    if os.path.exists(remeshed):
        return remeshed
    return obj_path


# ================================================================
# Phase 2: Point cloud sampling + occupancy queries -> .pt
# ================================================================

def process_one(args):
    """Process one OBJ -> .pt file."""
    obj_path, remesh_dir, output_dir, n_queries = args

    key = obj_path.encode()
    h = hashlib.md5(key).hexdigest()[:12]
    out_path = os.path.join(output_dir, f"{h}.pt")

    if os.path.exists(out_path):
        return "skip"

    try:
        import fpsample
        import meshiki

        # Use remeshed version if available (watertight)
        mesh_path = get_mesh_path(obj_path, remesh_dir)
        mesh = trimesh.load(mesh_path, force='mesh', process=True)
        mesh.merge_vertices()
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_mesh()
        if len(mesh.vertices) < 3 or len(mesh.faces) < 1:
            return "empty"

        verts = mesh.vertices.astype(np.float32)
        faces = mesh.faces

        # Point cloud sampling
        m = meshiki.Mesh(verts, faces)
        uniform_pts = m.uniform_point_sample(200000)
        uniform_pts = meshiki.fps(uniform_pts, 32768)
        salient_pts = m.salient_point_sample(16384, thresh_bihedral=15)

        # FPS indices
        uniform_pts = np.ascontiguousarray(uniform_pts)
        salient_pts = np.ascontiguousarray(salient_pts)
        n_fps = min(2048, len(uniform_pts))
        n_fps_s = min(2048, len(salient_pts))
        if n_fps < 16 or n_fps_s < 16:
            return "error:too_few_points"
        fps_idx = fpsample.bucket_fps_kdline_sampling(uniform_pts, n_fps, h=5, start_idx=0)
        fps_idx_s = fpsample.bucket_fps_kdline_sampling(salient_pts, n_fps_s, h=5, start_idx=0)
        if len(fps_idx) < 2048:
            fps_idx = np.pad(fps_idx, (0, 2048 - len(fps_idx)), mode='wrap')
        if len(fps_idx_s) < 2048:
            fps_idx_s = np.pad(fps_idx_s, (0, 2048 - len(fps_idx_s)), mode='wrap')

        # Query points
        n_near = n_queries // 2
        n_uniform = n_queries - n_near
        surface_pts, _ = trimesh.sample.sample_surface(mesh, n_near)
        noise = np.random.normal(0, 0.02, size=surface_pts.shape)
        near_pts = np.clip(surface_pts + noise, -1.0, 1.0).astype(np.float32)
        uniform_qpts = np.random.uniform(-1, 1, (n_uniform, 3)).astype(np.float32)
        query_pts = np.concatenate([near_pts, uniform_qpts], axis=0)

        # Occupancy GT
        if mesh.is_watertight:
            # Use mesh.contains for watertight meshes (accurate, but cache heavy)
            # Use KDTree method to avoid memory issues
            from scipy.spatial import cKDTree
            surf_pts, surf_face_idx = trimesh.sample.sample_surface(mesh, 100000)
            surf_normals = mesh.face_normals[surf_face_idx]
            tree = cKDTree(surf_pts)
            _, nn_idx = tree.query(query_pts)
            directions = query_pts - surf_pts[nn_idx]
            dots = np.sum(directions * surf_normals[nn_idx], axis=1)
            inside = dots < 0
            query_gt = np.where(inside, 1.0, -1.0).astype(np.float32)
            del tree, surf_pts, surf_normals, nn_idx, directions, dots
        else:
            # Fallback for non-watertight (shouldn't happen after remesh)
            from scipy.spatial import cKDTree
            surf_pts, surf_face_idx = trimesh.sample.sample_surface(mesh, 100000)
            surf_normals = mesh.face_normals[surf_face_idx]
            tree = cKDTree(surf_pts)
            _, nn_idx = tree.query(query_pts)
            directions = query_pts - surf_pts[nn_idx]
            dots = np.sum(directions * surf_normals[nn_idx], axis=1)
            inside = dots < 0
            query_gt = np.where(inside, 1.0, -1.0).astype(np.float32)
            del tree, surf_pts, surf_normals, nn_idx, directions, dots

        sample = {
            "pointcloud": torch.from_numpy(uniform_pts).float(),
            "fps_indices": torch.from_numpy(fps_idx).long(),
            "pointcloud_dorases": torch.from_numpy(salient_pts).float(),
            "fps_indices_dorases": torch.from_numpy(fps_idx_s).long(),
            "query_points": torch.from_numpy(query_pts).float(),
            "query_gt": torch.from_numpy(query_gt).float(),
            "obj_path": obj_path,
            "mesh_path": mesh_path,
            "is_watertight": bool(mesh.is_watertight),
        }

        torch.save(sample, out_path)
        return "ok"

    except (Exception, AssertionError, SystemError) as e:
        return f"error:{os.path.basename(obj_path)}:{type(e).__name__}:{e}"


# ================================================================
# Discovery
# ================================================================

def discover_objs(data_root, training_data_root=None):
    """Find part OBJ files."""
    precompute_root = os.path.abspath(data_root)

    if training_data_root is None:
        objs = []
        for cat in sorted(os.listdir(precompute_root)):
            cat_dir = os.path.join(precompute_root, cat)
            if not os.path.isdir(cat_dir):
                continue
            for seed in sorted(os.listdir(cat_dir)):
                seed_dir = os.path.join(cat_dir, seed)
                if not os.path.isdir(seed_dir):
                    continue
                for animode in sorted(os.listdir(seed_dir)):
                    anim_dir = os.path.join(seed_dir, animode)
                    for pname in ["part0.obj", "part1.obj"]:
                        obj_path = os.path.join(anim_dir, pname)
                        if os.path.exists(obj_path):
                            objs.append(obj_path)
        return objs

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


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/cpfs/yurh/Infinigen-Sim/precompute_output")
    parser.add_argument("--training_data", default="/mnt/data_ssd/infinigen-sim",
                        help="Only process animodes with gt_latent.pt here")
    parser.add_argument("--output_dir", default="/mnt/data_ssd/infinigen-sim/vae_cache")
    parser.add_argument("--remesh_dir", default="/mnt/data_ssd/infinigen-sim/vae_remesh")
    parser.add_argument("--voxel_size", type=float, default=0.01)
    parser.add_argument("--n_queries", type=int, default=16384)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--skip_remesh", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    td = args.training_data if args.training_data else None
    objs = discover_objs(args.data_root, training_data_root=td)
    print(f"Found {len(objs)} OBJ files")

    # Phase 1: Blender Voxel Remesh non-watertight OBJs
    if not args.skip_remesh:
        print(f"\n=== Phase 1: Voxel Remesh (voxel_size={args.voxel_size}) ===")
        remesh_non_watertight(objs, args.remesh_dir, args.voxel_size)
    else:
        print("Skipping remesh phase")

    # Phase 2: Point cloud + occupancy -> .pt
    print(f"\n=== Phase 2: Point cloud + occupancy ===")
    existing = len([f for f in os.listdir(args.output_dir) if f.endswith(".pt")])
    print(f"Already cached: {existing}")

    n_workers = args.workers if args.workers > 0 else min(cpu_count(), 16)
    print(f"Processing with {n_workers} workers...")

    tasks = [(p, args.remesh_dir, args.output_dir, args.n_queries) for p in objs]

    t0 = time.time()
    counts = {"ok": 0, "skip": 0, "empty": 0, "error": 0}

    with Pool(n_workers, maxtasksperchild=10) as pool:
        for i, result in enumerate(pool.imap_unordered(process_one, tasks, chunksize=1)):
            if result.startswith("error"):
                counts["error"] += 1
                if counts["error"] <= 5:
                    print(f"  ERROR: {result[:300]}")
            else:
                counts[result] += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(tasks)}] ok={counts['ok']} skip={counts['skip']} "
                      f"err={counts['error']} | {rate:.1f}/s ETA {eta/60:.0f}m")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}m")
    print(f"  OK: {counts['ok']}, Skip: {counts['skip']}, "
          f"Empty: {counts['empty']}, Error: {counts['error']}")

    total = len([f for f in os.listdir(args.output_dir) if f.endswith(".pt")])
    print(f"  Total cached: {total}")


if __name__ == "__main__":
    main()
