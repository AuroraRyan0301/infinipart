#!/usr/bin/env python3
"""
Pre-process part0/1.obj: detect thin/open meshes and solidify them.

Detection logic per connected component:
  1. If component is watertight → skip (already has inside/outside)
  2. If component has boundary edges (open surface) → solidify it
  3. Components are processed independently: a mixed OBJ with some watertight
     and some open components will only solidify the open ones.

Output: solidified OBJs in /mnt/data_ssd/infinigen-sim-data/precompute_solidified/
        Same directory structure as precompute/.

Uses Blender Solidify modifier via subprocess for reliability.
Suspicious cases logged for manual review.

Usage:
  python solidify_thin_meshes.py --num_workers 64
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import numpy as np

PRECOMPUTE = "/mnt/data_ssd/infinigen-sim-data/precompute"
OUTPUT = "/mnt/data_ssd/infinigen-sim-data/precompute_solidified"
BLENDER = "/mnt/cpfs/yurh/blender-4.2.18-linux-x64/blender"
THICKNESS = 0.02  # solidify thickness in normalized space [-1,1]

# Categories to process (IS + PhysXMobility, no PhysXNet)
CATEGORIES = None  # auto-detect, excluding PhysXNet


def detect_needs_solidify(obj_path):
    """
    Analyze mesh and determine if solidification is needed.

    Returns:
        action: "skip" | "solidify" | "suspicious"
        reason: explanation string
        stats: dict with mesh statistics
    """
    import trimesh

    try:
        mesh = trimesh.load(obj_path, process=False, force="mesh")
    except Exception as e:
        return "suspicious", f"load_error: {e}", {}

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)

    if n_faces == 0:
        return "suspicious", "empty mesh (0 faces)", {"verts": n_verts, "faces": 0}

    stats = {
        "verts": n_verts,
        "faces": n_faces,
        "watertight": mesh.is_watertight,
    }

    # Split into connected components
    components = mesh.split(only_watertight=False)
    n_components = len(components)
    stats["n_components"] = n_components

    # Check each component
    n_open = 0
    n_closed = 0
    n_tiny = 0
    total_open_faces = 0

    for comp in components:
        if len(comp.faces) < 2:
            n_tiny += 1
            continue
        if comp.is_watertight:
            n_closed += 1
        else:
            # Check boundary edges (open edges appear only once)
            edges = comp.edges_sorted
            unique, counts = np.unique(edges, axis=0, return_counts=True)
            n_boundary = (counts == 1).sum()
            if n_boundary > 0:
                n_open += 1
                total_open_faces += len(comp.faces)

    stats["n_open"] = n_open
    stats["n_closed"] = n_closed
    stats["n_tiny"] = n_tiny
    stats["open_faces"] = total_open_faces

    # Decision
    if n_open == 0 and n_tiny == 0:
        # All components are watertight
        return "skip", "all watertight", stats

    if n_open > 0:
        # Has open (thin) components that need solidifying
        return "solidify", f"{n_open} open components ({total_open_faces} faces)", stats

    if n_tiny > 0 and n_open == 0:
        # Only tiny degenerate components, but all "real" components are watertight
        if n_closed > 0:
            return "skip", f"watertight + {n_tiny} tiny degenerate", stats
        else:
            return "solidify", f"only {n_tiny} tiny degenerate components", stats

    return "suspicious", "unexpected state", stats


BLENDER_SOLIDIFY_SCRIPT = '''
import bpy
import sys
import os

argv = sys.argv[sys.argv.index("--") + 1:]
input_obj = argv[0]
output_obj = argv[1]
thickness = float(argv[2])

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.wm.obj_import(filepath=input_obj)
obj = bpy.context.selected_objects[0]

# Separate open vs closed components:
# Select non-manifold edges, grow selection, solidify only those
# Simpler approach: just solidify the whole mesh - watertight parts
# get thickened too but that's acceptable (adds slight thickness to already-solid parts)
mod = obj.modifiers.new(name='Solidify', type='SOLIDIFY')
mod.thickness = thickness
mod.offset = 0
mod.use_rim = True
mod.use_rim_only = False

bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier='Solidify')

# Recalculate normals
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent(inside=False)
bpy.ops.object.mode_set(mode='OBJECT')

bpy.ops.wm.obj_export(filepath=output_obj, export_selected_objects=True)
'''


def solidify_with_blender(input_obj, output_obj, thickness=THICKNESS):
    """Run Blender solidify on a single OBJ file."""
    script_path = "/tmp/solidify_script.py"
    with open(script_path, "w") as f:
        f.write(BLENDER_SOLIDIFY_SCRIPT)
    cmd = [BLENDER, "--background", "--python", script_path, "--",
           input_obj, output_obj, str(thickness)]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    return result.returncode == 0


def process_one(task_args):
    """Process a single part OBJ. Returns (rel_path, action, reason, stats)."""
    src_path, dst_path, rel_path = task_args

    if os.path.exists(dst_path):
        return rel_path, "exists", "already processed", {}

    action, reason, stats = detect_needs_solidify(src_path)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

    if action == "skip":
        # Copy as-is
        import shutil
        shutil.copy2(src_path, dst_path)
        return rel_path, "copied", reason, stats

    elif action == "solidify":
        ok = solidify_with_blender(src_path, dst_path)
        if ok:
            return rel_path, "solidified", reason, stats
        else:
            # Fallback: copy original
            import shutil
            shutil.copy2(src_path, dst_path)
            return rel_path, "solidify_failed", reason, stats

    else:  # suspicious
        import shutil
        shutil.copy2(src_path, dst_path)
        return rel_path, "suspicious", reason, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute", default=PRECOMPUTE)
    parser.add_argument("--output", default=OUTPUT)
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--thickness", type=float, default=THICKNESS)
    parser.add_argument("--log_dir", default="logs")
    args = parser.parse_args()

    # Discover all part OBJs (IS + PhysXMobility, no PhysXNet)
    tasks = []
    for cat in sorted(os.listdir(args.precompute)):
        if cat == "PhysXNet":
            continue
        cat_dir = os.path.join(args.precompute, cat)
        if not os.path.isdir(cat_dir):
            continue
        for seed in os.listdir(cat_dir):
            seed_dir = os.path.join(cat_dir, seed)
            if not os.path.isdir(seed_dir):
                continue
            # Check if this is direct animode level (PhysXMobility) or has subdirs
            for item in os.listdir(seed_dir):
                item_path = os.path.join(seed_dir, item)
                if os.path.isdir(item_path):
                    # animode directory
                    for part in ["part0.obj", "part1.obj"]:
                        src = os.path.join(item_path, part)
                        if os.path.isfile(src):
                            rel = os.path.relpath(src, args.precompute)
                            dst = os.path.join(args.output, rel)
                            tasks.append((src, dst, rel))
                elif item in ("part0.obj", "part1.obj"):
                    # Direct part file (shouldn't happen but handle it)
                    src = os.path.join(seed_dir, item)
                    rel = os.path.relpath(src, args.precompute)
                    dst = os.path.join(args.output, rel)
                    tasks.append((src, dst, rel))

    # Also copy metadata.json files
    meta_tasks = []
    for cat in sorted(os.listdir(args.precompute)):
        if cat == "PhysXNet":
            continue
        cat_dir = os.path.join(args.precompute, cat)
        if not os.path.isdir(cat_dir):
            continue
        for seed in os.listdir(cat_dir):
            meta_src = os.path.join(cat_dir, seed, "metadata.json")
            if os.path.isfile(meta_src):
                rel = os.path.relpath(meta_src, args.precompute)
                meta_dst = os.path.join(args.output, rel)
                if not os.path.exists(meta_dst):
                    os.makedirs(os.path.dirname(meta_dst), exist_ok=True)
                    import shutil
                    shutil.copy2(meta_src, meta_dst)
                    meta_tasks.append(rel)

    print(f"Found {len(tasks)} part OBJs to process, {len(meta_tasks)} metadata copied")
    print(f"Output: {args.output}")
    print(f"Workers: {args.num_workers}")
    print(f"Thickness: {args.thickness}")

    # Process with multiprocessing
    os.makedirs(args.log_dir, exist_ok=True)
    log_solidified = open(os.path.join(args.log_dir, "solidify_solidified.txt"), "w")
    log_suspicious = open(os.path.join(args.log_dir, "solidify_suspicious.txt"), "w")
    log_failed = open(os.path.join(args.log_dir, "solidify_failed.txt"), "w")

    counters = {"copied": 0, "solidified": 0, "solidify_failed": 0,
                "suspicious": 0, "exists": 0}
    t0 = time.time()

    with Pool(processes=args.num_workers) as pool:
        for i, (rel_path, action, reason, stats) in enumerate(
                pool.imap_unordered(process_one, tasks, chunksize=8)):
            counters[action] = counters.get(action, 0) + 1

            if action == "solidified":
                log_solidified.write(f"{rel_path} | {reason} | {stats}\n")
                log_solidified.flush()
            elif action == "suspicious":
                log_suspicious.write(f"{rel_path} | {reason} | {stats}\n")
                log_suspicious.flush()
            elif action == "solidify_failed":
                log_failed.write(f"{rel_path} | {reason} | {stats}\n")
                log_failed.flush()

            done = i + 1
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                eta = (len(tasks) - done) / max(rate, 0.01)
                print(f"  [{done}/{len(tasks)}] {rate:.1f}/s | "
                      f"copied={counters['copied']} solidified={counters['solidified']} "
                      f"suspicious={counters['suspicious']} failed={counters['solidify_failed']} | "
                      f"ETA={eta/60:.0f}min", flush=True)

    log_solidified.close()
    log_suspicious.close()
    log_failed.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}min | {len(tasks)/max(elapsed,1):.1f}/s")
    print(f"Results: {counters}")
    print(f"Output: {args.output}")
    print(f"Logs: {args.log_dir}/solidify_*.txt")


if __name__ == "__main__":
    main()
