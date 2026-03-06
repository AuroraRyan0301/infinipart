#!/usr/bin/env python3
"""
Cluster launch script for 100-node 4090 cluster.

Each node has 1x RTX 4090, identified by environment variable NODE_RANK (0..N-1)
and TOTAL_NODES. All nodes share a common filesystem (CPFS).

Pipeline:
  Phase 1: setup_physxnet_scene.py (CPU only, fast) — single node
  Phase 2: split_precompute.py (CPU only, per-object) — distributed across nodes
  Phase 3: render_animode.py (GPU, Blender Cycles) — distributed across nodes

Usage:
  # On each node (via SLURM / torchrun / manual):
  NODE_RANK=0 TOTAL_NODES=100 python cluster_launch.py --phase render

  # Or with SLURM:
  srun --nodes=100 --ntasks-per-node=1 python cluster_launch.py --phase all

Environment variables:
  NODE_RANK    — this node's index (0..TOTAL_NODES-1)
  TOTAL_NODES  — total number of nodes
  BLENDER_BIN  — path to Blender binary (default: /mnt/data/yurh/blender-4.2.18-linux-x64/blender)
  REPO_DIR     — path to Infinigen-Sim repo (default: auto-detect)
  DATA_DIR     — base data directory (default: /mnt/data)

  # SLURM auto-detection (if SLURM_PROCID / SLURM_NTASKS set, overrides NODE_RANK / TOTAL_NODES)
"""

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

# ======================================================================
# Path Configuration — all via environment variables for portability
# ======================================================================

REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", "/mnt/data")
BLENDER_BIN = os.environ.get("BLENDER_BIN",
    os.path.join(DATA_DIR, "yurh/blender-4.2.18-linux-x64/blender"))

# Dataset paths (relative to DATA_DIR)
PHYSXNET_BASE = os.path.join(DATA_DIR, "fulian/dataset/PhysXNet/version_1")
PHYSXMOB_BASE = os.path.join(DATA_DIR, "fulian/dataset/PhysX_mobility")

# Output directory (on shared filesystem)
OUTPUT_DIR = os.path.join(REPO_DIR, "precompute_output")


def get_node_info():
    """Get node rank and total nodes from env vars (SLURM or manual)."""
    # SLURM auto-detection
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        total = int(os.environ["SLURM_NTASKS"])
    else:
        rank = int(os.environ.get("NODE_RANK", 0))
        total = int(os.environ.get("TOTAL_NODES", 1))
    return rank, total


def shard_list(items, rank, total):
    """Deterministically shard a sorted list across nodes."""
    return [item for i, item in enumerate(items) if i % total == rank]


# ======================================================================
# Phase 0: Setup PhysX scenes (CPU only, fast, run on single node)
# ======================================================================

def phase_setup(args):
    rank, total = get_node_info()

    # Collect all PhysXNet IDs
    physxnet_ids = []
    urdf_dir = os.path.join(PHYSXNET_BASE, "urdf")
    if os.path.isdir(urdf_dir):
        for f in sorted(os.listdir(urdf_dir)):
            if f.endswith(".urdf"):
                physxnet_ids.append(("PhysXNet", "physxnet", f.replace(".urdf", "")))

    # Collect all PhysXMobility IDs
    physxmob_ids = []
    urdf_dir_mob = os.path.join(PHYSXMOB_BASE, "urdf")
    if os.path.isdir(urdf_dir_mob):
        for f in sorted(os.listdir(urdf_dir_mob)):
            if f.endswith(".urdf"):
                physxmob_ids.append(("PhysXMobility", "physx_mobility", f.replace(".urdf", "")))

    all_ids = physxnet_ids + physxmob_ids
    my_ids = shard_list(all_ids, rank, total)

    print(f"[Node {rank}/{total}] Phase setup: {len(my_ids)} / {len(all_ids)} objects")

    for factory, source, obj_id in my_ids:
        out_dir = os.path.join(REPO_DIR, "outputs", factory, obj_id)
        if os.path.exists(os.path.join(out_dir, "scene.urdf")):
            continue
        cmd = [
            sys.executable, os.path.join(REPO_DIR, "setup_physxnet_scene.py"),
            "--id", obj_id, "--factory", factory, "--source", source,
        ]
        subprocess.run(cmd, cwd=REPO_DIR)

    print(f"[Node {rank}/{total}] Phase setup: done")


# ======================================================================
# Phase 1: IS Factory Spawn (Blender, 1 GPU each, distributed)
# ======================================================================

IS_FACTORIES = [
    "dishwasher", "lamp", "cabinet", "drawer", "oven", "refrigerator",
    "box", "door", "toaster", "faucet", "plier", "window",
    "pepper_grinder", "trash", "door_handle", "stovetop",
    "soap_dispenser", "microwave",
]

def phase_spawn(args):
    rank, total = get_node_info()

    # Generate (factory, seed) pairs
    spawn_jobs = []
    for factory in IS_FACTORIES:
        for seed in range(args.is_seeds):
            spawn_jobs.append((factory, seed))

    my_jobs = shard_list(spawn_jobs, rank, total)
    print(f"[Node {rank}/{total}] Phase spawn: {len(my_jobs)} / {len(spawn_jobs)} IS jobs")

    for factory, seed in my_jobs:
        out_check = os.path.join(REPO_DIR, "sim_exports", "urdf", factory, str(seed))
        if os.path.isdir(out_check) and not args.force:
            print(f"  SKIP {factory}/{seed} (exists)")
            continue

        cmd = [
            BLENDER_BIN, "--background", "--python-expr",
            f"""
import sys
sys.path.insert(0, '{REPO_DIR}')
sys.argv = ['spawn_asset', '-n', '{factory}', '-s', '{seed}', '-exp', 'urdf', '-dir', './sim_exports']
exec(open('{REPO_DIR}/scripts/spawn_asset.py').read())
""",
        ]
        print(f"  Spawning {factory}/{seed}...")
        subprocess.run(cmd, cwd=REPO_DIR)

    print(f"[Node {rank}/{total}] Phase spawn: done")


# ======================================================================
# Phase 2: Precompute (CPU only, per-object, distributed)
# ======================================================================

def collect_all_objects(args):
    """Collect all objects to precompute: IS + PhysXNet + PhysXMobility."""
    objects = []

    # IS factories
    is_base = os.path.join(REPO_DIR, "sim_exports", "urdf")
    if os.path.isdir(is_base):
        for factory in sorted(os.listdir(is_base)):
            factory_path = os.path.join(is_base, factory)
            if not os.path.isdir(factory_path):
                continue
            for seed_dir in sorted(os.listdir(factory_path)):
                urdf = os.path.join(factory_path, seed_dir, f"{factory}.urdf")
                if os.path.exists(urdf):
                    objects.append(("IS", factory, seed_dir, is_base, ""))

    # PhysXNet
    physxnet_out = os.path.join(REPO_DIR, "outputs", "PhysXNet")
    if os.path.isdir(physxnet_out):
        for obj_id in sorted(os.listdir(physxnet_out)):
            scene = os.path.join(physxnet_out, obj_id, "scene.urdf")
            if os.path.exists(scene):
                objects.append(("PhysXNet", "PhysXNet", obj_id, "", "_PhysXnet"))

    # PhysXMobility
    physxmob_out = os.path.join(REPO_DIR, "outputs", "PhysXMobility")
    if os.path.isdir(physxmob_out):
        for obj_id in sorted(os.listdir(physxmob_out)):
            scene = os.path.join(physxmob_out, obj_id, "scene.urdf")
            if os.path.exists(scene):
                objects.append(("PhysXMobility", "PhysXMobility", obj_id, "", "_PhysXmobility"))

    return objects


def phase_precompute(args):
    rank, total = get_node_info()

    all_objects = collect_all_objects(args)
    my_objects = shard_list(all_objects, rank, total)
    print(f"[Node {rank}/{total}] Phase precompute: {len(my_objects)} / {len(all_objects)} objects")

    for source, factory, seed, base, suffix in my_objects:
        out_check = os.path.join(OUTPUT_DIR, factory, seed, "metadata.json")
        if os.path.exists(out_check) and not args.force:
            print(f"  SKIP {factory}/{seed} (exists)")
            continue

        cmd = [
            sys.executable, os.path.join(REPO_DIR, "split_precompute.py"),
            "--factory", factory,
            "--seed", seed,
            "--output_dir", OUTPUT_DIR,
        ]
        if base:
            cmd.extend(["--base", base])
        if suffix:
            cmd.extend(["--suffix", suffix])
        if args.force:
            cmd.append("--force")

        print(f"  Precomputing {factory}/{seed}...")
        subprocess.run(cmd, cwd=REPO_DIR)

    print(f"[Node {rank}/{total}] Phase precompute: done")


# ======================================================================
# Phase 3: Render (1 GPU per node, distributed by animode)
# ======================================================================

def collect_render_jobs(args):
    """Find all (metadata_path, animode_name) pairs to render."""
    jobs = []
    if not os.path.isdir(OUTPUT_DIR):
        return jobs

    for factory_dir in sorted(os.listdir(OUTPUT_DIR)):
        factory_path = os.path.join(OUTPUT_DIR, factory_dir)
        if not os.path.isdir(factory_path):
            continue
        for seed_dir in sorted(os.listdir(factory_path)):
            seed_path = os.path.join(factory_path, seed_dir)
            meta_path = os.path.join(seed_path, "metadata.json")
            if not os.path.exists(meta_path):
                continue

            # List animode directories (basic_*, senior_*, custom_*)
            for entry in sorted(os.listdir(seed_path)):
                animode_dir = os.path.join(seed_path, entry)
                if not os.path.isdir(animode_dir):
                    continue
                if not (entry.startswith("basic_") or entry.startswith("senior_")
                        or entry.startswith("custom_")):
                    continue

                # Check if already rendered (sentinel: first hemi view)
                sentinel = os.path.join(animode_dir, "hemi_00_nobg.mp4")
                if os.path.exists(sentinel) and not args.force:
                    continue

                jobs.append((meta_path, entry))

    return jobs


def phase_render(args):
    rank, total = get_node_info()

    all_jobs = collect_render_jobs(args)
    my_jobs = shard_list(all_jobs, rank, total)
    print(f"[Node {rank}/{total}] Phase render: {len(my_jobs)} / {len(all_jobs)} animode jobs")

    ok, fail = 0, 0
    for meta_path, animode_name in my_jobs:
        seed_dir = os.path.dirname(meta_path)
        factory = os.path.basename(os.path.dirname(seed_dir))
        seed = os.path.basename(seed_dir)
        label = f"{factory}/{seed}/{animode_name}"

        cmd = [
            BLENDER_BIN, "--background", "--python",
            os.path.join(REPO_DIR, "render_animode.py"), "--",
            "--metadata", meta_path,
            "--animode", animode_name,
            "--views", "all",
            "--color_mode", "both",
            "--bg_mode", "both",
            "--resolution", str(args.resolution),
            "--samples", str(args.samples),
            "--skip_existing",
        ]

        print(f"  [{rank}] START {label}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=args.timeout, cwd=REPO_DIR)
            if result.returncode == 0:
                print(f"  [{rank}] DONE  {label}")
                ok += 1
            else:
                err = result.stderr[-500:] if result.stderr else result.stdout[-500:]
                print(f"  [{rank}] FAIL  {label}: {err}")
                fail += 1
        except subprocess.TimeoutExpired:
            print(f"  [{rank}] TIMEOUT {label}")
            fail += 1

    print(f"[Node {rank}/{total}] Phase render: {ok} ok, {fail} fail")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cluster launch script for Infinigen-Sim pipeline")
    parser.add_argument("--phase", required=True,
                        choices=["setup", "spawn", "precompute", "render", "all"],
                        help="Pipeline phase to run")
    parser.add_argument("--is_seeds", type=int, default=100,
                        help="Number of seeds per IS factory (default: 100)")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-job timeout in seconds (default: 1800)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if output exists")
    args = parser.parse_args()

    rank, total = get_node_info()
    print(f"=== Node {rank}/{total} | Phase: {args.phase} ===")
    print(f"  REPO_DIR:    {REPO_DIR}")
    print(f"  DATA_DIR:    {DATA_DIR}")
    print(f"  BLENDER_BIN: {BLENDER_BIN}")
    print(f"  OUTPUT_DIR:  {OUTPUT_DIR}")

    if args.phase == "all":
        phase_setup(args)
        phase_spawn(args)
        phase_precompute(args)
        phase_render(args)
    elif args.phase == "setup":
        phase_setup(args)
    elif args.phase == "spawn":
        phase_spawn(args)
    elif args.phase == "precompute":
        phase_precompute(args)
    elif args.phase == "render":
        phase_render(args)


if __name__ == "__main__":
    main()
