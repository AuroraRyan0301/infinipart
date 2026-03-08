#!/usr/bin/env python3
"""
Cluster launch script for multi-node GPU cluster (e.g., 100x 8-GPU 4090 nodes).

Each node has N GPUs. The script runs ONE process per node, and dispatches
render jobs across all local GPUs via multiprocessing.

Pipeline:
  Phase 0: setup_physxnet_scene.py (CPU only, fast) — distributed across nodes
  Phase 1: spawn_asset.py (Blender, 1 GPU each) — distributed, multi-GPU per node
  Phase 2: split_precompute.py (CPU only) — distributed across nodes
  Phase 3: render_animode.py (GPU, Blender Cycles) — distributed, multi-GPU per node

Usage:
  # PET / torchrun cluster (auto-detects RANK/WORLD_SIZE or PET_NODE_RANK/PET_NNODES):
  python cluster_launch.py --phase all --is_seeds 100 --n_gpus 8

  # Manual:
  NODE_RANK=0 TOTAL_NODES=100 python cluster_launch.py --phase all --n_gpus 8

Environment variables (auto-detection priority):
  1. SLURM:   SLURM_PROCID / SLURM_NTASKS
  2. PET:     PET_NODE_RANK / PET_NNODES  (node-level, preferred for multi-GPU nodes)
  3. torchrun: RANK / WORLD_SIZE  (with PET_NPROC_PER_NODE to derive node rank)
  4. Manual:  NODE_RANK / TOTAL_NODES

  BLENDER_BIN  — path to Blender binary
  REPO_DIR     — path to Infinigen-Sim repo (default: auto-detect)
  DATA_DIR     — base data directory
"""

import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Pool

# ======================================================================
# Path Configuration — all via environment variables for portability
# ======================================================================

REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(REPO_DIR))
BLENDER_BIN = os.environ.get("BLENDER_BIN",
    os.path.join(DATA_DIR, "blender-4.2.18-linux-x64/blender"))

# Dataset paths (relative to DATA_DIR)
PHYSXNET_BASE = os.path.join(DATA_DIR, "PhysXNet/version_1")
PHYSXMOB_BASE = os.path.join(DATA_DIR, "PhysX_mobility")

# Output directory (on shared filesystem)
OUTPUT_DIR = os.path.join(REPO_DIR, "precompute_output")


def get_node_info():
    """Get node rank and total nodes from env vars.

    For multi-GPU nodes (e.g., 8 GPUs), we need NODE-level rank, not process rank.
    PET_NODE_RANK/PET_NNODES give this directly. If only RANK/WORLD_SIZE are
    available, we derive node rank using PET_NPROC_PER_NODE.
    """
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        total = int(os.environ["SLURM_NTASKS"])
    elif "PET_NODE_RANK" in os.environ and "PET_NNODES" in os.environ:
        # Best: direct node-level rank
        rank = int(os.environ["PET_NODE_RANK"])
        total = int(os.environ["PET_NNODES"])
    elif "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # torchrun: RANK is global, derive node rank
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        nproc = int(os.environ.get("PET_NPROC_PER_NODE", 1))
        rank = global_rank // nproc
        total = world_size // nproc
    else:
        rank = int(os.environ.get("NODE_RANK", 0))
        total = int(os.environ.get("TOTAL_NODES", 1))
    return rank, total


def is_primary_local_process():
    """Return True if this is the primary process on this node.

    For multi-GPU PET clusters, multiple processes may be launched per node.
    Only the primary (local_rank=0) should run our pipeline.
    """
    # If PET_NPROC_PER_NODE > 1, check local rank
    nproc = int(os.environ.get("PET_NPROC_PER_NODE", 1))
    if nproc <= 1:
        return True
    global_rank = int(os.environ.get("RANK", 0))
    local_rank = global_rank % nproc
    return local_rank == 0


def shard_list(items, rank, total):
    """Deterministically shard a sorted list across nodes."""
    return [item for i, item in enumerate(items) if i % total == rank]


def load_manifest(path):
    """Load subset manifest JSON. Returns None if path doesn't exist."""
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ======================================================================
# Phase 0: Setup PhysX scenes (CPU only, fast)
# ======================================================================

def phase_setup(args):
    rank, total = get_node_info()
    manifest = load_manifest(args.manifest)

    physxnet_ids = []
    urdf_dir = os.path.join(PHYSXNET_BASE, "urdf")
    if os.path.isdir(urdf_dir):
        allowed = set(manifest["physxnet_ids"]) if manifest and "physxnet_ids" in manifest else None
        for f in sorted(os.listdir(urdf_dir)):
            if f.endswith(".urdf"):
                obj_id = f.replace(".urdf", "")
                if allowed is not None and obj_id not in allowed:
                    continue
                physxnet_ids.append(("PhysXNet", "physxnet", obj_id))

    physxmob_ids = []
    urdf_dir_mob = os.path.join(PHYSXMOB_BASE, "urdf")
    if os.path.isdir(urdf_dir_mob):
        allowed = set(manifest["physxmob_ids"]) if manifest and "physxmob_ids" in manifest else None
        for f in sorted(os.listdir(urdf_dir_mob)):
            if f.endswith(".urdf"):
                obj_id = f.replace(".urdf", "")
                if allowed is not None and obj_id not in allowed:
                    continue
                physxmob_ids.append(("PhysXMobility", "physx_mobility", obj_id))

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
# Phase 1: IS Factory Spawn (Blender, multi-GPU per node)
# ======================================================================

IS_FACTORIES = [
    "dishwasher", "lamp", "cabinet", "drawer", "oven", "refrigerator",
    "box", "door", "toaster", "faucet", "plier", "window",
    "pepper_grinder", "trash", "door_handle", "stovetop",
    "soap_dispenser", "microwave",
]


def _run_spawn_job(args_tuple):
    """Run a single spawn job on assigned GPU."""
    factory, seed, gpu_id = args_tuple
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        BLENDER_BIN, "--background", "--python-expr",
        f"""
import sys
sys.path.insert(0, '{REPO_DIR}')
sys.argv = ['spawn_asset', '-n', '{factory}', '-s', '{seed}', '-exp', 'urdf', '-dir', './sim_exports']
exec(open('{REPO_DIR}/scripts/spawn_asset.py').read())
""",
    ]
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                timeout=600, cwd=REPO_DIR)
        if result.returncode == 0:
            print(f"  [GPU{gpu_id}] DONE spawn {factory}/{seed}")
            return True
        else:
            err = result.stderr[-300:] if result.stderr else result.stdout[-300:]
            print(f"  [GPU{gpu_id}] FAIL spawn {factory}/{seed}: {err}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [GPU{gpu_id}] TIMEOUT spawn {factory}/{seed}")
        return False


def phase_spawn(args):
    rank, total = get_node_info()
    manifest = load_manifest(args.manifest)
    is_seeds = manifest.get("is_seeds", args.is_seeds) if manifest else args.is_seeds

    spawn_jobs = []
    for factory in IS_FACTORIES:
        for seed in range(is_seeds):
            spawn_jobs.append((factory, seed))

    my_jobs = shard_list(spawn_jobs, rank, total)

    # Filter already done
    todo = []
    for factory, seed in my_jobs:
        out_check = os.path.join(REPO_DIR, "sim_exports", "urdf", factory, str(seed))
        if os.path.isdir(out_check) and not args.force:
            continue
        todo.append((factory, seed))

    print(f"[Node {rank}/{total}] Phase spawn: {len(todo)} todo / {len(my_jobs)} assigned / {len(spawn_jobs)} total")

    if not todo:
        print(f"[Node {rank}/{total}] Phase spawn: nothing to do")
        return

    # Assign GPUs round-robin
    gpu_ids = list(range(args.n_gpus))
    pool_args = [(f, s, gpu_ids[i % len(gpu_ids)]) for i, (f, s) in enumerate(todo)]

    with Pool(args.n_gpus) as pool:
        results = pool.map(_run_spawn_job, pool_args)

    ok = sum(1 for r in results if r)
    fail = sum(1 for r in results if not r)
    print(f"[Node {rank}/{total}] Phase spawn: {ok} ok, {fail} fail")


# ======================================================================
# Phase 2: Precompute (CPU only, per-object, distributed)
# ======================================================================

def collect_all_objects(args):
    """Collect all objects to precompute: IS + PhysXNet + PhysXMobility."""
    manifest = load_manifest(args.manifest)
    objects = []

    # IS factories
    is_base = os.path.join(REPO_DIR, "sim_exports", "urdf")
    if os.path.isdir(is_base):
        is_seeds = manifest.get("is_seeds", args.is_seeds) if manifest else args.is_seeds
        for factory in sorted(os.listdir(is_base)):
            factory_path = os.path.join(is_base, factory)
            if not os.path.isdir(factory_path):
                continue
            for seed_dir in sorted(os.listdir(factory_path)):
                if int(seed_dir) >= is_seeds:
                    continue
                urdf = os.path.join(factory_path, seed_dir, f"{factory}.urdf")
                if os.path.exists(urdf):
                    objects.append(("IS", factory, seed_dir, is_base, ""))

    # PhysXNet
    physxnet_out = os.path.join(REPO_DIR, "outputs", "PhysXNet")
    if os.path.isdir(physxnet_out):
        allowed = set(manifest["physxnet_ids"]) if manifest and "physxnet_ids" in manifest else None
        for obj_id in sorted(os.listdir(physxnet_out)):
            if allowed is not None and obj_id not in allowed:
                continue
            scene = os.path.join(physxnet_out, obj_id, "scene.urdf")
            if os.path.exists(scene):
                objects.append(("PhysXNet", "PhysXNet", obj_id, "", "_PhysXnet"))

    # PhysXMobility
    physxmob_out = os.path.join(REPO_DIR, "outputs", "PhysXMobility")
    if os.path.isdir(physxmob_out):
        allowed = set(manifest["physxmob_ids"]) if manifest and "physxmob_ids" in manifest else None
        for obj_id in sorted(os.listdir(physxmob_out)):
            if allowed is not None and obj_id not in allowed:
                continue
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
            continue

        cmd = [
            sys.executable, os.path.join(REPO_DIR, "split_precompute.py"),
            "--factory", factory,
            "--seed", seed,
            "--output_dir", OUTPUT_DIR,
        ]
        cmd.extend(["--base", base if base else REPO_DIR])
        if suffix:
            cmd.extend(["--suffix", suffix])
        if args.force:
            cmd.append("--force")
        if args.max_basic:
            cmd.extend(["--max_basic", str(args.max_basic)])
        if args.max_senior:
            cmd.extend(["--max_senior", str(args.max_senior)])

        print(f"  Precomputing {factory}/{seed}...")
        subprocess.run(cmd, cwd=REPO_DIR)

    print(f"[Node {rank}/{total}] Phase precompute: done")


# ======================================================================
# Phase 3: Render (multi-GPU per node, distributed by animode)
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

            for entry in sorted(os.listdir(seed_path)):
                animode_dir = os.path.join(seed_path, entry)
                if not os.path.isdir(animode_dir):
                    continue
                if not (entry.startswith("basic_") or entry.startswith("senior_")
                        or entry.startswith("custom_")):
                    continue

                # Check if already rendered (sentinel: first hemi view in fast set)
                sentinel = os.path.join(animode_dir, "hemi_01_nobg.mp4")
                if os.path.exists(sentinel) and not args.force:
                    continue

                jobs.append((meta_path, entry))

    return jobs


def _run_render_job(args_tuple):
    """Run a single render job on assigned GPU."""
    meta_path, animode_name, gpu_id, resolution, samples, timeout, views = args_tuple
    seed_dir = os.path.dirname(meta_path)
    factory = os.path.basename(os.path.dirname(seed_dir))
    seed = os.path.basename(seed_dir)
    label = f"{factory}/{seed}/{animode_name}"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        BLENDER_BIN, "--background", "--python",
        os.path.join(REPO_DIR, "render_animode.py"), "--",
        "--metadata", meta_path,
        "--animode", animode_name,
        "--views", views,
        "--color_mode", "both",
        "--bg_mode", "both",
        "--resolution", str(resolution),
        "--samples", str(samples),
        "--skip_existing",
        "--skip_probe",
    ]

    print(f"  [GPU{gpu_id}] START {label}")
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                timeout=timeout, cwd=REPO_DIR)
        if result.returncode == 0:
            print(f"  [GPU{gpu_id}] DONE  {label}")
            return True
        else:
            err = result.stderr[-500:] if result.stderr else result.stdout[-500:]
            print(f"  [GPU{gpu_id}] FAIL  {label}: {err}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [GPU{gpu_id}] TIMEOUT {label}")
        return False


def phase_render(args):
    rank, total = get_node_info()

    all_jobs = collect_render_jobs(args)
    my_jobs = shard_list(all_jobs, rank, total)
    print(f"[Node {rank}/{total}] Phase render: {len(my_jobs)} / {len(all_jobs)} animode jobs, {args.n_gpus} GPUs")

    if not my_jobs:
        print(f"[Node {rank}/{total}] Phase render: nothing to do")
        return

    # Assign GPUs round-robin
    gpu_ids = list(range(args.n_gpus))
    pool_args = [
        (meta, anim, gpu_ids[i % len(gpu_ids)],
         args.resolution, args.samples, args.timeout, args.views)
        for i, (meta, anim) in enumerate(my_jobs)
    ]

    with Pool(args.n_gpus) as pool:
        results = pool.map(_run_render_job, pool_args)

    ok = sum(1 for r in results if r)
    fail = sum(1 for r in results if not r)
    print(f"[Node {rank}/{total}] Phase render: {ok} ok, {fail} fail out of {len(my_jobs)}")


# ======================================================================
# Main
# ======================================================================

def main():
    # Exit early if not primary local process (multi-GPU PET clusters)
    if not is_primary_local_process():
        local_rank = int(os.environ.get("RANK", 0)) % int(os.environ.get("PET_NPROC_PER_NODE", 1))
        print(f"Skipping non-primary local rank {local_rank}")
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="Cluster launch script for Infinigen-Sim pipeline")
    parser.add_argument("--phase", required=True,
                        choices=["setup", "spawn", "precompute", "render", "all"],
                        help="Pipeline phase to run")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to subset_manifest.json (limits objects to process)")
    parser.add_argument("--is_seeds", type=int, default=100,
                        help="Number of seeds per IS factory (default: 100)")
    parser.add_argument("--n_gpus", type=int, default=8,
                        help="Number of GPUs per node (default: 8)")
    parser.add_argument("--views", type=str, default="fast",
                        help="View set for render: fast (4+2+2), all (16+8+8), hemi, etc.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=16,
                        help="Cycles samples (default: 16, with OIDN denoiser)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-job timeout in seconds (default: 1800)")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if output exists")
    parser.add_argument("--max_basic", type=int, default=10,
                        help="Cap basic animodes per object (0=unlimited, default: 10)")
    parser.add_argument("--max_senior", type=int, default=5,
                        help="Cap senior animodes per object (0=use split_precompute default, default: 5)")
    args = parser.parse_args()

    rank, total = get_node_info()
    print(f"=== Node {rank}/{total} | Phase: {args.phase} | GPUs: {args.n_gpus} ===")
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
