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
import queue
import subprocess
import sys
import tempfile
import threading
import time
from multiprocessing import Pool

# ======================================================================
# Path Configuration — all via environment variables for portability
# ======================================================================

REPO_DIR = os.environ.get("REPO_DIR", os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("DATA_DIR", os.path.dirname(REPO_DIR))
BLENDER_BIN = os.environ.get("BLENDER_BIN",
    os.path.join(DATA_DIR, "blender-4.2.18-linux-x64/blender"))

# Dataset paths
PHYSXNET_BASE = os.environ.get("PHYSXNET_BASE",
    "/mnt/data/fulian/dataset/PhysXNet/version_1")
PHYSXMOB_BASE = os.environ.get("PHYSXMOB_BASE",
    "/mnt/data/fulian/dataset/PhysX_mobility")

# Output directory (on shared filesystem)
OUTPUT_DIR = os.path.join(REPO_DIR, "precompute_output")


def _urdf_has_movable_joints(urdf_path):
    """Fast check (~1ms) whether a URDF has any movable joints.
    Avoids spawning a 5s subprocess just to discover 'no movable joints'."""
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(urdf_path)
        for j in tree.findall("joint"):
            if j.get("type") in ("revolute", "prismatic", "continuous"):
                return True
        return False
    except Exception:
        return True  # if can't parse, let split_precompute handle the error

# Stats directory for event-driven dashboard
STATS_DIR = "/mnt/data_ssd/infinigen-sim/.stats"


def write_stats(filename, data):
    """Atomically write JSON stats file for dashboard consumption."""
    os.makedirs(STATS_DIR, exist_ok=True)
    path = os.path.join(STATS_DIR, filename)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=STATS_DIR, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


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

    # Filter already-setup objects
    todo = []
    for factory, source, obj_id in my_ids:
        out_dir = os.path.join(REPO_DIR, "outputs", factory, obj_id)
        if not os.path.exists(os.path.join(out_dir, "scene.urdf")):
            todo.append((factory, source, obj_id))

    print(f"[Node {rank}/{total}] Phase setup: {len(todo)} new / {len(my_ids)} total objects")

    def _setup_one(item):
        factory, source, obj_id = item
        cmd = [
            sys.executable, os.path.join(REPO_DIR, "setup_physxnet_scene.py"),
            "--id", obj_id, "--factory", factory, "--source", source,
        ]
        subprocess.run(cmd, cwd=REPO_DIR, capture_output=True)

    n_workers = min(32, max(1, os.cpu_count() // 4))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        list(pool.map(_setup_one, todo))

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
    gpu_ids = args._gpu_ids
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
    """Find all metadata_path for objects that need rendering.

    Returns list of metadata paths (one per object, renders all animodes together).
    """
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

            if not args.force:
                # Check if any animode still needs rendering
                try:
                    with open(meta_path) as _mf:
                        splits = json.load(_mf).get("splits", {})
                except (json.JSONDecodeError, IOError):
                    continue
                all_done = True
                for animode_name in splits:
                    animode_dir = os.path.join(seed_path, animode_name)
                    sentinel = os.path.join(animode_dir, "hemi_01_nobg.mp4")
                    if not os.path.exists(sentinel):
                        all_done = False
                        break
                if all_done:
                    continue

            jobs.append(meta_path)

    return jobs


def _run_render_job(args_tuple):
    """Run render for one object (all animodes) on assigned GPU."""
    meta_path, gpu_id, resolution, samples, timeout, views = args_tuple
    seed_dir = os.path.dirname(meta_path)
    factory = os.path.basename(os.path.dirname(seed_dir))
    seed = os.path.basename(seed_dir)
    label = f"{factory}/{seed}"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        BLENDER_BIN, "--background", "--python",
        os.path.join(REPO_DIR, "render_animode.py"), "--",
        "--metadata", meta_path,
        "--animode", "all",
        "--views", views,
        "--color_mode", "both",
        "--bg_mode", "both",
        "--resolution", str(resolution),
        "--samples", str(samples),
        "--skip_existing",
        "--skip_probe",
    ]

    print(f"  [GPU{gpu_id}] RENDER {label}")
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                timeout=timeout, cwd=REPO_DIR)
        if result.returncode == 0:
            print(f"  [GPU{gpu_id}] DONE  {label}")
            return True
        else:
            err = (result.stderr or result.stdout or "")[-500:]
            print(f"  [GPU{gpu_id}] FAIL  {label}: {err}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [GPU{gpu_id}] TIMEOUT {label}")
        return False


def phase_render(args):
    rank, total = get_node_info()

    all_jobs = collect_render_jobs(args)
    my_jobs = shard_list(all_jobs, rank, total)
    print(f"[Node {rank}/{total}] Phase render: {len(my_jobs)} / {len(all_jobs)} objects, {args.n_gpus} GPUs")

    if not my_jobs:
        print(f"[Node {rank}/{total}] Phase render: nothing to do")
        return

    # Assign GPUs round-robin
    gpu_ids = args._gpu_ids
    pool_args = [
        (meta, gpu_ids[i % len(gpu_ids)],
         args.resolution, args.samples, args.timeout, args.views)
        for i, meta in enumerate(my_jobs)
    ]

    with Pool(args.n_gpus) as pool:
        results = pool.map(_run_render_job, pool_args)

    ok = sum(1 for r in results if r)
    fail = sum(1 for r in results if not r)
    print(f"[Node {rank}/{total}] Phase render: {ok} ok, {fail} fail out of {len(my_jobs)}")


# ======================================================================
# Phase "pipeline": Pipelined spawn+precompute (CPU) + render (GPU)
# ======================================================================

def _run_precompute_one(source, factory, seed, base, suffix, args):
    """Run precompute for a single object. Returns metadata path or None."""
    # Output dir includes suffix (e.g. PhysXNet_PhysXnet)
    out_factory = factory + suffix if suffix else factory
    out_check = os.path.join(OUTPUT_DIR, out_factory, seed, "metadata.json")
    if os.path.exists(out_check) and not args.force:
        return out_check  # already done, still needs render check

    cmd = [
        sys.executable, os.path.join(REPO_DIR, "split_precompute.py"),
        "--factory", factory,
        "--seed", seed,
        "--output_dir", OUTPUT_DIR,
        "--base", base if base else REPO_DIR,
    ]
    if suffix:
        cmd.extend(["--suffix", suffix])
    if args.force:
        cmd.append("--force")
    if args.max_basic:
        cmd.extend(["--max_basic", str(args.max_basic)])
    if args.max_senior:
        cmd.extend(["--max_senior", str(args.max_senior)])

    try:
        result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True,
                                timeout=300)  # 5 min timeout for precompute
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT precompute {factory}/{seed} (>300s)")
        return None
    if result.returncode == 0 and os.path.exists(out_check):
        return out_check
    else:
        err = (result.stderr or result.stdout or "")[-200:]
        print(f"  FAIL precompute {factory}/{seed}: {err}")
        return None


def _run_spawn_one(factory, seed, gpu_id=None):
    """Run IS factory spawn. Returns True on success."""
    out_check = os.path.join(REPO_DIR, "sim_exports", "urdf", factory, str(seed))
    if os.path.isdir(out_check):
        return True

    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [
        BLENDER_BIN, "--background", "--python-expr",
        f"import sys; sys.path.insert(0, '{REPO_DIR}'); "
        f"sys.argv = ['spawn_asset', '-n', '{factory}', '-s', '{seed}', '-exp', 'urdf', '-dir', './sim_exports']; "
        f"exec(open('{REPO_DIR}/scripts/spawn_asset.py').read())",
    ]
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                timeout=600, cwd=REPO_DIR)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT spawn {factory}/{seed}")
        return False


def _render_already_complete(meta_path):
    """Fast check: are all animodes already fully rendered?
    Checks if each animode dir has at least one _nobg.mp4 file.
    Returns True if everything is already done (skip Blender startup)."""
    try:
        seed_dir = os.path.dirname(meta_path)
        animode_dirs = [d for d in os.listdir(seed_dir)
                        if os.path.isdir(os.path.join(seed_dir, d))]
        if not animode_dirs:
            return False  # no animodes = needs work
        for ad in animode_dirs:
            ad_path = os.path.join(seed_dir, ad)
            has_nobg = any(f.endswith("_nobg.mp4") for f in os.listdir(ad_path))
            if not has_nobg:
                return False  # this animode needs rendering
        return True
    except Exception:
        return False


def _run_render_object(meta_path, gpu_id, args):
    """Render ALL animodes for one object on a single GPU (uses --animode all)."""
    seed_dir = os.path.dirname(meta_path)
    factory = os.path.basename(os.path.dirname(seed_dir))
    seed = os.path.basename(seed_dir)
    label = f"{factory}/{seed}"

    # Fast skip: if all animodes already have rendered videos, don't start Blender
    if _render_already_complete(meta_path):
        print(f"  [GPU{gpu_id}] SKIP  {label} (already rendered)")
        return True

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        BLENDER_BIN, "--background", "--python",
        os.path.join(REPO_DIR, "render_animode.py"), "--",
        "--metadata", meta_path,
        "--animode", "all",
        "--views", args.views,
        "--color_mode", "both",
        "--bg_mode", "both",
        "--resolution", str(args.resolution),
        "--samples", str(args.samples),
        "--skip_existing",
        "--skip_probe",
    ]

    print(f"  [GPU{gpu_id}] RENDER {label}")
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                timeout=args.timeout, cwd=REPO_DIR)
        if result.returncode == 0:
            print(f"  [GPU{gpu_id}] DONE  {label}")
            return True
        else:
            err = (result.stderr or result.stdout or "")[-500:]
            print(f"  [GPU{gpu_id}] FAIL  {label}: {err}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [GPU{gpu_id}] TIMEOUT {label}")
        return False


def phase_pipeline(args):
    """Pipelined execution: CPU spawn+precompute overlapped with GPU render.

    Architecture:
      - 1 CPU producer thread: spawn (IS only) → precompute → push to render queue
      - N GPU consumer threads: pull from queue → render all animodes per object
      - Since render (~20min/obj) >> precompute (~25s/obj), the queue stays full
        and precompute latency is fully hidden behind GPU rendering.
    """
    rank, total = get_node_info()

    # Collect PhysX objects from SOURCE urdf directories (auto-setup in pipeline)
    all_objects = []
    manifest = load_manifest(args.manifest)

    # PhysXNet: scan source urdf dir for all available objects
    physxnet_urdf_dir = os.path.join(PHYSXNET_BASE, "urdf")
    if os.path.isdir(physxnet_urdf_dir):
        allowed = set(manifest["physxnet_ids"]) if manifest and "physxnet_ids" in manifest else None
        for f in sorted(os.listdir(physxnet_urdf_dir)):
            if f.endswith(".urdf"):
                obj_id = f.replace(".urdf", "")
                if allowed is not None and obj_id not in allowed:
                    continue
                all_objects.append(("PhysXNet", "PhysXNet", obj_id, "", "_PhysXnet"))

    # PhysXMobility: scan source urdf dir
    physxmob_urdf_dir = os.path.join(PHYSXMOB_BASE, "urdf")
    if os.path.isdir(physxmob_urdf_dir):
        allowed = set(manifest["physxmob_ids"]) if manifest and "physxmob_ids" in manifest else None
        for f in sorted(os.listdir(physxmob_urdf_dir)):
            if f.endswith(".urdf"):
                obj_id = f.replace(".urdf", "")
                if allowed is not None and obj_id not in allowed:
                    continue
                all_objects.append(("PhysXMobility", "PhysXMobility", obj_id, "", "_PhysXmobility"))

    # IS factories (need spawn first)
    is_base = os.path.join(REPO_DIR, "sim_exports", "urdf")
    is_seeds = manifest.get("is_seeds", args.is_seeds) if manifest else args.is_seeds
    is_jobs = []
    for factory in IS_FACTORIES:
        for seed in range(is_seeds):
            is_jobs.append((factory, seed))

    my_objects = shard_list(all_objects, rank, total)
    my_is_jobs = shard_list(is_jobs, rank, total)

    total_work = len(my_objects) + len(my_is_jobs)
    print(f"[Node {rank}/{total}] Pipeline: {len(my_objects)} PhysX + {len(my_is_jobs)} IS = {total_work} objects, {args.n_gpus} GPUs")

    if total_work == 0:
        print(f"[Node {rank}/{total}] Pipeline: nothing to do")
        return

    # Shared render queue: CPU pushes metadata paths, GPU workers consume
    render_q = queue.Queue(maxsize=args.n_gpus * args.workers_per_gpu * 4)
    now = time.time()
    stats = {
        "precompute_ok": 0, "precompute_fail": 0, "precompute_skip": 0,
        "render_ok": 0, "render_fail": 0,
        "setup_ok": 0, "setup_fail": 0,
        "start_time": now,
        "total_objects": total_work,
        "total_physx": len(my_objects),
        "total_is": len(my_is_jobs),
        "last_render": None, "last_precompute": None,
        "recent_events": [],  # last 50 events
        # Per-phase timing for ETA
        "precompute_times": [],  # last 100 durations (seconds)
        "render_times": [],     # last 100 durations (seconds)
        # Currently active work items
        "active_renders": {},   # gpu_id -> {label, start_ts}
        "active_precomputes": [],  # [label, ...]
        "phase": "starting",    # starting / precompute / render / done
    }
    stats_lock = threading.Lock()

    def _emit_event(event_type, label, success=True, duration=None):
        """Record an event and flush stats to disk."""
        with stats_lock:
            evt = {"type": event_type, "label": label, "ok": success, "ts": time.time()}
            if duration is not None:
                evt["dur_s"] = round(duration, 1)
            stats["recent_events"].append(evt)
            if len(stats["recent_events"]) > 50:
                stats["recent_events"] = stats["recent_events"][-50:]
            if event_type == "render":
                stats["last_render"] = label
                if duration is not None:
                    stats["render_times"].append(duration)
                    if len(stats["render_times"]) > 100:
                        stats["render_times"] = stats["render_times"][-100:]
            elif event_type == "precompute":
                stats["last_precompute"] = label
                if duration is not None:
                    stats["precompute_times"].append(duration)
                    if len(stats["precompute_times"]) > 100:
                        stats["precompute_times"] = stats["precompute_times"][-100:]

            # Compute ETAs
            snap = {k: v for k, v in stats.items()}
            snap["elapsed_s"] = time.time() - stats["start_time"]
            snap["queue_size"] = render_q.qsize()

            # Precompute ETA
            pc_done = stats["precompute_ok"] + stats["precompute_fail"] + stats["precompute_skip"]
            pc_remain = total_work - pc_done
            if stats["precompute_times"]:
                avg_pc = sum(stats["precompute_times"]) / len(stats["precompute_times"])
                snap["precompute_eta_s"] = round(pc_remain * avg_pc / max(n_cpu_workers, 1))
                snap["precompute_avg_s"] = round(avg_pc, 1)
            snap["precompute_done"] = pc_done

            # Render ETA
            r_done = stats["render_ok"] + stats["render_fail"]
            r_total = stats["precompute_ok"]  # only successfully precomputed get rendered
            r_remain = max(0, r_total - r_done) + render_q.qsize()
            if stats["render_times"]:
                avg_r = sum(stats["render_times"]) / len(stats["render_times"])
                snap["render_eta_s"] = round(r_remain * avg_r / max(args.n_gpus, 1))
                snap["render_avg_s"] = round(avg_r, 1)
            snap["render_done"] = r_done
            snap["render_total"] = r_total

        write_stats("gen.json", snap)

    # Number of parallel CPU workers for spawn+precompute
    n_cpu_workers = min(16, max(1, os.cpu_count() // 8))

    def _process_one_object(item):
        """Process one object: setup (if PhysX) / spawn (if IS) + precompute. Returns metadata path or None."""
        kind = item[0]
        if kind == "physx":
            _, source, factory, seed, base, suffix = item
            # Auto-setup if not already done
            out_dir = os.path.join(REPO_DIR, "outputs", factory, seed)
            urdf_path = os.path.join(out_dir, "scene.urdf")
            if not os.path.exists(urdf_path):
                src_map = {"PhysXNet": "physxnet", "PhysXMobility": "physx_mobility"}
                cmd = [
                    sys.executable, os.path.join(REPO_DIR, "setup_physxnet_scene.py"),
                    "--id", seed, "--factory", factory,
                    "--source", src_map.get(factory, "physxnet"),
                ]
                result = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  FAIL setup {factory}/{seed}: {(result.stderr or '')[-200:]}")
                    return None
                urdf_path = os.path.join(out_dir, "scene.urdf")
            # Fast skip: check URDF for movable joints (~1ms vs ~5s subprocess)
            if not _urdf_has_movable_joints(urdf_path):
                return None
            return _run_precompute_one(source, factory, seed, base, suffix, args)
        else:  # IS
            _, factory, seed = item
            ok = _run_spawn_one(factory, seed)
            if not ok:
                return None
            return _run_precompute_one(
                "IS", factory, str(seed),
                os.path.join(REPO_DIR, "sim_exports", "urdf"), "", args)

    def producer():
        """CPU thread pool: spawn+precompute in parallel → push to render queue."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Build unified work list (PhysX first for fast queue fill, then IS)
        work = []
        for source, factory, seed, base, suffix in my_objects:
            work.append(("physx", source, factory, seed, base, suffix))
        for factory, seed in my_is_jobs:
            work.append(("is", factory, seed))

        with stats_lock:
            stats["phase"] = "precompute"

        # Track start times for duration calculation
        start_times = {}

        def _timed_process(item):
            label = f"{item[2]}/{item[3]}" if item[0] == "physx" else f"{item[1]}/{item[2]}"
            with stats_lock:
                stats["active_precomputes"].append(label)
            t0 = time.time()
            result = _process_one_object(item)
            dur = time.time() - t0
            with stats_lock:
                if label in stats["active_precomputes"]:
                    stats["active_precomputes"].remove(label)
            return result, label, dur

        with ThreadPoolExecutor(max_workers=n_cpu_workers) as pool:
            futures = {pool.submit(_timed_process, item): item for item in work}
            for future in as_completed(futures):
                meta, label, dur = future.result()
                if meta:
                    with stats_lock:
                        stats["precompute_ok"] += 1
                    _emit_event("precompute", label, True, duration=dur)
                    render_q.put(meta)
                else:
                    with stats_lock:
                        stats["precompute_fail"] += 1
                    _emit_event("precompute", label, False, duration=dur)

        # Poison pills for GPU workers (one per worker thread)
        n_render_workers = args.n_gpus * args.workers_per_gpu
        for _ in range(n_render_workers):
            render_q.put(None)

    def gpu_consumer(gpu_id):
        """GPU thread: pull objects from queue, render all animodes."""
        while True:
            meta_path = render_q.get()
            if meta_path is None:
                break
            seed_dir = os.path.dirname(meta_path)
            label = f"{os.path.basename(os.path.dirname(seed_dir))}/{os.path.basename(seed_dir)}"
            with stats_lock:
                stats["active_renders"][str(gpu_id)] = {"label": label, "start_ts": time.time()}
                stats["phase"] = "render"
            t0 = time.time()
            ok = _run_render_object(meta_path, gpu_id, args)
            dur = time.time() - t0
            with stats_lock:
                stats["active_renders"].pop(str(gpu_id), None)
                if ok:
                    stats["render_ok"] += 1
                else:
                    stats["render_fail"] += 1
            _emit_event("render", label, ok, duration=dur)

    # Launch threads
    producer_thread = threading.Thread(target=producer, name="producer")
    producer_thread.start()

    gpu_threads = []
    for gid in args._gpu_ids:
        for w in range(args.workers_per_gpu):
            t = threading.Thread(target=gpu_consumer, args=(gid,),
                                 name=f"gpu_{gid}_w{w}")
            t.start()
            gpu_threads.append(t)

    producer_thread.join()
    for t in gpu_threads:
        t.join()

    # Write final stats
    with stats_lock:
        stats["finished"] = True
        snap = {k: v for k, v in stats.items()}
        snap["elapsed_s"] = time.time() - stats["start_time"]
        snap["queue_size"] = 0
    write_stats("gen.json", snap)

    print(f"\n[Node {rank}/{total}] Pipeline done:")
    print(f"  Precompute: {stats['precompute_ok']} ok, {stats['precompute_fail']} fail")
    print(f"  Render:     {stats['render_ok']} ok, {stats['render_fail']} fail")


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
                        choices=["setup", "spawn", "precompute", "render", "all", "pipeline"],
                        help="Pipeline phase to run (pipeline = pipelined spawn+precompute+render)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Path to subset_manifest.json (limits objects to process)")
    parser.add_argument("--is_seeds", type=int, default=100,
                        help="Number of seeds per IS factory (default: 100)")
    parser.add_argument("--n_gpus", type=int, default=8,
                        help="Number of GPUs per node (default: 8)")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs to use (e.g. '2,3'). "
                             "Overrides --n_gpus.")
    parser.add_argument("--views", type=str, default="sample",
                        help="View set: sample (4+2+2 random/animode), all (16+8+8), fast (4+2+2 fixed)")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=16,
                        help="Cycles samples (default: 16, with OIDN denoiser)")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="Per-job timeout in seconds (default: 1800)")
    parser.add_argument("--workers_per_gpu", type=int, default=1,
                        help="Parallel render workers per GPU (default: 1). "
                             "Increase to fill GPU SM when single Blender underutilizes.")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration even if output exists")
    parser.add_argument("--max_basic", type=int, default=10,
                        help="Cap basic animodes per object (0=unlimited, default: 10)")
    parser.add_argument("--max_senior", type=int, default=5,
                        help="Cap senior animodes per object (0=use split_precompute default, default: 5)")
    args = parser.parse_args()

    # Resolve GPU IDs
    if args.gpu_ids:
        args._gpu_ids = [int(x) for x in args.gpu_ids.split(",")]
        args.n_gpus = len(args._gpu_ids)
    else:
        args._gpu_ids = list(range(args.n_gpus))

    rank, total = get_node_info()
    print(f"=== Node {rank}/{total} | Phase: {args.phase} | GPUs: {args._gpu_ids} ===")
    print(f"  REPO_DIR:    {REPO_DIR}")
    print(f"  DATA_DIR:    {DATA_DIR}")
    print(f"  BLENDER_BIN: {BLENDER_BIN}")
    print(f"  OUTPUT_DIR:  {OUTPUT_DIR}")

    if args.phase in ("all", "pipeline"):
        phase_setup(args)
        phase_pipeline(args)
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
