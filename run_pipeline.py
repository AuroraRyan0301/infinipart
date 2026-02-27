#!/usr/bin/env python3
"""
Unified 3-stage pipeline for articulated object rendering.

Stage 1: Asset generation + split precompute + verify
  - IM factories (50 seeds each, Blender-generated)
  - PhysXNet objects (all, seed=object_id)
  - PhysXMobility objects (all, seed=object_id)

Stage 2: Positive sample rendering (correct articulation)
  - 32 views: 16 hemi + 8 orbit + 8 sweep
  - Multiple animodes per factory

Stage 3: Negative sample rendering (wrong articulation)
  - 6 neg types: wrong_joint_type, wrong_axis, wrong_direction,
    over_motion, wrong_parts_moving, jitter
  - Same 32 views

Usage:
  # Generate all assets
  python run_pipeline.py --stage generate --source all --n_seeds 50

  # Generate only PhysXNet
  python run_pipeline.py --stage generate --source physxnet

  # Render positive samples (4 GPUs)
  python run_pipeline.py --stage render_positive --source all --n_gpus 4

  # Render negative samples (4 GPUs)
  python run_pipeline.py --stage render_negative --source all --n_gpus 4

  # Full pipeline
  python run_pipeline.py --stage all --source all --n_gpus 4
"""

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# ── Paths ──
BASE_DIR = "/mnt/data/yurh/Infinigen-Sim"
IM_BASE = "/mnt/data/yurh/Infinite-Mobility"
BLENDER = "/mnt/data/yurh/blender-3.6.0-linux-x64/blender"
PYTHONPATH = f"{BASE_DIR}:/mnt/data/yurh/infinigen"

# ── Views ──
HEMI_VIEWS = [f"hemi_{i:02d}" for i in range(16)]
ORBIT_VIEWS = [f"orbit_{i:02d}" for i in range(8)]
SWEEP_VIEWS = [f"sweep_{i:02d}" for i in range(8)]
ALL_MOVING = ORBIT_VIEWS + SWEEP_VIEWS
N_EXPECTED = len(HEMI_VIEWS) + len(ALL_MOVING)  # 32

# ── IM factories (original + sim_objects) ──
IM_FACTORIES = [
    "BeverageFridgeFactory", "MicrowaveFactory", "OvenFactory",
    "ToiletFactory", "KitchenCabinetFactory", "WindowFactory",
    "LiteDoorFactory", "OfficeChairFactory", "TapFactory",
    "LampFactory", "PotFactory", "BottleFactory", "DishwasherFactory",
    "BarChairFactory", "PanFactory", "TVFactory",
    # Infinigen-Sim sim_objects
    "SimDoorFactory", "DoorHandleFactory", "DrawerFactory",
    "BoxFactory", "CabinetFactory", "RefrigeratorFactory",
    "FaucetFactory", "StovetopFactory", "ToasterFactory",
    "PepperGrinderFactory", "PlierFactory", "SoapDispenserFactory",
    "TrashFactory",
]

FACTORY_ANIMODES = {
    "DishwasherFactory": 2, "BeverageFridgeFactory": 2,
    "MicrowaveFactory": 2, "OvenFactory": 2,
    "KitchenCabinetFactory": 2, "ToiletFactory": 3,
    "WindowFactory": 4, "LiteDoorFactory": 0,
    "OfficeChairFactory": 2, "TapFactory": 2,
    "LampFactory": 3, "PotFactory": 4,
    "BottleFactory": 3, "BarChairFactory": 2,
    "PanFactory": 0, "TVFactory": 2,
    "SimDoorFactory": 0, "DoorHandleFactory": 2,
    "DrawerFactory": 0, "BoxFactory": 2,
    "CabinetFactory": 2, "RefrigeratorFactory": 4,
    "FaucetFactory": 2, "StovetopFactory": 0,
    "ToasterFactory": 2, "PepperGrinderFactory": 0,
    "PlierFactory": 0, "SoapDispenserFactory": 2,
    "TrashFactory": 4,
}

# Load PartNet and PhysXNet factory lists dynamically
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

try:
    from partnet_factory_rules import PARTNET_FACTORY_LIST, PARTNET_ANIMODES
    IM_FACTORIES.extend([f for f in PARTNET_FACTORY_LIST if f not in IM_FACTORIES])
    FACTORY_ANIMODES.update(PARTNET_ANIMODES)
except ImportError:
    pass

PHYSXNET_FACTORIES = []
PHYSXMOB_FACTORIES = []
try:
    from physxnet_factory_rules import (
        get_all_physxnet_factories, get_all_physxmob_factories,
        get_physxnet_factory_ids, get_physxmob_factory_ids,
        ALL_ANIMODES as PHYSXNET_ALL_ANIMODES,
    )
    PHYSXNET_FACTORIES = list(get_all_physxnet_factories())
    PHYSXMOB_FACTORIES = list(get_all_physxmob_factories())
    FACTORY_ANIMODES.update(PHYSXNET_ALL_ANIMODES)
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def get_base_dir(factory):
    """Return asset base dir for a factory."""
    if "PhysXNet" in factory or "PhysXMobility" in factory:
        return BASE_DIR
    return IM_BASE


def count_videos(out_dir, animode):
    """Count rendered nobg videos for a specific animode."""
    if not os.path.isdir(out_dir):
        return 0
    suffix = f"_anim{animode}" if animode > 0 else ""
    count = 0
    for f in os.listdir(out_dir):
        if not f.endswith("_nobg.mp4"):
            continue
        if animode == 0 and "_anim" not in f:
            count += 1
        elif animode > 0 and f.endswith(f"{suffix}_nobg.mp4"):
            count += 1
    return count


# ══════════════════════════════════════════════════════════════
# Stage 1: Asset Generation
# ══════════════════════════════════════════════════════════════

def generate_im_asset(factory, seed):
    """Generate one IM factory/seed via Blender."""
    output_root = os.path.join(IM_BASE, "outputs")
    seed_dir = os.path.join(output_root, factory, str(seed))
    origins = os.path.join(seed_dir, "origins.json")
    urdf = os.path.join(seed_dir, "scene.urdf")

    if os.path.exists(origins) and os.path.exists(urdf):
        return (factory, seed, True, "exists")

    cmd = [
        BLENDER, "--background", "--python-use-system-env",
        "--python", "infinigen_examples/generate_individual_assets.py",
        "--", "--output_folder", f"outputs/{factory}",
        "-f", factory, "-n", "1", "--seed", str(seed),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH

    try:
        subprocess.run(cmd, cwd=BASE_DIR, env=env,
                       capture_output=True, text=True, timeout=1800)
        if os.path.exists(origins):
            return (factory, seed, True, "ok")
        return (factory, seed, False, "no output")
    except subprocess.TimeoutExpired:
        return (factory, seed, False, "timeout")
    except Exception as e:
        return (factory, seed, False, str(e))


def generate_physxnet_asset(factory, obj_id):
    """Prepare one PhysXNet/PhysXMobility scene (URDF + OBJ symlinks)."""
    try:
        from physxnet_loader import prepare_physxnet_scene
        info = prepare_physxnet_scene(factory, obj_id, os.path.join(BASE_DIR, "outputs"))
        if os.path.exists(info["urdf_path"]):
            return (factory, obj_id, True, "ok")
        return (factory, obj_id, False, "no URDF")
    except Exception as e:
        return (factory, obj_id, False, str(e))


def run_split_precompute(factory, seed, source):
    """Run split_precompute.py for one object."""
    base = get_base_dir(factory)
    cmd = [
        sys.executable, "split_precompute.py",
        "--factory", factory, "--seed", str(seed),
        "--source", source, "--base", base,
        "--output_dir", os.path.join(BASE_DIR, "precompute"),
    ]
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR,
                                capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            return (factory, seed, True, "ok")
        err = result.stderr.strip().split("\n")[-1:]
        return (factory, seed, False, f"rc={result.returncode}: {' '.join(err)}")
    except subprocess.TimeoutExpired:
        return (factory, seed, False, "timeout")
    except Exception as e:
        return (factory, seed, False, str(e))


def run_verify(factory, seed):
    """Run verify_split.py for one object."""
    cmd = [
        sys.executable, "verify_split.py",
        "--factory", factory, "--seed", str(seed),
        "--precompute_dir", os.path.join(BASE_DIR, "precompute"),
    ]
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR,
                                capture_output=True, text=True, timeout=120)
        return (factory, seed, result.returncode == 0, "ok" if result.returncode == 0 else "fail")
    except Exception as e:
        return (factory, seed, False, str(e))


def build_generate_jobs(source, n_seeds, seed_start):
    """Build list of (factory, seed, source_type) jobs."""
    jobs = []

    if source in ("im", "all"):
        for factory in IM_FACTORIES:
            for seed in range(seed_start, seed_start + n_seeds):
                jobs.append((factory, seed, "im"))

    if source in ("physxnet", "all"):
        for factory in PHYSXNET_FACTORIES:
            try:
                ids = get_physxnet_factory_ids(factory)
                for obj_id in ids:
                    jobs.append((factory, obj_id, "physxnet"))
            except Exception:
                pass

    if source in ("physxmob", "all"):
        for factory in PHYSXMOB_FACTORIES:
            try:
                ids = get_physxmob_factory_ids(factory)
                for obj_id in ids:
                    jobs.append((factory, obj_id, "physxmob"))
            except Exception:
                pass

    return jobs


def stage_generate(args):
    """Stage 1: Generate assets, run split precompute, verify."""
    jobs = build_generate_jobs(args.source, args.n_seeds, args.seed_start)
    print(f"\n{'='*60}")
    print(f"Stage 1: Asset Generation")
    print(f"{'='*60}")

    # Count by source
    counts = defaultdict(int)
    for _, _, src in jobs:
        counts[src] += 1
    for src, n in sorted(counts.items()):
        print(f"  {src}: {n} objects")
    print(f"  Total: {len(jobs)} objects")

    if args.shard:
        k, n = map(int, args.shard.split("/"))
        shard_size = (len(jobs) + n - 1) // n
        jobs = jobs[k * shard_size:(k + 1) * shard_size]
        print(f"  Shard {k}/{n}: {len(jobs)} objects")

    if args.dry_run:
        print("  [DRY RUN] Would generate above objects")
        return

    # Phase 1a: Generate assets
    print(f"\n--- Phase 1a: Generating assets ({args.n_workers} workers) ---")
    success, failed, skipped = 0, 0, 0
    t0 = time.time()

    for i, (factory, seed, src) in enumerate(jobs):
        if src == "im":
            f, s, ok, msg = generate_im_asset(factory, seed)
        else:
            f, s, ok, msg = generate_physxnet_asset(factory, seed)

        if "exists" in msg:
            skipped += 1
        elif ok:
            success += 1
        else:
            failed += 1

        if (i + 1) % 100 == 0 or not ok:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(jobs)}] {factory}/{seed} ({src}): {msg} "
                  f"({elapsed:.0f}s)")

    print(f"  Generate done: {success} ok, {skipped} skip, {failed} fail "
          f"({time.time()-t0:.0f}s)")

    # Phase 1b: Split precompute
    if not args.skip_split:
        print(f"\n--- Phase 1b: Split precompute ---")
        t0 = time.time()
        split_ok, split_fail = 0, 0
        for i, (factory, seed, src) in enumerate(jobs):
            f, s, ok, msg = run_split_precompute(factory, seed, src)
            if ok:
                split_ok += 1
            else:
                split_fail += 1
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(jobs)}] splits done ({time.time()-t0:.0f}s)")
        print(f"  Split done: {split_ok} ok, {split_fail} fail ({time.time()-t0:.0f}s)")

    # Phase 1c: Verify
    if not args.skip_verify:
        print(f"\n--- Phase 1c: Verify ---")
        t0 = time.time()
        verify_ok = 0
        for i, (factory, seed, src) in enumerate(jobs):
            f, s, ok, msg = run_verify(factory, seed)
            if ok:
                verify_ok += 1
        print(f"  Verify done: {verify_ok}/{len(jobs)} ({time.time()-t0:.0f}s)")


# ══════════════════════════════════════════════════════════════
# Stage 2: Positive Rendering
# ══════════════════════════════════════════════════════════════

_worker_gpu = None

def _init_gpu(gpu_queue):
    global _worker_gpu
    _worker_gpu = gpu_queue.get()


def render_positive_one(job):
    """Render one (factory, seed, animode) positive sample."""
    factory, seed, animode, out_dir, base_dir = job
    gpu_id = _worker_gpu

    n_existing = count_videos(out_dir, animode)
    if n_existing >= N_EXPECTED:
        return (factory, seed, animode, True, f"skip ({n_existing})")

    cmd = [
        BLENDER, "--background", "--python-use-system-env",
        "--python", "render_articulation.py", "--",
        "--factory", factory, "--seed", str(seed), "--device", "0",
        "--output_dir", out_dir, "--base", base_dir,
        "--resolution", "512", "--samples", "32",
        "--duration", "4.0", "--fps", "30",
        "--animode", str(animode),
        "--skip_bg",
        "--views", *HEMI_VIEWS,
        "--moving_views", *ALL_MOVING,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, env=env,
                                capture_output=True, text=True, timeout=7200)
        dur = time.time() - t0
        n_new = count_videos(out_dir, animode)
        if n_new >= N_EXPECTED:
            return (factory, seed, animode, True, f"ok ({n_new}v, {dur:.0f}s)")
        return (factory, seed, animode, False, f"partial ({n_new}/{N_EXPECTED})")
    except subprocess.TimeoutExpired:
        return (factory, seed, animode, False, "timeout")
    except Exception as e:
        return (factory, seed, animode, False, str(e))


def _enumerate_seeds(source, precompute_dir=None):
    """Enumerate (factory, seed, base_dir) from precompute or output dirs."""
    seeds = []

    if precompute_dir and os.path.isdir(precompute_dir):
        for factory in sorted(os.listdir(precompute_dir)):
            fdir = os.path.join(precompute_dir, factory)
            if not os.path.isdir(fdir):
                continue
            base = get_base_dir(factory)
            for ident in sorted(os.listdir(fdir)):
                meta = os.path.join(fdir, ident, "metadata.json")
                if not os.path.isfile(meta):
                    continue
                seed = int(ident) if ident.isdigit() else ident
                scene_dir = os.path.join(base, "outputs", factory, str(seed))
                if not os.path.exists(os.path.join(scene_dir, "origins.json")):
                    continue
                seeds.append((factory, seed, base))
    else:
        all_factories = []
        if source in ("im", "all"):
            all_factories.extend([(f, "im") for f in IM_FACTORIES])
        if source in ("physxnet", "all"):
            all_factories.extend([(f, "physxnet") for f in PHYSXNET_FACTORIES])
        if source in ("physxmob", "all"):
            all_factories.extend([(f, "physxmob") for f in PHYSXMOB_FACTORIES])

        for factory, src in all_factories:
            base = get_base_dir(factory)
            outputs_dir = os.path.join(base, "outputs", factory)
            if not os.path.isdir(outputs_dir):
                continue
            for ident in sorted(os.listdir(outputs_dir)):
                if not ident.replace("-", "").isdigit():
                    continue
                scene = os.path.join(outputs_dir, ident, "origins.json")
                if not os.path.exists(scene):
                    continue
                seed = int(ident) if ident.isdigit() else ident
                seeds.append((factory, seed, base))

    return seeds


def build_render_jobs(source, precompute_dir=None):
    """Build positive render job list: one job per (factory, seed, animode)."""
    jobs = []
    for factory, seed, base in _enumerate_seeds(source, precompute_dir):
        max_anim = FACTORY_ANIMODES.get(factory, 0)
        out_dir = os.path.join(BASE_DIR, "render_output", "positive",
                               factory, str(seed))
        for animode in range(max_anim + 1):
            jobs.append((factory, seed, animode, out_dir, base))
    return jobs


def build_negative_jobs(source, precompute_dir=None):
    """Build negative render job list: one job per (factory, seed).

    Negative samples iterate each movable joint internally, producing
    6 neg types per joint. No animode expansion needed here.
    """
    jobs = []
    for factory, seed, base in _enumerate_seeds(source, precompute_dir):
        out_dir = os.path.join(BASE_DIR, "render_output", "positive",
                               factory, str(seed))
        jobs.append((factory, seed, out_dir, base))
    return jobs


def stage_render_positive(args):
    """Stage 2: Render positive (correct) articulation samples."""
    precompute = os.path.join(BASE_DIR, "precompute")
    jobs = build_render_jobs(args.source, precompute if os.path.isdir(precompute) else None)

    if args.shard:
        k, n = map(int, args.shard.split("/"))
        shard_size = (len(jobs) + n - 1) // n
        jobs = jobs[k * shard_size:(k + 1) * shard_size]

    print(f"\n{'='*60}")
    print(f"Stage 2: Positive Rendering")
    print(f"{'='*60}")
    print(f"  Total jobs: {len(jobs)}")
    print(f"  Views per job: {N_EXPECTED}")
    print(f"  GPUs: {args.n_gpus}")

    # Count by factory
    fac_counts = defaultdict(int)
    for j in jobs:
        fac_counts[j[0]] += 1
    for f in sorted(fac_counts):
        print(f"  {f}: {fac_counts[f]} jobs")

    if args.dry_run:
        print("  [DRY RUN]")
        return

    # Filter already done
    to_render = [j for j in jobs if count_videos(j[3], j[2]) < N_EXPECTED]
    print(f"  To render: {len(to_render)} (skip {len(jobs)-len(to_render)} done)")

    if not to_render:
        print("  Nothing to render!")
        return

    gpu_queue = multiprocessing.Queue()
    for i in range(args.n_gpus):
        gpu_queue.put(i)

    ok, fail = 0, 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.n_gpus, initializer=_init_gpu,
                             initargs=(gpu_queue,)) as executor:
        futures = {executor.submit(render_positive_one, j): j for j in to_render}
        for i, fut in enumerate(as_completed(futures)):
            factory, seed, animode, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
            done = i + 1
            elapsed = time.time() - t0
            eta = elapsed / done * (len(to_render) - done)
            if done % 10 == 0 or not success:
                print(f"  [{done}/{len(to_render)}] {factory}/{seed} a{animode}: "
                      f"{msg} (eta={eta/60:.0f}m)")

    print(f"\n  Positive render done: {ok} ok, {fail} fail ({(time.time()-t0)/60:.1f}m)")


# ══════════════════════════════════════════════════════════════
# Stage 3: Negative Rendering
# ══════════════════════════════════════════════════════════════

def render_negative_one(job):
    """Render negative samples for one (factory, seed).

    The script internally iterates each movable joint and generates
    6 neg types per joint. No animode needed.
    """
    factory, seed, out_dir, base_dir = job
    gpu_id = _worker_gpu

    neg_dir = os.path.join(out_dir, "negatives")
    if os.path.isfile(os.path.join(neg_dir, "metadata.json")):
        return (factory, seed, True, "skip (done)")

    # All 32 views: 16 hemi (static) + 8 orbit + 8 sweep (moving)
    all_view_names = HEMI_VIEWS + ALL_MOVING

    cmd = [
        BLENDER, "--background", "--python-use-system-env",
        "--python", "render_negative_samples.py", "--",
        "--factory", factory, "--seed", str(seed), "--device", "0",
        "--output_dir", out_dir, "--base", base_dir,
        "--resolution", "512", "--samples", "32",
        "--duration", "4.0", "--fps", "30",
        "--skip_bg",
        "--neg_views", *all_view_names,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.time()
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, env=env,
                                capture_output=True, text=True, timeout=14400)
        dur = time.time() - t0
        if os.path.isfile(os.path.join(neg_dir, "metadata.json")):
            return (factory, seed, True, f"ok ({dur:.0f}s)")
        return (factory, seed, False, f"no metadata ({dur:.0f}s)")
    except subprocess.TimeoutExpired:
        return (factory, seed, False, "timeout")
    except Exception as e:
        return (factory, seed, False, str(e))


def stage_render_negative(args):
    """Stage 3: Render negative (wrong articulation) samples.

    One job per (factory, seed) — NOT per animode.
    Uses max animode as base so all joints are animated, then mutates them.
    """
    precompute = os.path.join(BASE_DIR, "precompute")
    jobs = build_negative_jobs(args.source,
                               precompute if os.path.isdir(precompute) else None)

    if args.shard:
        k, n = map(int, args.shard.split("/"))
        shard_size = (len(jobs) + n - 1) // n
        jobs = jobs[k * shard_size:(k + 1) * shard_size]

    print(f"\n{'='*60}")
    print(f"Stage 3: Negative Rendering")
    print(f"{'='*60}")
    print(f"  Total jobs: {len(jobs)} (1 per object, NOT per animode)")
    print(f"  GPUs: {args.n_gpus}")
    print(f"  Neg types: 6 (wrong_joint_type, wrong_axis, wrong_direction, "
          f"over_motion, wrong_parts_moving, jitter)")

    if args.dry_run:
        print("  [DRY RUN]")
        return

    to_render = []
    for j in jobs:
        neg_dir = os.path.join(j[3], "negatives")
        if not os.path.isfile(os.path.join(neg_dir, "metadata.json")):
            to_render.append(j)

    print(f"  To render: {len(to_render)} (skip {len(jobs)-len(to_render)} done)")

    if not to_render:
        print("  Nothing to render!")
        return

    gpu_queue = multiprocessing.Queue()
    for i in range(args.n_gpus):
        gpu_queue.put(i)

    ok, fail = 0, 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.n_gpus, initializer=_init_gpu,
                             initargs=(gpu_queue,)) as executor:
        futures = {executor.submit(render_negative_one, j): j for j in to_render}
        for i, fut in enumerate(as_completed(futures)):
            factory, seed, success, msg = fut.result()
            if success:
                ok += 1
            else:
                fail += 1
            done = i + 1
            elapsed = time.time() - t0
            eta = elapsed / done * (len(to_render) - done)
            if done % 10 == 0 or not success:
                print(f"  [{done}/{len(to_render)}] {factory}/{seed}: "
                      f"{msg} (eta={eta/60:.0f}m)")

    print(f"\n  Negative render done: {ok} ok, {fail} fail ({(time.time()-t0)/60:.1f}m)")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Unified 3-stage pipeline: generate + render+ + render-",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all IM assets (50 seeds)
  python run_pipeline.py --stage generate --source im --n_seeds 50

  # Generate all PhysXNet assets
  python run_pipeline.py --stage generate --source physxnet

  # Render positive on 4 GPUs
  python run_pipeline.py --stage render_positive --source all --n_gpus 4

  # Render negative on 4 GPUs
  python run_pipeline.py --stage render_negative --source all --n_gpus 4

  # Full pipeline with sharding (node 0 of 4)
  python run_pipeline.py --stage all --source all --n_gpus 4 --shard 0/4

  # Dry run to see job counts
  python run_pipeline.py --stage all --source all --dry_run
""")
    parser.add_argument("--stage", required=True,
                        choices=["generate", "render_positive", "render_negative", "all"],
                        help="Pipeline stage to run")
    parser.add_argument("--source", required=True,
                        choices=["im", "physxnet", "physxmob", "all"],
                        help="Data source to process")
    parser.add_argument("--n_seeds", type=int, default=50,
                        help="Seeds per IM factory (default: 50)")
    parser.add_argument("--seed_start", type=int, default=0,
                        help="Starting seed for IM factories")
    parser.add_argument("--n_gpus", type=int, default=4,
                        help="Number of GPUs for rendering")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="CPU workers for generation")
    parser.add_argument("--shard", type=str, default=None,
                        help="Process shard K/N (e.g. 0/4)")
    parser.add_argument("--skip_split", action="store_true",
                        help="Skip split precompute in generate stage")
    parser.add_argument("--skip_verify", action="store_true",
                        help="Skip verify in generate stage")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print plan without executing")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"  Infinigen-Sim Unified Pipeline")
    print(f"{'='*60}")
    print(f"  Stage:   {args.stage}")
    print(f"  Source:  {args.source}")
    print(f"  GPUs:    {args.n_gpus}")
    if args.shard:
        print(f"  Shard:   {args.shard}")
    print(f"  IM factories:       {len(IM_FACTORIES)}")
    print(f"  PhysXNet factories: {len(PHYSXNET_FACTORIES)}")
    print(f"  PhysXMob factories: {len(PHYSXMOB_FACTORIES)}")

    if args.stage in ("generate", "all"):
        stage_generate(args)

    if args.stage in ("render_positive", "all"):
        stage_render_positive(args)

    if args.stage in ("render_negative", "all"):
        stage_render_negative(args)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
