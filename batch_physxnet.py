#!/usr/bin/env python3
"""
Batch generate + render PhysXNet and PhysX_mobility assets.
Multi-view x multi-animode x multi-seed x multi-GPU.

Usage:
  # List available factories and object counts
  python batch_physxnet.py --list

  # Prepare + render 10 seeds of FurniturePhysXNetFactory (4 GPUs)
  python batch_physxnet.py --factory FurniturePhysXNetFactory --n_seeds 10 --n_gpus 4

  # Render all PhysX_mobility factories (4 seeds each)
  python batch_physxnet.py --dataset physxmob --n_seeds 4 --n_gpus 4

  # Prepare scenes only (no rendering)
  python batch_physxnet.py --factory ElectronicsPhysXMobilityFactory --n_seeds 5 --no_render

  # Material variant seeds (seed > N objects)
  python batch_physxnet.py --factory FurniturePhysXNetFactory --n_seeds 100 --mode both

  # Render only (scenes already prepared)
  python batch_physxnet.py --factory FurniturePhysXNetFactory --n_seeds 10 --render_only
"""

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

BLENDER = "/mnt/data/yurh/blender-3.6.0-linux-x64/blender"
BASE_DIR = "/mnt/data/yurh/Infinigen-Sim"
OUTPUT_ROOT = os.path.join(BASE_DIR, "outputs")

# Add script directory to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from physxnet_factory_rules import (
    PHYSXNET_FACTORY_LIST, PHYSXMOB_FACTORY_LIST,
    PHYSXNET_ANIMODES, PHYSXMOB_ANIMODES,
    ALL_ANIMODES, ALL_FACTORY_LIST,
    get_all_physxnet_factories, get_all_physxmob_factories,
    get_physxnet_factory_ids, get_physxmob_factory_ids,
)
from physxnet_loader import prepare_physxnet_scene

# Camera views (same as batch_generate_all.py)
HEMI_VIEWS = [f"hemi_{i:02d}" for i in range(16)]
ORBIT_VIEWS = [f"orbit_{i:02d}" for i in range(8)]
SWEEP_VIEWS = [f"sweep_{i:02d}" for i in range(8)]
ALL_MOVING = ORBIT_VIEWS + SWEEP_VIEWS
N_EXPECTED = len(HEMI_VIEWS) + len(ALL_MOVING)  # 32


# ── GPU assignment for render workers ──
_worker_gpu = None

def _init_gpu(gpu_queue):
    global _worker_gpu
    _worker_gpu = gpu_queue.get()


def count_videos(out_dir, animode):
    """Count rendered nobg videos for a specific animode."""
    if not os.path.isdir(out_dir):
        return 0
    suffix = f"_anim{animode}" if animode > 0 else ""
    count = 0
    for f in os.listdir(out_dir):
        if not f.endswith("_nobg.mp4"):
            continue
        if animode == 0:
            if "_anim" not in f:
                count += 1
        else:
            if f.endswith(f"{suffix}_nobg.mp4"):
                count += 1
    return count


# =====================================================================
# Stage 1: Prepare scenes (CPU only, no Blender needed)
# =====================================================================

def prepare_one(factory, seed, output_root):
    """Prepare one factory/seed scene (URDF + origins + OBJ symlinks)."""
    scene_dir = os.path.join(output_root, factory, str(seed))
    origins_path = os.path.join(scene_dir, "origins.json")
    urdf_path = os.path.join(scene_dir, "scene.urdf")

    if os.path.exists(origins_path) and os.path.exists(urdf_path):
        return (factory, seed, True, "skipped (exists)")

    try:
        info = prepare_physxnet_scene(factory, seed, output_root)
        if os.path.exists(info['urdf_path']) and os.path.exists(info['origins_path']):
            return (factory, seed, True,
                    f"ok ({info['dataset']} id={info['obj_id']}, variant={info['is_variant']})")
        else:
            return (factory, seed, False, "prepare returned but files missing")
    except Exception as e:
        return (factory, seed, False, str(e))


def run_prepare(jobs, output_root, n_workers):
    """Prepare all scenes."""
    print(f"\n{'='*60}")
    print(f"Preparing {len(jobs)} scenes ({n_workers} workers)")
    print(f"{'='*60}")

    existing = sum(1 for f, s in jobs
                   if os.path.exists(os.path.join(output_root, f, str(s), "origins.json"))
                   and os.path.exists(os.path.join(output_root, f, str(s), "scene.urdf")))
    print(f"Already prepared: {existing}/{len(jobs)}")
    print(f"To prepare: {len(jobs) - existing}")

    success, failed, skipped = 0, 0, 0
    failed_jobs = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(prepare_one, f, s, output_root): (f, s)
                   for f, s in jobs}

        for i, fut in enumerate(as_completed(futures)):
            factory, seed, ok, msg = fut.result()
            elapsed = time.time() - t0
            done = i + 1
            eta = elapsed / done * (len(jobs) - done) if done > 0 else 0

            if "skipped" in msg:
                skipped += 1
                status = "SKIP"
            elif ok:
                success += 1
                status = "OK  "
            else:
                failed += 1
                failed_jobs.append((factory, seed, msg))
                status = "FAIL"

            print(f"[{done}/{len(jobs)}] [{status}] {factory} s{seed}: {msg} "
                  f"(elapsed={elapsed:.0f}s, ETA={eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n=== Prepare Complete ({elapsed:.0f}s) ===")
    print(f"Success: {success}, Skipped: {skipped}, Failed: {failed}")

    if failed_jobs:
        print(f"\nFailed:")
        for f, s, m in failed_jobs:
            print(f"  {f} s{s}: {m}")
        fail_path = os.path.join(output_root, "failed_prepare_physxnet.json")
        with open(fail_path, "w") as fp:
            json.dump([{"factory": f, "seed": s, "error": m}
                       for f, s, m in failed_jobs], fp, indent=2)
        print(f"Saved to {fail_path}")


# =====================================================================
# Stage 2: Render (GPU)
# =====================================================================

def render_one(args_tuple):
    """Render one (factory, seed, animode) with all views."""
    factory, seed, animode, out_dir = args_tuple
    gpu_id = _worker_gpu

    # Check if already rendered
    n_existing = count_videos(out_dir, animode)
    if n_existing >= N_EXPECTED:
        return (factory, seed, animode, True, f"skipped ({n_existing} videos)")

    cmd = [
        BLENDER, "--background", "--python-use-system-env",
        "--python", "render_articulation.py", "--",
        "--factory", factory, "--seed", str(seed), "--device", "0",
        "--output_dir", out_dir,
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
        if result.returncode == 0 and n_new >= N_EXPECTED:
            return (factory, seed, animode, True,
                    f"ok ({n_new} videos, {dur:.0f}s, GPU{gpu_id})")
        elif n_new > n_existing:
            return (factory, seed, animode, False,
                    f"partial ({n_new}/{N_EXPECTED}, {dur:.0f}s, GPU{gpu_id})")
        else:
            lines = result.stdout.strip().split("\n")[-3:] if result.stdout else ["no output"]
            return (factory, seed, animode, False,
                    f"failed ({dur:.0f}s, GPU{gpu_id}): {' '.join(lines)}")
    except subprocess.TimeoutExpired:
        return (factory, seed, animode, False, f"timeout (GPU{gpu_id})")
    except Exception as e:
        return (factory, seed, animode, False, str(e))


def run_render_batch(render_jobs, n_gpus):
    """Render all jobs across N GPUs."""
    to_render = []
    already = 0
    for job in render_jobs:
        factory, seed, animode, out_dir = job
        if count_videos(out_dir, animode) >= N_EXPECTED:
            already += 1
        else:
            to_render.append(job)

    print(f"\n{'='*60}")
    print(f"Rendering: {len(to_render)} jobs on {n_gpus} GPUs")
    print(f"Already done: {already}, Total: {len(render_jobs)}")
    print(f"Views per job: {N_EXPECTED} (16 hemi + 8 orbit + 8 sweep)")
    print(f"{'='*60}")

    if not to_render:
        print("Nothing to render!")
        return

    gpu_queue = multiprocessing.Queue()
    for i in range(n_gpus):
        gpu_queue.put(i)

    success, failed = 0, 0
    failed_jobs = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_gpus, initializer=_init_gpu,
                             initargs=(gpu_queue,)) as executor:
        futures = {executor.submit(render_one, job): job for job in to_render}

        for i, fut in enumerate(as_completed(futures)):
            factory, seed, animode, ok, msg = fut.result()
            elapsed = time.time() - t0
            done = i + 1
            remaining = len(to_render) - done
            eta = elapsed / done * remaining if done > 0 else 0

            if ok:
                success += 1
                status = "OK  "
            else:
                failed += 1
                failed_jobs.append((factory, seed, animode, msg))
                status = "FAIL"

            print(f"[{done}/{len(to_render)}] [{status}] {factory} s{seed} a{animode}: {msg} "
                  f"(elapsed={elapsed/60:.0f}m, ETA={eta/60:.0f}m)")

    elapsed = time.time() - t0
    print(f"\n=== Rendering Complete ({elapsed/60:.1f}m) ===")
    print(f"Success: {success}, Failed: {failed}")

    if failed_jobs:
        print(f"\nFailed render jobs:")
        for f, s, a, m in failed_jobs:
            print(f"  {f} s{s} a{a}: {m}")
        fail_path = os.path.join(OUTPUT_ROOT, "failed_render_physxnet.json")
        with open(fail_path, "w") as fp:
            json.dump([{"factory": f, "seed": s, "animode": a, "error": m}
                       for f, s, a, m in failed_jobs], fp, indent=2)
        print(f"Saved to {fail_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch generate + render PhysXNet/PhysX_mobility assets")
    parser.add_argument("--list", action="store_true",
                        help="List available factories and exit")
    parser.add_argument("--dataset", default="all",
                        choices=["physxnet", "physxmob", "all"],
                        help="Which dataset to process (default: all)")
    parser.add_argument("--factory", default=None,
                        help="Only process this factory (default: all in dataset)")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of seeds per factory (default: 10)")
    parser.add_argument("--seed_start", type=int, default=0,
                        help="Starting seed (default: 0)")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="CPU workers for preparation (default: 4)")
    parser.add_argument("--n_gpus", type=int, default=4,
                        help="GPUs for rendering (default: 4)")
    parser.add_argument("--mode", default="retrieve",
                        choices=["retrieve", "variant", "both"],
                        help="Seed mode: retrieve=real objects, variant=material swaps, both=all")
    parser.add_argument("--render_only", action="store_true",
                        help="Skip preparation, only render")
    parser.add_argument("--no_render", action="store_true",
                        help="Skip rendering (prepare scenes only)")
    parser.add_argument("--views_only", nargs="+", default=None,
                        help="Only render these views (e.g. front side threequarter)")
    args = parser.parse_args()

    # ── List mode ──
    if args.list:
        print("=" * 72)
        print("PhysXNet + PhysX_mobility Factory Summary")
        print("=" * 72)

        pxn = get_all_physxnet_factories()
        pxm = get_all_physxmob_factories()

        print(f"\n-- PhysXNet: {sum(len(v) for v in pxn.values())} objects --")
        for name in sorted(pxn.keys()):
            n = len(pxn[name])
            a = PHYSXNET_ANIMODES.get(name, 0)
            print(f"  {name:<35} {n:>6} objects  animodes=0..{a}")

        print(f"\n-- PhysX_mobility: {sum(len(v) for v in pxm.values())} objects --")
        for name in sorted(pxm.keys()):
            n = len(pxm[name])
            a = PHYSXMOB_ANIMODES.get(name, 0)
            print(f"  {name:<40} {n:>5} objects  animodes=0..{a}")
        return

    # ── Determine factories to process ──
    if args.factory:
        factories = [args.factory]
    elif args.dataset == "physxnet":
        factories = PHYSXNET_FACTORY_LIST
    elif args.dataset == "physxmob":
        factories = PHYSXMOB_FACTORY_LIST
    else:
        factories = ALL_FACTORY_LIST

    # ── Determine seeds ──
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    # For "retrieve" mode, cap seeds to available objects
    if args.mode == "retrieve":
        print("Mode: retrieve (seeds map to actual objects only)")
    elif args.mode == "variant":
        print("Mode: variant (seeds include material swaps)")
    else:
        print("Mode: both (retrieve + variant)")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    print(f"\n=== PhysXNet/PhysX_mobility Batch Pipeline ===")
    print(f"Factories: {len(factories)}")
    print(f"Seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)} total)")
    print(f"Dataset: {args.dataset}")

    # Print plan per factory
    total_prepare = 0
    for factory in factories:
        # Get available object count for seed capping
        if factory in PHYSXNET_FACTORY_LIST or factory in get_all_physxnet_factories():
            n_objects = len(get_physxnet_factory_ids(factory))
        else:
            n_objects = len(get_physxmob_factory_ids(factory))

        if args.mode == "retrieve":
            n_seeds = min(len(seeds), n_objects)
        else:
            n_seeds = len(seeds)

        max_a = ALL_ANIMODES.get(factory, 0)
        total_prepare += n_seeds
        print(f"  {factory:<35} {n_objects:>5} objs, {n_seeds:>4} seeds, animodes=0..{max_a}")

    # ── Stage 1: Prepare scenes ──
    if not args.render_only:
        prep_jobs = []
        for factory in factories:
            if factory in PHYSXNET_FACTORY_LIST or factory in get_all_physxnet_factories():
                n_objects = len(get_physxnet_factory_ids(factory))
            else:
                n_objects = len(get_physxmob_factory_ids(factory))

            for seed in seeds:
                if args.mode == "retrieve" and seed >= n_objects:
                    continue
                prep_jobs.append((factory, seed))

        run_prepare(prep_jobs, OUTPUT_ROOT, args.n_workers)

    # ── Stage 2: Render ──
    if not args.no_render:
        render_jobs = []
        for factory in factories:
            max_animode = ALL_ANIMODES.get(factory, 0)

            if factory in PHYSXNET_FACTORY_LIST or factory in get_all_physxnet_factories():
                n_objects = len(get_physxnet_factory_ids(factory))
            else:
                n_objects = len(get_physxmob_factory_ids(factory))

            for seed in seeds:
                if args.mode == "retrieve" and seed >= n_objects:
                    continue

                scene_dir = os.path.join(OUTPUT_ROOT, factory, str(seed))
                origins = os.path.join(scene_dir, "origins.json")
                urdf = os.path.join(scene_dir, "scene.urdf")
                if not os.path.exists(origins) or not os.path.exists(urdf):
                    continue

                out_dir = os.path.join(OUTPUT_ROOT, "motion_videos", factory, str(seed))
                os.makedirs(out_dir, exist_ok=True)

                for animode in range(max_animode + 1):
                    render_jobs.append((factory, seed, animode, out_dir))

        print(f"\n=== Render Plan ===")
        print(f"Total render jobs: {len(render_jobs)}")
        est_hours = len(render_jobs) * 6 / 60 / args.n_gpus
        print(f"Est. time per job: ~6min (32 views)")
        print(f"Est. total: ~{est_hours:.0f}h on {args.n_gpus} GPUs")

        run_render_batch(render_jobs, args.n_gpus)

    print("\nDone!")


if __name__ == "__main__":
    main()
