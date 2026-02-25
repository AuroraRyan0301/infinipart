#!/usr/bin/env python3
"""
Batch generate + render all Infinite Mobility factories.
Multi-view x multi-animode x multi-seed x multi-GPU.

Usage:
  # Full pipeline: generate + render (4 GPUs, 10 seeds)
  python batch_generate_all.py --n_seeds 10 --n_gpus 4

  # Render only (assets already generated)
  python batch_generate_all.py --n_seeds 10 --n_gpus 4 --render_only

  # Single factory
  python batch_generate_all.py --n_seeds 10 --n_gpus 4 --factory OvenFactory --render_only

  # Generate only
  python batch_generate_all.py --n_seeds 10 --no_render --no_split
"""

import argparse
import json
import os
import subprocess
import sys
import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BLENDER = "/mnt/data/yurh/blender-3.6.0-linux-x64/blender"
BASE_DIR = "/mnt/data/yurh/Infinite-Mobility"
PYTHONPATH = f"{BASE_DIR}:/mnt/data/yurh/infinigen"

FACTORIES = [
    "BeverageFridgeFactory",
    "MicrowaveFactory",
    "OvenFactory",
    "ToiletFactory",
    "KitchenCabinetFactory",
    "WindowFactory",
    "LiteDoorFactory",
    "OfficeChairFactory",
    "TapFactory",
    "LampFactory",
    "PotFactory",
    "BottleFactory",
    "DishwasherFactory",
    "BarChairFactory",
    "PanFactory",
    "TVFactory",
]

# Max animode (inclusive) per factory. Animodes 0..N are rendered.
# Must match FACTORY_RULES["animode_joints"] in render_articulation.py.
FACTORY_ANIMODES = {
    "DishwasherFactory": 2,
    "BeverageFridgeFactory": 2,
    "MicrowaveFactory": 2,
    "OvenFactory": 2,
    "KitchenCabinetFactory": 2,
    "ToiletFactory": 3,
    "WindowFactory": 4,
    "LiteDoorFactory": 0,
    "OfficeChairFactory": 2,
    "TapFactory": 2,
    "LampFactory": 3,
    "PotFactory": 4,
    "BottleFactory": 2,
    "BarChairFactory": 2,
    "PanFactory": 0,
    "TVFactory": 2,
}

# Add PartNet-Mobility factories
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
try:
    from partnet_factory_rules import PARTNET_FACTORY_LIST, PARTNET_ANIMODES
    FACTORIES.extend([f for f in PARTNET_FACTORY_LIST if f not in FACTORIES])
    FACTORY_ANIMODES.update(PARTNET_ANIMODES)
except ImportError:
    pass

# Camera views: 16 fixed hemisphere + 8 orbit + 8 sweep = 32
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
            # Match files without _anim prefix (e.g. hemi_00_nobg.mp4)
            if "_anim" not in f:
                count += 1
        else:
            # Match files with exact suffix (e.g. hemi_00_anim2_nobg.mp4)
            if f.endswith(f"{suffix}_nobg.mp4"):
                count += 1
    return count


# ═══════════════════════════════════════════════════════════════
# Stage 1: Generate assets (CPU)
# ═══════════════════════════════════════════════════════════════

def generate_one(factory, seed, output_root):
    """Generate one factory/seed combination."""
    output_dir = os.path.join(output_root, factory)
    seed_dir = os.path.join(output_dir, str(seed))
    origins_path = os.path.join(seed_dir, "origins.json")
    urdf_path = os.path.join(seed_dir, "scene.urdf")

    if os.path.exists(origins_path) and os.path.exists(urdf_path):
        return (factory, seed, True, "skipped (exists)")

    # Clean stale partial outputs
    if os.path.isdir(seed_dir):
        for f in ["origins.json", "scene.urdf"]:
            p = os.path.join(seed_dir, f)
            if os.path.exists(p):
                os.remove(p)
        import shutil
        objs_dir = os.path.join(seed_dir, "objs")
        if os.path.isdir(objs_dir):
            shutil.rmtree(objs_dir)

    data_infos = os.path.join(output_dir, f"data_infos_{seed}.json")
    if os.path.exists(data_infos):
        os.remove(data_infos)

    cmd = [
        BLENDER, "--background", "--python-use-system-env",
        "--python", "infinigen_examples/generate_individual_assets.py",
        "--",
        "--output_folder", f"outputs/{factory}",
        "-f", factory,
        "-n", "1",
        "--seed", str(seed),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = PYTHONPATH

    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, env=env,
                                capture_output=True, text=True, timeout=1800)
        if os.path.exists(origins_path) and os.path.exists(urdf_path):
            return (factory, seed, True, "ok")
        elif os.path.exists(origins_path):
            return (factory, seed, True, "ok (no URDF)")
        else:
            err_lines = result.stderr.strip().split("\n")[-3:]
            return (factory, seed, False, f"failed: {' '.join(err_lines)}")
    except subprocess.TimeoutExpired:
        return (factory, seed, False, "timeout")
    except Exception as e:
        return (factory, seed, False, str(e))


def run_generate(jobs, output_root, n_workers):
    """Generate all factory/seed combinations."""
    print(f"\n{'='*60}")
    print(f"Generating {len(jobs)} assets ({n_workers} workers)")
    print(f"{'='*60}")

    existing = sum(1 for f, s in jobs
                   if os.path.exists(os.path.join(output_root, f, str(s), "origins.json"))
                   and os.path.exists(os.path.join(output_root, f, str(s), "scene.urdf")))
    print(f"Already generated: {existing}/{len(jobs)}")
    print(f"To generate: {len(jobs) - existing}")

    success, failed, skipped = 0, 0, 0
    failed_jobs = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(generate_one, f, s, output_root): (f, s)
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
    print(f"\n=== Generation Complete ({elapsed:.0f}s) ===")
    print(f"Success: {success}, Skipped: {skipped}, Failed: {failed}")

    if failed_jobs:
        print(f"\nFailed:")
        for f, s, m in failed_jobs:
            print(f"  {f} s{s}: {m}")
        fail_path = os.path.join(output_root, "failed_generate.json")
        with open(fail_path, "w") as fp:
            json.dump([{"factory": f, "seed": s, "error": m}
                       for f, s, m in failed_jobs], fp, indent=2)


# ═══════════════════════════════════════════════════════════════
# Stage 2: Render (GPU) — multi-view x multi-animode
# ═══════════════════════════════════════════════════════════════

def render_one(args):
    """Render one (factory, seed, animode) with all 32 views."""
    factory, seed, animode, out_dir = args
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
            lines = result.stdout.strip().split("\n")[-3:]
            return (factory, seed, animode, False,
                    f"failed ({dur:.0f}s, GPU{gpu_id}): {' '.join(lines)}")
    except subprocess.TimeoutExpired:
        return (factory, seed, animode, False, f"timeout (GPU{gpu_id})")
    except Exception as e:
        return (factory, seed, animode, False, str(e))


def run_render_batch(render_jobs, n_gpus):
    """Render all jobs across N GPUs."""
    # Filter already-rendered
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

    # GPU queue: each of N workers gets a unique GPU
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
        fail_path = os.path.join(BASE_DIR, "outputs", "failed_render.json")
        with open(fail_path, "w") as fp:
            json.dump([{"factory": f, "seed": s, "animode": a, "error": m}
                       for f, s, a, m in failed_jobs], fp, indent=2)
        print(f"Saved to {fail_path}")


# ═══════════════════════════════════════════════════════════════
# Stage 3: Split
# ═══════════════════════════════════════════════════════════════

def run_split(output_root):
    """Run the splitting pipeline for all generated factories."""
    print(f"\n{'='*60}")
    print("Running 2-part splitting pipeline...")
    print(f"{'='*60}")
    cmd = [sys.executable, "split_and_visualize.py",
           "--output_root", "outputs", "--base_path", BASE_DIR]
    subprocess.run(cmd, cwd=BASE_DIR)


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch generate + render Infinite Mobility assets (multi-view x multi-animode)")
    parser.add_argument("--n_seeds", type=int, default=10,
                        help="Number of seeds per factory (default: 10)")
    parser.add_argument("--seed_start", type=int, default=0,
                        help="Starting seed (default: 0)")
    parser.add_argument("--n_workers", type=int, default=4,
                        help="CPU workers for generation (default: 4)")
    parser.add_argument("--n_gpus", type=int, default=4,
                        help="GPUs for rendering (default: 4)")
    parser.add_argument("--factory", default=None,
                        help="Only process this factory (default: all)")
    parser.add_argument("--render_only", action="store_true",
                        help="Skip generation, only render")
    parser.add_argument("--no_render", action="store_true",
                        help="Skip rendering")
    parser.add_argument("--no_split", action="store_true",
                        help="Skip splitting")
    parser.add_argument("--split_only", action="store_true",
                        help="Only run splitting")
    args = parser.parse_args()

    output_root = os.path.join(BASE_DIR, "outputs")
    os.makedirs(output_root, exist_ok=True)

    if args.split_only:
        run_split(output_root)
        return

    factories = [args.factory] if args.factory else FACTORIES

    # Validate factory names
    for f in factories:
        if f not in FACTORY_ANIMODES:
            print(f"WARNING: {f} not in FACTORY_ANIMODES, will use animode 0 only")

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    print(f"=== Infinite Mobility Batch Pipeline ===")
    print(f"Factories: {len(factories)}")
    print(f"Seeds: {seeds[0]}-{seeds[-1]} ({len(seeds)} total)")
    print(f"Views per job: {N_EXPECTED} (16 hemi + 8 orbit + 8 sweep)")

    # ── Stage 1: Generate ──
    if not args.render_only:
        gen_jobs = [(f, s) for f in factories for s in seeds]
        run_generate(gen_jobs, output_root, args.n_workers)

    # ── Stage 2: Render ──
    if not args.no_render:
        # Build render jobs: (factory, seed, animode, out_dir)
        render_jobs = []
        for factory in factories:
            max_animode = FACTORY_ANIMODES.get(factory, 0)
            for seed in seeds:
                origins = os.path.join(output_root, factory, str(seed), "origins.json")
                if not os.path.exists(origins):
                    continue
                out_dir = os.path.join(output_root, "motion_videos", factory, str(seed))
                os.makedirs(out_dir, exist_ok=True)
                for animode in range(max_animode + 1):
                    render_jobs.append((factory, seed, animode, out_dir))

        # Print plan
        print(f"\n=== Render Plan ===")
        total_animodes = 0
        for factory in factories:
            max_a = FACTORY_ANIMODES.get(factory, 0)
            n_exist = sum(1 for s in seeds
                          if os.path.exists(os.path.join(output_root, factory, str(s), "origins.json")))
            n_jobs = n_exist * (max_a + 1)
            total_animodes += n_jobs
            print(f"  {factory}: {n_exist} seeds x {max_a+1} animodes = {n_jobs} jobs")
        print(f"Total render jobs: {len(render_jobs)}")
        print(f"Est. time per job: ~6min (32 views, persistent_data)")
        est_hours = len(render_jobs) * 6 / 60 / args.n_gpus
        print(f"Est. total: ~{est_hours:.0f}h on {args.n_gpus} GPUs")

        run_render_batch(render_jobs, args.n_gpus)

    # ── Stage 3: Split ──
    if not args.no_split:
        run_split(output_root)


if __name__ == "__main__":
    main()
