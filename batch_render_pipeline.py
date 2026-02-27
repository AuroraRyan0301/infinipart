#!/usr/bin/env python3
"""
Batch pipeline: Generate -> Split -> Render articulation motion videos.
Distributes rendering across multiple GPUs.

Usage:
  # Full pipeline: generate + split + render, seeds 0-99, 4 GPUs
  python batch_render_pipeline.py --seed_start 0 --seed_end 100 --n_gpus 4

  # Render only (assets already generated & split)
  python batch_render_pipeline.py --render_only --seed_start 0 --seed_end 10

  # Single factory
  python batch_render_pipeline.py --factory BottleFactory --seed_start 0 --seed_end 5

  # With all envmaps
  python batch_render_pipeline.py --render_only --all_envmaps --seed_start 0 --seed_end 10

  # Skip nobg renders (faster, bg only)
  python batch_render_pipeline.py --render_only --skip_nobg --seed_start 0 --seed_end 10

  # Custom paths (for running on different machines)
  python batch_render_pipeline.py --base /path/to/Infinite-Mobility --blender /path/to/blender
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from multiprocessing import Queue, Process


# ── Defaults (override with --base, --blender, etc.) ──
DEFAULT_BASE = "/mnt/data/yurh/Infinigen-Sim"
DEFAULT_BLENDER = "/mnt/data/yurh/blender-3.6.0-linux-x64/blender"
DEFAULT_ENVMAP_DIR = "/mnt/data/yurh/dataset3D/envmap/indoor"
DEFAULT_ENVMAP = "brown_photostudio_06_2k.exr"

FACTORIES = [
    # ── Original Infinite-Mobility factories ──
    "BeverageFridgeFactory",
    "BottleFactory",
    "DishwasherFactory",
    "KitchenCabinetFactory",
    "LampFactory",
    "LiteDoorFactory",
    "MicrowaveFactory",
    "OfficeChairFactory",
    "OvenFactory",
    "PotFactory",
    "TapFactory",
    "ToiletFactory",
    "WindowFactory",
    # ── Infinigen-Sim sim_objects ──
    "SimDoorFactory",
    "DoorHandleFactory",
    "DrawerFactory",
    "BoxFactory",
    "CabinetFactory",
    "RefrigeratorFactory",
    "FaucetFactory",
    "StovetopFactory",
    "ToasterFactory",
    "PepperGrinderFactory",
    "PlierFactory",
    "SoapDispenserFactory",
    "TrashFactory",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch pipeline: Generate + Split + Render articulation videos")

    # Paths
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="Infinite-Mobility base directory")
    parser.add_argument("--blender", default=DEFAULT_BLENDER,
                        help="Path to Blender executable")
    parser.add_argument("--output_root", default=None,
                        help="Output root for videos (default: {base}/outputs/motion_videos)")

    # Seed range
    parser.add_argument("--seed_start", type=int, default=0,
                        help="Starting seed (inclusive)")
    parser.add_argument("--seed_end", type=int, default=100,
                        help="Ending seed (exclusive)")

    # Factory selection
    parser.add_argument("--factory", default=None,
                        help="Only process this factory (default: all 13)")

    # GPU
    parser.add_argument("--n_gpus", type=int, default=4,
                        help="Number of GPUs to use")
    parser.add_argument("--gpu_offset", type=int, default=0,
                        help="Starting GPU index (e.g. 2 means use GPUs 2,3,4,5)")

    # Stage control
    parser.add_argument("--render_only", action="store_true",
                        help="Skip generation and splitting, only render")
    parser.add_argument("--split_only", action="store_true",
                        help="Skip generation, only split and render")
    parser.add_argument("--no_render", action="store_true",
                        help="Generate and split but skip rendering")
    parser.add_argument("--no_split", action="store_true",
                        help="Skip splitting stage")

    # Generation options
    parser.add_argument("--gen_workers", type=int, default=None,
                        help="Number of parallel generation workers (default: n_gpus)")
    parser.add_argument("--gen_timeout", type=int, default=600,
                        help="Timeout per generation job in seconds")

    # Render options
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--skip_nobg", action="store_true",
                        help="Skip transparent background renders")
    parser.add_argument("--skip_bg", action="store_true",
                        help="Skip opaque background renders")

    # Envmap options
    parser.add_argument("--envmap_dir", default=DEFAULT_ENVMAP_DIR)
    parser.add_argument("--envmap", default=DEFAULT_ENVMAP,
                        help="Single envmap filename (default: brown_photostudio_06_2k.exr)")
    parser.add_argument("--all_envmaps", action="store_true",
                        help="Use all envmaps in envmap_dir")
    parser.add_argument("--envmaps", default=None,
                        help="Comma-separated envmap filenames")

    # Gravity (BottleFactory)
    parser.add_argument("--bottle_gravity", action="store_true",
                        help="Also render BottleFactory with gravity")

    # Misc
    parser.add_argument("--dry_run", action="store_true",
                        help="Print jobs without executing")

    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════
# Stage 1: Generate assets
# ═══════════════════════════════════════════════════════════════

def generate_one(blender, base, factory, seed, timeout=600):
    """Generate one factory/seed. Returns (factory, seed, success, msg)."""
    seed_dir = os.path.join(base, "outputs", factory, str(seed))
    origins_path = os.path.join(seed_dir, "origins.json")
    urdf_path = os.path.join(seed_dir, "scene.urdf")

    if os.path.exists(origins_path) and os.path.exists(urdf_path):
        return (factory, seed, True, "skipped (exists)")

    pythonpath = f"{base}:/mnt/data/yurh/infinigen"
    cmd = [
        blender, "--background", "--python-use-system-env",
        "--python", "infinigen_examples/generate_individual_assets.py",
        "--",
        "--output_folder", f"outputs/{factory}",
        "-f", factory,
        "-n", "1",
        "--seed", str(seed),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath

    try:
        result = subprocess.run(
            cmd, cwd=base, env=env,
            capture_output=True, text=True, timeout=timeout,
        )
        if os.path.exists(origins_path):
            return (factory, seed, True, "ok")
        else:
            err = result.stderr.strip().split("\n")[-3:]
            return (factory, seed, False, f"failed: {' '.join(err)}")
    except subprocess.TimeoutExpired:
        return (factory, seed, False, "timeout")
    except Exception as e:
        return (factory, seed, False, str(e))


def run_generation(args, factories, seeds):
    """Stage 1: Generate all factory/seed combos."""
    print(f"\n{'='*60}")
    print("Stage 1: GENERATE ASSETS")
    print(f"{'='*60}")

    jobs = [(f, s) for f in factories for s in seeds]
    n_workers = args.gen_workers or args.n_gpus

    existing = sum(1 for f, s in jobs
                   if os.path.exists(os.path.join(args.base, "outputs", f, str(s), "origins.json"))
                   and os.path.exists(os.path.join(args.base, "outputs", f, str(s), "scene.urdf")))

    print(f"  Factories: {len(factories)}, Seeds: {len(seeds)}")
    print(f"  Total: {len(jobs)}, Already done: {existing}, To generate: {len(jobs)-existing}")
    print(f"  Workers: {n_workers}")

    if args.dry_run:
        print("  [DRY RUN] Skipping generation")
        return

    success, failed, skipped = 0, 0, 0
    failed_jobs = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for f, s in jobs:
            fut = executor.submit(generate_one, args.blender, args.base, f, s, args.gen_timeout)
            futures[fut] = (f, s)

        for i, fut in enumerate(as_completed(futures)):
            factory, seed, ok, msg = fut.result()
            done = i + 1
            elapsed = time.time() - t0
            eta = elapsed / done * (len(jobs) - done) if done else 0

            if "skipped" in msg:
                skipped += 1
                tag = "SKIP"
            elif ok:
                success += 1
                tag = "OK  "
            else:
                failed += 1
                failed_jobs.append((factory, seed, msg))
                tag = "FAIL"

            print(f"  [{done}/{len(jobs)}] [{tag}] {factory} seed={seed}: {msg} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    print(f"\n  Generation done: {success} ok, {skipped} skipped, {failed} failed")
    if failed_jobs:
        fail_path = os.path.join(args.base, "outputs", "failed_generation.json")
        with open(fail_path, "w") as fp:
            json.dump([{"factory": f, "seed": s, "error": m} for f, s, m in failed_jobs],
                      fp, indent=2)
        print(f"  Failed jobs saved to {fail_path}")


# ═══════════════════════════════════════════════════════════════
# Stage 2: Split into 2 parts
# ═══════════════════════════════════════════════════════════════

def run_splitting(args, factories):
    """Stage 2: Run split_and_visualize.py."""
    print(f"\n{'='*60}")
    print("Stage 2: SPLIT INTO 2 PARTS")
    print(f"{'='*60}")

    split_script = os.path.join(args.base, "split_and_visualize.py")
    if not os.path.exists(split_script):
        print(f"  ERROR: {split_script} not found")
        return

    if args.dry_run:
        print("  [DRY RUN] Skipping splitting")
        return

    for factory in factories:
        print(f"\n  Splitting {factory}...")
        cmd = [
            sys.executable, split_script,
            "--output_root", "outputs",
            "--base_path", args.base,
            "--factory", factory,
        ]
        result = subprocess.run(cmd, cwd=args.base, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[-300:]}")
        else:
            print(f"    Done")


# ═══════════════════════════════════════════════════════════════
# Stage 3: Render articulation videos
# ═══════════════════════════════════════════════════════════════

def check_render_complete(output_dir, skip_nobg=False, skip_bg=False):
    """Check if a render job is already complete (all mp4s exist)."""
    views = ["front", "side", "back", "threequarter"]
    variants = []
    if not skip_bg:
        variants.append("bg")
    if not skip_nobg:
        variants.append("nobg")

    for view in views:
        for var in variants:
            mp4 = os.path.join(output_dir, f"{view}_{var}.mp4")
            if not os.path.exists(mp4):
                return False
    return True


def render_one(blender, base, factory, seed, gpu_id, envmap_path,
               output_dir, resolution, fps, duration, samples,
               skip_nobg, skip_bg, gravity=False):
    """Render one factory/seed/envmap on a specific GPU."""
    # Check prerequisites
    scene_dir = os.path.join(base, "outputs", factory, str(seed))
    for f in ["scene.urdf", "origins.json"]:
        if not os.path.exists(os.path.join(scene_dir, f)):
            return (factory, seed, False, f"missing {f}")

    objs_dir = os.path.join(scene_dir, "outputs", factory, str(seed), "objs")
    if not os.path.isdir(objs_dir):
        return (factory, seed, False, "missing objs dir")

    if check_render_complete(output_dir, skip_nobg, skip_bg):
        return (factory, seed, True, "skipped (exists)")

    os.makedirs(output_dir, exist_ok=True)

    render_script = os.path.join(base, "render_articulation.py")
    cmd = [
        blender, "--background", "--python-use-system-env",
        "--python", render_script,
        "--",
        "--factory", factory,
        "--seed", str(seed),
        "--base", base,
        "--envmap", envmap_path,
        "--resolution", str(resolution),
        "--fps", str(fps),
        "--duration", str(duration),
        "--samples", str(samples),
        "--device", "0",  # always 0 because CUDA_VISIBLE_DEVICES remaps
        "--output_dir", output_dir,
    ]

    if skip_nobg:
        cmd.append("--skip_nobg")
    if skip_bg:
        cmd.append("--skip_bg")
    if gravity:
        cmd.append("--gravity")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        result = subprocess.run(
            cmd, cwd=base, env=env,
            capture_output=True, text=True,
            timeout=3600,  # 1 hour per render job
        )
        if check_render_complete(output_dir, skip_nobg, skip_bg):
            return (factory, seed, True, "ok")
        else:
            err = result.stderr.strip().split("\n")[-5:]
            return (factory, seed, False, f"render failed: {' '.join(err)}")
    except subprocess.TimeoutExpired:
        return (factory, seed, False, "timeout (1h)")
    except Exception as e:
        return (factory, seed, False, str(e))


def gpu_worker(gpu_id, job_queue, results_list, args, envmap_path, envmap_name):
    """Worker process: pulls jobs from queue, renders on assigned GPU."""
    while True:
        try:
            job = job_queue.get_nowait()
        except Exception:
            break

        factory, seed, gravity = job

        # Build output dir
        if envmap_name:
            output_dir = os.path.join(args.output_root, factory, str(seed), envmap_name)
        else:
            output_dir = os.path.join(args.output_root, factory, str(seed))

        # Add gravity suffix for gravity variant
        if gravity:
            output_dir = output_dir + "_gravity"

        result = render_one(
            args.blender, args.base, factory, seed, gpu_id, envmap_path,
            output_dir, args.resolution, args.fps, args.duration, args.samples,
            args.skip_nobg, args.skip_bg, gravity=gravity,
        )
        results_list.append(result)


def run_rendering(args, factories, seeds):
    """Stage 3: Render articulation videos across GPUs."""
    print(f"\n{'='*60}")
    print("Stage 3: RENDER ARTICULATION VIDEOS")
    print(f"{'='*60}")

    # Resolve envmaps
    if args.all_envmaps:
        envmap_files = sorted([
            f for f in os.listdir(args.envmap_dir)
            if f.endswith(".exr") or f.endswith(".hdr")
        ])
    elif args.envmaps:
        envmap_files = args.envmaps.split(",")
    else:
        envmap_files = [args.envmap]

    # Build job list: (factory, seed, gravity)
    render_jobs = []
    for factory in factories:
        for seed in seeds:
            render_jobs.append((factory, seed, False))
            # Optionally add gravity variant for BottleFactory
            if args.bottle_gravity and factory == "BottleFactory":
                render_jobs.append((factory, seed, True))

    use_envmap_subdir = len(envmap_files) > 1

    total_jobs = len(render_jobs) * len(envmap_files)
    gpus = [args.gpu_offset + i for i in range(args.n_gpus)]

    print(f"  Factories: {len(factories)}, Seeds: {len(seeds)}")
    print(f"  Envmaps: {len(envmap_files)}")
    print(f"  Jobs per envmap: {len(render_jobs)}")
    print(f"  Total render jobs: {total_jobs}")
    print(f"  GPUs: {gpus}")
    print(f"  Output: {args.output_root}")

    if args.dry_run:
        print("\n  [DRY RUN] Jobs:")
        for factory, seed, gravity in render_jobs[:20]:
            g = " [GRAVITY]" if gravity else ""
            print(f"    {factory} seed={seed}{g}")
        if len(render_jobs) > 20:
            print(f"    ... and {len(render_jobs)-20} more")
        return

    # Process each envmap
    t0_all = time.time()
    total_ok, total_skip, total_fail = 0, 0, 0
    all_failed = []

    for ei, envmap_file in enumerate(envmap_files):
        envmap_path = os.path.join(args.envmap_dir, envmap_file)
        if not os.path.exists(envmap_path):
            # If it's a full path already
            if os.path.exists(envmap_file):
                envmap_path = envmap_file
            else:
                print(f"\n  WARNING: envmap not found: {envmap_path}, skipping")
                continue

        envmap_name = os.path.splitext(os.path.basename(envmap_file))[0] if use_envmap_subdir else None

        print(f"\n  --- Envmap [{ei+1}/{len(envmap_files)}]: {os.path.basename(envmap_file)} ---")

        # Check how many already done
        already_done = 0
        for factory, seed, gravity in render_jobs:
            if envmap_name:
                od = os.path.join(args.output_root, factory, str(seed), envmap_name)
            else:
                od = os.path.join(args.output_root, factory, str(seed))
            if gravity:
                od += "_gravity"
            if check_render_complete(od, args.skip_nobg, args.skip_bg):
                already_done += 1

        print(f"  Already done: {already_done}/{len(render_jobs)}")

        if already_done == len(render_jobs):
            total_skip += already_done
            print(f"  All done, skipping")
            continue

        # Distribute jobs across GPUs using simple round-robin
        # Each GPU gets its own list of jobs to process sequentially
        gpu_jobs = {gpu: [] for gpu in gpus}
        pending_jobs = []
        for job in render_jobs:
            factory, seed, gravity = job
            if envmap_name:
                od = os.path.join(args.output_root, factory, str(seed), envmap_name)
            else:
                od = os.path.join(args.output_root, factory, str(seed))
            if gravity:
                od += "_gravity"
            if not check_render_complete(od, args.skip_nobg, args.skip_bg):
                pending_jobs.append(job)

        # Round-robin assignment
        for i, job in enumerate(pending_jobs):
            gpu = gpus[i % len(gpus)]
            gpu_jobs[gpu].append(job)

        print(f"  Pending: {len(pending_jobs)}, per GPU: "
              + ", ".join(f"GPU{g}={len(gpu_jobs[g])}" for g in gpus))

        # Launch one subprocess per GPU
        t0 = time.time()
        success, skipped, failed = 0, 0, 0
        failed_jobs = []

        # Use ProcessPoolExecutor: submit all pending jobs, each specifying its GPU
        with ProcessPoolExecutor(max_workers=args.n_gpus) as executor:
            futures = {}
            for i, job in enumerate(pending_jobs):
                factory, seed, gravity = job
                gpu = gpus[i % len(gpus)]

                if envmap_name:
                    output_dir = os.path.join(args.output_root, factory, str(seed), envmap_name)
                else:
                    output_dir = os.path.join(args.output_root, factory, str(seed))
                if gravity:
                    output_dir += "_gravity"

                fut = executor.submit(
                    render_one,
                    args.blender, args.base, factory, seed, gpu, envmap_path,
                    output_dir, args.resolution, args.fps, args.duration, args.samples,
                    args.skip_nobg, args.skip_bg, gravity,
                )
                futures[fut] = (factory, seed, gravity, gpu)

            for i, fut in enumerate(as_completed(futures)):
                factory, seed, ok, msg = fut.result()
                _, _, gravity, gpu = futures[fut]
                done = i + 1
                elapsed = time.time() - t0
                eta = elapsed / done * (len(pending_jobs) - done) if done else 0

                g = " [gravity]" if gravity else ""
                if "skipped" in msg:
                    skipped += 1
                    tag = "SKIP"
                elif ok:
                    success += 1
                    tag = "OK  "
                else:
                    failed += 1
                    failed_jobs.append((factory, seed, msg))
                    tag = "FAIL"

                print(f"  [{done}/{len(pending_jobs)}] [{tag}] GPU{gpu} "
                      f"{factory} seed={seed}{g}: {msg} "
                      f"({elapsed:.0f}s, ETA {eta:.0f}s)")

        total_ok += success
        total_skip += skipped + already_done
        total_fail += failed
        all_failed.extend(failed_jobs)

        elapsed_envmap = time.time() - t0
        print(f"  Envmap done: {success} ok, {skipped} skip, {failed} fail "
              f"({elapsed_envmap:.0f}s)")

    elapsed_total = time.time() - t0_all
    print(f"\n{'='*60}")
    print(f"RENDERING COMPLETE ({elapsed_total:.0f}s)")
    print(f"  OK: {total_ok}, Skipped: {total_skip}, Failed: {total_fail}")
    print(f"{'='*60}")

    if all_failed:
        fail_path = os.path.join(args.output_root, "failed_renders.json")
        with open(fail_path, "w") as fp:
            json.dump([{"factory": f, "seed": s, "error": m} for f, s, m in all_failed],
                      fp, indent=2)
        print(f"  Failed jobs saved to {fail_path}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Resolve paths
    if args.output_root is None:
        args.output_root = os.path.join(args.base, "outputs", "motion_videos")
    os.makedirs(args.output_root, exist_ok=True)

    # Resolve factories
    if args.factory:
        factories = [args.factory]
    else:
        factories = FACTORIES

    seeds = list(range(args.seed_start, args.seed_end))

    print(f"{'='*60}")
    print(f"Infinite Mobility Batch Pipeline")
    print(f"{'='*60}")
    print(f"  Base: {args.base}")
    print(f"  Blender: {args.blender}")
    print(f"  Factories: {len(factories)}")
    print(f"  Seeds: {args.seed_start}-{args.seed_end-1} ({len(seeds)} total)")
    print(f"  GPUs: {args.n_gpus} (offset={args.gpu_offset})")
    print(f"  Output: {args.output_root}")
    print(f"  Resolution: {args.resolution}, FPS: {args.fps}, Duration: {args.duration}s")
    print(f"  Samples: {args.samples}")

    # Validate blender exists
    if not os.path.exists(args.blender):
        print(f"\nERROR: Blender not found: {args.blender}")
        print("  Set --blender to your Blender path")
        sys.exit(1)

    # Stage 1: Generate
    if not args.render_only and not args.split_only:
        run_generation(args, factories, seeds)
    else:
        print(f"\n  [SKIP] Stage 1: Generation")

    # Stage 2: Split
    if not args.render_only and not args.no_split:
        run_splitting(args, factories)
    else:
        print(f"\n  [SKIP] Stage 2: Splitting")

    # Stage 3: Render
    if not args.no_render:
        run_rendering(args, factories, seeds)
    else:
        print(f"\n  [SKIP] Stage 3: Rendering")

    print(f"\nAll stages complete!")


if __name__ == "__main__":
    main()
