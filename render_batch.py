#!/usr/bin/env python3
"""
Batch render dispatcher for Infinigen-Sim precomputed animodes.

Walks output directory, finds all metadata.json, distributes render jobs
across multiple GPUs via CUDA_VISIBLE_DEVICES.

Usage:
  python render_batch.py --output_dir ./output --n_gpus 4 --views hemi --duration 4
"""

import argparse
import json
import os
import subprocess
import sys
from multiprocessing import Pool

BLENDER = "/mnt/data/yurh/blender-3.6.0-linux-x64/blender"
RENDER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_animode.py")


def find_render_jobs(output_dir, views, check_view="hemi_00"):
    """Find all (metadata_path, animode_name) pairs that need rendering."""
    jobs = []
    for factory_dir in sorted(os.listdir(output_dir)):
        factory_path = os.path.join(output_dir, factory_dir)
        if not os.path.isdir(factory_path):
            continue
        for seed_dir in sorted(os.listdir(factory_path)):
            seed_path = os.path.join(factory_path, seed_dir)
            meta_path = os.path.join(seed_path, "metadata.json")
            if not os.path.exists(meta_path):
                continue

            with open(meta_path) as f:
                metadata = json.load(f)

            for animode_name in sorted(metadata.get("splits", {}).keys()):
                # Check if already rendered (use first view as sentinel)
                sentinel = os.path.join(seed_path, animode_name, f"{check_view}_nobg.mp4")
                if os.path.exists(sentinel):
                    continue
                jobs.append((meta_path, animode_name))

    return jobs


def run_render_job(args_tuple):
    """Run a single render job (called by pool workers)."""
    meta_path, animode_name, gpu_id, views, duration, resolution, fps, samples = args_tuple

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        BLENDER, "--background", "--python", RENDER_SCRIPT, "--",
        "--metadata", meta_path,
        "--animode", animode_name,
        "--views", *views,
        "--resolution", str(resolution),
        "--fps", str(fps),
        "--duration", str(duration),
        "--samples", str(samples),
        "--skip_existing",
    ]

    label = f"[GPU{gpu_id}] {os.path.basename(os.path.dirname(os.path.dirname(meta_path)))}/{os.path.basename(os.path.dirname(meta_path))}/{animode_name}"
    print(f"  START {label}")

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            print(f"  DONE  {label}")
            return True
        else:
            err = result.stderr[-300:] if result.stderr else result.stdout[-300:]
            print(f"  FAIL  {label}: {err}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {label}")
        return False
    except Exception as e:
        print(f"  ERROR {label}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch render animode videos")
    parser.add_argument("--output_dir", required=True, help="Output directory with metadata.json files")
    parser.add_argument("--n_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3'). Overrides n_gpus.")
    parser.add_argument("--views", nargs="+", default=["hemi"])
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: n_gpus)")
    args = parser.parse_args()

    if args.gpu_ids:
        gpu_ids = [int(g) for g in args.gpu_ids.split(",")]
    else:
        gpu_ids = list(range(args.n_gpus))

    n_workers = args.workers if args.workers > 0 else len(gpu_ids)

    # Determine sentinel view for skip check
    check_view = args.views[0] if args.views[0] not in ("all", "hemi", "orbit", "sweep") else "hemi_00"

    jobs = find_render_jobs(args.output_dir, args.views, check_view)
    print(f"Found {len(jobs)} render jobs across {args.output_dir}")
    print(f"Using GPUs: {gpu_ids}, workers: {n_workers}")

    if not jobs:
        print("Nothing to render.")
        return

    # Assign GPUs round-robin
    job_args = []
    for i, (meta_path, animode_name) in enumerate(jobs):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        job_args.append((meta_path, animode_name, gpu_id,
                         args.views, args.duration, args.resolution,
                         args.fps, args.samples))

    # Run with process pool
    if n_workers == 1:
        results = [run_render_job(a) for a in job_args]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(run_render_job, job_args)

    ok = sum(1 for r in results if r)
    fail = sum(1 for r in results if not r)
    print(f"\nBatch complete: {ok} success, {fail} failed out of {len(jobs)} jobs")


if __name__ == "__main__":
    main()
