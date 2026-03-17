#!/usr/bin/env python3
"""
Batch negative-sample render dispatcher.

Walks precompute_output/, finds all metadata.json, distributes negative render
jobs across multiple GPUs. Each job = one object (all animodes x all neg_types).

Usage:
  python render_nega_batch.py --gpu_ids 1,2 --samples 8
  python render_nega_batch.py --gpu_ids 1,2 --factory lamp   # only lamp
"""

import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

BLENDER = os.environ.get("BLENDER_BIN", "/mnt/data/yurh/blender-4.2.18-linux-x64/blender")
NEGA_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_negative.py")


def find_nega_jobs(precompute_dir, nega_output_dir, factory_filter=None):
    """Find all metadata.json that need negative rendering.

    Returns list of (metadata_path, n_animodes).
    Skips objects that already have negative_metadata.json in nega_output_dir.
    """
    jobs = []
    for factory_dir in sorted(os.listdir(precompute_dir)):
        if factory_filter and factory_dir != factory_filter:
            continue
        factory_path = os.path.join(precompute_dir, factory_dir)
        if not os.path.isdir(factory_path):
            continue
        for seed_dir in sorted(os.listdir(factory_path)):
            seed_path = os.path.join(factory_path, seed_dir)
            meta_path = os.path.join(seed_path, "metadata.json")
            if not os.path.exists(meta_path):
                continue

            # Check if already done (negative_metadata.json in output dir)
            nega_done = os.path.join(nega_output_dir, factory_dir, seed_dir,
                                     "negative_metadata.json")
            if os.path.exists(nega_done):
                continue

            try:
                with open(meta_path) as f:
                    metadata = json.load(f)
                n_animodes = len(metadata.get("splits", {}))
                if n_animodes == 0:
                    continue
            except Exception:
                continue

            jobs.append((meta_path, n_animodes))

    return jobs


def run_nega_job(args_tuple):
    """Run negative rendering for one object (all animodes x all neg_types)."""
    meta_path, gpu_id, nega_output_dir, samples, views, resolution = args_tuple

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    cmd = [
        BLENDER, "--background", "--python", NEGA_SCRIPT, "--",
        "--metadata", meta_path,
        "--animode", "all",
        "--output_dir", nega_output_dir,
        "--views", *views,
        "--color_mode", "realistic",
        "--bg_mode", "nobg",
        "--resolution", str(resolution),
        "--samples", str(samples),
        "--skip_existing",
    ]

    # Extract label from path
    parts = meta_path.split("/")
    label = f"{parts[-3]}/{parts[-2]}" if len(parts) >= 3 else meta_path

    t0 = time.time()
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
        dt = time.time() - t0
        if result.returncode == 0:
            # Extract DONE line
            for line in result.stdout.split("\n"):
                if "DONE" in line:
                    print(f"  [GPU{gpu_id}] OK    {label} ({dt:.0f}s) {line.strip()}")
                    break
            else:
                print(f"  [GPU{gpu_id}] OK    {label} ({dt:.0f}s)")
            return True
        else:
            err = result.stderr[-200:] if result.stderr else result.stdout[-200:]
            print(f"  [GPU{gpu_id}] FAIL  {label} ({dt:.0f}s): {err.strip()}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  [GPU{gpu_id}] TIMEOUT {label}")
        return False
    except Exception as e:
        print(f"  [GPU{gpu_id}] ERROR {label}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch negative sample renderer")
    parser.add_argument("--precompute_dir", default="./precompute_output",
                        help="Input precompute directory")
    parser.add_argument("--output_dir", default="./precompute_nega_output",
                        help="Output directory for negatives")
    parser.add_argument("--gpu_ids", type=str, default="1,2",
                        help="Comma-separated GPU IDs")
    parser.add_argument("--factory", type=str, default=None,
                        help="Only process this factory (e.g. 'lamp')")
    parser.add_argument("--views", nargs="+", default=["hemi_05"],
                        help="Views to render")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0,
                        help="Parallel workers (default: n_gpus)")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpu_ids.split(",")]
    n_workers = args.workers if args.workers > 0 else len(gpu_ids)

    jobs = find_nega_jobs(args.precompute_dir, args.output_dir, args.factory)
    total_animodes = sum(n for _, n in jobs)
    print(f"Negative batch render")
    print(f"  Objects: {len(jobs)}, Animodes: {total_animodes}")
    print(f"  Est videos: ~{total_animodes * 6}")
    print(f"  GPUs: {gpu_ids}, workers: {n_workers}")
    print(f"  Output: {args.output_dir}")
    print()

    if not jobs:
        print("Nothing to render.")
        return

    # Assign GPUs round-robin
    job_args = []
    for i, (meta_path, _) in enumerate(jobs):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        job_args.append((meta_path, gpu_id, args.output_dir,
                         args.samples, args.views, args.resolution))

    t0 = time.time()
    if n_workers == 1:
        results = [run_nega_job(a) for a in job_args]
    else:
        with Pool(n_workers) as pool:
            results = pool.map(run_nega_job, job_args)

    ok = sum(1 for r in results if r)
    fail = sum(1 for r in results if not r)
    dt = time.time() - t0
    print(f"\nBatch complete: {ok} success, {fail} failed out of {len(jobs)} objects ({dt/3600:.1f}h)")


if __name__ == "__main__":
    main()
