#!/usr/bin/env python3
"""Batch render PhysXNet/PhysX_mobility test objects with multiple animodes and views.

Animode system:
  0 = revolute joints only
  1 = prismatic joints only
  2 = continuous joints only
  3 = all joints together
  10+N = per-joint mode: animate only the Nth movable joint (0-indexed)
"""
import subprocess
import os
import sys
import json
import time
from multiprocessing import Pool

BLENDER = "/mnt/data/yurh/blender-3.6.0-linux-x64/blender"
BASE_DIR = "/mnt/data/yurh/Infinigen-Sim"
OUTPUT_BASE = os.path.join(BASE_DIR, "test_output")
PHYSXNET_JSON_DIR = "/mnt/data/fulian/dataset/PhysXNet/version_1/finaljson"
PHYSXMOB_JSON_DIR = "/mnt/data/fulian/dataset/PhysX_mobility/finaljson"

# (factory, seed, object_desc, source)
OBJECTS = [
    # PhysXNet
    ("FurniturePhysXNetFactory", "46088", "SinkCabinet", "physxnet"),
    ("ChairPhysXNetFactory", "39882", "BarberChair", "physxnet"),
    ("LightingPhysXNetFactory", "17394", "TrackLight", "physxnet"),
    ("ElectronicsPhysXNetFactory", "6723", "SmartSpeaker", "physxnet"),
    ("AppliancePhysXNetFactory", "10459", "Refrigerator", "physxnet"),
    ("BagPhysXNetFactory", "8871", "Backpack", "physxnet"),
    ("DoorPhysXNetFactory", "9288", "SwingDoor", "physxnet"),
    ("ContainerPhysXNetFactory", "47219", "RectEnclosure", "physxnet"),
    # PhysX_mobility
    ("AgriPhysXMobilityFactory", "101064", "Cart", "physx_mobility"),
    ("ArchPhysXMobilityFactory", "103242", "Window", "physx_mobility"),
    ("BathPhysXMobilityFactory", "101517", "SoapDispenser", "physx_mobility"),
    ("BuildPhysXMobilityFactory", "102903", "SlidingWindow", "physx_mobility"),
]

NUM_GPUS = 4


def count_movable_joints(obj_id, source):
    """Count movable joints from group_info JSON."""
    if source == "physxnet":
        json_path = os.path.join(PHYSXNET_JSON_DIR, f"{obj_id}.json")
    else:
        json_path = os.path.join(PHYSXMOB_JSON_DIR, f"{obj_id}.json")

    if not os.path.exists(json_path):
        return 0

    with open(json_path) as f:
        data = json.load(f)

    gi = data.get("group_info", {})
    count = 0
    joint_types = {"revolute": 0, "prismatic": 0, "continuous": 0}
    for gid, val in gi.items():
        if isinstance(val, list) and len(val) >= 4 and isinstance(val[-1], str):
            jtype = val[-1]
            if jtype == 'C':
                count += 1
                joint_types["revolute"] += 1
            elif jtype == 'B':
                count += 1
                joint_types["prismatic"] += 1
            elif jtype == 'D':
                count += 1
                joint_types["revolute"] += 1
            elif jtype == 'CB':
                count += 1
                joint_types["revolute"] += 1
    return count, joint_types


def run_render(args):
    """Run a single Blender render."""
    gpu_id, factory, seed, animode, static_views, moving_views, out_dir, desc = args

    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        BLENDER, "--background", "--python-use-system-env",
        "--python", os.path.join(BASE_DIR, "render_articulation.py"),
        "--",
        "--factory", factory,
        "--seed", str(seed),
        "--device", "0",
        "--output_dir", out_dir,
        "--resolution", "256",
        "--samples", "16",
        "--duration", "2.0",
        "--fps", "15",
        "--animode", str(animode),
        "--skip_bg",
    ]

    if static_views:
        cmd += ["--views"] + static_views
    if moving_views:
        cmd += ["--moving_views"] + moving_views

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=BASE_DIR, env=env,
        )
        elapsed = time.time() - t0

        # Count output mp4s
        mp4_count = 0
        if os.path.isdir(out_dir):
            for f in os.listdir(out_dir):
                if f.endswith("_nobg.mp4"):
                    mp4_count += 1

        expected = len(static_views) + len(moving_views)
        ok = mp4_count >= expected
        info = f"{elapsed:.0f}s, {mp4_count}/{expected} mp4s"
        if not ok:
            # Extract key warnings/errors from stdout
            for line in result.stdout.split('\n'):
                if 'WARNING' in line or 'no matching' in line.lower():
                    info += f" | {line.strip()[:80]}"
                    break
            if result.returncode != 0:
                # Show last few lines of stderr
                stderr_lines = [l for l in result.stderr.split('\n') if l.strip()]
                if stderr_lines:
                    info += f" | stderr: {stderr_lines[-1][:80]}"
        return ok, f"GPU{gpu_id} {desc}_{seed} anim{animode}", info
    except subprocess.TimeoutExpired:
        return False, f"GPU{gpu_id} {desc}_{seed} anim{animode}", "TIMEOUT"
    except Exception as e:
        return False, f"GPU{gpu_id} {desc}_{seed} anim{animode}", str(e)[:80]


def main():
    # First, re-setup all scenes (ensures MTL files are copied etc.)
    print("Setting up scenes...")
    setup_script = os.path.join(BASE_DIR, "setup_physxnet_scene.py")
    for factory, seed, desc, source in OBJECTS:
        cmd = [sys.executable, setup_script, "--id", seed, "--factory", factory, "--source", source]
        subprocess.run(cmd, cwd=BASE_DIR, capture_output=True)
        print(f"  Setup: {desc}_{seed} ({source})")

    # Precompute dual-part splits for all objects
    print("\nPrecomputing dual-part splits...")
    split_script = os.path.join(BASE_DIR, "split_precompute.py")
    split_ok = 0
    for factory, seed, desc, source in OBJECTS:
        # Map batch_test source names to split_precompute source names
        src_arg = "physxmob" if source == "physx_mobility" else source
        cmd = [sys.executable, split_script,
               "--factory", factory, "--seed", seed,
               "--source", src_arg,
               "--output_dir", os.path.join(BASE_DIR, "precompute")]
        result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
        if result.returncode == 0:
            split_ok += 1
            print(f"  Split: {desc}_{seed} OK")
        else:
            stderr = result.stderr.strip().split('\n')[-1] if result.stderr.strip() else ""
            stdout = result.stdout.strip().split('\n')[-1] if result.stdout.strip() else ""
            print(f"  Split: {desc}_{seed} FAIL: {stdout or stderr}")
    print(f"  Splits: {split_ok}/{len(OBJECTS)} succeeded")

    # Build job list with per-object animode configs
    jobs = []
    gpu_idx = 0

    for factory, seed, desc, source in OBJECTS:
        n_joints, jtypes = count_movable_joints(seed, source)

        # Type-based animodes: only render if object has that joint type
        # 0=revolute, 1=prismatic, 2=continuous, 3=all
        has_rev = jtypes["revolute"] > 0
        has_pri = jtypes["prismatic"] > 0
        has_con = jtypes["continuous"] > 0

        type_configs = []
        if has_rev:
            type_configs.append((0, ["hemi_00", "hemi_08"], ["orbit_00"]))
        if has_pri:
            type_configs.append((1, ["hemi_00", "hemi_08"], ["orbit_00"]))
        if has_con:
            type_configs.append((2, ["hemi_00"], []))
        # animode 3 (all) only if object has at least 2 different joint types
        if sum([has_rev, has_pri, has_con]) >= 2:
            type_configs.append((3, ["hemi_00", "hemi_08"], ["orbit_00"]))

        for animode, static_views, moving_views in type_configs:
            out_dir = os.path.join(OUTPUT_BASE, f"{desc}_{seed}", f"anim{animode}")
            jobs.append((gpu_idx % NUM_GPUS, factory, seed, animode,
                         static_views, moving_views, out_dir, desc))
            gpu_idx += 1

        # Per-joint animodes (10+N) for objects with multiple movable joints
        if n_joints > 1:
            for joint_idx in range(n_joints):
                animode = 10 + joint_idx
                out_dir = os.path.join(OUTPUT_BASE, f"{desc}_{seed}", f"anim{animode}_joint{joint_idx}")
                jobs.append((gpu_idx % NUM_GPUS, factory, seed, animode,
                             ["hemi_00"], ["orbit_00"], out_dir, desc))
                gpu_idx += 1

    print(f"\nTotal render jobs: {len(jobs)} across {NUM_GPUS} GPUs")
    print(f"Output: {OUTPUT_BASE}\n")

    completed = 0
    failed = 0

    with Pool(processes=NUM_GPUS) as pool:
        for ok, label, info in pool.imap_unordered(run_render, jobs):
            completed += 1
            status = "OK" if ok else "FAIL"
            print(f"  [{completed:2d}/{len(jobs)}] {status} {label}: {info}", flush=True)
            if not ok:
                failed += 1

    print(f"\nDone: {completed - failed}/{completed} succeeded, {failed} failed")

    # Summary of generated videos
    total_mp4 = 0
    for root, dirs, files in os.walk(OUTPUT_BASE):
        for f in sorted(files):
            if f.endswith(".mp4"):
                total_mp4 += 1
    print(f"Total mp4 files: {total_mp4}")
    print(f"Output: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
