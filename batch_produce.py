#!/usr/bin/env python3
"""batch_produce.py - Production pipeline for articulated object data.

Pipeline per object:
  1. Setup scene (PhysXNet/PhysX_mobility: convert dataset to IM format)
  2. Precompute dual-part splits (normalize + split per animode)
  3. Render videos (all valid animodes x views)

Usage:
  python batch_produce.py                          # 6 test factories, 1 seed each
  python batch_produce.py --all --num_seeds 3      # ALL factories, 3 seeds each
  python batch_produce.py --num_seeds 100          # default 6 factories, 100 seeds
  python batch_produce.py --factories DishwasherFactory ElectronicsPhysXNetFactory
  python batch_produce.py --num_seeds 1000 --skip_render  # splits only
"""
import subprocess
import os
import sys
import json
import time
import argparse
from multiprocessing import Pool

# ======================================================================
# Paths
# ======================================================================
BLENDER = "/mnt/data/yurh/blender-3.6.0-linux-x64/blender"
BASE_DIR = "/mnt/data/yurh/Infinigen-Sim"
IM_BASE = "/mnt/cpfs/yurh/Infinite-Mobility"
PHYSXNET_JSON_DIR = "/mnt/data/fulian/dataset/PhysXNet/version_1/finaljson"
PHYSXMOB_JSON_DIR = "/mnt/data/fulian/dataset/PhysX_mobility/finaljson"

NUM_GPUS = 4

# ======================================================================
# Default test factories (2 IM + 2 PhysXNet + 2 PhysX_mobility)
# ======================================================================
DEFAULT_FACTORIES = [
    "DishwasherFactory",
    "MicrowaveFactory",
    "ElectronicsPhysXNetFactory",
    "FurniturePhysXNetFactory",
    "TransportPhysXMobilityFactory",
    "FurniturePhysXMobilityFactory",
]

# Render settings
RESOLUTION = 256
SAMPLES = 16
DURATION = 2.0
FPS = 15


# ======================================================================
# Source detection + seed enumeration
# ======================================================================

def detect_source(factory_name):
    """Detect factory source: 'im', 'physxnet', or 'physxmob'."""
    try:
        from physxnet_factory_rules import factory_dataset
        ds = factory_dataset(factory_name)
        if ds:
            return ds
    except ImportError:
        pass
    if "PhysXNet" in factory_name:
        return "physxnet"
    if "PhysXMobility" in factory_name:
        return "physxmob"
    return "im"


def get_seeds(factory_name, source, num_seeds):
    """Get seed list for a factory.

    IM: seeds are sequential integers (0, 1, 2, ...)
    PhysXNet/PhysX_mobility: seeds are object IDs from the dataset
    """
    if source == "im":
        # Check which seeds actually exist
        im_dir = os.path.join(IM_BASE, "outputs", factory_name)
        if os.path.isdir(im_dir):
            existing = sorted([int(d) for d in os.listdir(im_dir) if d.isdigit()])
            return [str(s) for s in existing[:num_seeds]]
        return [str(i) for i in range(num_seeds)]

    elif source == "physxnet":
        try:
            from physxnet_factory_rules import get_physxnet_factory_ids
            ids = get_physxnet_factory_ids(factory_name)
            return [str(oid) for oid in ids[:num_seeds]]
        except ImportError:
            return []

    elif source == "physxmob":
        try:
            from physxnet_factory_rules import get_physxmob_factory_ids
            ids = get_physxmob_factory_ids(factory_name)
            return [str(oid) for oid in ids[:num_seeds]]
        except ImportError:
            return []

    return []


def get_scene_base(source):
    """Get base directory for scene outputs."""
    if source == "im":
        return IM_BASE
    return BASE_DIR


def discover_all_factories():
    """Auto-discover all available factories from IM outputs + PhysXNet/PhysXMob rules.

    Returns sorted list of factory names.
    """
    factories = set()

    # 1. IM factories: scan outputs/ for dirs with scene.urdf
    im_outputs = os.path.join(IM_BASE, "outputs")
    if os.path.isdir(im_outputs):
        for fname in os.listdir(im_outputs):
            fdir = os.path.join(im_outputs, fname)
            if not os.path.isdir(fdir) or "PhysX" in fname:
                continue
            # Check at least one seed has a scene.urdf
            for seed_name in os.listdir(fdir):
                urdf = os.path.join(fdir, seed_name, "scene.urdf")
                if os.path.exists(urdf):
                    factories.add(fname)
                    break

    # 2. PhysXNet factories
    try:
        from physxnet_factory_rules import get_physxnet_factory_ids
        # Discover by scanning JSON categories → factory name mapping
        import glob, json as _json
        cat_to_factory = {}
        for jf in glob.glob(os.path.join(PHYSXNET_JSON_DIR, "*.json")):
            try:
                with open(jf) as f:
                    d = _json.load(f)
                cat = d.get("category", "")
                if cat and cat not in cat_to_factory:
                    factory_name = cat.replace(" ", "") + "PhysXNetFactory"
                    # Verify this factory can produce IDs
                    ids = get_physxnet_factory_ids(factory_name)
                    if ids:
                        cat_to_factory[cat] = factory_name
                        factories.add(factory_name)
            except Exception:
                continue
    except ImportError:
        pass

    # 3. PhysX_mobility factories
    try:
        from physxnet_factory_rules import get_physxmob_factory_ids
        import glob, json as _json
        cat_to_factory_mob = {}
        for jf in glob.glob(os.path.join(PHYSXMOB_JSON_DIR, "*.json")):
            try:
                with open(jf) as f:
                    d = _json.load(f)
                cat = d.get("category", "")
                if cat and cat not in cat_to_factory_mob:
                    factory_name = cat.replace(" ", "") + "PhysXMobilityFactory"
                    ids = get_physxmob_factory_ids(factory_name)
                    if ids:
                        cat_to_factory_mob[cat] = factory_name
                        factories.add(factory_name)
            except Exception:
                continue
    except ImportError:
        pass

    return sorted(factories)


# ======================================================================
# Phase 1: Setup scenes (PhysXNet/PhysX_mobility only)
# ======================================================================

def setup_scene(factory, seed, source):
    """Setup scene for a PhysXNet/PhysX_mobility object.

    Returns True on success.
    """
    if source == "im":
        return True  # IM scenes already exist

    setup_script = os.path.join(BASE_DIR, "setup_physxnet_scene.py")
    # setup_physxnet_scene uses "physx_mobility" not "physxmob"
    src_arg = "physx_mobility" if source == "physxmob" else source
    cmd = [sys.executable, setup_script,
           "--id", str(seed), "--factory", factory, "--source", src_arg]
    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    return result.returncode == 0


# ======================================================================
# Phase 2: Precompute splits
# ======================================================================

def precompute_split(factory, seed, source, output_dir, force=False):
    """Run split_precompute.py for one object.

    Returns True on success.
    """
    split_script = os.path.join(BASE_DIR, "split_precompute.py")
    base = get_scene_base(source)
    cmd = [sys.executable, split_script,
           "--factory", factory, "--seed", str(seed),
           "--source", source, "--base", base,
           "--output_dir", output_dir]
    if force:
        cmd.append("--force")
    result = subprocess.run(cmd, cwd=BASE_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        err = result.stdout.strip() or result.stderr.strip()
        if err:
            last_line = err.strip().split('\n')[-1]
            print(f"    split error: {last_line[:100]}")
    return result.returncode == 0


# ======================================================================
# Phase 3: Render videos
# ======================================================================

def build_render_jobs(factory, seed, source, output_dir):
    """Build render jobs from precompute metadata.

    Videos are placed in the SAME directory as the split OBJs (co-located).
    Returns list of render job dicts.
    """
    obj_base = os.path.join(output_dir, factory, str(seed))
    meta_path = os.path.join(obj_base, "metadata.json")
    if not os.path.exists(meta_path):
        return []

    with open(meta_path) as f:
        meta = json.load(f)

    splits = meta.get("splits", {})
    jobs = []

    for split_key, split_info in splits.items():
        if split_key == "default":
            animode = 3
            out_sub = "default"
        elif split_key.startswith("anim"):
            animode = int(split_key[4:])
            out_sub = split_key
        else:
            continue

        # Choose views based on animode type
        if split_info.get("type"):
            # Type-based animode (anim0/1/2)
            static_views = ["hemi_00", "hemi_08"]
            moving_views = ["orbit_00"]
        elif animode >= 10:
            # Per-joint animode
            static_views = ["hemi_00"]
            moving_views = ["orbit_00"]
        else:
            # Default (animode 3)
            static_views = ["hemi_00", "hemi_08"]
            moving_views = ["orbit_00"]

        # Videos go into same dir as splits: {output_dir}/{Factory}/{seed}/{animode}/
        out_dir = os.path.join(obj_base, out_sub)
        jobs.append({
            "factory": factory,
            "seed": str(seed),
            "source": source,
            "animode": animode,
            "static_views": static_views,
            "moving_views": moving_views,
            "out_dir": out_dir,
        })

    return jobs


def run_render(args):
    """Run a single Blender render. Called by multiprocessing Pool."""
    gpu_id = args["gpu_id"]
    factory = args["factory"]
    seed = args["seed"]
    source = args["source"]
    animode = args["animode"]
    static_views = args["static_views"]
    moving_views = args["moving_views"]
    out_dir = args["out_dir"]

    os.makedirs(out_dir, exist_ok=True)

    # IM scenes live under IM_BASE, PhysXNet/PhysX_mobility under BASE_DIR
    scene_base = IM_BASE if source == "im" else BASE_DIR

    cmd = [
        BLENDER, "--background", "--python-use-system-env",
        "--python", os.path.join(BASE_DIR, "render_articulation.py"),
        "--",
        "--factory", factory,
        "--seed", str(seed),
        "--base", scene_base,
        "--device", "0",
        "--output_dir", out_dir,
        "--resolution", str(RESOLUTION),
        "--samples", str(SAMPLES),
        "--duration", str(DURATION),
        "--fps", str(FPS),
        "--animode", str(animode),
        "--skip_bg",
    ]

    if static_views:
        cmd += ["--views"] + static_views
    if moving_views:
        cmd += ["--moving_views"] + moving_views

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    label = f"GPU{gpu_id} {factory}/{seed} anim{animode}"
    t0 = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=BASE_DIR, env=env,
        )
        elapsed = time.time() - t0

        mp4_count = 0
        if os.path.isdir(out_dir):
            for f in os.listdir(out_dir):
                if f.endswith("_nobg.mp4"):
                    mp4_count += 1

        expected = len(static_views) + len(moving_views)
        ok = mp4_count >= expected
        info = f"{elapsed:.0f}s, {mp4_count}/{expected} mp4s"
        if not ok:
            for line in result.stdout.split('\n'):
                if 'WARNING' in line or 'no matching' in line.lower():
                    info += f" | {line.strip()[:80]}"
                    break
            if result.returncode != 0:
                stderr_lines = [l for l in result.stderr.split('\n') if l.strip()]
                if stderr_lines:
                    info += f" | stderr: {stderr_lines[-1][:80]}"
        return ok, label, info

    except subprocess.TimeoutExpired:
        return False, label, "TIMEOUT"
    except Exception as e:
        return False, label, str(e)[:80]


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Production pipeline for articulated objects")
    parser.add_argument("--factories", nargs="+", default=None,
                        help="Factory names to process (default: 6 test factories)")
    parser.add_argument("--all", action="store_true",
                        help="Auto-discover and process ALL factories (IM + PhysXNet + PhysXMob)")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of seeds per factory (default: 1)")
    parser.add_argument("--output_dir", type=str,
                        default=os.path.join(BASE_DIR, "precompute"),
                        help="Unified output directory (splits + videos co-located)")
    parser.add_argument("--skip_render", action="store_true",
                        help="Only setup + split, skip rendering")
    parser.add_argument("--skip_setup", action="store_true",
                        help="Skip scene setup (assume already done)")
    parser.add_argument("--skip_split", action="store_true",
                        help="Skip split precompute (assume already done)")
    parser.add_argument("--force_split", action="store_true",
                        help="Force re-split even if metadata.json exists")
    parser.add_argument("--num_gpus", type=int, default=NUM_GPUS)
    args = parser.parse_args()

    if args.all:
        factories = discover_all_factories()
        print(f"Auto-discovered {len(factories)} factories")
    elif args.factories:
        factories = args.factories
    else:
        factories = DEFAULT_FACTORIES
    t_start = time.time()

    # ---- Enumerate all (factory, seed, source) tuples ----
    all_objects = []
    print(f"Enumerating objects for {len(factories)} factories, num_seeds={args.num_seeds}...")
    for factory in factories:
        source = detect_source(factory)
        seeds = get_seeds(factory, source, args.num_seeds)
        print(f"  {factory} ({source}): {len(seeds)} seeds")
        for seed in seeds:
            all_objects.append((factory, seed, source))

    print(f"\nTotal objects: {len(all_objects)}")

    # ---- Phase 1: Setup scenes ----
    if not args.skip_setup:
        print("\n" + "=" * 60)
        print("Phase 1: Setting up scenes")
        print("=" * 60)
        setup_ok = 0
        setup_skip = 0
        for factory, seed, source in all_objects:
            if source == "im":
                setup_skip += 1
                continue
            ok = setup_scene(factory, seed, source)
            status = "OK" if ok else "FAIL"
            print(f"  [{setup_ok + 1}] {status} {factory}/{seed}")
            if ok:
                setup_ok += 1
        print(f"  Setup: {setup_ok} done, {setup_skip} skipped (IM)")

    # ---- Phase 2: Precompute splits ----
    if not args.skip_split:
        print("\n" + "=" * 60)
        print("Phase 2: Precomputing dual-part splits")
        print("=" * 60)
        split_ok = 0
        split_skip = 0
        for i, (factory, seed, source) in enumerate(all_objects):
            meta_path = os.path.join(args.output_dir, factory, str(seed), "metadata.json")
            if not args.force_split and os.path.exists(meta_path):
                split_skip += 1
                print(f"  [{i+1}/{len(all_objects)}] SKIP {factory}/{seed} (exists)")
                continue
            ok = precompute_split(factory, seed, source, args.output_dir, force=args.force_split)
            status = "OK" if ok else "FAIL"
            print(f"  [{i+1}/{len(all_objects)}] {status} {factory}/{seed}")
            if ok:
                split_ok += 1
        print(f"  Splits: {split_ok} done, {split_skip} skipped")

    # ---- Phase 3: Render videos ----
    if args.skip_render:
        print("\nSkipping render (--skip_render)")
    else:
        print("\n" + "=" * 60)
        print("Phase 3: Rendering videos")
        print("=" * 60)

        # Build all render jobs from precompute metadata
        all_jobs = []
        gpu_idx = 0
        for factory, seed, source in all_objects:
            jobs = build_render_jobs(factory, seed, source, args.output_dir)
            for job in jobs:
                job["gpu_id"] = gpu_idx % args.num_gpus
                gpu_idx += 1
            all_jobs.extend(jobs)

        print(f"  Total render jobs: {len(all_jobs)} across {args.num_gpus} GPUs\n")

        completed = 0
        failed = 0
        with Pool(processes=args.num_gpus) as pool:
            for ok, label, info in pool.imap_unordered(run_render, all_jobs):
                completed += 1
                status = "OK" if ok else "FAIL"
                print(f"  [{completed:3d}/{len(all_jobs)}] {status} {label}: {info}",
                      flush=True)
                if not ok:
                    failed += 1

        print(f"\n  Render: {completed - failed}/{completed} succeeded, {failed} failed")

    # ---- Summary ----
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Pipeline complete in {elapsed:.0f}s")
    print("=" * 60)

    # Count outputs
    split_count = 0
    for factory, seed, source in all_objects:
        meta = os.path.join(args.output_dir, factory, str(seed), "metadata.json")
        if os.path.exists(meta):
            split_count += 1
    print(f"  Precompute: {split_count}/{len(all_objects)} objects")
    print(f"    -> {os.path.abspath(args.output_dir)}")

    if not args.skip_render:
        mp4_count = 0
        for root, dirs, files in os.walk(args.output_dir):
            for f in files:
                if f.endswith(".mp4"):
                    mp4_count += 1
        print(f"  Videos: {mp4_count} mp4 files")
        print(f"    -> {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
