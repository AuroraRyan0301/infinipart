#!/usr/bin/env python3
"""
Full test: IS (2 seeds) + PhysXNet (15) + PhysXMobility (15) precompute.
All output to /mnt/data/yurh/Infinigen-Sim/output/
"""

import os
import sys
import subprocess
import time
import json

BASE = "/mnt/data/yurh/Infinigen-Sim"
OUTPUT = os.path.join(BASE, "output")
PARTNET_DIR = "/mnt/data/yurh/dataset3D/Partnet"

# ======================================================================
# Phase 1: IS factories (generate 2 seeds each)
# ======================================================================
IS_FACTORIES = ["box", "cabinet", "dishwasher", "drawer", "lamp", "oven", "refrigerator"]
IS_SEEDS = [1001, 1002]

# ======================================================================
# Phase 2: PhysXMobility - 1 case per category (diverse), use PartNet URDFs
# ======================================================================
PHYSXMOB_CASES = [
    ("AppliancePhysXMobilityFactory", "100282"),      # WashingMachine 2j
    ("ClockPhysXMobilityFactory", "6500"),             # Clock 3j
    ("ContainerPhysXMobilityFactory", "100129"),       # Box 2j
    ("DoorPhysXMobilityFactory", "100982"),            # Door 3j
    ("ElectronicsPhysXMobilityFactory", "100064"),     # Electronics 1j
    ("FurniturePhysXMobilityFactory", "100520"),       # FoldingChair 1j
    ("KitchenPhysXMobilityFactory", "100015"),         # Kitchen 1j
    ("LightingPhysXMobilityFactory", "14563"),         # Lamp 2j
    ("LuggagePhysXMobilityFactory", "100550"),         # Luggage 2j
    ("OpticalPhysXMobilityFactory", "101284"),         # Optical 2j
    ("PlumbingPhysXMobilityFactory", "101319"),        # Plumbing 2j
    ("ToolPhysXMobilityFactory", "100142"),            # Pliers 1j
    ("TransportPhysXMobilityFactory", "100075"),       # Cart 2j
    ("WastePhysXMobilityFactory", "102153"),           # Trash 3j
    ("WritingPhysXMobilityFactory", "101685"),         # Pen 1j
]

# ======================================================================
# Phase 3: PhysXNet - 1 case per category (diverse), needs conversion
# ======================================================================
PHYSXNET_CASES = [
    ("AppliancePhysXNetFactory", "10007"),             # Appliance 2j
    ("AudioPhysXNetFactory", "10000"),                 # Audio 2j
    ("BagPhysXNetFactory", "13205"),                   # Bag 2j
    ("ChairPhysXNetFactory", "3353"),                  # Chair 2j
    ("ContainerPhysXNetFactory", "10349"),             # Container 1j
    ("DoorPhysXNetFactory", "12633"),                  # Door 2j
    ("ElectronicsPhysXNetFactory", "10005"),           # Electronics 3j
    ("FurniturePhysXNetFactory", "10048"),             # Furniture 3j
    ("KitchenPhysXNetFactory", "11239"),               # Kitchen 1j
    ("LightingPhysXNetFactory", "13292"),              # Lighting 2j
    ("PlumbingPhysXNetFactory", "1007"),               # Plumbing 2j
    ("ToolPhysXNetFactory", "10412"),                  # Tool 2j
    ("WastePhysXNetFactory", "10425"),                 # Waste 2j
    ("ClockPhysXNetFactory", "3474"),                  # Clock 1j
    ("ClothingPhysXNetFactory", "10096"),              # Clothing 2j
]


def run_cmd(cmd, cwd=None, timeout=300):
    """Run command, return (success, stdout)."""
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"


def generate_is_assets():
    """Generate IS factory URDF assets."""
    print("\n" + "=" * 70)
    print("PHASE 1: Generate IS factory assets")
    print("=" * 70)

    urdf_base = os.path.join(BASE, "sim_exports", "urdf")
    success, fail = 0, 0

    for factory in IS_FACTORIES:
        for seed in IS_SEEDS:
            urdf_dir = os.path.join(urdf_base, factory, str(seed))
            if os.path.exists(os.path.join(urdf_dir, f"{factory}.urdf")):
                print(f"  SKIP {factory}/{seed} (exists)")
                success += 1
                continue

            print(f"  Generating {factory}/{seed} ...", end=" ", flush=True)
            cmd = [sys.executable, "scripts/spawn_asset.py",
                   "-n", factory, "-s", str(seed), "-exp", "urdf",
                   "-dir", "./sim_exports"]
            ok, out = run_cmd(cmd, cwd=BASE, timeout=120)
            if ok:
                print("OK")
                success += 1
            else:
                # Check if it actually wrote the file
                if os.path.exists(os.path.join(urdf_dir, f"{factory}.urdf")):
                    print("OK (warning)")
                    success += 1
                else:
                    print(f"FAIL")
                    fail += 1

    print(f"  IS assets: {success} ok, {fail} failed")
    return success, fail


def convert_physxnet():
    """Convert PhysXNet JSON -> IM format (scene.urdf + objs)."""
    print("\n" + "=" * 70)
    print("PHASE 2: Convert PhysXNet cases")
    print("=" * 70)

    setup_script = os.path.join(BASE, "setup_physxnet_scene.py")
    # Copy from infinipart if not exists
    if not os.path.exists(setup_script):
        src = "/mnt/data/yurh/infinipart/setup_physxnet_scene.py"
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, setup_script)
            print(f"  Copied setup_physxnet_scene.py from infinipart")
        else:
            print(f"  ERROR: setup_physxnet_scene.py not found")
            return 0, len(PHYSXNET_CASES)

    success, fail = 0, 0
    for factory, oid in PHYSXNET_CASES:
        out_dir = os.path.join(BASE, "outputs", factory, oid)
        if os.path.exists(os.path.join(out_dir, "scene.urdf")):
            print(f"  SKIP {factory}/{oid} (exists)")
            success += 1
            continue

        print(f"  Converting {factory}/{oid} ...", end=" ", flush=True)
        cmd = [sys.executable, setup_script,
               "--id", oid, "--factory", factory]
        ok, out = run_cmd(cmd, cwd=BASE, timeout=60)
        if ok and os.path.exists(os.path.join(out_dir, "scene.urdf")):
            print("OK")
            success += 1
        else:
            print(f"FAIL")
            fail += 1

    print(f"  PhysXNet conversions: {success} ok, {fail} failed")
    return success, fail


def run_precompute():
    """Run split_precompute.py for all cases."""
    print("\n" + "=" * 70)
    print("PHASE 3: Run split_precompute")
    print("=" * 70)

    script = os.path.join(BASE, "split_precompute.py")
    success, fail, skip = 0, 0, 0
    results = []

    def run_split(factory, seed, base_dir, suffix=""):
        nonlocal success, fail, skip
        label = f"{factory}{suffix}/{seed}"
        out_dir_check = os.path.join(OUTPUT, f"{factory}{suffix}", str(seed))
        if os.path.exists(os.path.join(out_dir_check, "metadata.json")):
            print(f"  SKIP {label}")
            skip += 1
            results.append((label, "skip", 0))
            return

        print(f"  Processing {label} ...", end=" ", flush=True)
        cmd = [sys.executable, script,
               "--factory", factory, "--seed", str(seed),
               "--base", base_dir, "--output_dir", OUTPUT,
               "--suffix", suffix, "--force"]
        ok, out = run_cmd(cmd, cwd=BASE, timeout=180)

        # Parse animode count
        n_animodes = 0
        for line in out.split("\n"):
            if "Exported" in line and "animodes" in line:
                try:
                    n_animodes = int(line.split("Exported")[1].split("/")[0].strip())
                except:
                    pass

        if ok and n_animodes > 0:
            print(f"OK ({n_animodes} animodes)")
            success += 1
            results.append((label, "ok", n_animodes))
        elif "SKIP: no movable joints" in out or "SKIP: no joints" in out:
            print("SKIP (no joints)")
            skip += 1
            results.append((label, "no_joints", 0))
        else:
            err_lines = [l for l in out.strip().split("\n") if l.strip()]
            last = err_lines[-1][:80] if err_lines else "unknown"
            print(f"FAIL: {last}")
            fail += 1
            results.append((label, "fail", 0))

    # IS factories
    print("\n  --- IS Factories ---")
    is_base = os.path.join(BASE, "sim_exports", "urdf")
    for factory in IS_FACTORIES:
        for seed in IS_SEEDS:
            run_split(factory, seed, is_base)

    # PhysXMobility (PartNet URDFs)
    print("\n  --- PhysXMobility ---")
    for factory, oid in PHYSXMOB_CASES:
        run_split(factory, oid, PARTNET_DIR, suffix="_PhysXmobility")

    # PhysXNet (converted to IM format)
    print("\n  --- PhysXNet ---")
    for factory, oid in PHYSXNET_CASES:
        run_split(factory, oid, BASE, suffix="_PhysXnet")

    print(f"\n  Precompute: {success} ok, {fail} failed, {skip} skipped")
    return results


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    t0 = time.time()

    # Phase 1: IS assets
    is_ok, is_fail = generate_is_assets()

    # Phase 2: PhysXNet conversion
    pxn_ok, pxn_fail = convert_physxnet()

    # Phase 3: Precompute
    results = run_precompute()

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"DONE in {elapsed:.0f}s")
    print(f"Output: {OUTPUT}")
    print("=" * 70)

    # Summary table
    ok_count = sum(1 for _, s, _ in results if s == "ok")
    fail_count = sum(1 for _, s, _ in results if s == "fail")
    skip_count = sum(1 for _, s, _ in results if s in ("skip", "no_joints"))
    total_animodes = sum(n for _, s, n in results if s == "ok")
    print(f"\nSummary: {ok_count} success, {fail_count} failed, {skip_count} skipped")
    print(f"Total animodes: {total_animodes}")

    # Write summary
    summary = {
        "elapsed_s": round(elapsed, 1),
        "results": [{"label": l, "status": s, "animodes": n} for l, s, n in results],
        "totals": {"ok": ok_count, "fail": fail_count, "skip": skip_count,
                    "animodes": total_animodes},
    }
    with open(os.path.join(OUTPUT, "test_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved: {OUTPUT}/test_summary.json")


if __name__ == "__main__":
    main()
