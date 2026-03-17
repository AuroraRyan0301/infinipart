#!/usr/bin/env python3
"""
Generate a fixed train/test manifest from encoded training data.

Scans {data_root}/{category}/{seed_animode}/ directories, splits views
per animode into train vs test, and writes a JSON manifest.

Split strategy: for each animode, every Nth view (by index) goes to test,
the rest to train. Default N=4 (same as previous dynamic split).

Output JSON format:
{
    "created": "2026-03-17T...",
    "data_root": "/mnt/data_ssd/infinigen-sim",
    "test_view_interval": 4,
    "num_train": ...,
    "num_test": ...,
    "num_animodes": ...,
    "categories": [...],
    "train": [ {id, category, model_id, view_idx, gt_path, jepa_path}, ... ],
    "test":  [ ... ]
}

Usage:
    python generate_manifest.py --data_root /mnt/data_ssd/infinigen-sim \
        --output manifest.json --test_view_interval 4
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime


def discover_all_samples(data_root, test_view_interval=4):
    """Scan data_root and split views per animode into train/test."""
    train_samples = []
    test_samples = []
    categories = set()
    animode_count = 0
    skipped_no_gt = 0
    skipped_no_views = 0

    if not os.path.isdir(data_root):
        print(f"ERROR: data_root not found: {data_root}")
        sys.exit(1)

    for cat_name in sorted(os.listdir(data_root)):
        cat_dir = os.path.join(data_root, cat_name)
        if not os.path.isdir(cat_dir):
            continue
        # Skip hidden/meta dirs
        if cat_name.startswith("."):
            continue

        for model_id in sorted(os.listdir(cat_dir)):
            model_dir = os.path.join(cat_dir, model_id)
            if not os.path.isdir(model_dir):
                continue

            gt_path = os.path.join(model_dir, "gt_latent.pt")
            if not os.path.exists(gt_path):
                skipped_no_gt += 1
                continue

            views_dir = os.path.join(model_dir, "views")
            if not os.path.isdir(views_dir):
                skipped_no_views += 1
                continue

            jepa_files = sorted(glob.glob(
                os.path.join(views_dir, "v*_nobg_jepa.pt")))
            if not jepa_files:
                skipped_no_views += 1
                continue

            categories.add(cat_name)
            animode_count += 1

            for jepa_path in jepa_files:
                basename = os.path.basename(jepa_path)
                view_str = basename.split("_")[0]  # "vXX"
                try:
                    view_idx = int(view_str[1:])
                except ValueError:
                    continue

                sample = {
                    "id": f"{cat_name}/{model_id}/v{view_idx:02d}",
                    "category": cat_name,
                    "model_id": model_id,
                    "view_idx": view_idx,
                    "gt_path": gt_path,
                    "jepa_path": jepa_path,
                }

                if view_idx % test_view_interval == (test_view_interval - 1):
                    test_samples.append(sample)
                else:
                    train_samples.append(sample)

    return (train_samples, test_samples, sorted(categories),
            animode_count, skipped_no_gt, skipped_no_views)


def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed train/test manifest from encoded data")
    parser.add_argument("--data_root", type=str,
                        default="/mnt/data_ssd/infinigen-sim")
    parser.add_argument("--output", type=str, default="manifest.json",
                        help="Output manifest JSON path")
    parser.add_argument("--test_view_interval", type=int, default=4,
                        help="Every Nth view goes to test (default: 4)")
    args = parser.parse_args()

    print(f"Scanning {args.data_root} ...")
    (train, test, categories, n_animodes,
     skip_gt, skip_views) = discover_all_samples(
        args.data_root, args.test_view_interval)

    print(f"\n{'='*60}")
    print(f"  Data root:          {args.data_root}")
    print(f"  Test view interval: {args.test_view_interval}")
    print(f"  Categories:         {len(categories)}")
    print(f"  Animodes:           {n_animodes}")
    print(f"  Train views:        {len(train)}")
    print(f"  Test views:         {len(test)}")
    print(f"  Skipped (no GT):    {skip_gt}")
    print(f"  Skipped (no views): {skip_views}")
    print(f"{'='*60}\n")

    # Per-category breakdown
    train_cats = Counter(s["category"] for s in train)
    test_cats = Counter(s["category"] for s in test)
    print(f"{'Category':<25} {'Train':>7} {'Test':>7} {'Total':>7}")
    print(f"{'-'*25} {'-'*7} {'-'*7} {'-'*7}")
    for cat in categories:
        tr = train_cats.get(cat, 0)
        te = test_cats.get(cat, 0)
        print(f"{cat:<25} {tr:>7} {te:>7} {tr+te:>7}")
    print()

    manifest = {
        "created": datetime.now().isoformat(),
        "data_root": args.data_root,
        "test_view_interval": args.test_view_interval,
        "num_train": len(train),
        "num_test": len(test),
        "num_animodes": n_animodes,
        "categories": categories,
        "train": train,
        "test": test,
    }

    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest written to: {args.output}")
    print(f"  {len(train)} train + {len(test)} test = {len(train)+len(test)} total views")


if __name__ == "__main__":
    main()
