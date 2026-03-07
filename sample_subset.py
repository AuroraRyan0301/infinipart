#!/usr/bin/env python3
"""
Sample a stratified subset of PhysXNet and PhysXMobility objects.

Outputs a JSON manifest listing object IDs to process, ensuring category diversity.

Usage:
  python sample_subset.py --physxnet_n 1000 --physxmob_n 300 --seed 42
  # Output: subset_manifest.json
"""

import argparse
import collections
import json
import os
import random


def load_categories(finaljson_dir):
    """Load {object_id: category} from finaljson directory."""
    id_to_cat = {}
    for f in sorted(os.listdir(finaljson_dir)):
        if not f.endswith('.json'):
            continue
        obj_id = f.replace('.json', '')
        with open(os.path.join(finaljson_dir, f)) as fp:
            d = json.load(fp)
        cat = d.get('category', d.get('meta', {}).get('category', 'unknown'))
        id_to_cat[obj_id] = cat
    return id_to_cat


def stratified_sample(id_to_cat, n, rng):
    """Sample n objects with stratified category coverage.

    Strategy: first take 1 per category (ensure diversity), then fill remaining
    slots proportionally from each category.
    """
    cat_to_ids = collections.defaultdict(list)
    for obj_id, cat in id_to_cat.items():
        cat_to_ids[cat].append(obj_id)

    # Sort for determinism
    for cat in cat_to_ids:
        cat_to_ids[cat].sort()

    all_cats = sorted(cat_to_ids.keys())
    selected = set()

    # Phase 1: 1 per category (if n >= num_categories)
    for cat in all_cats:
        if len(selected) >= n:
            break
        pick = rng.choice(cat_to_ids[cat])
        selected.add(pick)

    # Phase 2: fill remaining proportionally
    remaining = n - len(selected)
    if remaining > 0:
        # Proportional allocation
        total = sum(len(ids) for ids in cat_to_ids.values())
        for cat in all_cats:
            available = [x for x in cat_to_ids[cat] if x not in selected]
            if not available:
                continue
            quota = max(1, int(remaining * len(cat_to_ids[cat]) / total))
            quota = min(quota, len(available))
            picks = rng.sample(available, quota)
            selected.update(picks)
            if len(selected) >= n:
                break

    # Phase 3: if still short, random fill
    if len(selected) < n:
        all_remaining = [x for x in id_to_cat if x not in selected]
        rng.shuffle(all_remaining)
        selected.update(all_remaining[:n - len(selected)])

    # Trim if over
    selected = sorted(selected)[:n]
    return selected


def main():
    parser = argparse.ArgumentParser(description="Sample stratified subset for pipeline")
    parser.add_argument("--physxnet_json", default="/mnt/data/fulian/dataset/PhysXNet/version_1/finaljson")
    parser.add_argument("--physxmob_json", default="/mnt/data/fulian/dataset/PhysX_mobility/finaljson")
    parser.add_argument("--physxnet_n", type=int, default=1000)
    parser.add_argument("--physxmob_n", type=int, default=300)
    parser.add_argument("--is_seeds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="subset_manifest.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    manifest = {"is_seeds": args.is_seeds}

    # PhysXNet
    if os.path.isdir(args.physxnet_json):
        id_to_cat = load_categories(args.physxnet_json)
        selected = stratified_sample(id_to_cat, args.physxnet_n, rng)
        # Category distribution
        cat_counts = collections.Counter(id_to_cat[x] for x in selected)
        print(f"PhysXNet: {len(selected)} objects from {len(cat_counts)} categories")
        print(f"  Top 10: {cat_counts.most_common(10)}")
        manifest["physxnet_ids"] = selected
    else:
        print(f"WARNING: PhysXNet finaljson not found at {args.physxnet_json}")
        manifest["physxnet_ids"] = []

    # PhysXMobility
    if os.path.isdir(args.physxmob_json):
        id_to_cat = load_categories(args.physxmob_json)
        selected = stratified_sample(id_to_cat, args.physxmob_n, rng)
        cat_counts = collections.Counter(id_to_cat[x] for x in selected)
        print(f"PhysXMobility: {len(selected)} objects from {len(cat_counts)} categories")
        print(f"  Top 10: {cat_counts.most_common(10)}")
        manifest["physxmob_ids"] = selected
    else:
        print(f"WARNING: PhysXMobility finaljson not found at {args.physxmob_json}")
        manifest["physxmob_ids"] = []

    # IS factories
    print(f"IS factories: 18 x {args.is_seeds} seeds = {18 * args.is_seeds} objects")

    # Summary
    total = len(manifest.get("physxnet_ids", [])) + len(manifest.get("physxmob_ids", [])) + 18 * args.is_seeds
    print(f"\nTotal: {total} objects")
    print(f"Manifest: {args.output}")

    with open(args.output, 'w') as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
