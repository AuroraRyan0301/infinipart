#!/usr/bin/env python3
"""
Analyze part0/part1 mesh surface area ratio for all animodes with gt_latent.
Loads both parts, combines and normalizes together, then computes area ratio.
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_and_combine_parts(anim_dir):
    """Load part0.obj and part1.obj, return (area0, area1) after joint normalization."""
    p0_path = os.path.join(anim_dir, "part0.obj")
    p1_path = os.path.join(anim_dir, "part1.obj")
    if not os.path.isfile(p0_path) or not os.path.isfile(p1_path):
        return None, None

    try:
        m0 = trimesh.load(p0_path, force='mesh', process=False)
        m1 = trimesh.load(p1_path, force='mesh', process=False)
    except Exception:
        return None, None

    if len(m0.vertices) == 0 and len(m1.vertices) == 0:
        return None, None

    # Combine to compute unified bounding box
    all_verts = []
    if len(m0.vertices) > 0:
        all_verts.append(m0.vertices)
    if len(m1.vertices) > 0:
        all_verts.append(m1.vertices)
    all_verts = np.concatenate(all_verts, axis=0)

    # Normalize: center + scale to unit cube (jointly)
    bbox_min = all_verts.min(axis=0)
    bbox_max = all_verts.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0
    extent = (bbox_max - bbox_min).max()
    if extent < 1e-8:
        return None, None
    scale = 1.0 / extent

    # Apply to both meshes
    if len(m0.vertices) > 0:
        m0.vertices = (m0.vertices - center) * scale
    if len(m1.vertices) > 0:
        m1.vertices = (m1.vertices - center) * scale

    area0 = m0.area if len(m0.vertices) > 0 else 0.0
    area1 = m1.area if len(m1.vertices) > 0 else 0.0

    return area0, area1


def find_animodes_with_gt(precompute_root, encoded_root):
    """Find animodes that have both part0/1.obj in precompute and gt_latent in encoded."""
    encoded_set = set()
    for cat in os.listdir(encoded_root):
        cat_dir = os.path.join(encoded_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for model_id in os.listdir(cat_dir):
            if os.path.isfile(os.path.join(cat_dir, model_id, "gt_latent.pt")):
                encoded_set.add((cat, model_id))
    print(f"  Encoded animodes with gt_latent: {len(encoded_set)}")

    results = []
    for cat, model_id in sorted(encoded_set):
        parts = model_id.split('_', 1)
        if len(parts) != 2:
            continue
        seed, animode = parts
        anim_dir = os.path.join(precompute_root, cat, seed, animode)
        if os.path.isdir(anim_dir):
            results.append((cat, seed, animode, anim_dir))

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute", default="/mnt/data/yurh/Infinigen-Sim/data_ssd/precompute_solidified")
    parser.add_argument("--encoded", default="/mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_solidified")
    parser.add_argument("--output", default="/mnt/data/yurh/Infinigen-Sim/data_ssd/mesh_area_ratio_dist.png")
    parser.add_argument("--max_animodes", type=int, default=None)
    args = parser.parse_args()

    print(f"Scanning {args.precompute} ...")
    entries = find_animodes_with_gt(args.precompute, args.encoded)
    print(f"Found {len(entries)} animodes")

    if args.max_animodes and len(entries) > args.max_animodes:
        import random
        random.seed(42)
        entries = random.sample(entries, args.max_animodes)
        print(f"Sampled {len(entries)} animodes")

    # part1 is the "moving part" (orange/yellow in group viz)
    p1_p0_ratios = []    # part1_area / part0_area
    p1_total_ratios = [] # part1_area / (part0_area + part1_area)
    labels = []
    skipped = 0

    for i, (cat, seed, animode, anim_dir) in enumerate(entries):
        area0, area1 = load_and_combine_parts(anim_dir)
        if area0 is None:
            skipped += 1
            continue

        total = area0 + area1
        if total < 1e-10:
            skipped += 1
            continue

        if area0 > 0:
            p1p0 = area1 / area0
        else:
            p1p0 = float('inf') if area1 > 0 else 0.0

        p1_p0_ratios.append(p1p0)
        p1_total_ratios.append(area1 / total)
        labels.append(f"{cat}/{seed}/{animode}")

        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(entries)}...")

    finite_ratios = [r for r in p1_p0_ratios if r != float('inf')]

    print(f"\n=== Results ({len(p1_p0_ratios)} animodes, {skipped} skipped) ===")
    print(f"Part1/Part0 area ratio:  median={np.median(finite_ratios):.4f}, "
          f"mean={np.mean(finite_ratios):.4f}, std={np.std(finite_ratios):.4f}")
    print(f"  min={np.min(finite_ratios):.4f}, max={np.max(finite_ratios):.4f}")
    print(f"  inf count (no part0): {len(p1_p0_ratios) - len(finite_ratios)}")
    print(f"Part1/Total area ratio: median={np.median(p1_total_ratios):.4f}, "
          f"mean={np.mean(p1_total_ratios):.4f}, std={np.std(p1_total_ratios):.4f}")
    print(f"  min={np.min(p1_total_ratios):.4f}, max={np.max(p1_total_ratios):.4f}")

    # Threshold stats
    print(f"\nPart1/Part0 threshold distribution:")
    for thresh in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
        cnt = sum(1 for r in p1_p0_ratios if r >= thresh)
        print(f"  >= {thresh:<5}: {cnt:>5} / {len(p1_p0_ratios)}  ({cnt/len(p1_p0_ratios)*100:.1f}%)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    clipped = np.clip(finite_ratios, 0, 10)
    ax.hist(clipped, bins=80, color='#e8a020', edgecolor='black', alpha=0.8)
    ax.axvline(np.median(finite_ratios), color='red', linestyle='--', label=f'median={np.median(finite_ratios):.3f}')
    ax.axvline(1.0, color='blue', linestyle=':', label='ratio=1.0')
    ax.set_xlabel('Part1 Area / Part0 Area')
    ax.set_ylabel('Count')
    ax.set_title(f'Part1/Part0 Area Ratio (N={len(finite_ratios)}, {len(p1_p0_ratios)-len(finite_ratios)} inf)')
    ax.legend()

    ax = axes[1]
    ax.hist(p1_total_ratios, bins=80, color='#4090d0', edgecolor='black', alpha=0.8)
    ax.axvline(np.median(p1_total_ratios), color='red', linestyle='--', label=f'median={np.median(p1_total_ratios):.3f}')
    ax.axvline(0.5, color='blue', linestyle=':', label='ratio=0.5')
    ax.set_xlabel('Part1 Area / Total Area')
    ax.set_ylabel('Count')
    ax.set_title(f'Part1/Total Area Ratio (N={len(p1_total_ratios)})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nPlot saved to: {args.output}")

    csv_path = args.output.replace('.png', '.csv')
    with open(csv_path, 'w') as f:
        f.write("category,seed,animode,part1_part0_ratio,part1_total_ratio\n")
        for label, r1, r2 in zip(labels, p1_p0_ratios, p1_total_ratios):
            parts = label.split('/')
            f.write(f"{parts[0]},{parts[1]},{parts[2]},{r1:.6f},{r2:.6f}\n")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
