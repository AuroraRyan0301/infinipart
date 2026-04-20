#!/usr/bin/env python3
"""
Analyze yellow/blue pixel ratio from group color video first frames.

Group color mapping (from render_animode.py):
  part0 groups -> blue-ish tones (GROUP_COLORS[0] = (0.22, 0.46, 0.72) = blue)
  part1 groups -> orange/yellow tones (GROUP_COLORS[1] = (0.89, 0.35, 0.13) = orange)

Due to lighting variance, we use HSV thresholds instead of exact RGB matching.
"""
import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def classify_pixels_hsv(frame_bgr):
    """Classify pixels as yellow/orange, blue, or other using HSV.

    Returns (n_yellow, n_blue, n_foreground, n_total)
    where foreground = non-black pixels.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Foreground mask: exclude black background (low value)
    fg_mask = v > 30  # threshold for non-black

    # Yellow/orange detection in HSV:
    # Hue: 10-35 (orange to yellow), Sat > 30, Val > 50
    yellow_mask = fg_mask & (h >= 8) & (h <= 35) & (s > 25) & (v > 50)

    # Blue detection in HSV:
    # Hue: 90-130 (blue range), Sat > 20, Val > 50
    blue_mask = fg_mask & (h >= 90) & (h <= 135) & (s > 15) & (v > 50)

    n_yellow = int(yellow_mask.sum())
    n_blue = int(blue_mask.sum())
    n_fg = int(fg_mask.sum())
    n_total = frame_bgr.shape[0] * frame_bgr.shape[1]

    return n_yellow, n_blue, n_fg, n_total


def find_group_videos(precompute_root, max_animodes=None, encoded_root=None):
    """Find group color videos that have matching gt_latent in encoded_root.

    Returns list of (category, seed, animode, [video_paths]).
    Each entry includes ALL group videos for that animode (all views).
    """
    # Build set of encoded animodes if encoded_root provided
    encoded_set = None
    if encoded_root and os.path.isdir(encoded_root):
        encoded_set = set()
        for cat in os.listdir(encoded_root):
            cat_dir = os.path.join(encoded_root, cat)
            if not os.path.isdir(cat_dir):
                continue
            for model_id in os.listdir(cat_dir):
                gt_path = os.path.join(cat_dir, model_id, "gt_latent.pt")
                if os.path.isfile(gt_path):
                    encoded_set.add((cat, model_id))
        print(f"  Encoded animodes with gt_latent: {len(encoded_set)}")

    results = []
    categories = sorted(os.listdir(precompute_root))
    for cat in categories:
        cat_dir = os.path.join(precompute_root, cat)
        if not os.path.isdir(cat_dir):
            continue
        for seed in sorted(os.listdir(cat_dir)):
            seed_dir = os.path.join(cat_dir, seed)
            if not os.path.isdir(seed_dir):
                continue
            for animode in sorted(os.listdir(seed_dir)):
                anim_dir = os.path.join(seed_dir, animode)
                if not os.path.isdir(anim_dir):
                    continue

                # Filter: only animodes with gt_latent
                if encoded_set is not None:
                    model_id = f"{seed}_{animode}"
                    if (cat, model_id) not in encoded_set:
                        continue

                group_vids = sorted(glob.glob(os.path.join(anim_dir, "*_group.mp4")))
                if group_vids:
                    results.append((cat, seed, animode, group_vids))

    if max_animodes and len(results) > max_animodes:
        random.seed(42)
        results = random.sample(results, max_animodes)

    return results


def extract_first_frame(video_path):
    """Extract first frame from video."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute", default="/mnt/data/yurh/Infinigen-Sim/data_ssd/precompute",
                        help="Precompute root directory")
    parser.add_argument("--max_animodes", type=int, default=None,
                        help="Max animodes to sample (None=all)")
    parser.add_argument("--encoded", default="/mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_solidified",
                        help="Encoded root (only animodes with gt_latent)")
    parser.add_argument("--output", default="/mnt/data/yurh/Infinigen-Sim/data_ssd/group_color_ratio_dist.png",
                        help="Output plot path")
    args = parser.parse_args()

    print(f"Scanning {args.precompute} ...")
    entries = find_group_videos(args.precompute, args.max_animodes, args.encoded)
    print(f"Found {len(entries)} animode views with group videos")

    if not entries:
        print("No group videos found!")
        return

    yellow_blue_ratios = []  # yellow / blue
    yellow_fg_ratios = []    # yellow / foreground
    labels = []
    skipped = 0

    for i, (cat, seed, animode, vid_paths) in enumerate(entries):
        # Aggregate across all views for this animode
        total_yellow, total_blue, total_fg = 0, 0, 0
        view_ok = 0
        for vid_path in vid_paths:
            frame = extract_first_frame(vid_path)
            if frame is None:
                continue
            n_yellow, n_blue, n_fg, _ = classify_pixels_hsv(frame)
            total_yellow += n_yellow
            total_blue += n_blue
            total_fg += n_fg
            view_ok += 1

        if view_ok == 0:
            skipped += 1
            continue

        if total_blue > 0:
            yb_ratio = total_yellow / total_blue
        else:
            yb_ratio = float('inf') if total_yellow > 0 else 0.0

        if total_fg > 0:
            yf_ratio = total_yellow / total_fg
        else:
            yf_ratio = 0.0

        yellow_blue_ratios.append(yb_ratio)
        yellow_fg_ratios.append(yf_ratio)
        labels.append(f"{cat}/{seed}/{animode}")

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(entries)}...")

    # Filter out inf for plotting
    finite_yb = [r for r in yellow_blue_ratios if r != float('inf')]

    print(f"\n=== Results ({len(yellow_blue_ratios)} animodes, {skipped} skipped) ===")
    print(f"Yellow/Blue ratio:  median={np.median(finite_yb):.3f}, "
          f"mean={np.mean(finite_yb):.3f}, std={np.std(finite_yb):.3f}")
    print(f"  min={np.min(finite_yb):.3f}, max={np.max(finite_yb):.3f}")
    print(f"  inf count (no blue): {len(yellow_blue_ratios) - len(finite_yb)}")
    print(f"Yellow/FG ratio:    median={np.median(yellow_fg_ratios):.3f}, "
          f"mean={np.mean(yellow_fg_ratios):.3f}, std={np.std(yellow_fg_ratios):.3f}")
    print(f"  min={np.min(yellow_fg_ratios):.3f}, max={np.max(yellow_fg_ratios):.3f}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Yellow/Blue ratio histogram
    ax = axes[0]
    clipped_yb = np.clip(finite_yb, 0, 50)  # clip for visualization
    ax.hist(clipped_yb, bins=60, color='#e8a020', edgecolor='black', alpha=0.8)
    ax.axvline(np.median(finite_yb), color='red', linestyle='--', label=f'median={np.median(finite_yb):.2f}')
    ax.axvline(1.0, color='blue', linestyle=':', label='ratio=1.0')
    ax.set_xlabel('Yellow Pixels / Blue Pixels')
    ax.set_ylabel('Count')
    ax.set_title(f'Yellow/Blue Ratio Distribution (N={len(finite_yb)}, {len(yellow_blue_ratios)-len(finite_yb)} inf)')
    ax.legend()

    # Yellow/Foreground ratio histogram
    ax = axes[1]
    ax.hist(yellow_fg_ratios, bins=60, color='#4090d0', edgecolor='black', alpha=0.8)
    ax.axvline(np.median(yellow_fg_ratios), color='red', linestyle='--', label=f'median={np.median(yellow_fg_ratios):.2f}')
    ax.axvline(0.5, color='blue', linestyle=':', label='ratio=0.5')
    ax.set_xlabel('Yellow Pixels / All Foreground Pixels')
    ax.set_ylabel('Count')
    ax.set_title(f'Yellow/Foreground Ratio Distribution (N={len(yellow_fg_ratios)})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nPlot saved to: {args.output}")

    # Also save raw data
    csv_path = args.output.replace('.png', '.csv')
    with open(csv_path, 'w') as f:
        f.write("category,seed,animode,yellow_blue_ratio,yellow_fg_ratio\n")
        for label, yb, yf in zip(labels, yellow_blue_ratios, yellow_fg_ratios):
            parts = label.split('/')
            f.write(f"{parts[0]},{parts[1]},{parts[2]},{yb:.6f},{yf:.6f}\n")
    print(f"CSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
