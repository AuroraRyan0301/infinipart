#!/usr/bin/env python3
"""
Create overview visualizations of negative-sample articulation videos.

For each factory:
  - A 1x6 grid video showing all 6 negative types side by side (front_bg view)

Overall:
  - A large grid video tiling one row per factory (factory label + 6 neg types)

Usage:
  python visualize_negatives.py
  python visualize_negatives.py --view threequarter_bg
  python visualize_negatives.py --factories DishwasherFactory BottleFactory
"""

import argparse
import os
import subprocess
import sys
import math
import json

BASE = os.path.dirname(os.path.abspath(__file__))
MOTION_TEST_DIR = os.path.join(BASE, "outputs", "motion_test")
VIS_DIR = os.path.join(MOTION_TEST_DIR, "visualizations")

FFMPEG = None
NEG_TYPES = [
    "wrong_joint_type",
    "wrong_axis",
    "wrong_direction",
    "over_motion",
    "wrong_parts_moving",
    "jitter",
]
NEG_LABELS = [
    "Wrong Joint Type",
    "Wrong Axis",
    "Wrong Direction",
    "Over Motion",
    "Wrong Parts",
    "Jitter",
]


def find_ffmpeg():
    """Find ffmpeg binary."""
    global FFMPEG
    # Check conda env first
    conda_ffmpeg = os.path.expanduser(
        "~/miniforge3/envs/blender_test/bin/ffmpeg"
    )
    if os.path.isfile(conda_ffmpeg):
        FFMPEG = conda_ffmpeg
        return
    # Check PATH
    try:
        result = subprocess.run(
            ["which", "ffmpeg"], capture_output=True, text=True
        )
        if result.returncode == 0:
            FFMPEG = result.stdout.strip()
            return
    except Exception:
        pass
    print("ERROR: ffmpeg not found")
    sys.exit(1)


def discover_factories(filter_list=None):
    """Find factories that have negative renders."""
    factories = []
    for name in sorted(os.listdir(MOTION_TEST_DIR)):
        factory_dir = os.path.join(MOTION_TEST_DIR, name)
        if not os.path.isdir(factory_dir):
            continue
        if name == "visualizations":
            continue
        if filter_list and name not in filter_list:
            continue
        # Find first seed with negatives
        for seed_name in sorted(os.listdir(factory_dir)):
            neg_dir = os.path.join(factory_dir, seed_name, "negatives")
            if os.path.isdir(neg_dir):
                factories.append((name, seed_name, neg_dir))
                break
    return factories


def make_per_factory_grid(factory_name, neg_dir, view, output_path, cell_size=256):
    """Create a 1x6 grid video for one factory: all neg types side by side with labels."""
    inputs = []
    for nt in NEG_TYPES:
        mp4 = os.path.join(neg_dir, nt, f"{view}.mp4")
        if os.path.isfile(mp4):
            inputs.append(mp4)
        else:
            inputs.append(None)

    if not any(inputs):
        print(f"  SKIP {factory_name}: no videos for view={view}")
        return False

    # Build ffmpeg command with filter_complex
    cmd = [FFMPEG, "-y"]

    # Add inputs (use color source for missing ones)
    input_idx = 0
    stream_map = {}
    for i, inp in enumerate(inputs):
        if inp:
            cmd += ["-i", inp]
            stream_map[i] = f"[{input_idx}:v]"
            input_idx += 1

    # Build filter graph
    filters = []
    labeled_streams = []

    for i, nt in enumerate(NEG_TYPES):
        label = NEG_LABELS[i]
        if i in stream_map:
            src = stream_map[i]
            # Scale to cell_size and add label
            filters.append(
                f"{src}scale={cell_size}:{cell_size},"
                f"drawtext=text='{label}':"
                f"fontsize=14:fontcolor=white:"
                f"borderw=1:bordercolor=black:"
                f"x=(w-text_w)/2:y=5[v{i}]"
            )
        else:
            # Create black placeholder with label
            filters.append(
                f"color=c=black:s={cell_size}x{cell_size}:d=1:r=8[bg{i}];"
                f"[bg{i}]drawtext=text='{label} (N/A)':"
                f"fontsize=14:fontcolor=gray:"
                f"x=(w-text_w)/2:y=(h-text_h)/2[v{i}]"
            )
        labeled_streams.append(f"[v{i}]")

    # Horizontal stack
    stack_input = "".join(labeled_streams)
    filters.append(f"{stack_input}hstack=inputs=6[out]")

    filter_str = ";".join(filters)
    cmd += [
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL {factory_name}: {result.stderr[-300:]}")
        return False
    return True


def make_overall_grid(factories, view, output_path, cell_size=192, max_cols=6):
    """Create an overall grid: one row per factory with factory name + neg types."""
    # Strategy: create per-factory row videos first, then vstack them
    # But with 62 factories that's too many inputs for ffmpeg.
    # Instead: render each factory row as a labeled image sequence, then tile.

    # Simpler approach: create a big montage image for each frame,
    # or create per-factory row videos and then concatenate vertically in batches.

    # Let's use a 2-pass approach:
    # Pass 1: create row video per factory (already done above)
    # Pass 2: tile rows into pages of ~10 factories each, then concat

    row_videos = []
    for factory_name, seed, neg_dir in factories:
        row_path = os.path.join(VIS_DIR, "rows", f"{factory_name}.mp4")
        if os.path.isfile(row_path):
            row_videos.append((factory_name, row_path))
            continue
        # We need to create the row with factory label on the left
        inputs_exist = False
        for nt in NEG_TYPES:
            mp4 = os.path.join(neg_dir, nt, f"{view}.mp4")
            if os.path.isfile(mp4):
                inputs_exist = True
                break
        if not inputs_exist:
            continue

        cmd = [FFMPEG, "-y"]
        input_idx = 0
        stream_map = {}
        for i, nt in enumerate(NEG_TYPES):
            mp4 = os.path.join(neg_dir, nt, f"{view}.mp4")
            if os.path.isfile(mp4):
                cmd += ["-i", mp4]
                stream_map[i] = f"[{input_idx}:v]"
                input_idx += 1

        filters = []
        labeled_streams = []

        # Factory name label column
        short_name = factory_name.replace("Factory", "").replace("Sapien", "S")
        filters.append(
            f"color=c=0x1a1a2e:s=120x{cell_size}:d=1:r=8,"
            f"drawtext=text='{short_name}':"
            f"fontsize=11:fontcolor=white:"
            f"x=(w-text_w)/2:y=(h-text_h)/2[label]"
        )
        labeled_streams.append("[label]")

        for i, nt in enumerate(NEG_TYPES):
            if i in stream_map:
                src = stream_map[i]
                filters.append(f"{src}scale={cell_size}:{cell_size}[v{i}]")
            else:
                filters.append(
                    f"color=c=black:s={cell_size}x{cell_size}:d=1:r=8[v{i}]"
                )
            labeled_streams.append(f"[v{i}]")

        stack_input = "".join(labeled_streams)
        filters.append(f"{stack_input}hstack=inputs=7[out]")

        filter_str = ";".join(filters)
        cmd += [
            "-filter_complex", filter_str,
            "-map", "[out]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            row_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAIL row {factory_name}: {result.stderr[-200:]}")
            continue
        row_videos.append((factory_name, row_path))

    if not row_videos:
        print("  No row videos to combine")
        return False

    # Now vstack in batches of PAGE_SIZE, then concat the pages
    PAGE_SIZE = 10
    pages = []
    for page_idx in range(0, len(row_videos), PAGE_SIZE):
        batch = row_videos[page_idx : page_idx + PAGE_SIZE]
        if len(batch) == 1:
            pages.append(batch[0][1])
            continue

        page_path = os.path.join(VIS_DIR, "rows", f"_page_{page_idx}.mp4")
        cmd = [FFMPEG, "-y"]
        for _, row_path in batch:
            cmd += ["-i", row_path]

        inputs_str = "".join(f"[{i}:v]" for i in range(len(batch)))
        cmd += [
            "-filter_complex",
            f"{inputs_str}vstack=inputs={len(batch)}[out]",
            "-map", "[out]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            page_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAIL page {page_idx}: {result.stderr[-200:]}")
            continue
        pages.append(page_path)

    if len(pages) == 1:
        # Just copy/rename
        os.replace(pages[0], output_path)
        return True

    # Concat pages using concat demuxer
    concat_list = os.path.join(VIS_DIR, "rows", "_concat.txt")
    # For vertical stacking of pages, we need to vstack them too
    # But they may have different heights. Use concat with vstack.
    if len(pages) <= 15:
        cmd = [FFMPEG, "-y"]
        for p in pages:
            cmd += ["-i", p]
        inputs_str = "".join(f"[{i}:v]" for i in range(len(pages)))
        cmd += [
            "-filter_complex",
            f"{inputs_str}vstack=inputs={len(pages)}[out]",
            "-map", "[out]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "20",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAIL final vstack: {result.stderr[-300:]}")
            return False
        return True
    else:
        # Too many pages, do hierarchical merge
        while len(pages) > 1:
            next_pages = []
            for i in range(0, len(pages), 2):
                if i + 1 >= len(pages):
                    next_pages.append(pages[i])
                    continue
                merged = os.path.join(
                    VIS_DIR, "rows", f"_merge_{i}.mp4"
                )
                cmd = [
                    FFMPEG, "-y",
                    "-i", pages[i],
                    "-i", pages[i + 1],
                    "-filter_complex",
                    "[0:v][1:v]vstack=inputs=2[out]",
                    "-map", "[out]",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-crf", "20",
                    merged,
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    print(f"  FAIL merge: {result.stderr[-200:]}")
                    next_pages.append(pages[i])
                else:
                    next_pages.append(merged)
            pages = next_pages
        os.replace(pages[0], output_path)
        return True


def make_per_factory_comparison(factory_name, neg_dir, view, output_path,
                                 cell_size=256):
    """Create a 2x3 grid video for one factory with labels."""
    cmd = [FFMPEG, "-y"]
    input_idx = 0
    stream_map = {}
    for i, nt in enumerate(NEG_TYPES):
        mp4 = os.path.join(neg_dir, nt, f"{view}.mp4")
        if os.path.isfile(mp4):
            cmd += ["-i", mp4]
            stream_map[i] = f"[{input_idx}:v]"
            input_idx += 1

    if input_idx == 0:
        return False

    filters = []
    for i, nt in enumerate(NEG_TYPES):
        label = NEG_LABELS[i]
        if i in stream_map:
            src = stream_map[i]
            filters.append(
                f"{src}scale={cell_size}:{cell_size},"
                f"drawtext=text='{label}':"
                f"fontsize=16:fontcolor=white:"
                f"borderw=2:bordercolor=black:"
                f"x=(w-text_w)/2:y=8[v{i}]"
            )
        else:
            filters.append(
                f"color=c=black:s={cell_size}x{cell_size}:d=1:r=8,"
                f"drawtext=text='{label} (N/A)':"
                f"fontsize=16:fontcolor=gray:"
                f"x=(w-text_w)/2:y=(h-text_h)/2[v{i}]"
            )

    # 2x3 grid: top row [0,1,2], bottom row [3,4,5]
    filters.append("[v0][v1][v2]hstack=inputs=3[top]")
    filters.append("[v3][v4][v5]hstack=inputs=3[bot]")
    filters.append("[top][bot]vstack=inputs=2[out]")

    filter_str = ";".join(filters)
    cmd += [
        "-filter_complex", filter_str,
        "-map", "[out]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL {factory_name} 2x3: {result.stderr[-300:]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Visualize negative-sample articulation videos"
    )
    parser.add_argument(
        "--view", default="front_bg",
        help="Which view to use (default: front_bg)"
    )
    parser.add_argument(
        "--factories", nargs="+", default=None,
        help="Filter to specific factories"
    )
    parser.add_argument(
        "--cell_size", type=int, default=256,
        help="Cell size for per-factory grids"
    )
    parser.add_argument(
        "--skip_overall", action="store_true",
        help="Skip the overall grid (only per-factory)"
    )
    args = parser.parse_args()

    find_ffmpeg()
    print(f"Using ffmpeg: {FFMPEG}")

    os.makedirs(VIS_DIR, exist_ok=True)
    os.makedirs(os.path.join(VIS_DIR, "per_factory"), exist_ok=True)
    os.makedirs(os.path.join(VIS_DIR, "rows"), exist_ok=True)

    factories = discover_factories(args.factories)
    print(f"Found {len(factories)} factories with negative renders\n")

    # Per-factory 2x3 grids
    print("=== Per-factory 2x3 grids ===")
    ok_count = 0
    for factory_name, seed, neg_dir in factories:
        out = os.path.join(
            VIS_DIR, "per_factory", f"{factory_name}_{args.view}.mp4"
        )
        if make_per_factory_comparison(
            factory_name, neg_dir, args.view, out, args.cell_size
        ):
            ok_count += 1
            print(f"  OK  {factory_name}")
        else:
            print(f"  SKIP {factory_name}")
    print(f"\n  Created {ok_count} per-factory grids")

    # Overall grid
    if not args.skip_overall:
        print("\n=== Overall grid ===")
        overall_out = os.path.join(VIS_DIR, f"all_negatives_{args.view}.mp4")
        if make_overall_grid(factories, args.view, overall_out, cell_size=128):
            print(f"\n  Overall grid: {overall_out}")
        else:
            print("\n  Failed to create overall grid")

    print(f"\nDone! Visualizations at: {VIS_DIR}")


if __name__ == "__main__":
    main()
