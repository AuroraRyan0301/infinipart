#!/usr/bin/env python3
"""
Fast JEPA encoding for PhysXMobility rendered videos.
Batched inference to maximize GPU utilization.

Usage:
  CUDA_VISIBLE_DEVICES=0 python encode_jepa_fast.py --batch_size 8
"""
import argparse
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VJEPA2_ROOT = "/mnt/cpfs/yurh/vjepa2"
VJEPA2_CKPT = os.path.join(VJEPA2_ROOT, "checkpoints", "vitg.pt")
PRECOMPUTE_ROOT = "/mnt/data_ssd/infinigen-sim-data/precompute"
OUTPUT_DIR = "/mnt/data_ssd/infinigen-sim-data/encoded"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

MIN_VIDEO_BYTES = 4096       # skip videos < 4KB
MIN_VIDEO_FRAMES = 10        # skip videos with < 10 frames


def load_vjepa2(device, num_frames=81, img_size=256):
    if VJEPA2_ROOT not in sys.path:
        sys.path.insert(0, VJEPA2_ROOT)
    from src.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        patch_size=16, embed_dim=1408, depth=40, num_heads=22,
        mlp_ratio=48 / 11, qkv_bias=True, use_rope=True, use_sdpa=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_size=(img_size, img_size), num_frames=num_frames,
    )
    ckpt = torch.load(VJEPA2_CKPT, weights_only=True, map_location="cpu")
    enc = {k.replace("module.", "").replace("backbone.", ""): v
           for k, v in ckpt["encoder"].items()}
    model.load_state_dict(enc, strict=True)
    del ckpt, enc
    model.eval().to(device).bfloat16()
    print(f"[VJEPA2] Loaded ViT-g on {device} (bf16)")
    return model


def discover_physxmobility_videos(precompute_root, output_dir):
    """Find all PhysXMobility nobg videos that need JEPA encoding."""
    pm_root = os.path.join(precompute_root, "PhysXMobility")
    if not os.path.isdir(pm_root):
        return []

    tasks = []
    for seed in os.listdir(pm_root):
        seed_dir = os.path.join(pm_root, seed)
        if not os.path.isdir(seed_dir):
            continue
        for animode in os.listdir(seed_dir):
            anim_dir = os.path.join(seed_dir, animode)
            if not os.path.isdir(anim_dir):
                continue
            # Must have part0.obj + part1.obj
            if not (os.path.isfile(os.path.join(anim_dir, "part0.obj")) and
                    os.path.isfile(os.path.join(anim_dir, "part1.obj"))):
                continue

            out_name = f"PhysXMobility/{seed}_{animode}"
            views_out = os.path.join(output_dir, out_name, "views")

            # Find nobg videos
            nobg_vids = sorted([f for f in os.listdir(anim_dir) if f.endswith("_nobg.mp4")])
            if not nobg_vids:
                continue

            for vid_name in nobg_vids:
                vid_path = os.path.join(anim_dir, vid_name)

                # Skip bad videos
                fsize = os.path.getsize(vid_path)
                if fsize < MIN_VIDEO_BYTES:
                    continue

                # Parse view name -> output path
                parts = vid_name.replace("_nobg.mp4", "").split("_")
                view_type, view_idx_str = parts[0], parts[1]
                view_num = int(view_idx_str)
                if view_type == "orbit":
                    view_num += 16
                elif view_type == "sweep":
                    view_num += 24

                out_path = os.path.join(views_out, f"v{view_num:02d}_nobg_jepa.pt")
                if os.path.exists(out_path):
                    continue  # already encoded

                tasks.append({
                    "video_path": vid_path,
                    "out_path": out_path,
                    "out_name": out_name,
                    "vid_name": vid_name,
                })

    return tasks


def load_video(video_path, num_frames=81, img_size=256):
    """Load and preprocess a single video. Returns None on failure."""
    try:
        from decord import VideoReader
        vr = VideoReader(video_path)
        total = len(vr)
        if total < MIN_VIDEO_FRAMES:
            return None
        indices = np.linspace(0, total - 1, num_frames, dtype=int)
        frames = vr.get_batch(indices).asnumpy()
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        frames = F.interpolate(frames, size=(img_size, img_size),
                               mode="bilinear", align_corners=False)
        frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, T, H, W]
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        return frames.squeeze(0).bfloat16()  # [3, T, H, W] bf16
    except Exception as e:
        return None


def _load_fn_global(t):
    """Top-level function for multiprocessing Pool (must be picklable)."""
    vid_path = t["video_path"]
    fsize = os.path.getsize(vid_path)
    if fsize < MIN_VIDEO_BYTES:
        return t, None, f"too_small | {vid_path} | {fsize} bytes"
    frames = load_video(vid_path)
    if frames is None:
        return t, None, f"load_failed | {vid_path} | could not decode"
    return t, frames, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute_root", default=PRECOMPUTE_ROOT)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="CPU workers for video loading")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--log_bad", default="logs/encode_jepa_bad_videos.txt",
                        help="Log file for skipped/failed videos")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_vjepa2(device, args.num_frames, args.img_size)

    print("Discovering PhysXMobility videos to encode...")
    tasks = discover_physxmobility_videos(args.precompute_root, args.output_dir)
    print(f"Found {len(tasks)} videos to encode")

    if not tasks:
        print("Nothing to do!")
        return

    # Bad video log
    os.makedirs(os.path.dirname(args.log_bad), exist_ok=True)
    bad_log = open(args.log_bad, "w")
    bad_log.write(f"# Bad videos log - {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    bad_log.write(f"# Format: reason | path | details\n\n")

    # Double-buffer: CPU prefetches batch N+1 while GPU processes batch N
    from multiprocessing import Pool
    import threading, queue

    load_fn = _load_fn_global

    encoded = 0
    skipped = 0
    failed = 0
    t0 = time.time()
    BS = args.batch_size

    # Queue holds ready-to-go (batch_tensor_cpu, batch_tasks) tuples
    batch_queue = queue.Queue(maxsize=3)

    def prefetch_worker():
        pool = Pool(processes=args.num_workers)
        batch_tasks_local = []
        batch_frames_local = []
        for task_item, frames, bad_reason in pool.imap_unordered(load_fn, tasks, chunksize=4):
            if frames is None:
                # Put bad result to main thread for logging
                batch_queue.put(("bad", bad_reason))
                continue
            batch_tasks_local.append(task_item)
            batch_frames_local.append(frames)
            if len(batch_frames_local) >= BS:
                tensor = torch.stack(batch_frames_local)  # stack on CPU
                batch_queue.put(("batch", (tensor, batch_tasks_local)))
                batch_tasks_local = []
                batch_frames_local = []
        # Remaining
        if batch_frames_local:
            tensor = torch.stack(batch_frames_local)
            batch_queue.put(("batch", (tensor, batch_tasks_local)))
        batch_queue.put(("done", None))
        pool.close()
        pool.join()

    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()

    while True:
        item = batch_queue.get()
        tag = item[0]

        if tag == "done":
            break
        elif tag == "bad":
            bad_log.write(f"{item[1]}\n")
            bad_log.flush()
            skipped += 1
            continue
        elif tag == "batch":
            batch_tensor_cpu, batch_tasks = item[1]
            try:
                batch_tensor = batch_tensor_cpu.to(device, non_blocking=True)
                with torch.inference_mode():
                    features = model(batch_tensor)
                for i, t in enumerate(batch_tasks):
                    os.makedirs(os.path.dirname(t["out_path"]), exist_ok=True)
                    torch.save(features[i].cpu().to(torch.bfloat16), t["out_path"])
                    encoded += 1
                del batch_tensor, batch_tensor_cpu, features
            except Exception as e:
                print(f"  [BATCH ERROR] {e}")
                failed += len(batch_tasks)

            if encoded % 100 == 0 and encoded > 0:
                elapsed = time.time() - t0
                rate = encoded / elapsed
                remaining = (len(tasks) - encoded - skipped - failed) / max(rate, 0.01)
                print(f"  [{encoded}/{len(tasks)}] {rate:.1f} vid/s | "
                      f"skip={skipped} fail={failed} | q={batch_queue.qsize()} | "
                      f"ETA={remaining/60:.0f}min", flush=True)

    prefetch_thread.join(timeout=10)

    bad_log.close()
    elapsed = time.time() - t0
    print(f"\nDone! Encoded: {encoded}, Skipped: {skipped}, Failed: {failed}")
    print(f"Time: {elapsed/60:.1f}min, Rate: {encoded/max(elapsed,1):.1f} vid/s")
    print(f"Output: {args.output_dir}/PhysXMobility/")
    print(f"Bad videos log: {args.log_bad}")


if __name__ == "__main__":
    main()
