#!/usr/bin/env python3
"""
Encode diff JEPA features for filtered animodes.

1. Load VJEPA2 ViT-g, encode nobg videos -> raw [10240, 1408]
2. Apply temporal diff: keep first 5 temporal tokens, diff the rest with lag=5
3. Save as *_nobg_diff_jepa.pt

Usage (4 GPU shards):
  CUDA_VISIBLE_DEVICES=0 python encode_diff_jepa.py --shard 0 --num_shards 4
  CUDA_VISIBLE_DEVICES=1 python encode_diff_jepa.py --shard 1 --num_shards 4
  CUDA_VISIBLE_DEVICES=2 python encode_diff_jepa.py --shard 2 --num_shards 4
  CUDA_VISIBLE_DEVICES=3 python encode_diff_jepa.py --shard 3 --num_shards 4
"""
import argparse
import csv
import glob
import os
import sys
import time
from functools import partial
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VJEPA2_ROOT = "/mnt/cpfs/yurh/vjepa2"
VJEPA2_CKPT = os.path.join(VJEPA2_ROOT, "checkpoints", "vitg.pt")

PRECOMPUTE_ROOT = "/mnt/data/yurh/Infinigen-Sim/data_ssd/precompute"
ENCODED_ROOT = "/mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_diff_jepa_filtered"
RATIO_CSV = "/mnt/data/yurh/Infinigen-Sim/data_ssd/mesh_area_ratio_dist.csv"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

MIN_VIDEO_BYTES = 4096
MIN_VIDEO_FRAMES = 10


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


def apply_diff_jepa(features):
    """Apply temporal diff transform to JEPA features.

    Input:  [10240, 1408] (40 temporal * 256 spatial)
    Output: [10240, 1408] (first 5 temporal tokens original, rest diffed with lag=5)
    """
    j_3d = features.reshape(40, 256, -1)  # [40, 256, 1408]
    result = torch.zeros_like(j_3d)
    result[:5] = j_3d[:5]  # keep first 5 temporal tokens as-is
    for t in range(5, 40):
        result[t] = j_3d[t] - j_3d[t - 5]  # diff with lag=5
    return result.reshape(10240, -1)  # [10240, 1408]


def load_video(video_path, num_frames=81, img_size=256):
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
        return frames.squeeze(0).bfloat16()  # [3, T, H, W]
    except Exception:
        return None


def _load_fn(task):
    vid_path = task["video_path"]
    fsize = os.path.getsize(vid_path)
    if fsize < MIN_VIDEO_BYTES:
        return task, None
    frames = load_video(vid_path)
    return task, frames


def discover_tasks(ratio_csv, precompute_root, encoded_root,
                   ratio_min=0.2, ratio_max=3.0):
    """Find all nobg videos for filtered animodes, output diff_jepa to encoded_root."""
    # Filter by area ratio
    filtered = set()
    with open(ratio_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = float(row['part1_part0_ratio'])
            if ratio_min <= r <= ratio_max:
                filtered.add((row['category'], row['seed'], row['animode']))

    print(f"Filtered animodes ({ratio_min} <= ratio <= {ratio_max}): {len(filtered)}")

    tasks = []
    for cat, seed, animode in sorted(filtered):
        anim_dir = os.path.join(precompute_root, cat, seed, animode)
        if not os.path.isdir(anim_dir):
            continue

        nobg_vids = sorted(glob.glob(os.path.join(anim_dir, "*_nobg.mp4")))
        for vid_path in nobg_vids:
            vid_name = os.path.basename(vid_path)
            # Parse view: hemi_01_nobg.mp4 -> v01, orbit_00_nobg.mp4 -> v16, sweep_00_nobg.mp4 -> v24
            parts = vid_name.replace("_nobg.mp4", "").split("_")
            view_type, view_idx_str = parts[0], parts[1]
            view_num = int(view_idx_str)
            if view_type == "orbit":
                view_num += 16
            elif view_type == "sweep":
                view_num += 24

            model_id = f"{seed}_{animode}"
            out_dir = os.path.join(encoded_root, cat, model_id, "views")
            # Use standard naming so train script can discover via *_nobg_jepa.pt glob
            out_path = os.path.join(out_dir, f"v{view_num:02d}_nobg_jepa.pt")

            if os.path.exists(out_path):
                continue

            tasks.append({
                "video_path": vid_path,
                "out_path": out_path,
                "cat": cat,
                "model_id": model_id,
            })

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute_root", default=PRECOMPUTE_ROOT)
    parser.add_argument("--encoded_root", default=ENCODED_ROOT)
    parser.add_argument("--ratio_csv", default=RATIO_CSV)
    parser.add_argument("--ratio_min", type=float, default=0.2)
    parser.add_argument("--ratio_max", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda:0")

    tasks = discover_tasks(args.ratio_csv, args.precompute_root, args.encoded_root,
                           args.ratio_min, args.ratio_max)
    print(f"Total tasks (before sharding): {len(tasks)}")

    # Shard
    tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    print(f"Shard {args.shard}/{args.num_shards}: {len(tasks)} tasks")

    if not tasks:
        print("Nothing to do!")
        return

    model = load_vjepa2(device)

    import threading
    import queue

    batch_queue = queue.Queue(maxsize=3)

    def prefetch_worker():
        pool = Pool(processes=args.num_workers)
        batch_tasks, batch_frames = [], []
        BS = args.batch_size
        for task_item, frames in pool.imap_unordered(_load_fn, tasks, chunksize=4):
            if frames is None:
                continue
            batch_tasks.append(task_item)
            batch_frames.append(frames)
            if len(batch_frames) >= BS:
                tensor = torch.stack(batch_frames)
                batch_queue.put((tensor, batch_tasks))
                batch_tasks, batch_frames = [], []
        if batch_frames:
            tensor = torch.stack(batch_frames)
            batch_queue.put((tensor, batch_tasks))
        batch_queue.put(None)
        pool.close()
        pool.join()

    prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
    prefetch_thread.start()

    encoded = 0
    t0 = time.time()

    while True:
        item = batch_queue.get()
        if item is None:
            break

        batch_tensor_cpu, batch_tasks = item
        batch_tensor = batch_tensor_cpu.to(device, non_blocking=True)

        with torch.inference_mode():
            features = model(batch_tensor)  # [B, 10240, 1408]

        for i, t in enumerate(batch_tasks):
            feat = features[i].cpu()  # [10240, 1408]
            diff_feat = apply_diff_jepa(feat)  # [10240, 1408]
            os.makedirs(os.path.dirname(t["out_path"]), exist_ok=True)
            torch.save(diff_feat.to(torch.bfloat16), t["out_path"])
            encoded += 1

        del batch_tensor, batch_tensor_cpu, features

        if encoded % 50 == 0:
            elapsed = time.time() - t0
            rate = encoded / elapsed
            remaining = (len(tasks) - encoded) / max(rate, 0.01)
            print(f"  [{encoded}/{len(tasks)}] {rate:.1f} vid/s | "
                  f"ETA={remaining/60:.0f}min", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone! Encoded {encoded} videos in {elapsed/60:.1f}min "
          f"({encoded/elapsed:.1f} vid/s)")


if __name__ == "__main__":
    main()
