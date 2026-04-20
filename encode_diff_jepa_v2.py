#!/usr/bin/env python3
"""
Encode concatenated JEPA features: 40 original + 35 diff = 19200 tokens.

Original: full VJEPA2 output [40, 256, 1408] = 10240 tokens
Diff:     temporal difference with lag=5 for t=5..39 → [35, 256, 1408] = 8960 tokens
Output:   concat([orig, diff]) = [19200, 1408]

Usage (4 GPU shards):
  CUDA_VISIBLE_DEVICES=0 python encode_diff_jepa_v2.py --shard 0 --num_shards 4
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
OUTPUT_ROOT = "/mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_jepa_v2"
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


def make_orig_plus_diff(features):
    """
    Input:  [10240, 1408] raw JEPA (40 temporal × 256 spatial)
    Output: [9600, 1408] = concat(orig_sub[5120], diff_sub[4480])

    orig: [10240, 1408] → stride-2 subsample → [5120, 1408]
    diff: [8960, 1408]  → stride-2 subsample → [4480, 1408]
    """
    j_3d = features.reshape(40, 256, -1)  # [40, 256, 1408]
    orig = j_3d.reshape(-1, j_3d.shape[-1])            # [10240, 1408]
    diff = (j_3d[5:] - j_3d[:35]).reshape(-1, j_3d.shape[-1])  # [8960, 1408]
    orig_sub = orig[::2]  # [5120, 1408]
    diff_sub = diff[::2]  # [4480, 1408]
    return torch.cat([orig_sub, diff_sub], dim=0)  # [9600, 1408]


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
        frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)
        frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
        return frames.squeeze(0).bfloat16()
    except Exception:
        return None


def _load_fn(task):
    vid_path = task["video_path"]
    fsize = os.path.getsize(vid_path)
    if fsize < MIN_VIDEO_BYTES:
        return task, None
    frames = load_video(vid_path)
    return task, frames


def discover_tasks(ratio_csv, precompute_root, output_root,
                   ratio_min=0.2, ratio_max=3.0, exclude_cats=None):
    if exclude_cats is None:
        exclude_cats = {"box"}
    filtered = set()
    with open(ratio_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = float(row['part1_part0_ratio'])
            cat = row['category']
            if ratio_min <= r <= ratio_max and cat not in exclude_cats:
                filtered.add((cat, row['seed'], row['animode']))

    print(f"Filtered animodes: {len(filtered)} (ratio [{ratio_min}, {ratio_max}], excl {exclude_cats})")

    tasks = []
    for cat, seed, animode in sorted(filtered):
        anim_dir = os.path.join(precompute_root, cat, seed, animode)
        if not os.path.isdir(anim_dir):
            continue
        # Only hemi views (fixed viewpoint)
        nobg_vids = sorted(glob.glob(os.path.join(anim_dir, "hemi_*_nobg.mp4")))
        for vid_path in nobg_vids:
            vid_name = os.path.basename(vid_path)
            parts = vid_name.replace("_nobg.mp4", "").split("_")
            view_num = int(parts[1])

            model_id = f"{seed}_{animode}"
            out_dir = os.path.join(output_root, cat, model_id, "views")
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
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    parser.add_argument("--ratio_csv", default=RATIO_CSV)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    tasks = discover_tasks(args.ratio_csv, args.precompute_root, args.output_root)
    print(f"Total tasks: {len(tasks)}")

    tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    print(f"Shard {args.shard}/{args.num_shards}: {len(tasks)} tasks")

    if not tasks:
        print("Nothing to do!")
        return

    model = load_vjepa2(device)

    import threading, queue
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
            combined = make_orig_plus_diff(feat)  # [19200, 1408]
            os.makedirs(os.path.dirname(t["out_path"]), exist_ok=True)
            torch.save(combined.to(torch.bfloat16), t["out_path"])
            encoded += 1

        del batch_tensor, batch_tensor_cpu, features

        if encoded % 50 == 0:
            elapsed = time.time() - t0
            rate = encoded / elapsed
            remaining = (len(tasks) - encoded) / max(rate, 0.01)
            print(f"  [{encoded}/{len(tasks)}] {rate:.1f} vid/s | ETA={remaining/60:.0f}min",
                  flush=True)

    elapsed = time.time() - t0
    print(f"\nDone! Encoded {encoded} videos in {elapsed/60:.1f}min ({encoded/elapsed:.1f} vid/s)")


if __name__ == "__main__":
    main()
