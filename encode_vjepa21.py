#!/usr/bin/env python3
"""
Encode V-JEPA 2.1 features for filtered animodes.

V-JEPA 2.1 ViT-g @384: output [23040, 1408] = 40 temporal × 576 spatial × 1408 dim
Uses autocast for inference (2.1 attention has dtype constraints).

Usage (4 GPU shards):
  CUDA_VISIBLE_DEVICES=0 python encode_vjepa21.py --shard 0 --num_shards 4
"""
import argparse
import csv
import glob
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F

VJEPA2_ROOT = "/mnt/cpfs/yurh/vjepa2"
VJEPA21_CKPT = os.path.join(VJEPA2_ROOT, "checkpoints", "vjepa2_1_vitg_384.pt")

PRECOMPUTE_ROOT = "/mnt/data/yurh/Infinigen-Sim/data_ssd/precompute"
OUTPUT_ROOT = "/mnt/data/yurh/Infinigen-Sim/data_ssd/encoded_vjepa21"
RATIO_CSV = "/mnt/data/yurh/Infinigen-Sim/data_ssd/mesh_area_ratio_dist.csv"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)

MIN_VIDEO_BYTES = 4096
MIN_VIDEO_FRAMES = 10


def load_vjepa21(device, num_frames=81, img_size=384):
    if VJEPA2_ROOT not in sys.path:
        sys.path.insert(0, VJEPA2_ROOT)
    from app.vjepa_2_1.models.vision_transformer import vit_giant_xformers

    model = vit_giant_xformers(
        img_size=(img_size, img_size), num_frames=num_frames,
        patch_size=16, tubelet_size=2,
        use_sdpa=True, use_SiLU=False, wide_SiLU=True,
        uniform_power=False, use_rope=True,
    )
    ckpt = torch.load(VJEPA21_CKPT, weights_only=True, map_location="cpu")
    enc = {k.replace("module.", "").replace("backbone.", ""): v
           for k, v in ckpt["target_encoder"].items()}
    model.load_state_dict(enc, strict=False)
    del ckpt, enc
    model.eval().to(device)
    print(f"[V-JEPA 2.1] Loaded ViT-g @{img_size} on {device}")
    return model


def load_video(video_path, num_frames=81, img_size=384):
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
        return frames.squeeze(0)  # [3, T, H, W] float32
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
        # Only hemi views
        nobg_vids = sorted(glob.glob(os.path.join(anim_dir, "hemi_*_nobg.mp4")))
        for vid_path in nobg_vids:
            vid_name = os.path.basename(vid_path)
            view_num = int(vid_name.replace("_nobg.mp4", "").split("_")[1])

            model_id = f"{seed}_{animode}"
            out_dir = os.path.join(output_root, cat, model_id, "views")
            out_path = os.path.join(out_dir, f"v{view_num:02d}_nobg_jepa.pt")

            if os.path.exists(out_path):
                continue

            tasks.append({
                "video_path": vid_path,
                "out_path": out_path,
            })

    return tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precompute_root", default=PRECOMPUTE_ROOT)
    parser.add_argument("--output_root", default=OUTPUT_ROOT)
    parser.add_argument("--ratio_csv", default=RATIO_CSV)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (smaller than v2.0 due to 384 resolution)")
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

    model = load_vjepa21(device)

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

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            features = model(batch_tensor)  # [B, 23040, 1408]

        for i, t in enumerate(batch_tasks):
            feat = features[i].cpu().to(torch.bfloat16)  # [23040, 1408]
            os.makedirs(os.path.dirname(t["out_path"]), exist_ok=True)
            torch.save(feat, t["out_path"])
            encoded += 1

        del batch_tensor, batch_tensor_cpu, features

        if encoded % 20 == 0:
            elapsed = time.time() - t0
            rate = encoded / elapsed
            remaining = (len(tasks) - encoded) / max(rate, 0.01)
            print(f"  [{encoded}/{len(tasks)}] {rate:.1f} vid/s | "
                  f"ETA={remaining/60:.0f}min", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone! Encoded {encoded} videos in {elapsed/60:.1f}min ({encoded/elapsed:.1f} vid/s)")


if __name__ == "__main__":
    main()
