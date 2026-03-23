#!/usr/bin/env python3
"""
Encode precomputed data for PartPacker training.

Two encoding steps:
  A) part0.obj + part1.obj -> PartPacker VAE -> gt_latent.pt [1, 8192, 64]
  B) *_nobg.mp4 videos -> V-JEPA2 ViT-g -> v{XX}_nobg_jepa.pt [10240, 1408]

Only processes animodes that have BOTH part meshes AND rendered nobg videos.

Output layout (compatible with train_partnet_vjepa_ddp.py discover_samples):
  {output_dir}/{factory}/{seed}_{animode}/
    gt_latent.pt
    views/
      v00_nobg_jepa.pt
      v01_nobg_jepa.pt
      ...

Modes:
  - One-shot (default): process all existing data, then exit
  - Watch (--watch): continuously scan for new data, encode, repeat

Usage (4 GPUs, one-shot):
  CUDA_VISIBLE_DEVICES=0 python encode_for_training.py --rank 0 --world_size 4 &
  CUDA_VISIBLE_DEVICES=1 python encode_for_training.py --rank 1 --world_size 4 &
  CUDA_VISIBLE_DEVICES=2 python encode_for_training.py --rank 2 --world_size 4 &
  CUDA_VISIBLE_DEVICES=3 python encode_for_training.py --rank 3 --world_size 4 &
  wait

Usage (watch mode, concurrent with cluster_launch.py):
  CUDA_VISIBLE_DEVICES=0 python encode_for_training.py --rank 0 --world_size 2 --watch &
  CUDA_VISIBLE_DEVICES=1 python encode_for_training.py --rank 1 --world_size 2 --watch &
"""

import argparse
import gc
import importlib
import os
import sys
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ================================================================
# Path config
# ================================================================

PARTPACKER_ROOT = "/mnt/cpfs/yurh/PartPacker"
VJEPA2_ROOT = "/mnt/cpfs/yurh/vjepa2"
VJEPA2_CKPT = os.path.join(VJEPA2_ROOT, "checkpoints", "vitg.pt")
VAE_CKPT = os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt")
VAE_CONFIG = "vae.configs.part_woenc"

PRECOMPUTE_ROOT = "/mnt/data_ssd/infinigen-sim-data/precompute"
DEFAULT_OUTPUT_DIR = "/mnt/data_ssd/infinigen-sim-data/encoded"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


# ================================================================
# Discovery: find animodes with both part meshes and nobg videos
# ================================================================

def discover_tasks(precompute_root):
    """Find all animode dirs that have part0.obj + part1.obj + *_nobg.mp4."""
    tasks = []
    if not os.path.isdir(precompute_root):
        return tasks
    for factory in sorted(os.listdir(precompute_root)):
        fd = os.path.join(precompute_root, factory)
        if not os.path.isdir(fd):
            continue
        for seed in sorted(os.listdir(fd)):
            sd = os.path.join(fd, seed)
            if not os.path.isdir(sd):
                continue
            for animode in sorted(os.listdir(sd)):
                ad = os.path.join(sd, animode)
                if not os.path.isdir(ad):
                    continue
                p0 = os.path.join(ad, "part0.obj")
                p1 = os.path.join(ad, "part1.obj")
                if not (os.path.isfile(p0) and os.path.isfile(p1)):
                    continue
                nobg_vids = sorted([
                    f for f in os.listdir(ad)
                    if f.endswith("_nobg.mp4")
                ])
                if not nobg_vids:
                    continue
                tasks.append({
                    "factory": factory,
                    "seed": seed,
                    "animode": animode,
                    "animode_dir": ad,
                    "part0_path": p0,
                    "part1_path": p1,
                    "nobg_videos": nobg_vids,
                    # 2-level layout: {factory}/{seed}_{animode}
                    "output_name": os.path.join(factory, f"{seed}_{animode}"),
                })
    return tasks


def _task_key(task):
    """Unique key for dedup."""
    return f"{task['factory']}/{task['seed']}/{task['animode']}"


# ================================================================
# VAE encoding: part0.obj + part1.obj -> gt_latent.pt [1, 8192, 64]
# ================================================================

def load_vae(device):
    """Load PartPacker VAE model."""
    sys.path.insert(0, PARTPACKER_ROOT)
    sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

    import fpsample  # noqa: F401 -- needed by prepare_input
    import meshiki  # noqa: F401

    from vae.model import Model

    ckpt = torch.load(VAE_CKPT, weights_only=True)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    config = importlib.import_module(VAE_CONFIG).make_config()
    model = Model(config).eval().to(device).to(torch.bfloat16)
    model.load_state_dict(ckpt, strict=True)
    del ckpt
    print(f"[VAE] Loaded from {VAE_CKPT}")
    return model


def prepare_vae_input(vertices, faces, num_fps=2048, num_fps_salient=2048):
    """Prepare VAE input from mesh vertices/faces."""
    import fpsample
    import meshiki

    mesh = meshiki.Mesh(vertices, faces)
    uniform_pts = mesh.uniform_point_sample(200000)
    uniform_pts = meshiki.fps(uniform_pts, 32768)
    salient_pts = mesh.salient_point_sample(16384, thresh_bihedral=15)

    sample = {}
    sample["pointcloud"] = torch.from_numpy(uniform_pts)

    fps_idx = fpsample.bucket_fps_kdline_sampling(
        uniform_pts, num_fps, h=5, start_idx=0
    )
    sample["fps_indices"] = torch.from_numpy(fps_idx).long()

    sample["pointcloud_dorases"] = torch.from_numpy(salient_pts)
    fps_idx_s = fpsample.bucket_fps_kdline_sampling(
        salient_pts, num_fps_salient, h=5, start_idx=0
    )
    sample["fps_indices_dorases"] = torch.from_numpy(fps_idx_s).long()

    return sample


def encode_part_mesh(obj_path, vae_model, device):
    """Encode a single part OBJ -> latent [1, 4096, 64]."""
    import trimesh

    mesh = trimesh.load(obj_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()

    # Our meshes are already normalized to [-1, 1] by split_precompute
    sample = prepare_vae_input(mesh.vertices.astype(np.float32), mesh.faces)
    for k in sample:
        sample[k] = sample[k].unsqueeze(0).to(device)

    with torch.inference_mode():
        posterior = vae_model.encode(sample)
        latent = posterior.mode()  # [1, 4096, 64]

    return latent


def encode_gt_latent(task, vae_model, device, output_dir):
    """Encode part0 + part1 -> gt_latent.pt [1, 8192, 64]."""
    out_path = os.path.join(output_dir, task["output_name"], "gt_latent.pt")
    if os.path.exists(out_path):
        return True  # already done

    try:
        lat0 = encode_part_mesh(task["part0_path"], vae_model, device)
        lat1 = encode_part_mesh(task["part1_path"], vae_model, device)
        gt_latent = torch.cat([lat0, lat1], dim=1)  # [1, 8192, 64]

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(gt_latent.cpu().float(), out_path)
        del lat0, lat1, gt_latent
        return True
    except Exception as e:
        print(f"  [VAE ERROR] {task['output_name']}: {e}")
        return False


# ================================================================
# VJEPA2 encoding: *_nobg.mp4 -> v{XX}_nobg_jepa.pt [10240, 1408]
# ================================================================

def load_vjepa2(device, num_frames=81, img_size=256):
    """Load V-JEPA2 ViT-g model.

    Default: 81 frames @ 256x256 -> 10240 tokens x 1408 dim.
    Token count: (81//2) * (256//16)^2 = 40 * 256 = 10240.
    """
    sys.path.insert(0, VJEPA2_ROOT)
    from src.models.vision_transformer import VisionTransformer

    model = VisionTransformer(
        patch_size=16,
        embed_dim=1408,
        depth=40,
        num_heads=22,
        mlp_ratio=48 / 11,
        qkv_bias=True,
        use_rope=True,
        use_sdpa=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_size=(img_size, img_size),
        num_frames=num_frames,
    )

    ckpt = torch.load(VJEPA2_CKPT, weights_only=True, map_location="cpu")
    enc = ckpt["encoder"]
    enc = {k.replace("module.", "").replace("backbone.", ""): v
           for k, v in enc.items()}
    model.load_state_dict(enc, strict=True)
    del ckpt, enc

    model.eval().to(device)
    print(f"[VJEPA2] Loaded ViT-g: "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M params, "
          f"{num_frames} frames -> {(num_frames // 2) * (img_size // 16) ** 2} tokens")
    return model


def load_video_frames(video_path, num_frames=81, img_size=256):
    """Load video, uniformly sample num_frames, resize and normalize."""
    from decord import VideoReader

    vr = VideoReader(video_path)
    total = len(vr)
    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # [T, H, W, 3]

    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    frames = F.interpolate(frames, size=(img_size, img_size),
                           mode="bilinear", align_corners=False)
    frames = frames.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, T, H, W]
    frames = (frames - IMAGENET_MEAN) / IMAGENET_STD
    return frames


def encode_videos(task, vjepa2_model, device, output_dir,
                  num_frames=81, img_size=256):
    """Encode all nobg videos for one animode -> v{XX}_nobg_jepa.pt."""
    views_dir = os.path.join(output_dir, task["output_name"], "views")
    os.makedirs(views_dir, exist_ok=True)

    encoded = 0
    for vid_name in task["nobg_videos"]:
        # Map video name to view index: hemi_00_nobg.mp4 -> v00
        # Pattern: {type}_{idx}_nobg.mp4
        parts = vid_name.replace("_nobg.mp4", "").split("_")
        # hemi_00, orbit_03, sweep_05 etc
        view_type = parts[0]
        view_idx_str = parts[1]

        # Global view numbering: hemi 0-15, orbit 16-23, sweep 24-31
        view_num = int(view_idx_str)
        if view_type == "orbit":
            view_num += 16
        elif view_type == "sweep":
            view_num += 24

        out_path = os.path.join(views_dir, f"v{view_num:02d}_nobg_jepa.pt")
        if os.path.exists(out_path):
            encoded += 1
            continue

        video_path = os.path.join(task["animode_dir"], vid_name)
        try:
            frames = load_video_frames(video_path, num_frames, img_size)
            frames = frames.to(device)

            with torch.inference_mode():
                features = vjepa2_model(frames)  # [1, N, 1408]

            # Save as bf16, drop batch dim
            torch.save(features[0].cpu().to(torch.bfloat16), out_path)
            del frames, features
            encoded += 1
        except Exception as e:
            print(f"  [JEPA ERROR] {task['output_name']}/{vid_name}: {e}")
            continue

    return encoded


# ================================================================
# Processing pipeline
# ================================================================

def process_tasks(tasks, vae_model, vjepa2_model, device, args):
    """Run VAE + JEPA encoding on a batch of tasks."""
    vae_ok, vae_fail, jepa_total = 0, 0, 0

    if not args.skip_vae and vae_model is not None:
        for task in tqdm(tasks, desc=f"[R{args.rank}] VAE", leave=False):
            if encode_gt_latent(task, vae_model, device, args.output_dir):
                vae_ok += 1
            else:
                vae_fail += 1
            torch.cuda.empty_cache()
        print(f"[R{args.rank}] VAE: {vae_ok} ok, {vae_fail} fail")

    if not args.skip_jepa and vjepa2_model is not None:
        for task in tqdm(tasks, desc=f"[R{args.rank}] JEPA", leave=False):
            n = encode_videos(task, vjepa2_model, device, args.output_dir,
                              args.num_frames, args.img_size)
            jepa_total += n
            torch.cuda.empty_cache()
        print(f"[R{args.rank}] JEPA: {jepa_total} videos encoded")

    return vae_ok, vae_fail, jepa_total


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Encode precomputed data for PartPacker training")
    parser.add_argument("--precompute_root", type=str, default=PRECOMPUTE_ROOT)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_frames", type=int, default=81,
                        help="Frames to sample per video (81 -> 10240 tokens)")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--skip_vae", action="store_true",
                        help="Skip VAE encoding (only do JEPA)")
    parser.add_argument("--skip_jepa", action="store_true",
                        help="Skip JEPA encoding (only do VAE)")
    # Watch mode
    parser.add_argument("--watch", action="store_true",
                        help="Daemon mode: continuously scan for new data")
    parser.add_argument("--watch_interval", type=int, default=60,
                        help="Seconds between scans in watch mode (default: 60)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load models once (they persist across watch iterations)
    vae_model = None
    vjepa2_model = None

    if not args.skip_vae:
        print(f"\n{'='*60}")
        print(f"Loading VAE model...")
        print(f"{'='*60}")
        vae_model = load_vae(device)

    if not args.skip_jepa:
        # Free VAE from GPU before loading JEPA if both are needed
        # They won't fit simultaneously on one GPU; run sequentially per task
        if vae_model is not None:
            vae_model = vae_model.cpu()
            torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print(f"Loading V-JEPA2 model...")
        print(f"{'='*60}")
        vjepa2_model = load_vjepa2(device, args.num_frames, args.img_size)

    os.makedirs(args.output_dir, exist_ok=True)
    completed_keys = set()

    # Discover already-completed tasks (skip them)
    _existing_tasks = discover_tasks(args.precompute_root)
    for t in _existing_tasks:
        out_dir = os.path.join(args.output_dir, t["output_name"])
        gt_exists = os.path.exists(os.path.join(out_dir, "gt_latent.pt"))
        views_dir = os.path.join(out_dir, "views")
        n_jepa = len([f for f in os.listdir(views_dir)
                       if f.endswith("_jepa.pt")]) if os.path.isdir(views_dir) else 0
        if gt_exists and n_jepa >= len(t["nobg_videos"]):
            completed_keys.add(_task_key(t))

    iteration = 0
    while True:
        iteration += 1
        all_tasks = discover_tasks(args.precompute_root)

        # Filter out completed tasks
        new_tasks = [t for t in all_tasks if _task_key(t) not in completed_keys]

        if not new_tasks:
            if not args.watch:
                if iteration == 1:
                    print(f"No tasks to process (found {len(all_tasks)} already completed)")
                break
            # Watch mode: wait and re-scan
            if iteration == 1 or (iteration % 10 == 0):
                print(f"[Watch] No new tasks. {len(completed_keys)} completed. "
                      f"Waiting {args.watch_interval}s...")
            time.sleep(args.watch_interval)
            continue

        total_vids = sum(len(t["nobg_videos"]) for t in new_tasks)
        print(f"\n[Iter {iteration}] Found {len(new_tasks)} new animodes "
              f"({total_vids} videos), {len(completed_keys)} already done")

        # Shard across workers
        my_tasks = new_tasks[args.rank::args.world_size]
        print(f"[R{args.rank}/{args.world_size}] Processing {len(my_tasks)} animodes")

        if my_tasks:
            # If both models needed, swap between GPU: VAE first, then JEPA
            if vae_model is not None and vjepa2_model is not None:
                # Phase 1: VAE on GPU
                vjepa2_model = vjepa2_model.cpu()
                torch.cuda.empty_cache()
                vae_model = vae_model.to(device)

                for task in tqdm(my_tasks, desc=f"[R{args.rank}] VAE", leave=False):
                    encode_gt_latent(task, vae_model, device, args.output_dir)
                    torch.cuda.empty_cache()

                # Phase 2: JEPA on GPU
                vae_model = vae_model.cpu()
                torch.cuda.empty_cache()
                vjepa2_model = vjepa2_model.to(device)

                for task in tqdm(my_tasks, desc=f"[R{args.rank}] JEPA", leave=False):
                    encode_videos(task, vjepa2_model, device, args.output_dir,
                                  args.num_frames, args.img_size)
                    torch.cuda.empty_cache()
            else:
                # Only one model needed
                if vae_model is not None:
                    vae_model = vae_model.to(device)
                if vjepa2_model is not None:
                    vjepa2_model = vjepa2_model.to(device)
                process_tasks(my_tasks, vae_model, vjepa2_model, device, args)

        # Mark completed
        for t in new_tasks:
            completed_keys.add(_task_key(t))

        if not args.watch:
            break
        else:
            print(f"[Watch] Scan complete. Waiting {args.watch_interval}s...")
            time.sleep(args.watch_interval)

    print(f"\n[R{args.rank}] All done! "
          f"Total completed: {len(completed_keys)} | Output: {args.output_dir}")


if __name__ == "__main__":
    main()
