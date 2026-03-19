#!/usr/bin/env python3
"""
Train DualPartSLatModel: finetune Shape SLat with part cross-attention + VJEPA conditioning.

Training data:
  - GT SLat latent (from preprocess_slat_gt.py)
  - VJEPA video features (from encode_for_training.py)
  - VAE 64³ occupancy coords (from PartPacker VAE decode)

Loss: Flow matching on SLat latent space.

Usage:
  # Overfit test (10 samples)
  CUDA_VISIBLE_DEVICES=2 python train_slat_refiner.py \
    --slat_cache /mnt/data_ssd/infinigen-sim/slat_cache \
    --data_root /mnt/data_ssd/infinigen-sim \
    --output_dir /mnt/data_ssd/infinigen-sim/slat_refiner_overfit \
    --max_samples 10 --steps 2000 --batch_size 1

  # Full training
  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29502 train_slat_refiner.py \
    --slat_cache /mnt/data_ssd/infinigen-sim/slat_cache \
    --data_root /mnt/data_ssd/infinigen-sim \
    --output_dir /mnt/data_ssd/infinigen-sim/slat_refiner \
    --steps 20000 --batch_size 2
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler

TRELLIS_ROOT = "/mnt/cpfs/yurh/TRELLIS.2"
sys.path.insert(0, TRELLIS_ROOT)
sys.path.insert(0, os.path.join(TRELLIS_ROOT, "o-voxel"))

from trellis2.modules.sparse import SparseTensor


# ================================================================
# Dataset
# ================================================================

class SLatPairDataset(Dataset):
    """Load paired (part0, part1) SLat GT latent + VJEPA features.

    Each sample is an animode: part0_slat + part1_slat + vjepa_features.
    """

    def __init__(self, slat_cache_dir, data_root, precompute_root, max_samples=None):
        self.slat_cache_dir = slat_cache_dir
        self.data_root = data_root
        self.precompute_root = precompute_root
        self.samples = []

        # Discover animode pairs that have both part0 + part1 SLat + VJEPA
        for cat in sorted(os.listdir(data_root)):
            cat_dir = os.path.join(data_root, cat)
            if not os.path.isdir(cat_dir) or cat.startswith("."):
                continue
            if cat in ("train_output", "train_output_v2", ".errors"):
                continue
            for mid in sorted(os.listdir(cat_dir)):
                md = os.path.join(cat_dir, mid)
                if not os.path.isdir(md):
                    continue
                gt_path = os.path.join(md, "gt_latent.pt")
                if not os.path.exists(gt_path):
                    continue

                parts = mid.split("_", 1)
                if len(parts) < 2:
                    continue
                seed, animode = parts[0], parts[1]

                # Check SLat cache for both parts
                p0_obj = os.path.join(precompute_root, cat, seed, animode, "part0.obj")
                p1_obj = os.path.join(precompute_root, cat, seed, animode, "part1.obj")
                h0 = hashlib.md5(p0_obj.encode()).hexdigest()[:12]
                h1 = hashlib.md5(p1_obj.encode()).hexdigest()[:12]
                p0_slat = os.path.join(slat_cache_dir, f"{h0}.pt")
                p1_slat = os.path.join(slat_cache_dir, f"{h1}.pt")

                if not os.path.exists(p0_slat) or not os.path.exists(p1_slat):
                    continue

                # Check VJEPA features
                views_dir = os.path.join(md, "views")
                if not os.path.isdir(views_dir):
                    continue
                jepa_files = sorted([
                    os.path.join(views_dir, f)
                    for f in os.listdir(views_dir) if f.endswith("_nobg_jepa.pt")
                ])
                if not jepa_files:
                    continue

                self.samples.append({
                    "id": f"{cat}/{mid}",
                    "p0_slat": p0_slat,
                    "p1_slat": p1_slat,
                    "jepa_files": jepa_files,
                })

        if max_samples and max_samples < len(self.samples):
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        print(f"[SLatPairDataset] {len(self.samples)} animode pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Load SLat GT for both parts
        p0 = torch.load(s["p0_slat"], weights_only=False)
        p1 = torch.load(s["p1_slat"], weights_only=False)

        # Load a random VJEPA feature
        jepa_path = random.choice(s["jepa_files"])
        jepa = torch.load(jepa_path, weights_only=False)
        if jepa.dim() == 2:
            jepa = jepa.unsqueeze(0)  # [1, T, D]

        return {
            "p0_feats": p0["slat_feats"],       # [N0, 32]
            "p0_coords": p0["slat_coords"],      # [N0, 4]
            "p0_grid_size": p0["grid_size"],
            "p1_feats": p1["slat_feats"],        # [N1, 32]
            "p1_coords": p1["slat_coords"],      # [N1, 4]
            "p1_grid_size": p1["grid_size"],
            "jepa": jepa.squeeze(0),              # [T, D]
        }


def collate_slat(batch):
    """Custom collate: can't stack variable-length sparse tensors."""
    # For now, batch_size=1 (sparse tensors have variable N)
    # TODO: batch>1 with padding
    return batch[0]


# ================================================================
# Flow matching
# ================================================================

def sample_flow_matching(gt_feats, t):
    """Sample noisy latent for flow matching.

    x_t = (1 - t) * noise + t * x_1
    velocity = x_1 - noise
    """
    noise = torch.randn_like(gt_feats)
    x_t = (1 - t) * noise + t * gt_feats
    velocity = gt_feats - noise
    return x_t, velocity


# ================================================================
# Training
# ================================================================

def train(args):
    # DDP setup
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")

    is_main = rank == 0

    # Load DualPartSLatModel
    sys.path.insert(0, "/mnt/cpfs/yurh/Infinigen-Sim")
    from dual_part_slat import build_dual_part_model

    model = build_dual_part_model(
        ckpt_dir="/mnt/data/yurh/TRELLIS.2-4B",
        device=device
    )

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    raw_model = model.module if ddp else model

    # Dataset
    dataset = SLatPairDataset(
        args.slat_cache, args.data_root,
        args.precompute_root, max_samples=args.max_samples)

    if ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset, batch_size=1,  # sparse tensors, batch=1
        shuffle=(sampler is None), sampler=sampler,
        num_workers=2, pin_memory=True,
        collate_fn=collate_slat,
    )

    # Optimizer (only trainable params)
    optimizer = torch.optim.AdamW(
        raw_model.trainable_parameters(),
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95))

    # LR schedule
    warmup_steps = min(500, args.steps // 10)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    step = 0
    t0 = time.time()
    running_loss = 0

    if is_main:
        print(f"\nStarting SLat Refiner training:")
        print(f"  Data: {len(dataset)} animode pairs")
        print(f"  Trainable: {raw_model.num_trainable_params()/1e6:.0f}M / {raw_model.num_total_params()/1e6:.0f}M")
        print(f"  Steps: {args.steps}, LR: {args.lr}")
        print(f"  Output: {args.output_dir}")
        print()

    while step < args.steps:
        if ddp and sampler is not None:
            sampler.set_epoch(step)

        for batch in loader:
            if step >= args.steps:
                break

            # Move to device
            p0_feats = batch["p0_feats"].to(device)
            p0_coords = batch["p0_coords"].to(device)
            p1_feats = batch["p1_feats"].to(device)
            p1_coords = batch["p1_coords"].to(device)
            jepa = batch["jepa"].unsqueeze(0).to(device)  # [1, T, D]

            # Sample timestep
            t_val = torch.rand(1, device=device).clamp(1e-5, 1 - 1e-5)
            t_broadcast = torch.tensor([1000 * t_val.item()], device=device)

            # Sample noisy latent for both parts
            x0_t, vel0 = sample_flow_matching(p0_feats, t_val)
            x1_t, vel1 = sample_flow_matching(p1_feats, t_val)

            # Build SparseTensors
            x0_st = SparseTensor(feats=x0_t.to(torch.bfloat16), coords=p0_coords)
            x1_st = SparseTensor(feats=x1_t.to(torch.bfloat16), coords=p1_coords)

            # Forward
            pred0, pred1 = raw_model(
                x0_st, x1_st, t_broadcast,
                jepa.to(torch.bfloat16))

            # Loss: MSE on predicted velocity
            loss0 = F.mse_loss(pred0.feats.float(), vel0)
            loss1 = F.mse_loss(pred1.feats.float(), vel1)
            loss = (loss0 + loss1) / 2

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.trainable_parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step += 1

            if is_main and (step % args.log_every == 0 or step == 1):
                n = min(step, args.log_every)
                avg_loss = running_loss / n
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                rate = step / elapsed
                eta = (args.steps - step) / rate if rate > 0 else 0
                print(f"  [{step}/{args.steps}] loss={avg_loss:.4f} "
                      f"(p0={loss0.item():.4f} p1={loss1.item():.4f}) "
                      f"lr={lr:.2e} | {rate:.2f} it/s ETA {eta/60:.0f}m")
                running_loss = 0

            # Save
            if is_main and (step % args.save_every == 0 or step == args.steps):
                ckpt = {
                    "vjepa_proj": raw_model.vjepa_proj.state_dict(),
                    "part_cross_attns": raw_model.part_cross_attns.state_dict(),
                    "step": step,
                    "args": vars(args),
                }
                path = os.path.join(args.output_dir, f"slat_refiner_step_{step}.pt")
                torch.save(ckpt, path)
                latest = os.path.join(args.output_dir, "slat_refiner_latest.pt")
                torch.save(ckpt, latest)
                print(f"  Saved: {path}")

            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    if is_main:
        print(f"\nDone in {elapsed/60:.1f}m ({step} steps)")

    if ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slat_cache", default="/mnt/data_ssd/infinigen-sim/slat_cache")
    parser.add_argument("--data_root", default="/mnt/data_ssd/infinigen-sim")
    parser.add_argument("--precompute_root", default="/mnt/cpfs/yurh/Infinigen-Sim/precompute_output")
    parser.add_argument("--output_dir", default="/mnt/data_ssd/infinigen-sim/slat_refiner")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
