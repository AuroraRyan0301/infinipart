#!/usr/bin/env python3
"""
Finetune PartPacker VAE on our articulated object dual-volume data.

Requires preprocessed .pt files from preprocess_vae_data.py.
Each .pt contains: pointcloud, fps_indices, pointcloud_dorases,
fps_indices_dorases, query_points, query_gt.

Usage:
  # Step 1: Preprocess (CPU, run once)
  python preprocess_vae_data.py --data_root ./precompute_output \
    --output_dir /mnt/data_ssd/infinigen-sim/vae_cache --workers 32

  # Step 2: Finetune (GPU)
  CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 finetune_vae.py \
    --cache_dir /mnt/data_ssd/infinigen-sim/vae_cache \
    --output_dir /mnt/data_ssd/infinigen-sim/vae_finetune \
    --lr 1e-5 --steps 20000 --batch_size 4
"""

import argparse
import importlib
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

PARTPACKER_ROOT = "/mnt/data/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))


# ================================================================
# Dataset (loads preprocessed .pt files)
# ================================================================

class CachedVAEDataset(Dataset):
    """Load preprocessed .pt files from preprocess_vae_data.py."""

    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.files = sorted([
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir) if f.endswith(".pt")
        ])
        print(f"[VAE Dataset] {len(self.files)} cached samples from {cache_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.files[idx], weights_only=False)
            # Remove non-tensor keys (can't be collated)
            for k in list(data.keys()):
                if not isinstance(data[k], torch.Tensor):
                    del data[k]
            return data
        except Exception:
            alt = random.randint(0, len(self.files) - 1)
            data = torch.load(self.files[alt], weights_only=False)
            for k in list(data.keys()):
                if not isinstance(data[k], torch.Tensor):
                    del data[k]
            return data


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

    # Load VAE
    from vae.model import Model as VAE
    vae_config = importlib.import_module("vae.configs.part_woenc").make_config()
    vae = VAE(vae_config).to(device, dtype=torch.bfloat16)

    # Load pretrained weights
    ckpt_path = os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt")
    ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    vae.load_state_dict(ckpt, strict=True)
    del ckpt
    if is_main:
        n_params = sum(p.numel() for p in vae.parameters()) / 1e6
        print(f"VAE loaded: {n_params:.1f}M params from {ckpt_path}")

    vae.train()

    if ddp:
        vae = DDP(vae, device_ids=[local_rank], find_unused_parameters=False)

    raw_model = vae.module if ddp else vae

    # Dataset
    dataset = CachedVAEDataset(args.cache_dir)
    if ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                      rank=rank, shuffle=True)
    else:
        sampler = None

    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        num_workers=args.num_workers, pin_memory=True,
        drop_last=True, persistent_workers=args.num_workers > 0,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        vae.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # LR schedule: cosine with warmup
    warmup_steps = min(500, args.steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    step = 0
    t0 = time.time()
    running_loss = 0
    running_mse = 0
    running_iou = 0

    if is_main:
        print(f"\nStarting VAE finetuning:")
        print(f"  Data: {len(dataset)} cached samples from {args.cache_dir}")
        print(f"  Batch: {args.batch_size} x {world_size} GPUs = {args.batch_size * world_size}")
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
            for k, v in batch.items():
                batch[k] = v.to(device)

            # Forward
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output, loss = raw_model.training_step(batch, step)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Logging
            loss_val = loss.item()
            running_loss += loss_val
            running_mse += output["scalar"]["loss_mse"].item()
            running_iou += output["scalar"]["iou_fg"].item()

            step += 1

            if is_main and (step % args.log_every == 0 or step == 1):
                n = min(step, args.log_every)
                avg_loss = running_loss / n
                avg_mse = running_mse / n
                avg_iou = running_iou / n
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                rate = step / elapsed
                eta = (args.steps - step) / rate if rate > 0 else 0

                print(f"  [{step}/{args.steps}] loss={avg_loss:.4f} "
                      f"mse={avg_mse:.4f} iou_fg={avg_iou:.3f} "
                      f"lr={lr:.2e} | {rate:.2f} it/s ETA {eta/60:.0f}m")

                running_loss = 0
                running_mse = 0
                running_iou = 0

            # Save checkpoint
            if is_main and (step % args.save_every == 0 or step == args.steps):
                ckpt = {
                    "model": raw_model.state_dict(),
                    "step": step,
                    "args": vars(args),
                }
                path = os.path.join(args.output_dir, f"vae_ft_step_{step}.pt")
                torch.save(ckpt, path)
                # Also save as latest
                latest_path = os.path.join(args.output_dir, "vae_ft_latest.pt")
                torch.save(ckpt, latest_path)
                print(f"  Saved: {path}")

    elapsed = time.time() - t0
    if is_main:
        print(f"\nDone in {elapsed/60:.1f}m ({step} steps)")

    if ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Finetune PartPacker VAE")
    parser.add_argument("--cache_dir", type=str,
                        default="/mnt/data_ssd/infinigen-sim/vae_cache")
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/data_ssd/infinigen-sim/vae_finetune")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
