#!/usr/bin/env python3
"""
Train DualPartSLatModel with progressive cross-attention expansion.

Strategy: start with cross-attn on last 1 layer (block 29), expand by 1 layer
every --expand_every steps until reaching --cross_attn_min.

Multi-GPU via DDP. Batch size = 1 per GPU (sparse tensors, variable length).

Usage:
  # 3-GPU training
  CUDA_VISIBLE_DEVICES=1,2,3 torchrun --nproc_per_node=3 --master_port=29503 \
    train_slat_progressive.py --steps 20000 --output_dir /mnt/data_ssd/infinigen-sim/slat_progressive_full

  # Single GPU
  CUDA_VISIBLE_DEVICES=1 python train_slat_progressive.py --steps 20000 \
    --output_dir /mnt/data_ssd/infinigen-sim/slat_progressive_full
"""
import argparse
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
sys.path.insert(0, "/mnt/cpfs/yurh/Infinigen-Sim")

from trellis2.modules.sparse import SparseTensor

# SLat normalization constants (from TRELLIS 2 pipeline.json)
SLAT_NORM_MEAN = torch.tensor([
    0.781296, 0.018091, -0.495192, -0.558457, 1.060530, 0.093252, 1.518149, -0.933218,
    -0.732996, 2.604095, -0.118341, -2.143904, 0.495076, -2.179512, -2.130751, -0.996944,
    0.261421, -2.217463, 1.260067, -0.150213, 3.790713, 1.481266, -1.046058, -1.523667,
    -0.059621, 2.220780, 1.621212, 0.877230, 0.567247, -3.175944, -3.186688, 1.578665
])
SLAT_NORM_STD = torch.tensor([
    5.972266, 4.706852, 5.445010, 5.209927, 5.320220, 4.547237, 5.020802, 5.444004,
    5.226681, 5.683095, 4.831436, 5.286469, 5.652043, 5.367606, 5.525084, 4.730578,
    4.805265, 5.124013, 5.530808, 5.619001, 5.103930, 5.417670, 5.269677, 5.547194,
    5.634698, 5.235274, 6.110351, 5.511298, 6.237273, 4.879207, 5.347008, 5.405691
])
SIGMA_MIN = 1e-5


def normalize_slat(feats, device):
    return (feats - SLAT_NORM_MEAN.to(device)) / SLAT_NORM_STD.to(device)


def sample_flow_matching(x_0, t):
    noise = torch.randn_like(x_0)
    x_t = (1 - t) * x_0 + (SIGMA_MIN + (1 - SIGMA_MIN) * t) * noise
    velocity = (1 - SIGMA_MIN) * noise - x_0
    return x_t, velocity


# ================================================================
# Dataset using slat_gt/ format
# ================================================================

class SLatGTDataset(Dataset):
    """Load paired SLat GT + VJEPA from slat_gt/{cat}/{seed}_{animode}.pt format."""

    SKIP_DIRS = {"train_output", "train_output_v2", ".errors", "slat_gt",
                 "slat_overfit_v3", "slat_progressive", "vae_cache",
                 "vae_finetune", "vae_finetune_v2", "vae_finetune_v3", "vae_finetune_v4"}

    def __init__(self, data_root, slat_gt_root, max_samples=None, max_tokens=20000):
        self.samples = []

        for cat in sorted(os.listdir(slat_gt_root)):
            cat_gt = os.path.join(slat_gt_root, cat)
            cat_data = os.path.join(data_root, cat)
            if not os.path.isdir(cat_gt) or not os.path.isdir(cat_data):
                continue

            for f in sorted(os.listdir(cat_gt)):
                if not f.endswith('.pt'):
                    continue
                mid = f[:-3]  # e.g., "0_basic_0"

                # Check VJEPA
                views_dir = os.path.join(cat_data, mid, "views")
                if not os.path.isdir(views_dir):
                    continue
                jepa_files = sorted([
                    os.path.join(views_dir, j)
                    for j in os.listdir(views_dir) if j.endswith("_nobg_jepa.pt")
                ])
                if not jepa_files:
                    continue

                self.samples.append({
                    "id": f"{cat}/{mid}",
                    "gt_path": os.path.join(cat_gt, f),
                    "jepa_files": jepa_files,
                })

        if max_samples and max_samples < len(self.samples):
            random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        self.max_tokens = max_tokens

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for attempt in range(5):
            try:
                i = idx if attempt == 0 else random.randint(0, len(self.samples) - 1)
                s = self.samples[i]
                gt = torch.load(s["gt_path"], weights_only=False, map_location="cpu")
                p0 = gt["p0_lr"]  # 512-resolution coords
                p1 = gt["p1_lr"]
                total_tokens = p0["feats"].shape[0] + p1["feats"].shape[0]
                if total_tokens > self.max_tokens:
                    i = random.randint(0, len(self.samples) - 1)
                    continue

                jepa_path = random.choice(s["jepa_files"])
                jepa = torch.load(jepa_path, weights_only=False, map_location="cpu")
                if jepa.dim() == 2:
                    jepa = jepa.unsqueeze(0)

                return {
                    "p0_feats": p0["feats"],
                    "p0_coords": p0["coords"],
                    "p1_feats": p1["feats"],
                    "p1_coords": p1["coords"],
                    "jepa": jepa.squeeze(0),
                }
            except (EOFError, RuntimeError, KeyError):
                continue
        return self.__getitem__(0)


def collate_slat(batch):
    """Batch size = 1 for sparse tensors."""
    return batch[0]


# ================================================================
# Training
# ================================================================

def train(args):
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

    # Build model
    from dual_part_slat import build_dual_part_model
    model = build_dual_part_model(
        device=device, resolution="512",
        cross_attn_start_block=args.cross_attn_start,
    )
    raw_model = model

    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
        raw_model = model.module

    # Dataset
    dataset = SLatGTDataset(
        args.data_root, args.slat_gt_root,
        max_samples=args.max_samples, max_tokens=args.max_tokens,
    )
    if is_main:
        print(f"[Dataset] {len(dataset)} samples")

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(
        dataset, batch_size=1, shuffle=(sampler is None), sampler=sampler,
        num_workers=4, pin_memory=True, collate_fn=collate_slat,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        raw_model.trainable_parameters(),
        lr=args.lr, weight_decay=0.01, betas=(0.9, 0.95),
    )
    warmup = min(500, args.steps // 10)

    # Resume
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.vjepa_proj.load_state_dict(ckpt["vjepa_proj"])
        raw_model.part_cross_attns.load_state_dict(ckpt["part_cross_attns"])
        start_step = ckpt.get("step", 0)
        raw_model.cross_attn_start_block = ckpt.get("cross_attn_start_block", args.cross_attn_start)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if is_main:
            print(f"Resumed from step {start_step}, cross_attn_start={raw_model.cross_attn_start_block}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Training loop
    step = start_step
    t0 = time.time()
    running_loss = 0
    current_start = raw_model.cross_attn_start_block

    if is_main:
        print(f"\n=== Progressive SLat Refiner Training ===")
        print(f"  GPUs: {world_size}, Data: {len(dataset)} samples")
        print(f"  Trainable: {raw_model.num_trainable_params()/1e6:.0f}M / {raw_model.num_total_params()/1e6:.0f}M")
        print(f"  Steps: {args.steps}, LR: {args.lr}")
        print(f"  Progressive: block {args.cross_attn_start}→{args.cross_attn_min}, "
              f"expand every {args.expand_every} steps")
        print(f"  Output: {args.output_dir}\n")

    while step < args.steps:
        if ddp and sampler is not None:
            sampler.set_epoch(step // len(dataset))

        for batch in loader:
            if step >= args.steps:
                break

            # Progressive expansion
            desired_start = max(
                args.cross_attn_min,
                args.cross_attn_start - (step - start_step) // args.expand_every
            )
            if desired_start < current_start:
                current_start = desired_start
                raw_model.cross_attn_start_block = current_start
                if is_main:
                    active = 30 - current_start
                    print(f"\n  >>> Step {step}: expanded to {active} cross-attn layers "
                          f"(blocks {current_start}-29)\n")

            # Move to device
            p0_feats = batch["p0_feats"].to(device)
            p0_coords = batch["p0_coords"].to(device)
            p1_feats = batch["p1_feats"].to(device)
            p1_coords = batch["p1_coords"].to(device)
            jepa = batch["jepa"].unsqueeze(0).to(device)

            # Normalize
            p0_feats = normalize_slat(p0_feats, device)
            p1_feats = normalize_slat(p1_feats, device)

            # Flow matching
            t_val = torch.rand(1, device=device).clamp(1e-5, 1 - 1e-5)
            t_broadcast = torch.tensor([1000 * t_val.item()], device=device)
            x0_t, vel0 = sample_flow_matching(p0_feats, t_val)
            x1_t, vel1 = sample_flow_matching(p1_feats, t_val)

            x0_st = SparseTensor(feats=x0_t, coords=p0_coords)
            x1_st = SparseTensor(feats=x1_t, coords=p1_coords)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                pred0, pred1 = raw_model(x0_st, x1_st, t_broadcast, jepa)

            loss = (F.mse_loss(pred0.feats.float(), vel0) +
                    F.mse_loss(pred1.feats.float(), vel1)) / 2

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(raw_model.trainable_parameters(), 1.0)

            # LR with warmup
            lr_scale = min(1.0, (step + 1) / warmup) if warmup > 0 else 1.0
            for pg in optimizer.param_groups:
                pg["lr"] = args.lr * lr_scale
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if is_main and (step % args.log_every == 0 or step == 1):
                n = min(step - start_step, args.log_every)
                avg = running_loss / max(n, 1)
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                rate = (step - start_step) / elapsed
                eta = (args.steps - step) / rate if rate > 0 else 0
                active = 30 - current_start
                print(f"  [{step}/{args.steps}] loss={avg:.4f} lr={lr:.2e} "
                      f"active={active} | {rate:.2f} it/s ETA {eta/60:.0f}m")
                running_loss = 0

            if is_main and (step % args.save_every == 0 or step == args.steps):
                ckpt = {
                    "vjepa_proj": raw_model.vjepa_proj.state_dict(),
                    "part_cross_attns": raw_model.part_cross_attns.state_dict(),
                    "cross_attn_start_block": current_start,
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "args": vars(args),
                }
                path = os.path.join(args.output_dir, f"step_{step}.pt")
                torch.save(ckpt, path)
                torch.save(ckpt, os.path.join(args.output_dir, "latest.pt"))
                print(f"  Saved: {path}")

            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    if is_main:
        print(f"\nDone in {elapsed/60:.1f}m ({step} steps). "
              f"Final active layers: {30 - current_start} (blocks {current_start}-29)")

    if ddp:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/mnt/data_ssd/infinigen-sim")
    parser.add_argument("--slat_gt_root", default="/mnt/data_ssd/infinigen-sim/slat_gt")
    parser.add_argument("--output_dir", default="/mnt/data_ssd/infinigen-sim/slat_progressive_full")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=20000)
    parser.add_argument("--cross_attn_start", type=int, default=29,
                        help="Initial cross_attn_start_block (29=last 1 layer)")
    parser.add_argument("--cross_attn_min", type=int, default=15,
                        help="Minimum cross_attn_start_block (15=last 15 layers)")
    parser.add_argument("--expand_every", type=int, default=500,
                        help="Expand cross-attn by 1 layer every N steps")
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
