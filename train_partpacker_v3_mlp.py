#!/usr/bin/env python3
"""
PartPacker Flow DiT training v3: MLP-only projector.
No self-attention, no AdaLN injection — just a 2-layer MLP projector.

Feature: [9600, 1408] = orig_sub(5120) + diff_sub(4480)
Projector: MLP(1408 → 1536, GELU, 1536 → 1536)
DiT: 1249.5M, loaded from Exp1 step_70000

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_partpacker_v3_mlp.py \
    --manifest /path/to/manifest.json \
    --output_dir /path/to/checkpoints \
    --resume /path/to/step_70000.pt \
    --batch_size 8 --steps 110000 --warmup_proj_steps 10000
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

PARTPACKER_ROOT = "/mnt/cpfs/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

DIT_DIM = 1536


class MLPProjector(nn.Module):
    """2-layer MLP projector. Simple, stable, no attention."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TrainWrapper(nn.Module):
    def __init__(self, dit, cond_dim=1408):
        super().__init__()
        self.dit = dit
        self.proj = MLPProjector(cond_dim, DIT_DIM)

    def forward(self, noisy_latent, vjepa_feat, timesteps):
        cond = self.proj(vjepa_feat)
        return self.dit(noisy_latent, cond, timesteps)


class LatentDataset(torch.utils.data.Dataset):
    def __init__(self, samples, gt_cache_size=512, cond_dim=1408, cond_tokens=9600):
        self._samples = samples
        self._gt_cache = {}
        self._gt_cache_order = []
        self._gt_cache_size = gt_cache_size
        self.cond_dim = cond_dim
        self._cond_tokens = cond_tokens

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        s = self._samples[idx]
        try:
            return self._load_sample(s)
        except Exception:
            for _ in range(10):
                try:
                    return self._load_sample(random.choice(self._samples))
                except Exception:
                    continue
            return (torch.zeros(8192, 64),
                    torch.zeros(self._cond_tokens, self.cond_dim, dtype=torch.bfloat16))

    def _load_sample(self, sample):
        gt_path = sample["gt_path"]
        if gt_path in self._gt_cache:
            gt = self._gt_cache[gt_path]
        else:
            gt = torch.load(gt_path, weights_only=False, map_location="cpu").float()
            if gt.dim() == 2:
                gt = gt.unsqueeze(0)
            if len(self._gt_cache) >= self._gt_cache_size:
                old_key = self._gt_cache_order.pop(0)
                self._gt_cache.pop(old_key, None)
            self._gt_cache[gt_path] = gt
            self._gt_cache_order.append(gt_path)

        vj = torch.load(sample["jepa_path"], weights_only=False,
                         map_location="cpu").to(torch.bfloat16)
        if vj.dim() == 2:
            vj = vj.unsqueeze(0)
        return gt.squeeze(0), vj.squeeze(0)


def collate_latents(batch):
    gts, vjs = zip(*batch)
    return torch.stack(gts), torch.stack(vjs)


def train(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    is_main = (rank == 0)

    np.random.seed(42 + rank)
    torch.manual_seed(42 + rank)

    with open(args.manifest) as f:
        manifest = json.load(f)
    train_samples = manifest["train"]
    test_samples = manifest["test"]
    categories = manifest.get("categories", [])

    eff_bs = args.batch_size * world_size

    if is_main:
        from collections import Counter
        print(f"\n{'='*70}")
        print(f"V3: MLP-only Projector")
        print(f"  Cond: [{args.cond_dim}] -> MLP -> [{DIT_DIM}]")
        print(f"  GPUs: {world_size}, batch/GPU: {args.batch_size}, eff_bs: {eff_bs}")
        print(f"  Train: {len(train_samples)}, Test: {len(test_samples)}")
        print(f"  Steps: {args.steps}, LR: {args.lr}")
        print(f"  Warmup proj: {args.warmup_proj_steps} steps")
        print(f"{'='*70}")
        for cat, n in Counter(s["category"] for s in train_samples).most_common():
            print(f"  {cat}: {n} train")
        print()

    train_dataset = LatentDataset(train_samples, cond_dim=args.cond_dim,
                                  cond_tokens=args.cond_tokens)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=collate_latents,
        drop_last=True, persistent_workers=True,
    )

    from flow.flow_matching import FlowMatchingScheduler
    from flow.modules.dit import DiT

    dit = DiT(
        hidden_dim=1536, num_heads=16, num_layers=24,
        latent_size=4096, latent_dim=64,
        qknorm=True, qknorm_type="RMSNorm",
        use_pos_embed=False, use_parts=True, part_embed_mode="part2_only",
    ).to(device, dtype=torch.bfloat16)

    scheduler = FlowMatchingScheduler(shift=3.0)
    scheduler.to(device)

    wrapper = TrainWrapper(dit, cond_dim=args.cond_dim).to(device, dtype=torch.bfloat16)

    # Load DiT from checkpoint
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        wrapper.dit.load_state_dict(ckpt["dit"])
        step_loaded = ckpt.get("step", "?")
        del ckpt
        if is_main:
            print(f"[Resume] DiT from {args.resume} (step {step_loaded}), proj fresh")

    dit_params = sum(p.numel() for p in wrapper.dit.parameters())
    proj_params = sum(p.numel() for p in wrapper.proj.parameters())
    if is_main:
        print(f"DiT: {dit_params/1e6:.1f}M | proj: {proj_params/1e6:.2f}M | "
              f"Total: {(dit_params+proj_params)/1e6:.1f}M")

    warmup_proj = args.warmup_proj_steps
    model = DDP(wrapper, device_ids=[rank],
                find_unused_parameters=(warmup_proj > 0))

    if warmup_proj > 0:
        for p in model.module.dit.parameters():
            p.requires_grad = False
        trainable_params = list(model.module.proj.parameters())
        if is_main:
            print(f"  Phase 1: freeze DiT, train proj only for {warmup_proj} steps")
    else:
        trainable_params = list(model.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01)

    model.train()
    os.makedirs(args.output_dir, exist_ok=True)

    use_wandb = is_main and args.wandb_project is not None
    if use_wandb:
        import wandb
        run_name = args.wandb_run_name or f"v3_mlp_{world_size}gpu"
        wandb.init(project=args.wandb_project, name=run_name, config={
            "proj_type": "mlp_only",
            "cond_dim": args.cond_dim,
            "cond_tokens": args.cond_tokens,
            "steps": args.steps, "lr": args.lr,
            "batch_size_per_gpu": args.batch_size,
            "eff_batch_size": eff_bs,
            "num_train": len(train_samples),
            "warmup_proj_steps": warmup_proj,
        })

    losses = []
    pbar = tqdm(range(args.steps), desc="Training", disable=not is_main)
    train_iter = iter(train_loader)

    for step in pbar:
        # Phase transition
        if warmup_proj > 0 and step == warmup_proj:
            if is_main:
                print(f"\n  Phase 2: unfreeze DiT at step {step}")
            for p in model.module.dit.parameters():
                p.requires_grad = True
            trainable_params = list(model.parameters())
            optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.steps - step, eta_min=args.lr * 0.01)
            warmup_proj = 0

        optimizer.zero_grad()

        try:
            gt_latent, jepa_input = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            gt_latent, jepa_input = next(train_iter)

        gt_latent = gt_latent.to(device)
        jepa_input = jepa_input.to(device)

        if np.random.rand() < 0.1:
            jepa_input = torch.zeros_like(jepa_input)

        with torch.no_grad():
            noisy_latent, noise, timesteps = scheduler.add_noise(gt_latent, 1.0, 1.0)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            model_pred = model(
                noisy_latent.to(dtype=torch.bfloat16),
                jepa_input,
                timesteps,
            )
            target = (noise - gt_latent).to(dtype=torch.bfloat16)
            loss = F.mse_loss(model_pred, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        lr_sched.step()

        loss_val = loss.item()
        loss_tensor = torch.tensor([loss_val], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        loss_val = loss_tensor.item()
        losses.append(loss_val)

        if is_main:
            pbar.set_postfix(loss=f"{loss_val:.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if use_wandb:
            import wandb
            wandb.log({"loss": loss_val, "lr": optimizer.param_groups[0]["lr"]}, step=step)

        if is_main and (step + 1) % 100 == 0:
            avg = np.mean(losses[-100:])
            print(f"  Step {step+1}/{args.steps} | avg100: {avg:.6f} | "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e}")
            if use_wandb:
                wandb.log({"avg100_loss": avg}, step=step)

        if is_main and (step + 1) % 1000 == 0:
            save_dict = {
                "dit": model.module.dit.state_dict(),
                "proj": model.module.proj.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_sched": lr_sched.state_dict(),
                "step": step + 1,
                "loss": loss_val,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "latest.pt"))

        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            dist.barrier()
            if is_main:
                save_dict = {
                    "dit": model.module.dit.state_dict(),
                    "proj": model.module.proj.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_sched": lr_sched.state_dict(),
                    "step": step + 1,
                    "loss": loss_val,
                }
                ckpt_path = os.path.join(args.output_dir, f"step_{step+1}.pt")
                torch.save(save_dict, ckpt_path)
                torch.save(save_dict, os.path.join(args.output_dir, "latest.pt"))
                print(f"  Saved: {ckpt_path}")
            dist.barrier()

        del gt_latent, noisy_latent, noise, jepa_input, model_pred, target

    if use_wandb:
        import wandb
        wandb.finish()
    if is_main:
        print("Training complete!")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=110000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_proj_steps", type=int, default=10000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--cond_dim", type=int, default=1408)
    parser.add_argument("--cond_tokens", type=int, default=9600)
    parser.add_argument("--wandb_project", default=None)
    parser.add_argument("--wandb_run_name", default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
