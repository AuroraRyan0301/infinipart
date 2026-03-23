#!/usr/bin/env python3
"""
Finetune PartPacker DiT on Infinigen-Sim data.

Adapted from PartPacker/train_partnet_vjepa_ddp.py for our data format:
  - gt_latent: [1, 4096, 64] (2048 per part, from our VAE config)
  - jepa: [8192, 1408] (64-frame VJEPA2, 256x256)
  - DiT: latent_size=2048 (per-part), total 4096

Data layout:
  training_data/{factory}_{seed}_{animode}/
    gt_latent.pt          [1, 4096, 64]
    views/
      v{XX}_nobg_jepa.pt  [8192, 1408]

Usage:
  # Single GPU test
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29501 \
    train_infinipart.py --preload

  # 4 GPU DDP
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=29501 \
    train_infinipart.py --preload
"""

import contextlib
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

PARTPACKER_ROOT = "/mnt/data/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

# Our data dimensions
VJEPA_DIM = 1408        # VJEPA2 feature dim (same)
VJEPA_TOKENS = 8192     # 64 frames @ 256x256 -> 8192 tokens (vs original 10240)
DIT_DIM = 1536          # DiT hidden dim
LATENT_SIZE = 2048      # Per-part latent tokens (our VAE: 2048, original: 4096)
LATENT_DIM = 64         # Latent feature dim
TOTAL_LATENT = LATENT_SIZE * 2  # 4096 total (two parts)

DEFAULT_DATA_ROOT = "/mnt/cpfs/yurh/Infinigen-Sim/training_data"


# ================================================================
# TrainWrapper
# ================================================================

class TrainWrapper(nn.Module):
    def __init__(self, dit, vjepa_dim=VJEPA_DIM, dit_dim=DIT_DIM):
        super().__init__()
        self.dit = dit
        self.proj = nn.Linear(vjepa_dim, dit_dim)

    def forward(self, noisy_latent, vjepa_feat, timesteps):
        cond = self.proj(vjepa_feat)
        return self.dit(noisy_latent, cond, timesteps)


# ================================================================
# Data discovery
# ================================================================

def discover_samples(data_root, test_view_interval=4):
    """Discover all samples from training_data/."""
    train_samples = []
    test_samples = []
    categories = []

    for cat_name in sorted(os.listdir(data_root)):
        cat_dir = os.path.join(data_root, cat_name)
        if not os.path.isdir(cat_dir):
            continue

        gt_path = os.path.join(cat_dir, "gt_latent.pt")
        if not os.path.exists(gt_path):
            continue

        views_dir = os.path.join(cat_dir, "views")
        if not os.path.isdir(views_dir):
            continue

        # Extract category from name: {factory}_{seed}_{animode}
        parts = cat_name.split("_")
        category = parts[0]  # factory name
        if category not in categories:
            categories.append(category)

        import glob as glob_mod
        jepa_files = sorted(glob_mod.glob(
            os.path.join(views_dir, "v*_nobg_jepa.pt")))

        for jepa_path in jepa_files:
            basename = os.path.basename(jepa_path)
            view_str = basename.split("_")[0]  # "vXX"
            try:
                view_idx = int(view_str[1:])
            except ValueError:
                continue

            sample = {
                "id": f"{cat_name}/v{view_idx:02d}",
                "category": category,
                "model_id": cat_name,
                "view_idx": view_idx,
                "gt_path": gt_path,
                "jepa_path": jepa_path,
            }
            if view_idx % test_view_interval == (test_view_interval - 1):
                test_samples.append(sample)
            else:
                train_samples.append(sample)

    categories.sort()
    return train_samples, test_samples, categories


def select_eval_subset(train_samples, test_samples, max_per_split=16):
    def _pick(samples, n):
        if len(samples) <= n:
            return list(samples)
        seen = set()
        picked = []
        for s in samples:
            key = s["model_id"]
            if key not in seen:
                picked.append(s)
                seen.add(key)
                if len(picked) >= n:
                    break
        if len(picked) < n:
            for s in samples:
                if s not in picked:
                    picked.append(s)
                    if len(picked) >= n:
                        break
        return picked
    return _pick(train_samples, max_per_split), _pick(test_samples, max_per_split)


# ================================================================
# Flow matching inference
# ================================================================

def run_flow_inference(dit, cond, device, num_steps=20, cfg_scale=5.0):
    cond_null = torch.zeros_like(cond)
    cond_input = torch.cat([cond, cond_null], dim=0)
    x = torch.randn(1, TOTAL_LATENT, LATENT_DIM, device=device)
    sigmas = np.linspace(1, 0, num_steps + 1)
    sigmas = 3.0 * sigmas / (1 + (3.0 - 1) * sigmas)
    sigmas_pair = [(sigmas[i], sigmas[i + 1]) for i in range(num_steps)]
    with torch.inference_mode():
        for sigma, sigma_prev in sigmas_pair:
            timesteps = torch.tensor(
                [1000 * sigma] * 2, device=device, dtype=torch.float32)
            x_input = torch.cat([x, x], dim=0).to(dtype=torch.bfloat16)
            pred = dit(x_input, cond_input, timesteps).float()
            cond_v, uncond_v = pred.chunk(2, dim=0)
            pred_v = uncond_v + (cond_v - uncond_v) * cfg_scale
            x = x - (sigma - sigma_prev) * pred_v
    return x


def evaluate_latent_mse(wrapper, eval_samples, device, num_steps=20,
                        cfg_scale=5.0, preloaded=None):
    mse_list = []
    for sample in eval_samples:
        try:
            if preloaded and sample["gt_path"] in preloaded["gt"]:
                gt = preloaded["gt"][sample["gt_path"]].to(device).float()
            else:
                gt = torch.load(
                    sample["gt_path"], weights_only=False, map_location=device
                ).float()
            if gt.dim() == 2:
                gt = gt.unsqueeze(0)

            if preloaded and sample["id"] in preloaded["jepa"]:
                vj = preloaded["jepa"][sample["id"]].to(device, dtype=torch.bfloat16)
            else:
                vj = torch.load(
                    sample["jepa_path"], weights_only=False, map_location=device
                ).to(dtype=torch.bfloat16)
            if vj.dim() == 2:
                vj = vj.unsqueeze(0)

            with torch.no_grad():
                cond = wrapper.proj(vj)

            pred = run_flow_inference(wrapper.dit, cond, device, num_steps, cfg_scale)
            mse = F.mse_loss(pred.float(), gt).item()
            mse_list.append(mse)
            del gt, vj, cond, pred
        except Exception as e:
            print(f"  [eval skip] {sample['id']}: {e}")
            continue
    return float(np.mean(mse_list)) if mse_list else -1.0


# ================================================================
# Preloading
# ================================================================

def preload_all(samples, is_main=True):
    gt_cache = {}
    jepa_cache = {}
    if is_main:
        print("Preloading all features into CPU RAM...")
    for s in tqdm(samples, desc="Preload", disable=not is_main):
        if s["gt_path"] not in gt_cache:
            g = torch.load(s["gt_path"], weights_only=False, map_location="cpu").float()
            if g.dim() == 2:
                g = g.unsqueeze(0)
            gt_cache[s["gt_path"]] = g
        vj = torch.load(s["jepa_path"], weights_only=False, map_location="cpu").to(torch.bfloat16)
        if vj.dim() == 2:
            vj = vj.unsqueeze(0)
        jepa_cache[s["id"]] = vj
    if is_main:
        gt_mb = sum(v.numel() * v.element_size() for v in gt_cache.values()) / 1e6
        vj_mb = sum(v.numel() * v.element_size() for v in jepa_cache.values()) / 1e6
        print(f"  Cached: {len(gt_cache)} GT ({gt_mb:.0f}MB), "
              f"{len(jepa_cache)} JEPA ({vj_mb:.0f}MB)")
    return {"gt": gt_cache, "jepa": jepa_cache}


# ================================================================
# Training
# ================================================================

def train(args):
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    is_main = (rank == 0)

    np.random.seed(42 + rank)
    torch.manual_seed(42 + rank)

    train_samples, test_samples, categories = discover_samples(
        args.data_root, test_view_interval=args.test_view_interval)

    eff_bs = args.batch_size * args.grad_accum * world_size

    if is_main:
        print(f"\n{'='*70}")
        print(f"Infinigen-Sim DiT Training: VJEPA2 Only")
        print(f"  Latent: [{TOTAL_LATENT}, {LATENT_DIM}] = 2 x [{LATENT_SIZE}, {LATENT_DIM}]")
        print(f"  VJEPA2: [*, {VJEPA_DIM}] (variable token count)")
        print(f"  Categories: {categories}")
        print(f"  GPUs: {world_size}, batch_size/GPU: {args.batch_size}")
        print(f"  Grad accum: {args.grad_accum}")
        print(f"  Effective batch size: {eff_bs}")
        print(f"  Training samples: {len(train_samples)}")
        print(f"  Test samples: {len(test_samples)}")
        print(f"  Steps: {args.steps}, LR: {args.lr}")
        print(f"  Init from pretrained: {args.init_from_pretrained}")
        print(f"{'='*70}\n")

        from collections import Counter
        train_cats = Counter(s["category"] for s in train_samples)
        test_cats = Counter(s["category"] for s in test_samples)
        for cat in categories:
            print(f"  {cat}: {train_cats.get(cat, 0)} train, "
                  f"{test_cats.get(cat, 0)} test views")
        print()

    if len(train_samples) == 0:
        if is_main:
            print("ERROR: No training samples found.")
        dist.destroy_process_group()
        return

    eval_train, eval_test = select_eval_subset(train_samples, test_samples)
    if is_main:
        print(f"  Eval subset: {len(eval_train)} train, {len(eval_test)} test")

    preloaded = None
    if args.preload:
        all_samples = train_samples + test_samples
        preloaded = preload_all(all_samples, is_main)

    # Build model — adapted for our latent size
    from flow.flow_matching import FlowMatchingScheduler
    from flow.modules.dit import DiT

    dit = DiT(
        hidden_dim=DIT_DIM, num_heads=16, num_layers=24,
        latent_size=LATENT_SIZE, latent_dim=LATENT_DIM,
        qknorm=True, qknorm_type="RMSNorm",
        use_pos_embed=False, use_parts=True, part_embed_mode="part2_only",
    ).to(device, dtype=torch.bfloat16)

    scheduler = FlowMatchingScheduler(shift=3.0)
    scheduler.to(device)

    # Load pretrained (partial — latent_size mismatch expected)
    if args.init_from_pretrained:
        flow_ckpt_path = os.path.join(PARTPACKER_ROOT, "pretrained", "flow.pt")
        if os.path.exists(flow_ckpt_path):
            ckpt = torch.load(flow_ckpt_path, weights_only=False, map_location=device)
            if "model" in ckpt:
                ckpt = ckpt["model"]
            dit_state = {
                k.replace("dit.", ""): v
                for k, v in ckpt.items() if k.startswith("dit.")
            }
            param_dict = dict(dit.named_parameters())
            loaded, skipped = 0, 0
            for k, v in dit_state.items():
                if k in param_dict and param_dict[k].shape == v.shape:
                    param_dict[k].data.copy_(v)
                    loaded += 1
                else:
                    skipped += 1
            del ckpt, dit_state
            if is_main:
                print(f"[Init] Pretrained DiT: loaded {loaded}, "
                      f"skipped {skipped} (shape mismatch expected for latent_size change)")

    dit_params = sum(p.numel() for p in dit.parameters())
    proj_params = VJEPA_DIM * DIT_DIM + DIT_DIM
    if is_main:
        print(f"DiT: {dit_params/1e6:.1f}M | proj: {proj_params} "
              f"| Total: {(dit_params+proj_params)/1e6:.1f}M")

    wrapper = TrainWrapper(dit).to(device, dtype=torch.bfloat16)
    model = DDP(wrapper, device_ids=[rank])

    trainable_params = list(model.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01)

    # Resume
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.module.dit.load_state_dict(ckpt["dit"])
        if "proj" in ckpt:
            model.module.proj.load_state_dict(ckpt["proj"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "lr_sched" in ckpt:
            lr_sched.load_state_dict(ckpt["lr_sched"])
        start_step = ckpt.get("step", 0)
        del ckpt
        if is_main:
            print(f"[Resume] Step {start_step}, lr={lr_sched.get_last_lr()[0]:.2e}")

    model.train()
    os.makedirs(args.output_dir, exist_ok=True)

    # wandb
    use_wandb = is_main and args.wandb_project is not None
    if use_wandb:
        import wandb
        run_name = args.wandb_run_name or f"infinipart_ddp{world_size}_{len(train_samples)}s"
        wandb.init(project=args.wandb_project, name=run_name, config={
            "conditioning": "vjepa2_only",
            "latent_size_per_part": LATENT_SIZE,
            "total_latent": TOTAL_LATENT,
            "vjepa_dim": VJEPA_DIM,
            "categories": categories,
            "steps": args.steps, "lr": args.lr,
            "batch_size_per_gpu": args.batch_size,
            "eff_batch_size": eff_bs,
            "num_train": len(train_samples),
            "num_test": len(test_samples),
            "dit_params_M": dit_params / 1e6,
        })

    # Training loop
    losses = []
    bs = args.batch_size
    ga = args.grad_accum
    pbar = tqdm(range(start_step, args.steps), desc="Training", disable=not is_main)

    for step in pbar:
        optimizer.zero_grad()
        accum_loss = 0.0

        for _accum in range(ga):
            is_last = (_accum == ga - 1)
            sync_ctx = contextlib.nullcontext() if is_last else model.no_sync()

            with sync_ctx:
                gt_list, jepa_list = [], []
                idxs = np.random.randint(len(train_samples), size=bs)
                for idx in idxs:
                    sample = train_samples[idx]
                    try:
                        if preloaded and sample["gt_path"] in preloaded["gt"]:
                            g = preloaded["gt"][sample["gt_path"]].to(device)
                        else:
                            g = torch.load(
                                sample["gt_path"], weights_only=False
                            ).to(device).float()
                        if g.dim() == 2:
                            g = g.unsqueeze(0)

                        if preloaded and sample["id"] in preloaded["jepa"]:
                            vj = preloaded["jepa"][sample["id"]].to(device, dtype=torch.bfloat16)
                        else:
                            vj = torch.load(
                                sample["jepa_path"], weights_only=False
                            ).to(device, dtype=torch.bfloat16)
                        if vj.dim() == 2:
                            vj = vj.unsqueeze(0)
                    except Exception:
                        continue

                    gt_list.append(g)
                    jepa_list.append(vj)

                if len(gt_list) == 0:
                    continue

                gt_latent = torch.cat(gt_list, dim=0)
                jepa_input = torch.cat(jepa_list, dim=0)
                del gt_list, jepa_list

                # CFG dropout (10%)
                if np.random.rand() < 0.1:
                    jepa_input = torch.zeros_like(jepa_input)

                with torch.no_grad():
                    noisy_latent, noise, timesteps = scheduler.add_noise(
                        gt_latent, 1.0, 1.0)

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    model_pred = model(
                        noisy_latent.to(dtype=torch.bfloat16),
                        jepa_input,
                        timesteps,
                    )
                    del jepa_input
                    target = (noise - gt_latent).to(dtype=torch.bfloat16)
                    loss = F.mse_loss(model_pred, target) / ga
                loss.backward()
                accum_loss += loss.item() * ga
                del gt_latent, noisy_latent, noise, model_pred, target

        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()
        lr_sched.step()

        loss_val = accum_loss / ga
        loss_tensor = torch.tensor([loss_val], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        loss_val = loss_tensor.item()
        losses.append(loss_val)

        if is_main:
            pbar.set_postfix(loss=f"{loss_val:.4f}",
                             lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if use_wandb:
            wandb.log({"loss": loss_val,
                       "lr": optimizer.param_groups[0]["lr"]}, step=step)

        if is_main and (step + 1) % 100 == 0:
            avg = np.mean(losses[-100:])
            print(f"  Step {step+1}/{args.steps} | avg100: {avg:.6f} | "
                  f"lr: {optimizer.param_groups[0]['lr']:.2e}")
            if use_wandb:
                wandb.log({"avg100_loss": avg}, step=step)

        # Eval
        if args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            dist.barrier()
            if is_main:
                model.eval()
                train_mse = evaluate_latent_mse(
                    model.module, eval_train, device,
                    num_steps=args.eval_steps, preloaded=preloaded)
                test_mse = evaluate_latent_mse(
                    model.module, eval_test, device,
                    num_steps=args.eval_steps, preloaded=preloaded)
                print(f"  [Eval@{step+1}] train_mse={train_mse:.4f} | "
                      f"test_mse={test_mse:.4f}")
                if use_wandb:
                    wandb.log({"eval/train_mse": train_mse,
                               "eval/test_mse": test_mse}, step=step)
                model.train()
                torch.cuda.empty_cache()
            dist.barrier()

        # Save
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
                    "categories": categories,
                    "num_train": len(train_samples),
                    "num_test": len(test_samples),
                    "latent_size": LATENT_SIZE,
                    "total_latent": TOTAL_LATENT,
                }
                ckpt_path = os.path.join(args.output_dir, f"step_{step+1}.pt")
                torch.save(save_dict, ckpt_path)
                torch.save(save_dict, os.path.join(args.output_dir, "latest.pt"))
                print(f"  Saved: {ckpt_path}")
            dist.barrier()

    if use_wandb:
        wandb.finish()
    if is_main:
        final_avg = np.mean(losses[-100:]) if losses else 0
        print(f"\nTraining complete. Final avg100: {final_avg:.6f}")
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(
        description="Infinigen-Sim PartPacker DiT Training (DDP)")
    parser.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=str,
                        default="/mnt/cpfs/yurh/Infinigen-Sim/output/infinipart_train")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--eval_every", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=20)
    parser.add_argument("--test_view_interval", type=int, default=4)
    parser.add_argument("--init_from_pretrained", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
