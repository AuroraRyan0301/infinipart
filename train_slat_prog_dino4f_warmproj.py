#!/usr/bin/env python3
"""Progressive SLat 2-animode: DINOv2 4-frame.
Phase 1 (step 1-1000): Only train projector, freeze everything else.
Phase 2 (step 1001+): Unfreeze cross-attn, progressive unlock."""
import os, sys, time, random
import torch
import torch.nn.functional as F
import wandb

TRELLIS_ROOT = "/mnt/cpfs/yurh/TRELLIS.2"
sys.path.insert(0, TRELLIS_ROOT)
sys.path.insert(0, os.path.join(TRELLIS_ROOT, "o-voxel"))
sys.path.insert(0, "/mnt/cpfs/yurh/Infinigen-Sim")

from trellis2.modules.sparse import SparseTensor
from dual_part_slat import build_dual_part_model

SLAT_NORM_MEAN = torch.tensor([0.781296,0.018091,-0.495192,-0.558457,1.060530,0.093252,1.518149,-0.933218,-0.732996,2.604095,-0.118341,-2.143904,0.495076,-2.179512,-2.130751,-0.996944,0.261421,-2.217463,1.260067,-0.150213,3.790713,1.481266,-1.046058,-1.523667,-0.059621,2.220780,1.621212,0.877230,0.567247,-3.175944,-3.186688,1.578665])
SLAT_NORM_STD = torch.tensor([5.972266,4.706852,5.445010,5.209927,5.320220,4.547237,5.020802,5.444004,5.226681,5.683095,4.831436,5.286469,5.652043,5.367606,5.525084,4.730578,4.805265,5.124013,5.530808,5.619001,5.103930,5.417670,5.269677,5.547194,5.634698,5.235274,6.110351,5.511298,6.237273,4.879207,5.347008,5.405691])
SIGMA_MIN = 1e-5
DATA_ROOT = "/mnt/data_ssd/infinigen-sim-data"
WARMUP_STEPS = 1000

device = torch.device("cuda")
torch.manual_seed(42)
wandb.init(project="infinipart", name="prog_dino4f_warmproj_2anim",
           config={"cond": "dino4f", "warmup_steps": WARMUP_STEPS, "steps": 8000})

def normalize_slat(feats, dev):
    return (feats - SLAT_NORM_MEAN.to(dev)) / SLAT_NORM_STD.to(dev)

def sample_flow_matching(x_0, t):
    noise = torch.randn_like(x_0)
    x_t = (1 - t) * x_0 + (SIGMA_MIN + (1 - SIGMA_MIN) * t) * noise
    velocity = (1 - SIGMA_MIN) * noise - x_0
    return x_t, velocity

# Build model with cross_attn disabled initially (start_block=999 = no cross-attn active)
model = build_dual_part_model(device=device, resolution="512",
                               cross_attn_start_block=999, vjepa_dim=1536)

# Phase 1: freeze cross-attn, only train projector
for p in model.part_cross_attns.parameters():
    p.requires_grad = False

proj_params = list(model.vjepa_proj.parameters())
optimizer = torch.optim.AdamW(proj_params, lr=1e-4, weight_decay=0.01)

# Load data
samples = []
for animode in ["basic_2", "basic_5"]:
    gt = torch.load(os.path.join(DATA_ROOT, f"slat_gt/PhysXMobility/22367_{animode}.pt"),
                    weights_only=False, map_location=device)
    vdir = os.path.join(DATA_ROOT, f"encoded_dino/PhysXMobility/22367_{animode}/views")
    cf = sorted([f for f in os.listdir(vdir) if f.endswith("_dino.pt")])[0]
    cond = torch.load(os.path.join(vdir, cf), weights_only=False, map_location=device)
    if cond.dim() == 2: cond = cond.unsqueeze(0)
    samples.append({
        'name': animode,
        'p0_feats': normalize_slat(gt['p0_lr']['feats'].to(device), device),
        'p0_coords': gt['p0_lr']['coords'].to(device),
        'p1_feats': normalize_slat(gt['p1_lr']['feats'].to(device), device),
        'p1_coords': gt['p1_lr']['coords'].to(device),
        'cond': cond,
    })
    print(f"  {animode}: p0={gt['p0_lr']['feats'].shape[0]}, p1={gt['p1_lr']['feats'].shape[0]}, cond={cond.shape}")

out_dir = os.path.join(DATA_ROOT, "checkpoints/prog_dino4f_warmproj_2anim")
os.makedirs(out_dir, exist_ok=True)

total_steps = 8000
t0 = time.time(); running_loss = 0; current_block = 999
per_sample_loss = {s['name']: [] for s in samples}
phase = 1

print(f"\nPhase 1: Projector warmup ({WARMUP_STEPS} steps, cross-attn frozen)\n", flush=True)

for step in range(1, total_steps + 1):
    # Phase transition
    if step == WARMUP_STEPS + 1 and phase == 1:
        phase = 2
        print(f"\n  [Step {step}] Phase 2: Unfreeze cross-attn, start progressive\n", flush=True)
        # Unfreeze cross-attn
        for p in model.part_cross_attns.parameters():
            p.requires_grad = True
        # New optimizer with all trainable params
        all_params = list(model.vjepa_proj.parameters()) + list(model.part_cross_attns.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0.01)
        current_block = 29
        model.cross_attn_start_block = current_block
        wandb.log({"phase": 2, "cross_attn_start_block": current_block}, step=step)

    # Progressive unlock (only in phase 2)
    if phase == 2:
        steps_in_phase2 = step - WARMUP_STEPS
        new_block = max(15, 29 - (steps_in_phase2 // 200))
        if new_block != current_block:
            current_block = new_block
            model.cross_attn_start_block = current_block
            wandb.log({"cross_attn_start_block": current_block}, step=step)

    s = random.choice(samples)
    t_val = torch.rand(1, device=device).clamp(1e-5, 1 - 1e-5)
    t_broadcast = torch.tensor([1000 * t_val.item()], device=device)
    x0_t, vel0 = sample_flow_matching(s['p0_feats'], t_val)
    x1_t, vel1 = sample_flow_matching(s['p1_feats'], t_val)
    x0_st = SparseTensor(feats=x0_t, coords=s['p0_coords'])
    x1_st = SparseTensor(feats=x1_t, coords=s['p1_coords'])

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        pred0, pred1 = model(x0_st, x1_st, t_broadcast, s['cond'])
    loss = (F.mse_loss(pred0.feats.float(), vel0) + F.mse_loss(pred1.feats.float(), vel1)) / 2

    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
    optimizer.step()

    lv = loss.item()
    running_loss += lv
    per_sample_loss[s['name']].append(lv)
    wandb.log({"loss": lv, f"loss_{s['name']}": lv, "phase": phase}, step=step)

    if step % 50 == 0:
        avg = running_loss / 50
        per_avg = {k: sum(v[-25:])/max(len(v[-25:]),1) for k, v in per_sample_loss.items()}
        wandb.log({"avg50_loss": avg, **{f"avg_{k}": v for k, v in per_avg.items()}}, step=step)
        blk_str = f"blk={current_block}" if phase == 2 else "proj_only"
        print(f"  [{step}/{total_steps}] loss={avg:.6f} b2={per_avg['basic_2']:.4f} b5={per_avg['basic_5']:.4f} {blk_str} | {step/(time.time()-t0):.2f} it/s", flush=True)
        running_loss = 0

    if step % 2000 == 0:
        torch.save({
            "vjepa_proj": model.vjepa_proj.state_dict(),
            "part_cross_attns": model.part_cross_attns.state_dict(),
            "cross_attn_start_block": current_block, "step": step, "phase": phase,
        }, os.path.join(out_dir, f"step_{step}.pt"))

    torch.cuda.empty_cache()

print(f"\nDone in {(time.time()-t0)/60:.1f}m")
wandb.finish()
