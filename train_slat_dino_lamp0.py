#!/usr/bin/env python3
"""SLat module + DINOv2 4-frame overfit: table 22367 basic_0 vs basic_5."""
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

DATA_ROOT = "/mnt/data_ssd/infinigen-sim-data"
DINO_ROOT = "/mnt/data_ssd/infinigen-sim-data/encoded_dino/lamp"

device = torch.device("cuda:0")
torch.manual_seed(42)
wandb.init(project="infinipart", name="slat_dino_lamp0_basic0", config={"sample": "lamp/0_basic_0", "cond": "dino4f"})

def normalize_slat(feats, dev):
    return (feats - SLAT_NORM_MEAN.to(dev)) / SLAT_NORM_STD.to(dev)

def sample_flow_matching(x_0, t):
    noise = torch.randn_like(x_0)
    x_t = (1 - t) * x_0 + (SIGMA_MIN + (1 - SIGMA_MIN) * t) * noise
    velocity = (1 - SIGMA_MIN) * noise - x_0
    return x_t, velocity

# Build model with DINOv2 dim (1536 instead of 1408)
model = build_dual_part_model(device=device, resolution="512", cross_attn_start_block=15, vjepa_dim=1536)
print(f"SLat model built (vjepa_dim=1536 for DINOv2)")

# Load 2 samples: basic_0 and basic_5
samples = []
for animode in ["basic_0"]:
    gt = torch.load(os.path.join(DATA_ROOT, f"slat_gt/lamp/0_{animode}.pt"),
                    weights_only=False, map_location=device)
    # Find a DINOv2 feature file
    dino_dir = os.path.join(DINO_ROOT, f"0_{animode}/views")
    dino_files = sorted([f for f in os.listdir(dino_dir) if f.endswith("_dino.pt")])
    dino = torch.load(os.path.join(dino_dir, dino_files[0]), weights_only=False, map_location=device)
    if dino.dim() == 2:
        dino = dino.unsqueeze(0)  # [1, 5480, 1536]

    p0_feats = gt['p0_lr']['feats'].to(device)
    p0_coords = gt['p0_lr']['coords'].to(device)
    p1_feats = gt['p1_lr']['feats'].to(device)
    p1_coords = gt['p1_lr']['coords'].to(device)

    samples.append({
        'name': animode,
        'p0_feats': normalize_slat(p0_feats, device),
        'p0_coords': p0_coords,
        'p1_feats': normalize_slat(p1_feats, device),
        'p1_coords': p1_coords,
        'dino': dino,
    })
    print(f"  {animode}: p0={p0_feats.shape[0]}, p1={p1_feats.shape[0]}, dino={dino.shape}")

optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=0.01)
out_dir = os.path.join(DATA_ROOT, "checkpoints/slat_dino_lamp0")
os.makedirs(out_dir, exist_ok=True)

total_steps = 5000
t0 = time.time()
running_loss = 0

print(f"\nSLat + DINOv2 overfit: basic_0 vs basic_5, {total_steps} steps\n", flush=True)

for step in range(1, total_steps + 1):
    s = random.choice(samples)
    t_val = torch.rand(1, device=device).clamp(1e-5, 1 - 1e-5)
    t_broadcast = torch.tensor([1000 * t_val.item()], device=device)

    x0_t, vel0 = sample_flow_matching(s['p0_feats'], t_val)
    x1_t, vel1 = sample_flow_matching(s['p1_feats'], t_val)
    x0_st = SparseTensor(feats=x0_t, coords=s['p0_coords'])
    x1_st = SparseTensor(feats=x1_t, coords=s['p1_coords'])

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        pred0, pred1 = model(x0_st, x1_st, t_broadcast, s['dino'])
    loss = (F.mse_loss(pred0.feats.float(), vel0) + F.mse_loss(pred1.feats.float(), vel1)) / 2

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
    optimizer.step()
    running_loss += loss.item()
    wandb.log({"loss": loss.item()}, step=step)

    if step % 100 == 0:
        avg = running_loss / 100
        wandb.log({"avg100_loss": avg}, step=step)
        elapsed = time.time() - t0
        rate = step / elapsed
        print(f"  [{step}/{total_steps}] loss={avg:.6f} | {rate:.2f} it/s", flush=True)
        running_loss = 0

    if step % 5000 == 0:
        path = os.path.join(out_dir, f"step_{step}.pt")
        torch.save({
            "vjepa_proj": model.vjepa_proj.state_dict(),
            "part_cross_attns": model.part_cross_attns.state_dict(),
            "cross_attn_start_block": model.cross_attn_start_block,
            "step": step,
            "cond_dim": 1536,
        }, path)
        print(f"  Saved: {path}", flush=True)

    torch.cuda.empty_cache()

print(f"\nDone in {(time.time()-t0)/60:.1f}m")
