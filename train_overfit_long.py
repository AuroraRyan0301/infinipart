#!/usr/bin/env python3
"""Continue overfit training from checkpoint. Eval lamp at each save."""
import os, sys, random, time
import numpy as np
import torch
import torch.nn.functional as F

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

def normalize_slat(feats, device):
    return (feats - SLAT_NORM_MEAN.to(device)) / SLAT_NORM_STD.to(device)

def sample_flow_matching(x_0, t):
    noise = torch.randn_like(x_0)
    x_t = (1 - t) * x_0 + (SIGMA_MIN + (1 - SIGMA_MIN) * t) * noise
    velocity = (1 - SIGMA_MIN) * noise - x_0
    return x_t, velocity

def discover_samples(slat_gt_root, encoded_root, categories, max_per_cat=3, max_tokens=20000):
    samples = []
    for cat in categories:
        gt_dir = os.path.join(slat_gt_root, cat)
        data_dir = os.path.join(encoded_root, cat)
        if not os.path.isdir(gt_dir) or not os.path.isdir(data_dir):
            continue
        count = 0
        for f in sorted(os.listdir(gt_dir)):
            if not f.endswith('.pt'): continue
            mid = f[:-3]
            views_dir = os.path.join(data_dir, mid, "views")
            if not os.path.isdir(views_dir): continue
            jepa_files = [os.path.join(views_dir, j) for j in os.listdir(views_dir) if j.endswith("_nobg_jepa.pt")]
            if not jepa_files: continue
            gt = torch.load(os.path.join(gt_dir, f), map_location='cpu', weights_only=False)
            p0, p1 = gt['p0_lr'], gt['p1_lr']
            total = p0['feats'].shape[0] + p1['feats'].shape[0]
            if total > max_tokens: continue
            samples.append({
                'id': f'{cat}/{mid}',
                'p0_feats': p0['feats'], 'p0_coords': p0['coords'],
                'p1_feats': p1['feats'], 'p1_coords': p1['coords'],
                'jepa_path': jepa_files[0],
            })
            count += 1
            if count >= max_per_cat: break
    return samples

device = torch.device("cuda:0")
torch.manual_seed(42); random.seed(42)

# Resume
ckpt_path = os.path.join(DATA_ROOT, "checkpoints/slat_overfit_long/latest.pt")
ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
start_step = ckpt["step"]
cross_attn_start = ckpt["cross_attn_start_block"]

model = build_dual_part_model(device=device, resolution="512", cross_attn_start_block=cross_attn_start)
model.vjepa_proj.load_state_dict(ckpt["vjepa_proj"])
model.part_cross_attns.load_state_dict(ckpt["part_cross_attns"])
print(f"Resumed from step {start_step}, cross_attn_start={cross_attn_start}")

categories = "dishwasher,cabinet,lamp,faucet,drawer,PhysXMobility".split(",")
samples = discover_samples(
    os.path.join(DATA_ROOT, "slat_gt"),
    os.path.join(DATA_ROOT, "encoded"),
    categories, max_per_cat=3)
print(f"Samples: {len(samples)}")

# Cache on GPU
for s in samples:
    j = torch.load(s['jepa_path'], weights_only=False, map_location=device)
    if j.dim() == 2: j = j.unsqueeze(0)
    s['jepa'] = j
    s['p0_feats_gpu'] = s['p0_feats'].to(device)
    s['p0_coords_gpu'] = s['p0_coords'].to(device)
    s['p1_feats_gpu'] = s['p1_feats'].to(device)
    s['p1_coords_gpu'] = s['p1_coords'].to(device)

optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=0.01)
out_dir = os.path.join(DATA_ROOT, "checkpoints/slat_overfit_long")
os.makedirs(out_dir, exist_ok=True)

total_steps = 38000
step = start_step
t0 = time.time()
running_loss = 0

print(f"\nTraining: step {start_step} → {total_steps}, save every 3000\n", flush=True)

while step < total_steps:
    for s in samples:
        if step >= total_steps: break
        gt0_norm = normalize_slat(s['p0_feats_gpu'], device)
        gt1_norm = normalize_slat(s['p1_feats_gpu'], device)
        t_val = torch.rand(1, device=device).clamp(1e-5, 1 - 1e-5)
        t_broadcast = torch.tensor([1000 * t_val.item()], device=device)
        x0_t, vel0 = sample_flow_matching(gt0_norm, t_val)
        x1_t, vel1 = sample_flow_matching(gt1_norm, t_val)
        x0_st = SparseTensor(feats=x0_t, coords=s['p0_coords_gpu'])
        x1_st = SparseTensor(feats=x1_t, coords=s['p1_coords_gpu'])

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            pred0, pred1 = model(x0_st, x1_st, t_broadcast, s['jepa'])
        loss = (F.mse_loss(pred0.feats.float(), vel0) + F.mse_loss(pred1.feats.float(), vel1)) / 2

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        step += 1

        if step % 100 == 0:
            avg = running_loss / 100
            elapsed = time.time() - t0
            rate = (step - start_step) / elapsed
            print(f"  [{step}/{total_steps}] loss={avg:.6f} | {rate:.2f} it/s", flush=True)
            running_loss = 0

        if step % 3000 == 0:
            path = os.path.join(out_dir, f"step_{step}.pt")
            torch.save({
                "vjepa_proj": model.vjepa_proj.state_dict(),
                "part_cross_attns": model.part_cross_attns.state_dict(),
                "cross_attn_start_block": cross_attn_start,
                "step": step,
            }, path)
            torch.save({
                "vjepa_proj": model.vjepa_proj.state_dict(),
                "part_cross_attns": model.part_cross_attns.state_dict(),
                "cross_attn_start_block": cross_attn_start,
                "step": step,
            }, os.path.join(out_dir, "latest.pt"))
            print(f"  Saved: {path}", flush=True)

        torch.cuda.empty_cache()

print(f"\nDone in {(time.time()-t0)/60:.1f}m")
