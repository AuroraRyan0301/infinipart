#!/usr/bin/env python3
"""Single-sample overfit: lamp_0_basic_0 only. Resume from step_21000 ckpt."""
import os, sys, time
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

device = torch.device("cuda:0")
torch.manual_seed(42)

# Load from step_21000
ckpt_path = os.path.join(DATA_ROOT, "checkpoints/slat_overfit_long/step_21000.pt")
ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
cross_attn_start = ckpt["cross_attn_start_block"]

model = build_dual_part_model(device=device, resolution="512", cross_attn_start_block=cross_attn_start)
model.vjepa_proj.load_state_dict(ckpt["vjepa_proj"])
model.part_cross_attns.load_state_dict(ckpt["part_cross_attns"])
print(f"Loaded step_21000, cross_attn_start={cross_attn_start}")

# Load single sample: lamp/0_basic_0
gt_path = os.path.join(DATA_ROOT, "slat_gt/lamp/0_basic_0.pt")
gt = torch.load(gt_path, weights_only=False, map_location=device)
p0_feats = gt['p0_lr']['feats'].to(device)
p0_coords = gt['p0_lr']['coords'].to(device)
p1_feats = gt['p1_lr']['feats'].to(device)
p1_coords = gt['p1_lr']['coords'].to(device)
print(f"Sample: lamp/0_basic_0, p0={p0_feats.shape[0]}, p1={p1_feats.shape[0]} tokens")

# Find jepa
views_dir = os.path.join(DATA_ROOT, "encoded/lamp/0_basic_0/views")
jepa_files = sorted([f for f in os.listdir(views_dir) if f.endswith("_nobg_jepa.pt")])
jepa = torch.load(os.path.join(views_dir, jepa_files[0]), weights_only=False, map_location=device)
if jepa.dim() == 2:
    jepa = jepa.unsqueeze(0)
print(f"JEPA: {jepa.shape}")

gt0_norm = normalize_slat(p0_feats, device)
gt1_norm = normalize_slat(p1_feats, device)

optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=0.01)
out_dir = os.path.join(DATA_ROOT, "checkpoints/slat_overfit_single_lamp")
os.makedirs(out_dir, exist_ok=True)

total_steps = 5000
t0 = time.time()
running_loss = 0

print(f"\nSingle-sample overfit: 0 -> {total_steps}, save every 500\n", flush=True)

for step in range(1, total_steps + 1):
    t_val = torch.rand(1, device=device).clamp(1e-5, 1 - 1e-5)
    t_broadcast = torch.tensor([1000 * t_val.item()], device=device)
    x0_t, vel0 = sample_flow_matching(gt0_norm, t_val)
    x1_t, vel1 = sample_flow_matching(gt1_norm, t_val)
    x0_st = SparseTensor(feats=x0_t, coords=p0_coords)
    x1_st = SparseTensor(feats=x1_t, coords=p1_coords)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        pred0, pred1 = model(x0_st, x1_st, t_broadcast, jepa)
    loss = (F.mse_loss(pred0.feats.float(), vel0) + F.mse_loss(pred1.feats.float(), vel1)) / 2

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
    optimizer.step()

    running_loss += loss.item()

    if step % 50 == 0:
        avg = running_loss / 50
        elapsed = time.time() - t0
        rate = step / elapsed
        print(f"  [{step}/{total_steps}] loss={avg:.6f} | {rate:.2f} it/s", flush=True)
        running_loss = 0

    if step % 500 == 0:
        path = os.path.join(out_dir, f"step_{step}.pt")
        torch.save({
            "vjepa_proj": model.vjepa_proj.state_dict(),
            "part_cross_attns": model.part_cross_attns.state_dict(),
            "cross_attn_start_block": cross_attn_start,
            "step": 21000 + step,
        }, path)
        print(f"  Saved: {path}", flush=True)

    torch.cuda.empty_cache()

print(f"\nDone in {(time.time()-t0)/60:.1f}m")
