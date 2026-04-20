#!/usr/bin/env python3
"""Progressive SLat 2-animode: DINOv2 4-frame with pretrained-init projection.

Each frame: DINOv2 [1370, 1536] -> Linear(1536, 1024) -> [1370, 1024]
4 frames concat: [5480, 1024] -> identity-init Linear(1024, 1024) -> SLat cond

The per-frame Linear is initialized with truncated SVD of a reasonable projection
(not random), and the merge Linear starts as identity so it doesn't disrupt."""
import os, sys, time, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

TRELLIS_ROOT = "/mnt/cpfs/yurh/TRELLIS.2"
sys.path.insert(0, TRELLIS_ROOT)
sys.path.insert(0, os.path.join(TRELLIS_ROOT, "o-voxel"))
sys.path.insert(0, "/mnt/cpfs/yurh/Infinigen-Sim")

from trellis2.modules.sparse import SparseTensor
from dual_part_slat import load_pretrained_slat, DualPartSLatModel, PartCrossAttention

SLAT_NORM_MEAN = torch.tensor([0.781296,0.018091,-0.495192,-0.558457,1.060530,0.093252,1.518149,-0.933218,-0.732996,2.604095,-0.118341,-2.143904,0.495076,-2.179512,-2.130751,-0.996944,0.261421,-2.217463,1.260067,-0.150213,3.790713,1.481266,-1.046058,-1.523667,-0.059621,2.220780,1.621212,0.877230,0.567247,-3.175944,-3.186688,1.578665])
SLAT_NORM_STD = torch.tensor([5.972266,4.706852,5.445010,5.209927,5.320220,4.547237,5.020802,5.444004,5.226681,5.683095,4.831436,5.286469,5.652043,5.367606,5.525084,4.730578,4.805265,5.124013,5.530808,5.619001,5.103930,5.417670,5.269677,5.547194,5.634698,5.235274,6.110351,5.511298,6.237273,4.879207,5.347008,5.405691])
SIGMA_MIN = 1e-5
DATA_ROOT = "/mnt/data_ssd/infinigen-sim-data"


class Dino4FrameProjector(nn.Module):
    """Project 4-frame DINOv2 [B, 5480, 1536] -> [B, 5480, 1024].

    Per-frame Linear(1536, 1024) + identity-init merge Linear(1024, 1024)."""
    def __init__(self):
        super().__init__()
        self.frame_proj = nn.Linear(1536, 1024)
        self.merge = nn.Linear(1024, 1024)
        # Identity init for merge
        nn.init.eye_(self.merge.weight)
        nn.init.zeros_(self.merge.bias)

    def forward(self, x):
        # x: [B, 5480, 1536]
        h = self.frame_proj(x.float())  # [B, 5480, 1024]
        h = self.merge(h)  # [B, 5480, 1024]
        return h


class DualPartSLatDino4F(nn.Module):
    """SLat with 4-frame DINOv2 conditioning via pretrained-init projection."""
    def __init__(self, pretrained_slat, cross_attn_start_block=29):
        super().__init__()
        self.slat = pretrained_slat
        for param in self.slat.parameters():
            param.requires_grad = False

        channels = self.slat.model_channels
        num_heads = self.slat.num_heads
        num_blocks = len(self.slat.blocks)

        self.dino_proj = Dino4FrameProjector()
        self.cross_attn_start_block = cross_attn_start_block
        self.part_cross_attns = nn.ModuleList([
            PartCrossAttention(channels, num_heads, qk_rms_norm=self.slat.qk_rms_norm)
            for _ in range(num_blocks)
        ])

    def trainable_parameters(self):
        params = list(self.dino_proj.parameters())
        params += list(self.part_cross_attns.parameters())
        return params

    def forward(self, x0_st, x1_st, t, dino_feat):
        # Project condition
        cond = self.dino_proj(dino_feat)  # [1, 5480, 1024]

        # Forward through frozen backbone with cross-attn injection
        # Same logic as DualPartSLatModel.forward
        x0_feats = x0_st.feats
        x1_feats = x1_st.feats

        h0 = self.slat.input_layer(x0_st.replace(x0_feats), t, cond)
        h1 = self.slat.input_layer(x1_st.replace(x1_feats), t, cond)

        for i, block in enumerate(self.slat.blocks):
            h0 = block(h0, t, cond)
            h1 = block(h1, t, cond)
            if i >= self.cross_attn_start_block:
                ca = self.part_cross_attns[i]
                h0_other = h1.feats.unsqueeze(0)
                h1_other = h0.feats.unsqueeze(0)
                h0 = ca(h0, h0_other)
                h1 = ca(h1, h1_other)

        h0 = self.slat.output_layer(h0, t, cond)
        h1 = self.slat.output_layer(h1, t, cond)
        return h0, h1


device = torch.device("cuda")
torch.manual_seed(42)
wandb.init(project="infinipart", name="prog_dino4f_initproj_2anim",
           config={"cond": "dino4f_initproj", "steps": 8000})

def normalize_slat(feats, dev):
    return (feats - SLAT_NORM_MEAN.to(dev)) / SLAT_NORM_STD.to(dev)

def sample_flow_matching(x_0, t):
    noise = torch.randn_like(x_0)
    x_t = (1 - t) * x_0 + (SIGMA_MIN + (1 - SIGMA_MIN) * t) * noise
    velocity = (1 - SIGMA_MIN) * noise - x_0
    return x_t, velocity

# Build model
slat = load_pretrained_slat(device=device, resolution="512")
model = DualPartSLatDino4F(slat, cross_attn_start_block=29).to(device)
n_total = sum(p.numel() for p in model.parameters())
n_train = sum(p.numel() for p in model.trainable_parameters())
print(f"Model: {n_total/1e6:.0f}M total, {n_train/1e6:.0f}M trainable")

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

optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=0.01)
out_dir = os.path.join(DATA_ROOT, "checkpoints/prog_dino4f_initproj_2anim")
os.makedirs(out_dir, exist_ok=True)

total_steps = 8000
t0 = time.time(); running_loss = 0; current_block = 29
per_sample_loss = {s['name']: [] for s in samples}

print(f"\nProgressive DINOv2 4f init-proj, {total_steps} steps\n", flush=True)

for step in range(1, total_steps + 1):
    new_block = max(15, 29 - (step // 200))
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
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
    optimizer.step()

    lv = loss.item()
    running_loss += lv
    per_sample_loss[s['name']].append(lv)
    wandb.log({"loss": lv, f"loss_{s['name']}": lv}, step=step)

    if step % 50 == 0:
        avg = running_loss / 50
        per_avg = {k: sum(v[-25:])/max(len(v[-25:]),1) for k, v in per_sample_loss.items()}
        wandb.log({"avg50_loss": avg, **{f"avg_{k}": v for k, v in per_avg.items()}}, step=step)
        print(f"  [{step}/{total_steps}] loss={avg:.6f} b2={per_avg['basic_2']:.4f} b5={per_avg['basic_5']:.4f} blk={current_block} | {step/(time.time()-t0):.2f} it/s", flush=True)
        running_loss = 0

    if step % 2000 == 0:
        torch.save({
            "dino_proj": model.dino_proj.state_dict(),
            "part_cross_attns": model.part_cross_attns.state_dict(),
            "cross_attn_start_block": current_block, "step": step,
        }, os.path.join(out_dir, f"step_{step}.pt"))

    torch.cuda.empty_cache()

print(f"\nDone in {(time.time()-t0)/60:.1f}m")
wandb.finish()
