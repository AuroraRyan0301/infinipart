#!/usr/bin/env python3
"""Progressive cross-attn SLat overfit. Unlock one block every 200 steps."""
import os, sys, time, argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--sample", required=True)
parser.add_argument("--cond", choices=["jepa", "dino"], required=True)
parser.add_argument("--steps", type=int, default=8000)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--unlock_interval", type=int, default=200)
parser.add_argument("--start_block", type=int, default=29)
parser.add_argument("--end_block", type=int, default=15)
parser.add_argument("--wandb_name", type=str, default=None)
args = parser.parse_args()

cat = args.sample.split("/")[0]
sample_id = args.sample.split("/")[1]
cond_dim = 1408 if args.cond == "jepa" else 1536
device = torch.device("cuda")
torch.manual_seed(42)

run_name = args.wandb_name or f"prog_{args.cond}_{sample_id}"
wandb.init(project="infinipart", name=run_name, config=vars(args))

def normalize_slat(feats, dev):
    return (feats - SLAT_NORM_MEAN.to(dev)) / SLAT_NORM_STD.to(dev)

def sample_flow_matching(x_0, t):
    noise = torch.randn_like(x_0)
    x_t = (1 - t) * x_0 + (SIGMA_MIN + (1 - SIGMA_MIN) * t) * noise
    velocity = (1 - SIGMA_MIN) * noise - x_0
    return x_t, velocity

# Build model starting from last block only
model = build_dual_part_model(device=device, resolution="512",
                               cross_attn_start_block=args.start_block,
                               vjepa_dim=cond_dim)

# Load GT
gt = torch.load(os.path.join(DATA_ROOT, f"slat_gt/{cat}/{sample_id}.pt"),
                weights_only=False, map_location=device)
p0_feats = normalize_slat(gt['p0_lr']['feats'].to(device), device)
p0_coords = gt['p0_lr']['coords'].to(device)
p1_feats = normalize_slat(gt['p1_lr']['feats'].to(device), device)
p1_coords = gt['p1_lr']['coords'].to(device)

# Load condition
if args.cond == "jepa":
    views_dir = os.path.join(DATA_ROOT, f"encoded/{cat}/{sample_id}/views")
    cond_file = sorted([f for f in os.listdir(views_dir) if f.endswith("_nobg_jepa.pt")])[0]
    cond = torch.load(os.path.join(views_dir, cond_file), weights_only=False, map_location=device)
else:
    views_dir = os.path.join(DATA_ROOT, f"encoded_dino/{cat}/{sample_id}/views")
    cond_file = sorted([f for f in os.listdir(views_dir) if f.endswith("_dino.pt")])[0]
    cond = torch.load(os.path.join(views_dir, cond_file), weights_only=False, map_location=device)
if cond.dim() == 2:
    cond = cond.unsqueeze(0)

print(f"Sample: {args.sample}, p0={p0_feats.shape[0]}, p1={p1_feats.shape[0]}, cond={cond.shape} ({args.cond})")
print(f"Progressive: block {args.start_block}→{args.end_block}, unlock every {args.unlock_interval} steps")

optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.01)
out_dir = os.path.join(DATA_ROOT, f"checkpoints/prog_{args.cond}_{sample_id}")
os.makedirs(out_dir, exist_ok=True)

t0 = time.time()
running_loss = 0
current_block = args.start_block

for step in range(1, args.steps + 1):
    # Progressive unlock
    new_block = max(args.end_block, args.start_block - (step // args.unlock_interval))
    if new_block != current_block:
        current_block = new_block
        model.cross_attn_start_block = current_block
        print(f"  [Step {step}] Unlocked → cross_attn from block {current_block}", flush=True)
        wandb.log({"cross_attn_start_block": current_block}, step=step)

    t_val = torch.rand(1, device=device).clamp(1e-5, 1 - 1e-5)
    t_broadcast = torch.tensor([1000 * t_val.item()], device=device)
    x0_t, vel0 = sample_flow_matching(p0_feats, t_val)
    x1_t, vel1 = sample_flow_matching(p1_feats, t_val)
    x0_st = SparseTensor(feats=x0_t, coords=p0_coords)
    x1_st = SparseTensor(feats=x1_t, coords=p1_coords)

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        pred0, pred1 = model(x0_st, x1_st, t_broadcast, cond)
    loss = (F.mse_loss(pred0.feats.float(), vel0) + F.mse_loss(pred1.feats.float(), vel1)) / 2

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
    optimizer.step()

    running_loss += loss.item()
    wandb.log({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"]}, step=step)

    if step % 50 == 0:
        avg = running_loss / 50
        wandb.log({"avg50_loss": avg}, step=step)
        print(f"  [{step}/{args.steps}] loss={avg:.6f} block={current_block} | {step/(time.time()-t0):.2f} it/s", flush=True)
        running_loss = 0

    if step % 2000 == 0:
        torch.save({
            "vjepa_proj": model.vjepa_proj.state_dict(),
            "part_cross_attns": model.part_cross_attns.state_dict(),
            "cross_attn_start_block": current_block,
            "step": step,
        }, os.path.join(out_dir, f"step_{step}.pt"))

    torch.cuda.empty_cache()

print(f"\nDone in {(time.time()-t0)/60:.1f}m")
wandb.finish()
