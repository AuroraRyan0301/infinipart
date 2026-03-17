# Pipeline Status Report — 2026-03-17 (updated)

## Git Branches

| Branch | Description | Status |
|--------|-------------|--------|
| `new_v3` | Full positive-sample pipeline (spawn → precompute → render → encode → train) | **Active** on this machine |
| `nega` | `new_v3` + `render_negative.py` for negative sample rendering (6 mutation types) | **Pushed**, to run on another machine |

### nega Branch — Negative Sample Rendering
- **Script**: `render_negative.py` — Blender script, imports from `render_animode.py`
- **6 mutation types**: `wrong_joint_type`, `wrong_axis`, `wrong_direction`, `over_motion`, `wrong_parts_moving`, `jitter`
- **Key difference**: NO BVH collision detection (intentional interpenetration)
- **No new precompute needed** — reuses existing part0/part1 splits, only mutates joint parameters at render time
- **Output**: `{animode_dir}/negatives/{neg_type}/{view}_{suffix}.mp4`
- **Status**: Script written, **NOT YET TESTED** — needs smoke test on target machine

```bash
# Smoke test
CUDA_VISIBLE_DEVICES=0 /path/to/blender --background --python render_negative.py -- \
  --metadata ./precompute_output/lamp/0/metadata.json \
  --animode basic_0 --neg_types wrong_axis --views hemi_05 --bg_mode both
```

## Running Processes (4x tmux sessions)

| Component | GPU | Conda Env | Command | Status |
|-----------|-----|-----------|---------|--------|
| Training (DDP) | 0,1 | `partpacker_wan` | `torchrun --nproc_per_node=2 train_partnet_vjepa_ddp.py` | Step ~20,298, loss ~0.49 |
| Generation | 2,3 | `infinigen-sim` | `cluster_launch.py --phase pipeline --gpu_ids 2,3` | Processing PhysXNet ~10100+ |
| Encode watch | 2 | `partpacker_wan` | `encode_for_training.py --watch --watch_interval 300` | Running |
| Dashboard | CPU | `infinigen-sim` | `dashboard.py --port 8501` | Running |

## Data Volume

### Encoded Training Data (on SSD)

| Source | Animodes | Views | Note |
|--------|----------|-------|------|
| IS factories (18 types) | ~1,164 | ~15,000+ | Main training data |
| PhysXNet | ~718 | ~2,400+ | Growing (gen pipeline active) |
| **Total** | **1,882** | **17,493** | train: 13,627 / test: 3,866 |

### Encoded Data by Factory (gt_latent.pt counts)

| Factory | Count | | Factory | Count |
|---------|-------|-|---------|-------|
| PhysXNet_PhysXnet | 718 | | refrigerator | 52 |
| box | 141 | | soap_dispenser | 37 |
| toaster | 123 | | lamp | 31 |
| drawer | 121 | | cabinet | 28 |
| window | 113 | | door_handle | 20 |
| stovetop | 112 | | door | 18 |
| dishwasher | 101 | | trash | 17 |
| microwave | 82 | | PhysXNet (misc) | 13 |
| oven | 70 | | plier | 10 |
| faucet | 59 | | pepper_grinder | 10 |

### Generation Pipeline Progress

| Phase | IS Factory | PhysXNet | PhysXMobility |
|-------|-----------|----------|---------------|
| Total objects | 1,800 (18×100) | 32,041 | 2,024 |
| Precomputed | ~1,251 | ~7,611 (3% yield) | ~1,815 |
| Encoded | ~1,164 animodes | ~731 animodes | included above |

### IS Factory Gaps
- door_handle: 10/100, soap_dispenser: 15/100, microwave: 19/100, pepper_grinder: 24/100

## Training Status

- **Model**: PartPacker Flow DiT (1249.5M params) + VJEPA2 projector
- **Step**: ~20,298 (resumed from 20,000, target 30,000)
- **Loss (avg)**: ~0.49 (fluctuating 0.32–0.67)
- **LR**: cosine annealing, ~2.45e-05
- **Batch size**: 16/GPU × 2 GPU = 32 effective
- **Speed**: ~8.9s/step on 2x H200
- **Data**: 13,627 train views, dynamic refresh every 500 steps (auto-discovers new encoded data)
- **Error tolerance**: corrupt `.pt` files auto-skipped + logged to `/mnt/data_ssd/infinigen-sim/.errors/`
- **Checkpoints**: every 2000 steps at `/mnt/data_ssd/infinigen-sim/train_output/`

## Key File Locations

| Item | Path |
|------|------|
| Training script | `/mnt/cpfs/yurh/PartPacker/train_partnet_vjepa_ddp.py` |
| Training output | `/mnt/data_ssd/infinigen-sim/train_output/` |
| Encoded data | `/mnt/data_ssd/infinigen-sim/{category}/{animode_id}/` |
| Error logs | `/mnt/data_ssd/infinigen-sim/.errors/` |
| Precompute output | `/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/` |
| IS spawn output | `/mnt/cpfs/yurh/Infinigen-Sim/sim_exports/urdf/` |
| Gen log | `/mnt/cpfs/yurh/Infinigen-Sim/gen_log.txt` |
| Train log | `/mnt/cpfs/yurh/Infinigen-Sim/train_log_dynamic.txt` |
| Dashboard stats | `/mnt/data_ssd/infinigen-sim/.stats/gen.json` |
| Dashboard | `http://localhost:8501` |

## Known Issues

1. **PhysXNet 97% waste**: 32,041 objects, only ~3% have movable joints → mostly skipped
2. **Training data repetition**: 17,493 views from 1,882 animodes → some repetition per epoch
3. **Inference quality**: step 20,000 MSE ~1.7-2.1, shapes recognizable but coarse
4. **IS factory gaps**: some factories partially failed (door_handle 10/100, etc.)
5. **render_negative.py untested**: needs smoke test before batch deployment
