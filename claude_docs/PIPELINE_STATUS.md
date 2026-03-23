# Pipeline Status Report — 2026-03-17 (evening update)

## Git Branches

| Branch | Description | Status |
|--------|-------------|--------|
| `new_v3` | Full positive-sample pipeline + inference comparison | **Active** on this machine |
| `nega` | `new_v3` + `render_negative.py` + `render_nega_batch.py` | **Pushed**, to run on another machine |

## Running Processes

| Component | GPU | Conda Env | Status |
|-----------|-----|-----------|--------|
| Training (DDP) | 0,1 | `partpacker_wan` | **PAUSED** at step 22000 (stopped for inference) |
| Generation | 2,3 | `infinigen-sim` | Running, processing PhysXNet ~10100+ |
| Encode watch | 2 | `partpacker_wan` | Running |
| Dashboard | CPU | `infinigen-sim` | Running (port 8501) |

GPU 0,1 currently idle (training paused for inference testing).

## Inference Comparison Results (step 22000)

Compared **our video-conditioned model** (VJEPA2 → DiT) vs **original PartPacker** (single image → DINOv2-giant → DiT).

| Sample | Ours (video) | PP first | PP mid1 | PP mid2 | PP last |
|--------|-------------|----------|---------|---------|---------|
| dishwasher/8/senior_3 | **2.080** | 2.136 | 2.199 | 2.160 | 2.235 |
| lamp/1/senior_0 | **1.782** | 1.844 | 2.177 | 1.990 | 2.017 |
| drawer/0/custom_1_flip | 2.076 | 2.074 | **2.059** | **2.043** | 2.050 |

**Key findings**:
- Our video model beats original PartPacker on **dishwasher** (2.08 vs 2.14-2.23) and **lamp** (1.78 vs 1.84-2.18)
- On **drawer** results are essentially tied (2.08 vs 2.04-2.07)
- Only 22k steps trained — still early; expect improvement with more training + data
- Original PartPacker uses DINOv2-giant (1.1B params) for image encoding; our model uses VJEPA2 video features

**Comparison panels**: `output/infer_vis/compare_{sample}.png`

## Data Volume

### Encoded Training Data (on SSD)

| Metric | Count |
|--------|-------|
| Total animodes | 2,116 |
| Total views | 19,195 |
| Train split | ~15,300 |
| Test split | ~3,900 |

### Encoded Data by Factory

| Factory | Animodes | | Factory | Animodes |
|---------|----------|-|---------|----------|
| PhysXNet_PhysXnet | 952 | | refrigerator | 52 |
| box | 141 | | soap_dispenser | 37 |
| toaster | 123 | | lamp | 31 |
| drawer | 121 | | cabinet | 28 |
| window | 113 | | door_handle | 20 |
| stovetop | 112 | | door | 18 |
| dishwasher | 101 | | trash | 17 |
| microwave | 82 | | PhysXNet (misc) | 13 |
| oven | 70 | | plier | 10 |
| faucet | 59 | | pepper_grinder | 10 |
| | | | PhysXMobility | 6 |

### Generation Pipeline Progress

| Phase | IS Factory | PhysXNet | PhysXMobility |
|-------|-----------|----------|---------------|
| Total objects | 1,800 (18×100) | 32,041 | 2,024 |
| Precomputed | ~1,251 | ~7,611 (3% yield) | ~1,815 |
| Encoded | ~1,164 animodes | ~965 animodes | 6 animodes |

## Training Status

- **Model**: PartPacker Flow DiT (1249.5M params) + VJEPA2 projector
- **Latest checkpoint**: step 22,000
- **Loss**: ~0.49 avg (fluctuating 0.32–0.67)
- **LR**: cosine annealing, ~2.45e-05
- **Batch size**: 16/GPU × 2 GPU = 32 effective
- **Speed**: ~8.9s/step on 2x H200
- **Data**: dynamic refresh every 500 steps
- **Checkpoints**: every 2000 steps (step_2000 through step_22000)
- **Error tolerance**: corrupt `.pt` auto-skipped + logged to `.errors/`

## Negative Sample Branch (nega)

- `render_negative.py`: 6 mutation types (wrong_joint_type, wrong_axis, wrong_direction, over_motion, wrong_parts_moving, jitter)
- `render_nega_batch.py`: multi-GPU batch renderer
- No BVH collision detection (intentional interpenetration)
- No new precompute needed — reuses existing splits
- **NOT YET TESTED** — to run on another machine

## Key File Locations

| Item | Path |
|------|------|
| Training script | `/mnt/cpfs/yurh/PartPacker/train_partnet_vjepa_ddp.py` |
| Inference comparison | `/mnt/cpfs/yurh/Infinigen-Sim/infer_visualize.py` |
| Comparison panels | `/mnt/cpfs/yurh/Infinigen-Sim/output/infer_vis/compare_*.png` |
| Training checkpoints | `/mnt/data_ssd/infinigen-sim/train_output/` |
| Encoded data | `/mnt/data_ssd/infinigen-sim/{category}/{animode_id}/` |
| Error logs | `/mnt/data_ssd/infinigen-sim/.errors/` |
| Precompute output | `/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/` |
| IS spawn output | `/mnt/cpfs/yurh/Infinigen-Sim/sim_exports/urdf/` |
| Gen log | `/mnt/cpfs/yurh/Infinigen-Sim/gen_log.txt` |
| Train log | `/mnt/cpfs/yurh/Infinigen-Sim/train_log_dynamic.txt` |
| Dashboard | `http://localhost:8501` |

## Restart Commands

```bash
# Kill all
pkill -f "train_partnet_vjepa" ; pkill -f "cluster_launch" ; pkill -f "encode_for_training" ; pkill -f "dashboard.py" ; pkill -f "blender.*render_animode"

# Dashboard
tmux new-session -d -s dashboard "bash -c 'source /mnt/data/yurh/miniconda3/etc/profile.d/conda.sh && conda activate infinigen-sim && cd /mnt/cpfs/yurh/Infinigen-Sim && python dashboard.py --port 8501'"

# Training (GPU 0,1) — resume from latest
tmux new-session -d -s train "bash -c 'source /mnt/data/yurh/miniconda3/etc/profile.d/conda.sh && conda activate partpacker_wan && cd /mnt/cpfs/yurh/PartPacker && CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_partnet_vjepa_ddp.py --data_root /mnt/data_ssd/infinigen-sim --output_dir /mnt/data_ssd/infinigen-sim/train_output --dynamic --refresh_interval 500 --steps 30000 --batch_size 16 --lr 1e-4 --save_every 2000 --eval_every 5000 --resume /mnt/data_ssd/infinigen-sim/train_output/latest.pt --error_dir /mnt/data_ssd/infinigen-sim/.errors 2>&1 | tee /mnt/cpfs/yurh/Infinigen-Sim/train_log_dynamic.txt'"

# Generation (GPU 2,3)
tmux new-session -d -s gen "bash -c 'source /mnt/data/yurh/miniconda3/etc/profile.d/conda.sh && conda activate infinigen-sim && cd /mnt/cpfs/yurh/Infinigen-Sim && python -u cluster_launch.py --phase pipeline --gpu_ids 2,3 --is_seeds 100 --views sample --samples 16 2>&1 | tee gen_log.txt'"

# Encode watch (GPU 2)
tmux new-session -d -s encode "bash -c 'source /mnt/data/yurh/miniconda3/etc/profile.d/conda.sh && conda activate partpacker_wan && cd /mnt/cpfs/yurh/Infinigen-Sim && CUDA_VISIBLE_DEVICES=2 python encode_for_training.py --precompute_root /mnt/cpfs/yurh/Infinigen-Sim/precompute_output --output_dir /mnt/data_ssd/infinigen-sim --device cuda:0 --watch --watch_interval 300 2>&1 | tee encode_watch_log.txt'"
```

## Known Issues
1. **PhysXNet 97% waste**: 32,041 objects, only ~3% have movable joints
2. **Training paused**: GPU 0,1 idle — need to restart training
3. **Early training**: only 22k steps, MSE ~2.0 — shapes recognizable but coarse
4. **IS factory gaps**: door_handle 10/100, soap_dispenser 15/100, etc.
5. **render_negative.py untested**: needs smoke test on target machine
