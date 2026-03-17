# Pipeline Status Report — 2024-03-16

## Running Processes

| Component | GPU | Command | Status |
|-----------|-----|---------|--------|
| Training (DDP) | 0, 1 | `torchrun --nproc_per_node=2 train_partnet_vjepa_ddp.py --steps 999999999 --resume latest.pt --dynamic` | Step 20,700, loss ~0.50 |
| Generation | 2, 3 | `cluster_launch.py --phase pipeline --gpu_ids 2,3 --is_seeds 100 --views sample --samples 16` | Running, processing PhysXNet |
| Encode watch | 2 | `encode_for_training.py --device cuda:2 --watch --watch_interval 300` | Running |
| Dashboard | CPU | `dashboard.py --port 8501` | Running (event-driven, ~1ms response) |

## Data Volume

### Encoded (on SSD, ready for training)

| Source | Animodes | Views | Note |
|--------|----------|-------|------|
| IS factories (18 types) | ~1,100 | ~10,000+ | Main training data |
| PhysXNet | 16 | ~130 | Just started |
| PhysXMobility | 6 | ~50 | Barely started |
| **Total** | **~1,155** | **~11,046** | train split ~8,700 views |

### Generation Pipeline Progress

| Phase | IS Factory | PhysXNet | PhysXMobility |
|-------|-----------|----------|---------------|
| Total objects | 1,800 (18×100) | 32,041 | 2,024 |
| Spawned | **~200 (11%)** | N/A (pre-existing) | N/A |
| Has movable joints | ~200 | **948 (3%)** | ~unknown |
| Precomputed | ~200 | 948 | 1 |
| Rendered (has videos) | ~18 | ~11 | ~0 |
| Encoded | ~1,100 animodes | 16 animodes | 6 animodes |

### IS Factory Spawn Breakdown

| Factory | Spawned/Target | Precomputed | Encoded |
|---------|---------------|-------------|---------|
| dishwasher | 100/100 | 99 | 92 |
| lamp | 33/100 | 23 | 31 |
| box | 10/100 | 10 | 141* |
| cabinet | 10/100 | 10 | 28 |
| door | 10/100 | 10 | 18 |
| door_handle | 10/100 | 10 | 20 |
| drawer | 10/100 | 10 | 121* |
| faucet | 10/100 | 10 | 59 |
| microwave | 10/100 | 10 | 82 |
| oven | 10/100 | 10 | 70 |
| pepper_grinder | 10/100 | 10 | 10 |
| plier | 10/100 | 10 | 10 |
| refrigerator | 10/100 | 10 | 52 |
| soap_dispenser | 10/100 | 10 | 37 |
| stovetop | 10/100 | 10 | 112* |
| toaster | 10/100 | 10 | 123* |
| trash | 10/100 | 9 | 17 |
| window | 10/100 | 10 | 113* |

\* Encoded animode count > precomputed object count because each object has multiple animodes.

## Training Status

- **Model**: PartPacker Flow DiT (1249.5M params)
- **Step**: 20,700 (resumed from 20,000)
- **Loss (avg100)**: ~0.50, not yet showing clear downward trend
- **LR**: 2.27e-05 (cosine schedule)
- **Batch size**: 16 per GPU × 2 GPU = 32 samples/step
- **Speed**: ~9s/step
- **Data**: 8,700 train views, dynamic refresh every 5000 steps
- **Throughput**: ~13,000 sample uses/hr → each view reused ~1.5×/hr (high repetition)

## Known Issues

### 1. IS Factory Spawn Bottleneck (Critical)
- 16 of 18 factories only have 10 seeds spawned (target: 100)
- **1,600 seeds missing** — the largest gap in the pipeline
- IS is the highest-quality data source (full PBR materials)
- Spawn requires Blender + GPU, but GPUs are occupied by rendering
- Pipeline processes PhysX before IS, so IS spawn hasn't been reached

### 2. PhysXNet 97% Waste Rate
- 32,041 objects → only 948 (3%) have movable joints
- Pipeline spends time on setup + precompute for 31,000+ useless objects
- No pre-filtering — every object goes through full setup_physxnet_scene.py + split_precompute.py
- Fix: scan URDFs for revolute/prismatic joints before running setup

### 3. Render Throughput
- 2 GPUs rendering, each object takes 5-30 min depending on animode count
- 948 PhysXNet objects × ~15 min avg = ~120 GPU-hours for PhysXNet alone
- IS factory rendering also incomplete (dishwasher 12/99, lamp 6/23)
- H200 has no RT cores — ray tracing is slower than consumer GPUs

### 4. PhysXMobility Not Started
- 2,024 objects available, only 1 processed
- Queued after PhysXNet in pipeline, hasn't been reached yet

### 5. Training Data Repetition
- 8,700 views with 32 samples/step × ~400 steps/hr = ~13,000/hr
- Each view seen ~1.5 times per hour → potential overfitting risk
- Need more data volume to improve generalization

## Recommended Actions

1. **Reorder pipeline**: IS Factory first → PhysXMobility → PhysXNet (descending quality)
2. **Pre-filter PhysXNet**: Parse URDFs for joint types before setup, skip objects with no revolute/prismatic joints
3. **Parallelize IS spawn**: Use idle GPU time (GPU3 often at 0%) for IS factory spawning
4. **Reduce render load**: Consider fewer views per animode (4 instead of 8) for initial data ramp-up
5. **Monitor loss curve**: If loss plateaus after more data, investigate model/data quality issues

## File Locations

| Item | Path |
|------|------|
| Training output | `/mnt/data_ssd/infinigen-sim/train_output/` |
| Encoded data | `/mnt/data_ssd/infinigen-sim/{category}/{animode_id}/` |
| Precompute output | `/mnt/cpfs/yurh/Infinigen-Sim/precompute_output/` |
| IS spawn output | `/mnt/cpfs/yurh/Infinigen-Sim/sim_exports/urdf/` |
| Gen log | `/mnt/cpfs/yurh/Infinigen-Sim/gen_log.txt` |
| Train log | `/mnt/cpfs/yurh/Infinigen-Sim/train_log_dynamic.txt` |
| Dashboard stats | `/mnt/data_ssd/infinigen-sim/.stats/gen.json` |
| Dashboard | `http://localhost:8501` |
