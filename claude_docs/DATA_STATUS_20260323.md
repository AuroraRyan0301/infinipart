# Data Status — 2026-03-23

## Data Path
All data under `/mnt/data_ssd/infinigen-sim-data/`

## Pipeline Stages per Animode

```
part0/1.obj (precompute)
    → mp4 video (render)
        → gt_latent.pt [1,8192,64] (PP VAE encode)
        → *_jepa.pt [10240,1408] (VJEPA2 encode)
            → slat_gt.pt (SLat Encoder, coords+feats at 512 & 1024)
                → vae_coords.pt (PP VAE decode → 32³ occ → coords)
```

## Per-Source Breakdown

### IS Factory (18 types × 100 seeds)

| Category | precompute | gt_latent | jepa | slat_gt | vae_coords |
|----------|-----------|-----------|------|---------|------------|
| box | 1,484 | 141 | 141 | 141 | 48 |
| cabinet | 204 | 28 | 28 | 28 | 0 |
| dishwasher | 877 | 101 | 101 | 101 | 41 |
| drawer | 673 | 121 | 121 | 121 | 121 |
| faucet | 586 | 59 | 59 | 59 | 52 |
| lamp | 405 | 31 | 31 | 31 | 20 |
| oven | 666 | 70 | 70 | 70 | 37 |
| plier | 100 | 10 | 10 | 10 | 5 |
| soap_dispenser | 54 | 37 | 37 | 37 | 13 |
| trash | 84 | 17 | 17 | 17 | 6 |
| window | 740 | 113 | 113 | 113 | 19 |
| door* | — | 18 | 18 | 18 | 7 |
| door_handle* | — | 20 | 20 | 20 | 20 |
| microwave* | — | 82 | 82 | 82 | 43 |
| pepper_grinder* | — | 10 | 10 | 10 | 9 |
| refrigerator* | — | 52 | 52 | 52 | 40 |
| stovetop* | — | 112 | 112 | 112 | 10 |
| toaster* | — | 123 | 123 | 123 | 65 |
| **IS Total** | **5,873** | **1,145** | **1,145** | **1,145** | **556** |

\* 标注的 7 个类别 precompute 数据在搬迁时丢失，需要重新 spawn+precompute+render。encoded/slat_gt 完好。

### PhysXMobility (2,024 objects)

| Stage | Count | Note |
|-------|-------|------|
| precompute | 5,917 | 完成 |
| 有 mp4 视频 | 5,877 | 几乎全部已渲染 |
| gt_latent | 1,012 | **正在 4 卡编码中** (从 6 → 1012, 继续增长) |
| jepa | 6 | 编码脚本先跑 VAE，JEPA 还没开始 |
| slat_gt | 6 | 等 jepa 编完后需要补跑 SLat Encoder |
| vae_coords | 2 | 等 PP VAE 改善后重新生成 |

### PhysXNet (32,041 objects, ~3% 有可动关节)

| Stage | Count | Note |
|-------|-------|------|
| precompute | 23,833 | 完成 |
| 有 mp4 视频 | ~1,218 | 大部分没渲染 |
| gt_latent | 1,154 | 完成 |
| jepa | 1,137 | 完成 |
| slat_gt | 1,154 | 完成 |
| vae_coords | 745 | 65% survive |

## Training Task Data Availability

| 训练任务 | 需要什么 cache | IS | PhysXMob | PhysXNet | 合计 |
|---------|---------------|-----|----------|----------|------|
| **PP VAE Finetune** | part0/1.obj | 5,873 | 5,917 | 23,833 | **35,623** |
| **SLat Flow Model** | slat_gt + jepa | 859 | 6* | 1,006 | **1,871** |

\* PhysXMobility 编码中，预计完成后 SLat Flow 可用量增至 ~3,000+

## Active Processes

| GPU | Task | Status |
|-----|------|--------|
| 0,1,2,3 | encode_for_training.py (4-way parallel) | 正在编码 PhysXMobility，~2.2s/animode/GPU |

## Key Bottlenecks

1. **PP VAE 质量** — 43% 样本 VAE occ coords 有空 part，cabinet 全灭。**端到端推理的核心阻塞。需要 finetune PP VAE。**
2. **PhysXMobility JEPA** — gt_latent 编码中，jepa 还没开始，slat_gt 需要补跑
3. **IS 7 类 precompute 丢失** — door/door_handle/microwave/pepper_grinder/refrigerator/stovetop/toaster 需重新生成
4. **PhysXNet 渲染覆盖率低** — 23K precompute 只渲染了 1.2K
