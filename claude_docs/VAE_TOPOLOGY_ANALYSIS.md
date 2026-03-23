# VAE Topology Bottleneck Analysis

## Summary

The PartPacker VAE (340M params) preserves geometry well but **destroys topology** — specifically, it merges spatially separated connected components into single blobs. This is the primary bottleneck for our dual-volume topology prediction pipeline.

## Quantitative Results (200 samples)

### VAE Encode-Decode Roundtrip

|              | Chamfer Dist | OBJ CC (true) | VAE CC (decoded) | CC Error | CC Exact Match |
|---|---|---|---|---|---|
| **Part0**    | 0.049        | 8.34           | 1.81             | 6.74     | 26.7%          |
| **Part1**    | 0.040        | 5.37           | 2.35             | 3.92     | 26.1%          |
| **Both**     |              |                |                  |          | **12.8%**      |

- Chamfer Distance is small (good geometry reconstruction)
- CC Error is massive: VAE loses 6.7 + 3.9 = **10.6 components per sample** on average
- Only **12.8%** of samples have both parts' CC exactly preserved

### Impact on End-to-End Pipeline (Pred vs OBJ vs GT VAE)

|                     | Ours (video 22k) | Original PartPacker |
|---|---|---|
| **Pred vs OBJ**     | 9.2% exact        | 12.3% exact         |
| **Pred vs GT VAE**  | 27.7% exact       | 22.1% exact         |
| **VAE vs OBJ**      | 13.8% exact       | 13.8% exact         |

- VAE roundtrip itself only gets 13.8% topo exact — this is the **upper bound** for any model using this VAE
- Models (9-12%) perform close to the VAE ceiling (13.8%), meaning **VAE is the bottleneck, not the DiT**

### Per-Category Breakdown

| Category             | N   | OBJ CC (p0/p1) | VAE CC (p0/p1) | CC Error | CD (p0/p1)    |
|---|---|---|---|---|---|
| PhysXNet             | 115 | 5.5 / 4.1      | 1.7 / 2.4      | 7.08     | 0.030 / 0.036 |
| PhysXMobility        | 27  | 9.0 / 6.8      | 1.7 / 2.1      | 12.52    | 0.070 / 0.027 |
| box                  | 1   | 31.0 / 38.0    | 2.0 / 2.0      | 65.00    | 0.164 / 0.413 |
| cabinet              | 1   | 27.0 / 12.0    | 1.0 / 9.0      | 29.00    | 0.057 / 0.130 |
| dishwasher           | 2   | 23.0 / 3.5     | 2.5 / 2.5      | 21.50    | 0.274 / 0.084 |
| drawer               | 8   | 11.6 / 4.2     | 4.4 / 2.1      | 9.38     | 0.079 / 0.050 |
| faucet               | 4   | 2.5 / 2.5      | 1.5 / 1.5      | 2.00     | 0.047 / 0.097 |
| oven                 | 4   | 32.2 / 15.8    | 2.0 / 4.5      | 43.50    | 0.175 / 0.145 |
| stovetop             | 5   | 28.6 / 12.4    | 1.0 / 2.6      | 37.40    | 0.051 / 0.012 |
| toaster              | 8   | 6.6 / 2.5      | 1.1 / 1.5      | 6.50     | 0.082 / 0.002 |
| window               | 1   | 42.0 / 34.0    | 1.0 / 1.0      | 74.00    | 0.019 / 0.008 |

IS factory objects (box, cabinet, oven, stovetop, window) have extremely high OBJ CC (20-40+) because procedural geometry generates many isolated mesh fragments (screws, hinges, decorations).

## Qualitative Observations

See `output/vae_compare/*_final.png` for Blender-rendered comparisons (blue=part0, orange=part1).

Key patterns:
1. **Small isolated parts vanish**: screws, hinges, knobs, decorations → all absorbed into nearest large body
2. **Internal structure lost**: shelf dividers, oven racks, drawer separators → merged into outer shell
3. **Overall silhouette preserved**: the object is still recognizable, just "smoothed out" topologically
4. **Low-CC objects fare well**: faucet (CC 5→3), toaster (CC 13→2.6) retain most structure

## Root Cause

The VAE uses a point cloud encoder (32K uniform + 16K salient points → FPS → perceiver) that inherently blurs spatial separation. The decoder (FlexiCubes at resolution 64) can only represent ~64^3 = 262K voxels, which cannot faithfully reconstruct many small separated components.

## Implications for Training

1. **Topo loss against OBJ CC is futile** — the VAE cannot represent the true topology anyway
2. **Topo loss against GT VAE CC is viable** — optimize Pred CC to match GT VAE CC (the best the VAE can do)
3. **VAE finetuning** could improve topology preservation, especially if we add a CC-aware loss term
4. **Resolution increase** (64→128) would help but at significant compute cost

## Next Steps

- [ ] Finetune VAE on our articulated object data with topology-aware loss
- [ ] Evaluate higher VAE decode resolution (128)
- [ ] Consider separate VAE per-part with explicit CC preservation
