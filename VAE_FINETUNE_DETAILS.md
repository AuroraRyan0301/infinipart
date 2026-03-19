# VAE Finetune Technical Details

## PartPacker VAE Architecture

### Overview
PartPacker VAE (340M params) is a Neural Implicit Function — it learns to predict whether any point in 3D space is inside or outside a mesh. It does NOT predict point positions or use Chamfer Distance.

### Encoder (Perceiver)

Input: surface point cloud → Output: latent [B, 4096, 64]

Two parallel branches, each using cross-attention:

**Uniform branch:**
- KV: 32768 uniformly sampled surface points [B, 32768, 3]
- Q: 2048 FPS-selected points from the above (indices into KV)
- Cross-attention: each Q aggregates info from all 32768 KV points
- Output: 2048 latent tokens

**Salient branch (DoRaSa):**
- KV: 16384 salient points (sampled near sharp edges, dihedral angle > 15°)
- Q: 2048 FPS-selected points from the above
- Output: 2048 latent tokens

Combined: 2048 + 2048 = 4096 latent tokens × 64 dim = [B, 4096, 64]

**Why salient points?** Uniform sampling allocates points by surface area — large flat faces get many points, small edges get few. Salient sampling concentrates on sharp features (hinges, door edges, seams) that are critical for shape detail.

**Why cross-attention supports variable point cloud size:**
```
Q:  [B, 2048, d]  ← fixed (determines latent size)
K:  [B, N, d]     ← variable (any number of surface points)
V:  [B, N, d]     ← variable

scores = Q @ K^T / sqrt(d)  → [B, 2048, N]
weights = softmax(scores)    → [B, 2048, N]  (each row sums to 1)
output  = weights @ V        → [B, 2048, d]  (N gets summed away)
```
N can be any size. The model just needs exactly 2048 FPS indices for the query.

### Decoder

Input: latent [B, 4096, 64] → 24 layers self-attention → hidden [B, 4096, 1024]

Then for any query point in [-1,1]³:
- query_point [B, Q, 3] → fourier encoding → cross-attention with hidden → pred [B, Q] (occupancy value)

### Mesh Extraction (inference only)
1. Create 64³ grid of query points in [-1,1]³
2. Query all 262144 points → occupancy field
3. FlexiCubes / Marching Cubes on the occupancy field → extract mesh surface

### Training Loss
```
loss = MSE(pred_occupancy, gt_occupancy)
     + L1(pred_occupancy, gt_occupancy)
     + 0.001 * KL(latent_posterior, N(0,1))
```
- MSE + L1: reconstruction accuracy (predicted inside/outside vs ground truth)
- KL: regularize latent distribution (small weight, prevents latent collapse)
- GT occupancy: +1 = inside mesh, -1 = outside mesh

## Data Preprocessing Pipeline

### What it does
Converts each `part0.obj` / `part1.obj` into tensors that VAE can consume.

### Steps
```
Input: part0.obj (mesh with vertices and faces)

1. Load mesh: trimesh.load() → vertices [N,3], faces [M,3]

2. Sample 200K surface points uniformly
   meshiki.uniform_point_sample(200000) → [200000, 3]

3. FPS downsample to 32768 (for encoder KV)
   meshiki.fps(points, 32768) → uniform_pts [32768, 3]

4. Sample 16384 salient points (near sharp edges)
   meshiki.salient_point_sample(16384, thresh=15°) → salient_pts [16384, 3]

5. FPS select 2048 query indices from uniform points
   fpsample(uniform_pts, 2048) → fps_indices [2048]

6. FPS select 2048 query indices from salient points
   fpsample(salient_pts, 2048) → fps_indices_dorases [2048]

7. Sample 16384 occupancy query points in [-1,1]³
   - 8192 near-surface (surface point + Gaussian noise σ=0.02)
   - 8192 uniform random
   → query_points [16384, 3]

8. Compute GT occupancy via ray casting
   mesh.contains(query_points) → inside/outside → query_gt [16384]
   (+1 = inside, -1 = outside)

Output: .pt file with all 6 tensors
```

### Output tensor shapes
| Key | Shape | Purpose |
|-----|-------|---------|
| pointcloud | [32768, 3] | Encoder KV (uniform) |
| fps_indices | [2048] | Encoder Q indices (uniform) |
| pointcloud_dorases | [16384, 3] | Encoder KV (salient) |
| fps_indices_dorases | [2048] | Encoder Q indices (salient) |
| query_points | [16384, 3] | Decoder query positions |
| query_gt | [16384] | Training GT: +1 inside, -1 outside |

### FPS (Farthest Point Sampling)
NOT frames per second. Greedy algorithm that selects K points maximally spread out:
1. Pick random first point
2. Pick the point farthest from all selected points
3. Repeat until K points selected

Ensures uniform spatial coverage, unlike random sampling which can cluster.

### Near-surface query sampling
Pure uniform sampling in [-1,1]³ is inefficient — most points are far from the mesh surface and trivially outside. Near-surface points concentrate around the inside/outside boundary where the model actually needs to learn.

### Occupancy GT quality
Our part OBJs go through watertight processing in `split_precompute.py`:
1. `_clean_mesh()`: merge duplicate vertices, remove degenerate faces, fix normals
2. `_stitch_open_mesh()`: find open boundary loops, close coplanar convex holes

This is conservative (only stitches safe boundaries). Some OBJs remain non-watertight, but `mesh.contains()` (ray casting: odd crossings = inside) still works reliably for most cases.

## Finetune Setup

### Why finetune?
Original VAE was trained on ShapeNet/Objaverse data where each "part" is typically one connected semantic component. Our dual-volume parts come from URDF topology 2-coloring — one part can contain multiple spatially separated pieces (e.g., lamp base + head both in part0). The VAE has a prior towards merging disconnected components, which breaks our topology.

### Data
- Only process animodes that have training data (gt_latent.pt exists): ~2318 animodes × 2 parts = ~4636 OBJs
- Preprocess to .pt cache files (CPU, one-time)
- Training loads .pt directly (instant, GPU-bound)

### Training config
- 2 GPU DDP, batch_size 4 per GPU = 8 effective
- lr 1e-5, cosine schedule with 500-step warmup
- 20K steps, save every 2K
- AdamW, weight_decay 0.01, grad_clip 1.0

### Scripts
| Step | Script | Compute |
|------|--------|---------|
| Preprocess | `preprocess_vae_data.py` | CPU (multiprocess) |
| Train | `finetune_vae.py` | GPU 2,3 (DDP) |
| Evaluate | `eval_vae_roundtrip.py --vae_ckpt <path>` | GPU |

### Checkpoints
- Output: `/mnt/data_ssd/infinigen-sim/vae_finetune_v2/`
- Cache: `/mnt/data_ssd/infinigen-sim/vae_cache/`
