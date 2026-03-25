# Articulated Object Generation — Benchmark Survey

## 1. Related Methods Overview

| Method | Input | Output | Core Approach |
|---|---|---|---|
| **SINGAPO** (CVPR 2025) | Single image | Articulated 3D object (parts + joints) | Diffusion on part-level representation, retrieval-based part mesh |
| **PAct** (2025) | 1-2 images | Articulated 3D object with texture | Two-stage: structure prediction + geometry/appearance generation |
| **Articulate-Anything** (2024) | Text / image / video | URDF articulated object | LLM/VLM pipeline: part decomposition → joint prediction → iterative refinement |
| **URDFormer** (2023) | Single image | URDF structure | Transformer predicts URDF graph from image features |
| **NAP** (2023) | Category prior | Articulated object | Neural part generation with joint-aware priors |
| **Ours (Infinipart)** | Video | Dual volume (part0 + part1 topology split) | VJEPA2 video features → PartPacker Flow DiT → VAE decode → dual part mesh |

## 2. Standard Datasets

### PartNet-Mobility (Primary benchmark, all methods use)
- Source: SAPIEN (https://sapien.ucsd.edu/browse)
- 2,346 articulated objects, 46 categories
- Standard train/test split: **by object ID** (unseen objects in test)
- SINGAPO/PAct use 77 test objects
- Each object has URDF + per-part meshes + joint annotations

### ACD (Articulated Common Dataset, S2O)
- 135 objects from HSSD + ABO
- Used for **zero-shot generalization** (not in training)
- PAct and SINGAPO evaluate on this

### Our Data (Infinipart)
- IS Factories: 18 procedural categories (lamp, cabinet, dishwasher, etc.)
- PhysXMobility: 2,024 objects from PhysX_mobility dataset (overlaps with PartNet-Mobility IDs)
- Category labels available via `finaljson/{id}.json` → `object_name` field

## 3. Evaluation Metrics

### Geometry Metrics (measured in both Resting State RS and Articulated State AS)

| Metric | Formula / Description | Used by |
|---|---|---|
| **Chamfer Distance (dCD)** | Symmetric distance between 2048 sampled points on predicted vs GT mesh surfaces. Lower = better. | SINGAPO, PAct |
| **Generalized IoU (dgIoU)** | `1 - gIoU` of part bounding boxes. Measures part position/scale accuracy at bbox level. Lower = better. | SINGAPO, PAct |
| **Centroid Distance (dcDist)** | Euclidean distance between part centroids. More sensitive to thin/small parts than IoU. Lower = better. | SINGAPO, PAct |

### Articulation Quality Metrics

| Metric | Description | Used by |
|---|---|---|
| **Average Overlapping Ratio (AOR)** | Fraction of sibling parts that collide/interpenetrate during articulation. Detects unrealistic motion. Lower = better. | SINGAPO, PAct |
| **Joint Success Rate** | % of objects where predicted joint type + parameters produce valid articulation. Higher = better. | Articulate-Anything |
| **Graph Accuracy (Acc%)** | % of objects where predicted kinematic graph topology exactly matches GT. | SINGAPO |

### Visual / Perceptual Metrics

| Metric | Description | Used by |
|---|---|---|
| **CLIP Similarity** | Cosine similarity between CLIP embeddings of rendered predictions vs input image. Measured across 5 viewpoints × 6 articulation states. Higher = better. | PAct |
| **User Study** | Human raters score realism + articulation plausibility (1-3 scale). | SINGAPO, Articulate-Anything |

### Topology Metrics (proposed for our method)

| Metric | Description |
|---|---|
| **Connected Components (CC)** | Number of connected components per part. GT should be 1 per solid part. Measures mesh fragmentation. |
| **Watertight Rate** | % of decoded parts that are watertight (closed surface). |
| **Dual Volume IoU** | Volumetric IoU between predicted part0/part1 and GT part0/part1 occupancy. |

## 4. Train/Test Split Methodology

### Standard practice (SINGAPO, PAct)
- **Split by object ID**: test objects are completely unseen during training
- NOT by view: different views of the same object are NOT used as test
- SINGAPO defines the standard split in `data/data_split.json`
- Multiple views per test object (typically 2 random views per object for evaluation)

### Our split (Infinipart overfit v2)
- **Train objects**: 31 PhysXMobility objects (table, cabinet, kitchen pot, trashcan, toilet, etc.)
- **Test held-out views**: same train objects, held-out views (view_idx % 4 == 3)
- **Test OOD objects**: 11 completely unseen objects (1 per category)
- Manifest: `/mnt/data_ssd/infinigen-sim-data/checkpoints/overfit_physxmob_v2/manifest.json`

## 5. Evaluation Protocol Details

### Part matching (SINGAPO, PAct)
- Use **Hungarian algorithm** with centroid distance as cost matrix
- Optimal bipartite matching between predicted parts and GT parts
- Handles variable number of predicted parts

### Multi-state evaluation
- Evaluate at resting state (RS) + multiple articulated states (AS)
- AS states are sampled by varying joint angles (e.g., 6 evenly spaced states)
- Report metrics averaged across states

### Mesh comparison pipeline
1. Predict articulated object (parts + joints)
2. For each evaluation state: apply joint transforms to get part positions
3. Sample 2048 points per part surface
4. Match predicted ↔ GT parts via Hungarian
5. Compute dCD, dgIoU, dcDist per matched pair
6. Compute AOR across all sibling part pairs
7. Average across states and objects

## 6. Baselines to Compare Against

| Method | Code | Key result |
|---|---|---|
| SINGAPO | https://github.com/3dlg-hcvc/singapo | State-of-art on PartNet-Mobility |
| PAct | (paper only, 2025) | Beats SINGAPO on most metrics |
| Articulate-Anything | https://github.com/vlongle/articulate-anything | 75% joint success rate |
| URDFormer | https://github.com/yzhangec/URDFormer | Oracle + DINO variants |
| PartPacker (original, no video) | https://github.com/NVlabs/PartPacker | Our backbone, image-conditioned |

## 7. Key Differences: Our Method vs Others

| Aspect | Others (SINGAPO, PAct) | Ours (Infinipart) |
|---|---|---|
| Input | 1-2 static images | Video (81 frames, captures motion) |
| Output | Full articulated object (all parts + all joints) | Dual volume: part0 + part1 per animode (binary topology split) |
| Joint prediction | Explicit (type, axis, limits) | Implicit (captured in which parts move together) |
| Part count | Variable (matches object complexity) | Always 2 (binary split, alternating coloring) |
| Evaluation focus | Part geometry + joint accuracy | Topology correctness + volume quality |
| Training data | PartNet-Mobility only | IS factories + PhysXMobility (procedural + real) |
