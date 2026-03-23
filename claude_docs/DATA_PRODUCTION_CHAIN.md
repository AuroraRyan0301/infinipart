# Data Production Chain

## Three Data Sources

### IS Factory (18 types × 100 seeds)
```
Blender spawn_asset.py (infinigen procedural generation)
    → sim_exports/urdf/{name}/{seed}/
        ├── {name}.urdf          (baked PBR materials)
        └── assets/*.obj         (per-link mesh)
    → split_precompute.py
        ← URDF topology + OBJ mesh
        ← BVH collision → joint classify (active/passive/fixed)
        ← bipartite 2-coloring → part0/1.obj + metadata.json
    → render_animode.py (Blender Cycles)
        ← metadata + part0/1.obj + URDF joint params
        → mp4 video (multi-view, motion trajectory)
```
Self-contained. No external data dependency.

### PhysXMobility (2,024 objects)
```
Original data (fulian):
    /mnt/cpfs/fulian/dataset/PhysX_mobility/
        ├── urdf/{id}.urdf
        ├── partseg/{id}/objs/original-*.obj + MTL (native colors)
        └── finaljson/{id}.json (semantic labels)

    → setup_physxnet_scene.py
        → scenes/outputs/PhysXMobility/{id}/
            ├── scene.urdf     (rewritten absolute mesh paths)
            ├── origins.json   (link-to-group mapping)
            └── objs/          (grouped OBJ copies)
    → split_precompute.py → precompute/PhysXMobility/{id}/{animode}/part0/1.obj
    → render_animode.py → mp4 videos
        ⚠ render reads scene.urdf from metadata["scene_dir"]
```

### PhysXNet (32,041 objects, ~3% movable)
```
Original data (fulian):
    /mnt/cpfs/fulian/dataset/PhysXNet/version_1/
        ├── urdf/{id}.urdf
        ├── partseg/{id}/objs/*.obj (plain geometry, no MTL)
        └── finaljson/{id}.json

    Material sources (for render only):
        ShapeNet → /mnt/cpfs/yurh/dataset3D/ShapeNetCore/
        PartNet  → /mnt/cpfs/yurh/dataset3D/Partnet/ (color fallback)
        ambientCG → /mnt/cpfs/yurh/infinipart/pbr_textures/
        overlap map → physxnet_partnet_overlap.json

    → setup_physxnet_scene.py (same as PhysXMobility)
    → split_precompute.py (same)
    → render_animode.py (same, but material from ShapeNet/ambientCG)
```

## Common Encoding Pipeline (after precompute)

```
precompute/{cat}/{seed}/{animode}/
    ├── part0.obj, part1.obj
    ├── metadata.json (joint params, scene_dir, animode info)
    └── *_nobg.mp4 (rendered video, multi-view)

    → encode_for_training.py (partpacker_wan env)
        ├── PP VAE Encoder: part0/1.obj → gt_latent.pt [1, 8192, 64]
        └── VJEPA2 ViT-g:   mp4 → *_jepa.pt [10240, 1408]

    → preprocess_slat_gt.py (trellis2 env)
        └── SLat Encoder: part0/1.obj → slat_gt.pt
            ├── p0_lr/p1_lr: coords [N,4] + feats [N,32]  (res=512)
            └── p0_hr/p1_hr: coords [M,4] + feats [M,32]  (res=1024)

    → precompute_vae_coords.py (partpacker_wan env)
        └── PP VAE Decoder: gt_latent → 32³ occ grid → coords [0,31]
            → vae_coords.pt: p0_coords [N,4] + p1_coords [M,4]
```

## Dependency Graph

```
fulian original data ──→ setup_physxnet_scene.py ──→ scenes/
                                                        │
IS spawn_asset.py ──→ sim_exports/                      │
                         │                              │
                         ▼                              ▼
                    split_precompute.py ──→ precompute/ (part0/1.obj + metadata)
                                                │
                                    render_animode.py (needs scene_dir from metadata)
                                                │
                                                ▼
                                        precompute/ (+ mp4 videos)
                                                │
                            ┌───────────────────┼───────────────────┐
                            ▼                   ▼                   ▼
                    PP VAE encode        VJEPA2 encode       SLat Encoder
                    → gt_latent.pt       → *_jepa.pt         → slat_gt.pt
                            │                                       │
                            ▼                                       │
                    PP VAE decode                                   │
                    → vae_coords.pt                                 │
                            │                                       │
                            └───────────────┬───────────────────────┘
                                            │
                                    Training / Inference
```

## File Locations

| Data | Location |
|------|----------|
| Original URDF+OBJ (fulian) | `/mnt/cpfs/fulian/dataset/PhysX*/` (read-only) |
| Scene setup (URDF rewrite) | `/mnt/data_ssd/infinigen-sim-data/scenes/` |
| Precompute + videos | `/mnt/data_ssd/infinigen-sim-data/precompute/` |
| Encoded (VAE + JEPA) | `/mnt/data_ssd/infinigen-sim-data/encoded/` |
| SLat GT cache | `/mnt/data_ssd/infinigen-sim-data/slat_gt/` |
| VAE occ coords | `/mnt/data_ssd/infinigen-sim-data/vae_coords/` |
| Checkpoints | `/mnt/data_ssd/infinigen-sim-data/checkpoints/` |
| IS spawn outputs | needs re-generation (sim_exports was deleted) |
