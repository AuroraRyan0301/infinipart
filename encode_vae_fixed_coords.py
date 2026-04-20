#!/usr/bin/env python3
"""Re-encode all gt_latent with correct GLB coordinate transform.

GT mesh vertices are transformed with TRIMESH_GLB_EXPORT before VAE encode,
so that latent space matches PartPacker's flow model pretrained weights.

Usage (4 GPUs):
  for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu python encode_vae_fixed_coords.py --shard $gpu --n_shards 4 &
  done
  wait
"""
import argparse, os, sys, time, importlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/mnt/cpfs/yurh/PartPacker")
sys.path.insert(0, "/mnt/cpfs/yurh/PartPacker/vae")
import fpsample, meshiki, trimesh
from vae.model import Model as VAEModel

# PartPacker's coordinate transform: OBJ Y-up -> GLB coord for VAE latent space
TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)

SOLIDIFIED = "/mnt/data_ssd/infinigen-sim-data/precompute_solidified"
OUTPUT = "/mnt/data_ssd/infinigen-sim-data/encoded_solidified"


def prepare_input(obj_path):
    """Load mesh, apply GLB coord transform, sample points, return VAE input."""
    mesh = trimesh.load(obj_path, process=False, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    v = mesh.vertices.astype(np.float32)

    # Apply GLB coordinate transform BEFORE encoding
    v = v @ TRIMESH_GLB_EXPORT

    f = mesh.faces
    m = meshiki.Mesh(v, f)
    uniform_pts = meshiki.fps(m.uniform_point_sample(200000), 32768)
    salient_pts = m.salient_point_sample(16384, thresh_bihedral=15)
    if salient_pts.ndim != 2 or salient_pts.shape[0] < 2048:
        salient_pts = uniform_pts.copy()
    return {
        "pointcloud": torch.from_numpy(uniform_pts),
        "fps_indices": torch.from_numpy(
            fpsample.bucket_fps_kdline_sampling(uniform_pts, 2048, h=5, start_idx=0)
        ).long(),
        "pointcloud_dorases": torch.from_numpy(salient_pts),
        "fps_indices_dorases": torch.from_numpy(
            fpsample.bucket_fps_kdline_sampling(salient_pts, 2048, h=5, start_idx=0)
        ).long(),
    }


class EncodeDataset(Dataset):
    def __init__(self, tasks):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        p0_path, p1_path, out_path = self.tasks[idx]
        try:
            s0 = prepare_input(p0_path)
            s1 = prepare_input(p1_path)
            return s0, s1, out_path, ""
        except Exception as e:
            return None, None, out_path, f"{p0_path} | {e}"


def collate_fn(batch):
    s0, s1, out_path, err = batch[0]
    return s0, s1, out_path, err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=16)
    args = parser.parse_args()

    # Load VAE
    ckpt = torch.load("/mnt/cpfs/yurh/PartPacker/pretrained/vae.pt", weights_only=True)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    config = importlib.import_module("vae.configs.part_woenc").make_config()
    model = VAEModel(config).cuda().bfloat16().eval()
    model.load_state_dict(ckpt, strict=True)
    del ckpt
    print(f"[Shard {args.shard}] VAE loaded", flush=True)

    # Discover tasks: find all animodes in encoded_solidified that need gt_latent
    # Also include encoded/ (IS factories)
    ENCODED = "/mnt/data_ssd/infinigen-sim-data/encoded"
    tasks = []

    # IS factories (from encoded/)
    for cat in sorted(os.listdir(ENCODED)):
        if "PhysXNet" in cat:
            continue
        cat_dir = os.path.join(ENCODED, cat)
        if not os.path.isdir(cat_dir):
            continue
        for sample in sorted(os.listdir(cat_dir)):
            out_path = os.path.join(OUTPUT, cat, sample, "gt_latent.pt")
            if os.path.exists(out_path):
                continue
            # Check if solidified parts exist
            seed = sample.split("_")[0]
            animode = "_".join(sample.split("_")[1:])
            p0 = os.path.join(SOLIDIFIED, cat, seed, animode, "part0.obj")
            p1 = os.path.join(SOLIDIFIED, cat, seed, animode, "part1.obj")
            if os.path.isfile(p0) and os.path.isfile(p1):
                tasks.append((p0, p1, out_path))

    # PhysXMobility (from encoded/)
    pm_encoded = os.path.join(ENCODED, "PhysXMobility")
    if os.path.isdir(pm_encoded):
        for sample in sorted(os.listdir(pm_encoded)):
            out_path = os.path.join(OUTPUT, "PhysXMobility", sample, "gt_latent.pt")
            if os.path.exists(out_path):
                continue
            seed = sample.split("_")[0]
            animode = "_".join(sample.split("_")[1:])
            p0 = os.path.join(SOLIDIFIED, "PhysXMobility", seed, animode, "part0.obj")
            p1 = os.path.join(SOLIDIFIED, "PhysXMobility", seed, animode, "part1.obj")
            if os.path.isfile(p0) and os.path.isfile(p1):
                tasks.append((p0, p1, out_path))

    # Shard
    tasks = [t for i, t in enumerate(tasks) if i % args.n_shards == args.shard]
    print(f"[Shard {args.shard}] {len(tasks)} tasks", flush=True)

    ds = EncodeDataset(tasks)
    dl = DataLoader(
        ds,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        prefetch_factor=8,
        persistent_workers=True,
    )

    encoded = failed = 0
    t0 = time.time()
    bad_f = open(f"logs/encode_vae_fixed_shard{args.shard}.txt", "w")

    for s0, s1, out_path, err in dl:
        if s0 is None:
            bad_f.write(f"{err}\n")
            bad_f.flush()
            failed += 1
            continue
        try:
            for k in s0:
                s0[k] = s0[k].unsqueeze(0).cuda()
            for k in s1:
                s1[k] = s1[k].unsqueeze(0).cuda()
            with torch.inference_mode():
                lat0 = model.encode(s0).mode()
                lat1 = model.encode(s1).mode()
            gt = torch.cat([lat0, lat1], dim=1)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(gt.cpu(), out_path)
            encoded += 1
            del lat0, lat1, gt, s0, s1
        except Exception as e:
            bad_f.write(f"{out_path} | {e}\n")
            bad_f.flush()
            failed += 1

        if encoded % 100 == 0 and encoded > 0:
            el = time.time() - t0
            r = encoded / el
            rem = (len(tasks) - encoded - failed) / max(r, 0.01)
            print(
                f"  [Shard {args.shard}] [{encoded}/{len(tasks)}] {r:.1f}/s | fail={failed} | ETA={rem/60:.0f}min",
                flush=True,
            )

    bad_f.close()
    el = time.time() - t0
    print(
        f"\n[Shard {args.shard}] Done! {encoded} encoded, {failed} failed | {el/60:.1f}min | {encoded/max(el,1):.1f}/s"
    )


if __name__ == "__main__":
    main()
