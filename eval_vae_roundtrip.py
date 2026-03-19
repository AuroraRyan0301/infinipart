"""Evaluate VAE encode-decode roundtrip: Chamfer Distance + CC error on part0/part1 OBJs."""
import argparse, glob, json, os, random, sys, time
import numpy as np
import torch
import trimesh

PARTPACKER_ROOT = "/mnt/data/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

PRECOMPUTE_ROOT = "/mnt/cpfs/yurh/Infinigen-Sim/precompute_output"


def chamfer_distance(pts_a, pts_b, n_sample=8192):
    """Chamfer distance between two point clouds (numpy)."""
    if len(pts_a) == 0 or len(pts_b) == 0:
        return -1.0
    if len(pts_a) > n_sample:
        idx = np.random.choice(len(pts_a), n_sample, replace=False)
        pts_a = pts_a[idx]
    if len(pts_b) > n_sample:
        idx = np.random.choice(len(pts_b), n_sample, replace=False)
        pts_b = pts_b[idx]
    # a->b
    from scipy.spatial import cKDTree
    tree_b = cKDTree(pts_b)
    d_ab, _ = tree_b.query(pts_a)
    tree_a = cKDTree(pts_a)
    d_ba, _ = tree_a.query(pts_b)
    return float(np.mean(d_ab**2) + np.mean(d_ba**2))


def count_cc(mesh_or_path, threshold_ratio=0.01, min_faces=10):
    """Count connected components, filtering tiny fragments."""
    if isinstance(mesh_or_path, str):
        mesh = trimesh.load(mesh_or_path, force='mesh', process=False)
    else:
        mesh = mesh_or_path
    if len(mesh.faces) == 0:
        return 0
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        total = sum(len(c.faces) for c in components)
        thresh = max(total * threshold_ratio, min_faces)
        components = [c for c in components if len(c.faces) >= thresh]
    return len(components)


def discover_samples(max_samples=200, seed=42):
    """Find part0.obj/part1.obj pairs from precompute output."""
    rng = random.Random(seed)
    pairs = []
    for cat in sorted(os.listdir(PRECOMPUTE_ROOT)):
        cat_dir = os.path.join(PRECOMPUTE_ROOT, cat)
        if not os.path.isdir(cat_dir):
            continue
        for seed_name in sorted(os.listdir(cat_dir)):
            seed_dir = os.path.join(cat_dir, seed_name)
            if not os.path.isdir(seed_dir):
                continue
            for animode in sorted(os.listdir(seed_dir)):
                anim_dir = os.path.join(seed_dir, animode)
                p0 = os.path.join(anim_dir, "part0.obj")
                p1 = os.path.join(anim_dir, "part1.obj")
                if os.path.exists(p0) and os.path.exists(p1):
                    pairs.append({
                        "category": cat,
                        "seed": seed_name,
                        "animode": animode,
                        "part0_obj": p0,
                        "part1_obj": p1,
                        "id": f"{cat}/{seed_name}/{animode}",
                    })
    rng.shuffle(pairs)
    pairs = pairs[:max_samples]
    print(f"Found {len(pairs)} animode pairs")
    return pairs


def load_vae(device, ckpt_override=None):
    import importlib
    from vae.model import Model as VAE
    vae_config = importlib.import_module("vae.configs.part_woenc").make_config()
    vae = VAE(vae_config).eval().to(device, dtype=torch.bfloat16)
    ckpt_path = ckpt_override or os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt")
    ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    vae.load_state_dict(ckpt, strict=True)
    del ckpt
    print(f"VAE loaded from {ckpt_path} ({sum(p.numel() for p in vae.parameters())/1e6:.1f}M params)")
    return vae


def prepare_vae_input(vertices, faces, num_fps=2048, num_fps_salient=2048):
    """Prepare VAE input from mesh vertices/faces (from encode_for_training.py)."""
    import fpsample
    import meshiki

    mesh = meshiki.Mesh(vertices, faces)
    uniform_pts = mesh.uniform_point_sample(200000)
    uniform_pts = meshiki.fps(uniform_pts, 32768)
    salient_pts = mesh.salient_point_sample(16384, thresh_bihedral=15)

    sample = {}
    sample["pointcloud"] = torch.from_numpy(uniform_pts)

    fps_idx = fpsample.bucket_fps_kdline_sampling(
        uniform_pts, num_fps, h=5, start_idx=0
    )
    sample["fps_indices"] = torch.from_numpy(fps_idx).long()

    sample["pointcloud_dorases"] = torch.from_numpy(salient_pts)
    fps_idx_s = fpsample.bucket_fps_kdline_sampling(
        salient_pts, num_fps_salient, h=5, start_idx=0
    )
    sample["fps_indices_dorases"] = torch.from_numpy(fps_idx_s).long()

    return sample


def vae_encode_decode(vae, obj_path, device, resolution=64):
    """Encode OBJ -> latent -> decode -> mesh. Returns (decoded_mesh, latent)."""
    mesh = trimesh.load(obj_path, force='mesh', process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.to_mesh()
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        return None, None

    sample = prepare_vae_input(mesh.vertices.astype(np.float32), mesh.faces)
    for k in sample:
        sample[k] = sample[k].unsqueeze(0).to(device)

    with torch.inference_mode():
        posterior = vae.encode(sample)
        latent = posterior.mode()  # [1, 4096, 64]

        dec_data = {"latent": latent}
        dec_result = vae(dec_data, resolution=resolution)

    if "meshes" not in dec_result or len(dec_result["meshes"]) == 0:
        return None, latent
    verts, faces = dec_result["meshes"][0]
    if len(verts) == 0:
        return None, latent
    decoded = trimesh.Trimesh(verts, faces, process=False)
    return decoded, latent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--output", type=str, default="eval_vae_roundtrip.json")
    parser.add_argument("--vae_ckpt", type=str, default=None,
                        help="Override VAE checkpoint path (default: pretrained)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    np.random.seed(args.seed)

    samples = discover_samples(args.max_samples, args.seed)
    vae = load_vae(device, ckpt_override=args.vae_ckpt)
    torch.cuda.empty_cache()

    results = []
    t0 = time.time()

    for i, s in enumerate(samples):
        try:
            r = {"id": s["id"], "category": s["category"]}

            for part in ["part0", "part1"]:
                obj_path = s[f"{part}_obj"]
                orig_mesh = trimesh.load(obj_path, force='mesh', process=False)
                orig_cc = count_cc(orig_mesh)

                decoded, latent = vae_encode_decode(vae, obj_path, device, args.resolution)

                if decoded is not None:
                    dec_cc = count_cc(decoded)
                    cd = chamfer_distance(orig_mesh.vertices, decoded.vertices)
                    r[f"{part}_obj_cc"] = orig_cc
                    r[f"{part}_vae_cc"] = dec_cc
                    r[f"{part}_cc_error"] = abs(orig_cc - dec_cc)
                    r[f"{part}_chamfer"] = cd
                    r[f"{part}_obj_verts"] = len(orig_mesh.vertices)
                    r[f"{part}_vae_verts"] = len(decoded.vertices)
                else:
                    r[f"{part}_obj_cc"] = orig_cc
                    r[f"{part}_vae_cc"] = -1
                    r[f"{part}_cc_error"] = -1
                    r[f"{part}_chamfer"] = -1.0

                del decoded, latent
                torch.cuda.empty_cache()

            results.append(r)

            if (i + 1) % 10 == 0 or i == 0:
                recent = results[-10:]
                valid = [r for r in recent if r["part0_chamfer"] >= 0 and r["part1_chamfer"] >= 0]
                if valid:
                    cd0 = np.mean([r["part0_chamfer"] for r in valid])
                    cd1 = np.mean([r["part1_chamfer"] for r in valid])
                    occ0 = np.mean([r["part0_obj_cc"] for r in valid])
                    occ1 = np.mean([r["part1_obj_cc"] for r in valid])
                    vcc0 = np.mean([r["part0_vae_cc"] for r in valid])
                    vcc1 = np.mean([r["part1_vae_cc"] for r in valid])
                    ce0 = np.mean([r["part0_cc_error"] for r in valid])
                    ce1 = np.mean([r["part1_cc_error"] for r in valid])
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed
                    eta = (len(samples) - i - 1) / rate if rate > 0 else 0
                    print(f"  [{i+1}/{len(samples)}] "
                          f"CD={cd0:.4f}/{cd1:.4f} "
                          f"OBJ_CC={occ0:.1f}/{occ1:.1f} "
                          f"VAE_CC={vcc0:.1f}/{vcc1:.1f} "
                          f"CC_err={ce0:.1f}/{ce1:.1f} "
                          f"| {rate:.1f} it/s, ETA {eta/60:.0f}m")

        except Exception as e:
            print(f"  [{i+1}] ERROR {s['id']}: {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed/60:.1f}m ({len(results)} results)")

    # Aggregate
    valid = [r for r in results if r["part0_chamfer"] >= 0 and r["part1_chamfer"] >= 0]
    agg = {
        "n_total": len(results),
        "n_valid": len(valid),
        "elapsed_min": elapsed / 60,
        "part0_chamfer_mean": float(np.mean([r["part0_chamfer"] for r in valid])),
        "part1_chamfer_mean": float(np.mean([r["part1_chamfer"] for r in valid])),
        "part0_obj_cc_mean": float(np.mean([r["part0_obj_cc"] for r in valid])),
        "part1_obj_cc_mean": float(np.mean([r["part1_obj_cc"] for r in valid])),
        "part0_vae_cc_mean": float(np.mean([r["part0_vae_cc"] for r in valid])),
        "part1_vae_cc_mean": float(np.mean([r["part1_vae_cc"] for r in valid])),
        "part0_cc_error_mean": float(np.mean([r["part0_cc_error"] for r in valid])),
        "part1_cc_error_mean": float(np.mean([r["part1_cc_error"] for r in valid])),
        # CC exact match rate
        "part0_cc_exact": float(np.mean([r["part0_cc_error"] == 0 for r in valid])),
        "part1_cc_exact": float(np.mean([r["part1_cc_error"] == 0 for r in valid])),
        "both_cc_exact": float(np.mean([r["part0_cc_error"] == 0 and r["part1_cc_error"] == 0 for r in valid])),
        "results": results,
    }

    # Per-category
    from collections import defaultdict
    cats = defaultdict(list)
    for r in valid:
        cats[r["category"]].append(r)

    print(f"\n{'='*80}")
    print(f"  VAE Roundtrip: {agg['n_valid']} samples")
    print(f"\n  {'':>12} {'Chamfer':>10} {'OBJ CC':>10} {'VAE CC':>10} {'CC err':>10} {'CC exact':>10}")
    print(f"  {'Part0':>12} {agg['part0_chamfer_mean']:>10.4f} {agg['part0_obj_cc_mean']:>10.2f} "
          f"{agg['part0_vae_cc_mean']:>10.2f} {agg['part0_cc_error_mean']:>10.2f} {agg['part0_cc_exact']:>9.1%}")
    print(f"  {'Part1':>12} {agg['part1_chamfer_mean']:>10.4f} {agg['part1_obj_cc_mean']:>10.2f} "
          f"{agg['part1_vae_cc_mean']:>10.2f} {agg['part1_cc_error_mean']:>10.2f} {agg['part1_cc_exact']:>9.1%}")
    print(f"  Both CC exact: {agg['both_cc_exact']:.1%}")

    print(f"\n  Per-category:")
    print(f"  {'Category':<22} {'N':>4} {'CD0':>8} {'CD1':>8} {'OCC':>8} {'VCC':>8} {'CCerr':>8} {'Exact%':>7}")
    agg["per_category"] = {}
    for cat, rs in sorted(cats.items()):
        ci = {
            "n": len(rs),
            "cd0": float(np.mean([r["part0_chamfer"] for r in rs])),
            "cd1": float(np.mean([r["part1_chamfer"] for r in rs])),
            "obj_cc": f"{np.mean([r['part0_obj_cc'] for r in rs]):.1f}/{np.mean([r['part1_obj_cc'] for r in rs]):.1f}",
            "vae_cc": f"{np.mean([r['part0_vae_cc'] for r in rs]):.1f}/{np.mean([r['part1_vae_cc'] for r in rs]):.1f}",
            "cc_err": float(np.mean([r["part0_cc_error"] + r["part1_cc_error"] for r in rs])),
            "both_exact": float(np.mean([r["part0_cc_error"] == 0 and r["part1_cc_error"] == 0 for r in rs])),
        }
        agg["per_category"][cat] = ci
        print(f"  {cat:<22} {ci['n']:>4} {ci['cd0']:>8.4f} {ci['cd1']:>8.4f} "
              f"{ci['obj_cc']:>8} {ci['vae_cc']:>8} {ci['cc_err']:>8.2f} {ci['both_exact']:>6.0%}")
    print(f"{'='*80}")

    with open(args.output, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
