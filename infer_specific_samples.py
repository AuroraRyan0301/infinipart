#!/usr/bin/env python3
"""Infer specific samples listed by (category, model_id). For apples-to-apples ckpt comparison."""
import argparse, glob, importlib, os, subprocess, sys, time
import numpy as np
import torch
import torch.nn as nn

PARTPACKER_ROOT = "/mnt/cpfs/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))
from infer_quick import load_flow_model, load_vae, flow_sample, decode_latent_to_obj, render_blender


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--jepa_root", required=True)
    parser.add_argument("--precompute_root", default="/mnt/data/yurh/Infinigen-Sim/data_ssd/precompute_solidified")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--samples", nargs="+", required=True,
                        help="List of cat/seed/animode, e.g. faucet/9/senior_0")
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--no_render", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda")
    dit, proj, step = load_flow_model(args.ckpt, device)
    vae = load_vae(device)

    out_dir = os.path.join(args.output_dir, f"step_{step}")
    os.makedirs(out_dir, exist_ok=True)

    for spec in args.samples:
        cat, seed, animode = spec.split("/")
        mid = f"{seed}_{animode}"
        tag = f"{cat}_{mid}"
        print(f"\n[{spec}]")

        # Find jepa feature
        jepa_files = sorted(glob.glob(os.path.join(args.jepa_root, cat, mid, "views", "*.pt")))
        if not jepa_files:
            print(f"  No jepa found for {spec}, skipping")
            continue
        jepa = torch.load(jepa_files[0], map_location=device, weights_only=False).to(torch.bfloat16)

        t0 = time.time()
        pred = flow_sample(dit, proj, jepa, num_steps=args.num_steps, cfg_scale=args.cfg_scale, device=device)
        print(f"  Flow: {time.time()-t0:.1f}s")

        pred_p0 = os.path.join(out_dir, f"{tag}_pred_p0.obj")
        pred_p1 = os.path.join(out_dir, f"{tag}_pred_p1.obj")
        ok0 = decode_latent_to_obj(vae, pred[:, :4096, :], pred_p0, args.resolution)
        ok1 = decode_latent_to_obj(vae, pred[:, 4096:, :], pred_p1, args.resolution)
        print(f"  Decode: p0={'OK' if ok0 else 'FAIL'}, p1={'OK' if ok1 else 'FAIL'}")

        if not args.no_render and ok0 and ok1:
            gt_p0 = os.path.join(args.precompute_root, cat, seed, animode, "part0.obj")
            gt_p1 = os.path.join(args.precompute_root, cat, seed, animode, "part1.obj")
            pred_png = os.path.join(out_dir, f"{tag}_pred.png")
            gt_png = os.path.join(out_dir, f"{tag}_gt.png")
            compare_png = os.path.join(out_dir, f"{tag}_compare.png")
            render_blender(pred_p0, pred_p1, pred_png, f"Pred {tag}")
            render_blender(gt_p0, gt_p1, gt_png, f"GT {tag}")
            if os.path.isfile(pred_png) and os.path.isfile(gt_png):
                subprocess.run(["ffmpeg", "-y", "-i", pred_png, "-i", gt_png,
                                "-filter_complex", "hstack=inputs=2", compare_png],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  Rendered: {compare_png}")

        torch.cuda.empty_cache()

    print(f"\nDone! {out_dir}")


if __name__ == "__main__":
    main()
