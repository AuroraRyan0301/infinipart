#!/usr/bin/env python3
"""
Run inference on a single video input: VJEPA2 features -> DiT -> PartPacker VAE decode -> mesh.

Usage:
  CUDA_VISIBLE_DEVICES=3 python infer_one.py \
    --jepa_path training_data/box_0_basic_3/views/v00_nobg_jepa.pt \
    --gt_path training_data/box_0_basic_3/gt_latent.pt \
    --output_dir output/infer_box_0_basic_3
"""

import argparse
import importlib
import os
import sys

import numpy as np
import torch

PARTPACKER_ROOT = "/mnt/data/yurh/PartPacker"
sys.path.insert(0, PARTPACKER_ROOT)
sys.path.insert(0, os.path.join(PARTPACKER_ROOT, "vae"))

LATENT_SIZE = 2048
LATENT_DIM = 64
TOTAL_LATENT = LATENT_SIZE * 2  # 4096
VJEPA_DIM = 1408
DIT_DIM = 1536


def build_model(device, ckpt_path=None):
    """Build DiT + projector, load weights."""
    from flow.modules.dit import DiT

    dit = DiT(
        hidden_dim=DIT_DIM, num_heads=16, num_layers=24,
        latent_size=LATENT_SIZE, latent_dim=LATENT_DIM,
        qknorm=True, qknorm_type="RMSNorm",
        use_pos_embed=False, use_parts=True, part_embed_mode="part2_only",
    ).to(device, dtype=torch.bfloat16)

    proj = torch.nn.Linear(VJEPA_DIM, DIT_DIM).to(device, dtype=torch.bfloat16)

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        dit.load_state_dict(ckpt["dit"])
        if "proj" in ckpt:
            proj.load_state_dict(ckpt["proj"])
        print(f"  Step: {ckpt.get('step', '?')}, Loss: {ckpt.get('loss', '?')}")
    else:
        # Load pretrained PartPacker flow weights (partial match)
        flow_ckpt = os.path.join(PARTPACKER_ROOT, "pretrained", "flow.pt")
        print(f"Loading pretrained: {flow_ckpt}")
        ckpt = torch.load(flow_ckpt, weights_only=False, map_location=device)
        if "model" in ckpt:
            ckpt = ckpt["model"]
        dit_state = {
            k.replace("dit.", ""): v for k, v in ckpt.items() if k.startswith("dit.")
        }
        param_dict = dict(dit.named_parameters())
        loaded, skipped = 0, 0
        for k, v in dit_state.items():
            if k in param_dict and param_dict[k].shape == v.shape:
                param_dict[k].data.copy_(v)
                loaded += 1
            else:
                skipped += 1
        print(f"  Loaded {loaded}, skipped {skipped}")
        del ckpt, dit_state

    dit.eval()
    proj.eval()
    return dit, proj


def flow_inference(dit, cond, device, num_steps=50, cfg_scale=5.0):
    """Flow matching sampling."""
    cond_null = torch.zeros_like(cond)
    cond_input = torch.cat([cond, cond_null], dim=0)
    x = torch.randn(1, TOTAL_LATENT, LATENT_DIM, device=device)

    sigmas = np.linspace(1, 0, num_steps + 1)
    sigmas = 3.0 * sigmas / (1 + (3.0 - 1) * sigmas)

    with torch.inference_mode():
        for i in range(num_steps):
            sigma, sigma_prev = sigmas[i], sigmas[i + 1]
            timesteps = torch.tensor(
                [1000 * sigma] * 2, device=device, dtype=torch.float32)
            x_input = torch.cat([x, x], dim=0).to(dtype=torch.bfloat16)
            pred = dit(x_input, cond_input, timesteps).float()
            cond_v, uncond_v = pred.chunk(2, dim=0)
            pred_v = uncond_v + (cond_v - uncond_v) * cfg_scale
            x = x - (sigma - sigma_prev) * pred_v
    return x


def decode_latent_to_mesh(latent, device, output_dir, grid_res=256):
    """Decode latent [1, 4096, 64] -> two meshes via PartPacker VAE."""
    import trimesh
    from vae.model import Model
    from vae.utils import postprocess_mesh

    vae_ckpt = os.path.join(PARTPACKER_ROOT, "pretrained", "vae.pt")
    ckpt = torch.load(vae_ckpt, weights_only=True)
    if "model" in ckpt:
        ckpt = ckpt["model"]
    config = importlib.import_module("vae.configs.part_woenc").make_config()
    vae = Model(config).eval().to(device).to(torch.bfloat16)
    vae.load_state_dict(ckpt, strict=True)
    del ckpt
    print("VAE loaded for decoding")

    # Split into two parts
    lat0 = latent[:, :LATENT_SIZE, :]  # [1, 2048, 64]
    lat1 = latent[:, LATENT_SIZE:, :]  # [1, 2048, 64]

    TRIMESH_GLB_EXPORT = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).astype(np.float32)

    os.makedirs(output_dir, exist_ok=True)

    for i, lat in enumerate([lat0, lat1]):
        print(f"  Decoding part {i}...")
        data = {"latent": lat.to(device, dtype=torch.bfloat16)}
        with torch.inference_mode():
            results = vae(data, resolution=grid_res)

        if "meshes" in results and len(results["meshes"]) > 0:
            vertices, faces = results["meshes"][0]
            mesh = trimesh.Trimesh(vertices, faces)
            mesh.vertices = mesh.vertices @ TRIMESH_GLB_EXPORT.T
            out_path = os.path.join(output_dir, f"part{i}.obj")
            mesh.export(out_path)
            print(f"  Saved: {out_path} ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")
        else:
            print(f"  Part {i}: no mesh extracted")

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa_path", type=str, required=True,
                        help="Path to VJEPA2 feature .pt file")
    parser.add_argument("--gt_path", type=str, default=None,
                        help="Path to gt_latent.pt (for comparison)")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Training checkpoint (default: pretrained)")
    parser.add_argument("--output_dir", type=str, default="output/infer_test")
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--decode", action="store_true",
                        help="Also decode latent to mesh via VAE")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load VJEPA2 features
    print(f"Loading JEPA features: {args.jepa_path}")
    vj = torch.load(args.jepa_path, weights_only=False, map_location=device)
    if vj.dim() == 2:
        vj = vj.unsqueeze(0)
    vj = vj.to(dtype=torch.bfloat16)
    print(f"  Shape: {vj.shape}")

    # Build model
    dit, proj = build_model(device, args.ckpt)

    # Project JEPA features
    with torch.inference_mode():
        cond = proj(vj)
    print(f"  Projected cond: {cond.shape}")

    # Run flow inference
    print(f"Running flow inference ({args.num_steps} steps, cfg={args.cfg_scale})...")
    pred_latent = flow_inference(dit, cond, device, args.num_steps, args.cfg_scale)
    print(f"  Predicted latent: {pred_latent.shape}")

    # Save predicted latent
    torch.save(pred_latent.cpu(), os.path.join(args.output_dir, "pred_latent.pt"))

    # Compare with GT if available
    if args.gt_path and os.path.exists(args.gt_path):
        gt = torch.load(args.gt_path, weights_only=False, map_location=device).float()
        if gt.dim() == 2:
            gt = gt.unsqueeze(0)
        mse = torch.nn.functional.mse_loss(pred_latent.float(), gt).item()
        print(f"  GT MSE: {mse:.6f}")

        # Also decode GT for comparison
        if args.decode:
            print("\nDecoding GT latent...")
            gt_dir = os.path.join(args.output_dir, "gt")
            decode_latent_to_mesh(gt, device, gt_dir)

    # Decode predicted latent
    if args.decode:
        print("\nDecoding predicted latent...")
        pred_dir = os.path.join(args.output_dir, "pred")
        decode_latent_to_mesh(pred_latent, device, pred_dir)

    # Free DiT memory before decode
    del dit, proj, cond, pred_latent
    torch.cuda.empty_cache()

    print(f"\nDone! Output: {args.output_dir}")


if __name__ == "__main__":
    main()
