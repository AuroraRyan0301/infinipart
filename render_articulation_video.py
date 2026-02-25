#!/usr/bin/env python3
"""
Render articulation videos for Infinite Mobility generated objects.

For each split object, animate the moving part (part1) along its dominant
joint axis and render to video.

Usage:
  python render_articulation_video.py --output_root outputs
  python render_articulation_video.py --factory DishwasherFactory
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation

# Joint info per factory: axis, type, range for the dominant joint
FACTORY_JOINTS = {
    "DishwasherFactory": {
        "type": "revolute",
        "axis": np.array([0.0, 1.0, 0.0]),
        "lower": 0.0,
        "upper": math.pi / 2,
        "origin_shift": "bottom",  # hinge at bottom of door
    },
    "BeverageFridgeFactory": {
        "type": "revolute",
        "axis": np.array([0.0, 0.0, 1.0]),
        "lower": 0.0,
        "upper": math.pi / 2,
        "origin_shift": "side",  # hinge at side of door
    },
    "MicrowaveFactory": {
        "type": "revolute",
        "axis": np.array([0.0, 0.0, 1.0]),
        "lower": -math.pi / 2,
        "upper": 0.0,
        "origin_shift": "side",
    },
    "OvenFactory": {
        "type": "revolute",
        "axis": np.array([0.0, 1.0, 0.0]),
        "lower": 0.0,
        "upper": math.pi / 2,
        "origin_shift": "bottom",
    },
    "ToiletFactory": {
        "type": "revolute",
        "axis": np.array([1.0, 0.0, 0.0]),
        "lower": -math.pi / 2,
        "upper": 0.0,
        "origin_shift": "back",
    },
    "KitchenCabinetFactory": {
        "type": "prismatic",
        "axis": np.array([0.0, 1.0, 0.0]),
        "lower": 0.0,
        "upper": 0.75,
        "origin_shift": None,
    },
    "WindowFactory": {
        "type": "prismatic",
        "axis": np.array([0.0, 0.0, 1.0]),
        "lower": 0.0,
        "upper": 0.5,
        "origin_shift": None,
    },
    "LiteDoorFactory": {
        "type": "revolute",
        "axis": np.array([0.0, 0.0, 1.0]),
        "lower": 0.0,
        "upper": math.pi / 2,
        "origin_shift": "side",
    },
    "OfficeChairFactory": {
        "type": "revolute",
        "axis": np.array([1.0, 0.0, 0.0]),
        "lower": -math.pi / 10,
        "upper": 0.0,
        "origin_shift": None,
    },
    "TapFactory": {
        "type": "revolute",
        "axis": np.array([0.0, 0.0, 1.0]),
        "lower": 0.0,
        "upper": math.pi / 4,
        "origin_shift": None,
    },
    "PotFactory": {
        "type": "prismatic",
        "axis": np.array([0.0, 0.0, 1.0]),
        "lower": 0.0,
        "upper": 0.3,
        "origin_shift": None,
    },
    "BottleFactory": {
        "type": "prismatic",
        "axis": np.array([0.0, 0.0, 1.0]),
        "lower": 0.0,
        "upper": 0.2,
        "origin_shift": None,
    },
}


def find_hinge_point(part1_verts, joint_info):
    """Estimate the hinge point for revolute joints based on part geometry."""
    vmin = part1_verts.min(axis=0)
    vmax = part1_verts.max(axis=0)
    vcenter = (vmin + vmax) / 2

    origin = joint_info.get("origin_shift")
    if origin == "bottom":
        # Hinge at bottom edge (min Z)
        return np.array([vcenter[0], vcenter[1], vmin[2]])
    elif origin == "side":
        # Hinge at the side furthest from body center (max Y or min Y)
        return np.array([vcenter[0], vmax[1], vcenter[2]])
    elif origin == "back":
        # Hinge at back edge
        return np.array([vcenter[0], vmax[1], vcenter[2]])
    else:
        return vcenter


def apply_articulation(vertices, joint_info, q, hinge_point=None):
    """Apply joint articulation to vertices at position q."""
    jtype = joint_info["type"]
    axis = joint_info["axis"]

    if jtype == "prismatic":
        offset = q * axis
        return vertices + offset
    elif jtype == "revolute":
        if hinge_point is None:
            hinge_point = np.zeros(3)
        # Translate to hinge, rotate, translate back
        centered = vertices - hinge_point
        R = Rotation.from_rotvec(q * axis).as_matrix()
        rotated = (R @ centered.T).T
        return rotated + hinge_point
    return vertices


def render_frame(part0_verts, part0_faces, part1_verts, part1_faces,
                 elev, azim, ax, max_faces=3000):
    """Render a single frame to a matplotlib axis."""
    ax.clear()

    def subsample(verts, faces, max_f):
        if faces.shape[0] <= max_f:
            return verts, faces
        idx = np.random.choice(faces.shape[0], max_f, replace=False)
        return verts, faces[idx]

    v0, f0 = subsample(part0_verts, part0_faces, max_faces)
    v1, f1 = subsample(part1_verts, part1_faces, max_faces)

    # Draw part0 (blue/body)
    polys0 = v0[f0]
    pc0 = Poly3DCollection(polys0, alpha=0.5, linewidths=0.1, edgecolors='navy')
    pc0.set_facecolor((0.3, 0.5, 0.85, 0.5))
    ax.add_collection3d(pc0)

    # Draw part1 (red/moving)
    polys1 = v1[f1]
    pc1 = Poly3DCollection(polys1, alpha=0.7, linewidths=0.1, edgecolors='darkred')
    pc1.set_facecolor((0.85, 0.3, 0.3, 0.7))
    ax.add_collection3d(pc1)

    all_v = np.concatenate([v0, v1])
    vmin_all, vmax_all = all_v.min(0) - 0.2, all_v.max(0) + 0.2
    ax.set_xlim(vmin_all[0], vmax_all[0])
    ax.set_ylim(vmin_all[1], vmax_all[1])
    ax.set_zlim(vmin_all[2], vmax_all[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def render_video(part0_path, part1_path, factory_name, out_path,
                 n_frames=30, fps=10):
    """Render an articulation video as MP4."""
    joint_info = FACTORY_JOINTS.get(factory_name)
    if joint_info is None:
        print(f"  [SKIP] No joint info for {factory_name}")
        return False

    # Load meshes
    part0 = trimesh.load(part0_path, force="mesh", process=False)
    part1 = trimesh.load(part1_path, force="mesh", process=False)

    if isinstance(part0, trimesh.Scene):
        part0 = part0.to_mesh()
    if isinstance(part1, trimesh.Scene):
        part1 = part1.to_mesh()

    # Find hinge point
    hinge = find_hinge_point(part1.vertices, joint_info)

    # Generate joint positions: 0 -> upper -> 0 (open and close)
    q_lower = joint_info["lower"]
    q_upper = joint_info["upper"]
    half = n_frames // 2
    qs_open = np.linspace(q_lower, q_upper, half)
    qs_close = np.linspace(q_upper, q_lower, n_frames - half)
    qs = np.concatenate([qs_open, qs_close])

    # Render frames
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    frame_dir = out_path.replace('.mp4', '_frames')
    os.makedirs(frame_dir, exist_ok=True)

    for i, q in enumerate(qs):
        moved_verts = apply_articulation(part1.vertices.copy(), joint_info, q, hinge)
        render_frame(part0.vertices, part0.faces,
                     moved_verts, part1.faces,
                     elev=25, azim=45 + i * 2, ax=ax)
        ax.set_title(f'{factory_name} | q={q:.2f}', fontsize=12)
        frame_path = os.path.join(frame_dir, f'frame_{i:04d}.png')
        fig.savefig(frame_path, dpi=100)

    plt.close(fig)

    # Create GIF using imageio
    import imageio
    gif_path = out_path.replace('.mp4', '.gif')
    frame_files = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')])
    images = [imageio.imread(f) for f in frame_files]
    imageio.mimsave(gif_path, images, duration=1000 // fps, loop=0)
    print(f"  GIF saved: {gif_path}")

    # Also try MP4 with ffmpeg if available
    try:
        import subprocess
        mp4_result = subprocess.run(
            ['ffmpeg', '-y', '-framerate', str(fps),
             '-i', os.path.join(frame_dir, 'frame_%04d.png'),
             '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
             '-crf', '23', out_path],
            capture_output=True, text=True
        )
        if mp4_result.returncode == 0:
            print(f"  MP4 saved: {out_path}")
    except FileNotFoundError:
        pass

    # Clean up frames
    import shutil
    shutil.rmtree(frame_dir, ignore_errors=True)

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="outputs")
    parser.add_argument("--base_path", default="/mnt/data/yurh/Infinite-Mobility")
    parser.add_argument("--factory", default=None)
    parser.add_argument("--n_frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    base = args.base_path
    root = os.path.join(base, args.output_root)

    # Load split summary
    summary_path = os.path.join(root, "split_summary.json")
    if not os.path.exists(summary_path):
        print(f"[ERROR] No split_summary.json found. Run split_and_visualize.py first.")
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    for entry in summary:
        factory = entry["factory"]
        if args.factory and factory != args.factory:
            continue

        out_dir = entry["out_dir"]
        part0_path = os.path.join(out_dir, "part0.obj")
        part1_path = os.path.join(out_dir, "part1.obj")

        if not os.path.exists(part0_path) or not os.path.exists(part1_path):
            print(f"[SKIP] Missing split files for {factory} seed={entry['seed']}")
            continue

        print(f"\n--- Rendering {factory} seed={entry['seed']} ---")
        video_path = os.path.join(out_dir, "articulation.mp4")
        render_video(part0_path, part1_path, factory, video_path,
                     n_frames=args.n_frames, fps=args.fps)


if __name__ == "__main__":
    main()
