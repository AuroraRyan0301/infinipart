#!/usr/bin/env python3
"""
Render CFG comparison grid for presentation.
Layout: rows = objects, columns = cfg values + GT
Supports --seed to select different random seed runs.
"""
import argparse, subprocess, os, sys, glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BLENDER = "/mnt/cpfs/yurh/blender-4.2.18-linux-x64/blender"
BASE = "/mnt/cpfs/yurh/Infinigen-Sim/output"
PRECOMPUTE = "/mnt/data/yurh/Infinigen-Sim/data_ssd/precompute_solidified"

CFG_SCALES = [0.0, 1.0, 2.0, 4.0, 5.0, 7.9, 11.0]

BLENDER_SCRIPT = r'''
import bpy, sys, os, math, mathutils
argv = sys.argv[sys.argv.index("--") + 1:]
p0_obj, p1_obj, output_png = argv[0], argv[1], argv[2]
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()
for c in bpy.data.collections: bpy.data.collections.remove(c)
def import_obj(path, color, name):
    bpy.ops.wm.obj_import(filepath=path)
    obj = bpy.context.selected_objects[0]
    obj.name = name
    mat = bpy.data.materials.new(name=f"mat_{name}")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = 0.5
    bsdf.inputs["Metallic"].default_value = 0.1
    if obj.data.materials: obj.data.materials[0] = mat
    else: obj.data.materials.append(mat)
    return obj
o0 = import_obj(p0_obj, (0.15, 0.35, 0.85, 1.0), "part0")
o1 = import_obj(p1_obj, (0.95, 0.55, 0.1, 1.0), "part1")
all_verts = []
for obj in [o0, o1]:
    for v in obj.data.vertices: all_verts.append(obj.matrix_world @ v.co)
if all_verts:
    bbox_min = mathutils.Vector((min(v.x for v in all_verts), min(v.y for v in all_verts), min(v.z for v in all_verts)))
    bbox_max = mathutils.Vector((max(v.x for v in all_verts), max(v.y for v in all_verts), max(v.z for v in all_verts)))
    center = (bbox_min + bbox_max) / 2
    size = max((bbox_max - bbox_min).x, (bbox_max - bbox_min).y, (bbox_max - bbox_min).z)
    scale = 2.0 / size if size > 0 else 1.0
    for obj in [o0, o1]:
        obj.location -= center
        obj.scale *= scale
cam_data = bpy.data.cameras.new("Camera")
cam_data.lens = 50
cam = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam
cam.location = (2.5, -2.5, 2.0)
cam.rotation_euler = (math.radians(60), 0, math.radians(45))
light_data = bpy.data.lights.new("Light", type='SUN')
light_data.energy = 3
light = bpy.data.objects.new("Light", light_data)
bpy.context.scene.collection.objects.link(light)
light.location = (3, -3, 5)
light.rotation_euler = (math.radians(30), math.radians(15), 0)
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
bg.inputs["Color"].default_value = (0.95, 0.95, 0.95, 1.0)
bg.inputs["Strength"].default_value = 0.5
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.samples = 64
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.film_transparent = False
scene.render.filepath = output_png
scene.render.image_settings.file_format = 'PNG'
bpy.ops.render.render(write_still=True)
'''


def render_obj(p0, p1, out_png):
    if os.path.isfile(out_png):
        return True
    if not os.path.isfile(p0) or not os.path.isfile(p1):
        return False
    script_path = "/tmp/pp_render_grid.py"
    with open(script_path, "w") as f:
        f.write(BLENDER_SCRIPT)
    subprocess.run([BLENDER, "--background", "--python", script_path, "--", p0, p1, out_png],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
    return os.path.isfile(out_png)


def get_font(size):
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    return ImageFont.load_default()


def add_label(img, label, font_size=36):
    font = get_font(font_size)
    w, h = img.size
    label_h = font_size + 20
    new_img = Image.new("RGB", (w, h + label_h), (255, 255, 255))
    new_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(new_img)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    tx = (w - tw) // 2
    ty = h + 5
    draw.text((tx, ty), label, fill=(0, 0, 0), font=font)
    return new_img


def discover_tags(seed_val):
    """Discover which tags were generated for this seed."""
    # Check the cfg=5.0 dir (always exists)
    cfg5_dir = os.path.join(BASE, f"infer_seed{seed_val}_cfg5.0")
    step_dirs = glob.glob(os.path.join(cfg5_dir, "step_*"))
    if not step_dirs:
        return [], ""
    step_dir = step_dirs[0]
    step_name = os.path.basename(step_dir)

    objs = glob.glob(os.path.join(step_dir, "*_pred_p0.obj"))
    tags = []
    for obj_path in sorted(objs):
        name = os.path.basename(obj_path)
        tag = name.replace("_pred_p0.obj", "")
        # Parse display name: faucet_9_senior_0 -> Faucet (9/senior_0)
        parts = tag.split("_", 2)
        cat = parts[0].capitalize()
        rest = tag[len(parts[0])+1:]
        display = f"{cat} ({rest})"
        tags.append((tag, display))
    return tags, step_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    tags, step_name = discover_tags(args.seed)
    if not tags:
        print(f"No results found for seed={args.seed}")
        return

    print(f"Seed {args.seed}, {step_name}, {len(tags)} objects")

    out_dir = os.path.join(BASE, f"cfg_grid_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    cell_images = []

    for tag, display_name in tags:
        row = []
        print(f"  {display_name}...", flush=True)

        parts = tag.split("_", 2)
        cat = parts[0]
        obj_seed = parts[1]
        animode = "_".join(tag.split("_")[2:])

        for cfg in CFG_SCALES:
            infer_dir = os.path.join(BASE, f"infer_seed{args.seed}_cfg{cfg}", step_name)

            p0 = os.path.join(infer_dir, f"{tag}_pred_p0.obj")
            p1 = os.path.join(infer_dir, f"{tag}_pred_p1.obj")
            png = os.path.join(out_dir, f"{tag}_cfg{cfg}.png")

            ok = render_obj(p0, p1, png)
            if ok:
                img = Image.open(png)
                img = add_label(img, f"CFG = {cfg}")
                row.append(img)
            else:
                img = Image.new("RGB", (512, 512 + 56), (200, 200, 200))
                row.append(img)

        # GT
        gt_p0 = os.path.join(PRECOMPUTE, cat, obj_seed, animode, "part0.obj")
        gt_p1 = os.path.join(PRECOMPUTE, cat, obj_seed, animode, "part1.obj")
        gt_png = os.path.join(out_dir, f"{tag}_gt.png")
        ok = render_obj(gt_p0, gt_p1, gt_png)
        if ok:
            img = Image.open(gt_png)
            img = add_label(img, "Ground Truth")
            row.append(img)
        else:
            img = Image.new("RGB", (512, 512 + 56), (200, 200, 200))
            row.append(img)

        cell_images.append((display_name, row))

    # Assemble grid
    n_cols = len(CFG_SCALES) + 1
    cell_w = cell_images[0][1][0].size[0]
    cell_h = cell_images[0][1][0].size[1]
    row_label_w = 350
    padding = 4

    total_w = row_label_w + n_cols * (cell_w + padding) + padding
    total_h = len(cell_images) * (cell_h + padding) + padding + 60  # extra for title

    grid = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    # Title
    title_font = get_font(44)
    title = f"CFG Scale Comparison | {step_name} | seed={args.seed}"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    tw = bbox[2] - bbox[0]
    draw.text(((total_w - tw) // 2, 8), title, fill=(0, 0, 0), font=title_font)

    label_font = get_font(34)
    y_offset = 60

    for ri, (display_name, row) in enumerate(cell_images):
        y = y_offset + padding + ri * (cell_h + padding)
        # Row label
        bbox = draw.textbbox((0, 0), display_name, font=label_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (row_label_w - tw) // 2
        ty = y + (cell_h - th) // 2
        draw.text((tx, ty), display_name, fill=(0, 0, 0), font=label_font)
        # Cells
        for ci, img in enumerate(row):
            x = row_label_w + padding + ci * (cell_w + padding)
            grid.paste(img, (x, y))

    out_path = args.output or os.path.join(out_dir, "cfg_comparison_grid.png")
    grid.save(out_path, quality=95)
    print(f"\nSaved: {out_path} ({grid.size[0]}x{grid.size[1]})")


if __name__ == "__main__":
    main()
