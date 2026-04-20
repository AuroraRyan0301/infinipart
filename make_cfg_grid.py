#!/usr/bin/env python3
"""
Render all cfg variants + GT, add labels, assemble into presentation grid.
Layout: rows = objects, columns = cfg values + GT
"""
import subprocess, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BLENDER = "/mnt/cpfs/yurh/blender-4.2.18-linux-x64/blender"
BASE = "/mnt/cpfs/yurh/Infinigen-Sim/output"
PRECOMPUTE = "/mnt/data/yurh/Infinigen-Sim/data_ssd/precompute_solidified"

CFG_SCALES = [0.0, 1.0, 2.0, 4.0, 5.0, 7.9, 11.0]
TAGS = [
    ("dishwasher_0_senior_1", "Dishwasher"),
    ("cabinet_2_senior_0", "Cabinet"),
    ("oven_7_senior_1", "Oven"),
    ("faucet_9_senior_0", "Faucet"),
]

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


def add_label(img, label, font_size=36):
    """Add text label below image, return new image."""
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()

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


def main():
    out_dir = os.path.join(BASE, "cfg_grid")
    os.makedirs(out_dir, exist_ok=True)

    cell_images = []  # list of rows, each row is list of PIL images

    for tag, display_name in TAGS:
        row = []
        print(f"Processing {display_name}...")

        # Parse seed/animode for GT
        parts = tag.split("_", 2)
        cat = parts[0]
        seed = parts[1]
        animode = "_".join(tag.split("_")[2:])

        for cfg in CFG_SCALES:
            cfg_str = f"{cfg}" if cfg != int(cfg) else f"{int(cfg)}.0"
            infer_dir = os.path.join(BASE, f"infer_cfg{cfg}", f"step_69000")
            if cfg == 5.0:
                infer_dir = os.path.join(BASE, "infer_quick", "step_69000")
            elif cfg == 0.0:
                infer_dir = os.path.join(BASE, "infer_cfg0", "step_69000")

            p0 = os.path.join(infer_dir, f"{tag}_pred_p0.obj")
            p1 = os.path.join(infer_dir, f"{tag}_pred_p1.obj")
            png = os.path.join(out_dir, f"{tag}_cfg{cfg}.png")

            ok = render_obj(p0, p1, png)
            if ok:
                img = Image.open(png)
                label = f"CFG = {cfg}"
                img = add_label(img, label)
                row.append(img)
            else:
                # placeholder
                img = Image.new("RGB", (512, 512), (200, 200, 200))
                img = add_label(img, f"CFG={cfg} N/A")
                row.append(img)

        # GT
        gt_p0 = os.path.join(PRECOMPUTE, cat, seed, animode, "part0.obj")
        gt_p1 = os.path.join(PRECOMPUTE, cat, seed, animode, "part1.obj")
        gt_png = os.path.join(out_dir, f"{tag}_gt.png")
        ok = render_obj(gt_p0, gt_p1, gt_png)
        if ok:
            img = Image.open(gt_png)
            img = add_label(img, "Ground Truth")
            row.append(img)
        else:
            img = Image.new("RGB", (512, 512), (200, 200, 200))
            img = add_label(img, "GT N/A")
            row.append(img)

        cell_images.append((display_name, row))

    # Assemble grid
    n_cols = len(CFG_SCALES) + 1  # cfg variants + GT
    cell_w = cell_images[0][1][0].size[0]
    cell_h = cell_images[0][1][0].size[1]
    row_label_w = 300
    padding = 4

    total_w = row_label_w + n_cols * (cell_w + padding) + padding
    total_h = len(cell_images) * (cell_h + padding) + padding

    grid = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        label_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40)
    except:
        try:
            label_font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf", 40)
        except:
            label_font = ImageFont.load_default()

    for ri, (display_name, row) in enumerate(cell_images):
        y = padding + ri * (cell_h + padding)
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

    out_path = os.path.join(out_dir, "cfg_comparison_grid.png")
    grid.save(out_path, quality=95)
    print(f"\nSaved: {out_path}")
    print(f"Size: {grid.size}")


if __name__ == "__main__":
    main()
