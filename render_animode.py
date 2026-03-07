#!/usr/bin/env python3
"""
Blender render script for Infinigen-Sim precomputed animodes.

Reads metadata.json from precompute pipeline, loads URDF + OBJs,
applies forward kinematics per frame, renders 32 camera views.

Run with:
  CUDA_VISIBLE_DEVICES=2 /mnt/data/yurh/blender-4.2.18-linux-x64/blender \
      --background --python render_animode.py -- \
      --metadata ./output/dishwasher/1001/metadata.json \
      --animode basic_0 --views hemi --resolution 512 --duration 4
"""

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque

import numpy as np
import bpy
from mathutils import Matrix, Vector, Euler

# ======================================================================
# Constants
# ======================================================================

ENVMAP_DIR = "/mnt/data/yurh/dataset3D/envmap/indoor"
FFMPEG_BIN = "/usr/local/bin/ffmpeg"

# Static-skip detection
PROBE_RESOLUTION   = 64    # probe render resolution (px)
PROBE_SAMPLE_COUNT = 1     # Cycles samples for probe (deterministic w/ fixed seed)
PROBE_FRAME_COUNT  = 4     # probe frames per view (evenly spaced)
STATIC_SKIP_M1_FILE    = "static_skip_method1.json"   # joint-value check record
STATIC_SKIP_PROBE_FILE = "static_skip_probe.json"     # probe render check record
PROBE_SAVE_VIEW_KEYS   = ["hemi_00", "hemi_05", "hemi_11"]  # 3 views saved for inspection

# PhysXNet / PhysXMobility JSON data paths
PHYSXNET_JSON_DIR = "/mnt/data/fulian/dataset/PhysXNet/version_1/finaljson"
PHYSXMOB_JSON_DIR = "/mnt/data/fulian/dataset/PhysX_mobility/finaljson"

# Material source data paths
PARTNET_BASE = "/mnt/data/yurh/dataset3D/Partnet"
SHAPENET_BASE = "/mnt/data/yurh/dataset3D/ShapeNetCore"
OVERLAP_MAP_PATH = "/mnt/data/yurh/infinipart/physxnet_partnet_overlap.json"
PBR_TEXTURES_DIR = "/mnt/data/yurh/infinipart/pbr_textures"

# PartNet model_cat -> ShapeNet synset_id mapping
PARTNET_TO_SYNSET = {
    "StorageFurniture": "02933112",
    "Table": "04379243",
    "Faucet": "03325088",
    "Chair": "03001627",
    "Bottle": "02876657",
    "Laptop": "03642806",
    "Dishwasher": "03207941",
    "Lamp": "03636649",
    "Keyboard": "03085013",
    "Display": "03211117",
    "Clock": "03046257",
    "TrashCan": "02747177",
    "Microwave": "03761084",
    "Oven": "04330267",
    "KitchenPot": "03991062",
    "Refrigerator": "04074963",
    "Scissors": "04225987",
    "Door": "03691459",
    "Box": "02801938",
    "Kettle": "03991062",
}

# Material keyword -> PBR texture category (checked in order, first match wins)
MATERIAL_KEYWORD_MAP = [
    ("stainless", "metal"), ("steel", "metal"), ("iron", "metal"),
    ("aluminum", "metal"), ("aluminium", "metal"), ("chrome", "metal"),
    ("copper", "metal"), ("brass", "metal"), ("bronze", "metal"),
    ("titanium", "metal"), ("gold", "metal"), ("silver", "metal"),
    ("zinc", "metal"), ("nickel", "metal"), ("metal", "metal"),
    ("glass", "glass"),
    ("bamboo", "wood"), ("plywood", "wood"), ("mdf", "wood"),
    ("oak", "wood"), ("walnut", "wood"), ("laminate", "wood"),
    ("wood", "wood"), ("bentwood", "wood"),
    ("ceramic", "ceramic"), ("porcelain", "ceramic"),
    ("marble", "marble"), ("granite", "stone"), ("stone", "stone"),
    ("concrete", "concrete"),
    ("leather", "leather"), ("leatherette", "leather"),
    ("fabric", "fabric"), ("cotton", "fabric"), ("polyester", "fabric"),
    ("nylon", "fabric"), ("felt", "fabric"), ("silk", "fabric"),
    ("mesh", "fabric"), ("canvas", "fabric"), ("foam", "fabric"),
    ("rubber", "rubber"), ("tpu", "rubber"), ("silicone", "rubber"),
    ("paper", "paper"), ("cardboard", "paper"), ("cork", "paper"),
    ("abs", "plastic"), ("polycarbonate", "plastic"),
    ("polypropylene", "plastic"), ("pvc", "plastic"),
    ("acrylic", "plastic"), ("resin", "plastic"), ("plastic", "plastic"),
    ("vinyl", "plastic"), ("fiberglass", "plastic"),
]

# Fallback PBR values per texture category
CATEGORY_DEFAULTS = {
    "wood":     {"metallic": 0.00, "roughness": 0.55, "color": (0.55, 0.35, 0.20)},
    "metal":    {"metallic": 0.90, "roughness": 0.25, "color": (0.60, 0.60, 0.60)},
    "plastic":  {"metallic": 0.00, "roughness": 0.40, "color": (0.50, 0.50, 0.50)},
    "glass":    {"metallic": 0.00, "roughness": 0.05, "color": (0.90, 0.92, 0.95)},
    "ceramic":  {"metallic": 0.00, "roughness": 0.35, "color": (0.85, 0.82, 0.78)},
    "marble":   {"metallic": 0.00, "roughness": 0.20, "color": (0.90, 0.88, 0.85)},
    "stone":    {"metallic": 0.00, "roughness": 0.55, "color": (0.50, 0.48, 0.45)},
    "concrete": {"metallic": 0.00, "roughness": 0.70, "color": (0.60, 0.58, 0.55)},
    "leather":  {"metallic": 0.00, "roughness": 0.60, "color": (0.30, 0.18, 0.10)},
    "fabric":   {"metallic": 0.00, "roughness": 0.80, "color": (0.55, 0.50, 0.45)},
    "rubber":   {"metallic": 0.00, "roughness": 0.80, "color": (0.15, 0.15, 0.15)},
    "paper":    {"metallic": 0.00, "roughness": 0.85, "color": (0.90, 0.88, 0.83)},
}

# PBR material properties: material_name -> (metallic, roughness, color_rgb)
MATERIAL_PBR = {
    # Metals
    "Steel":            (0.95, 0.25, (0.56, 0.57, 0.58)),
    "Stainless Steel":  (0.95, 0.20, (0.63, 0.64, 0.65)),
    "Iron":             (0.90, 0.35, (0.50, 0.50, 0.50)),
    "Cast Iron":        (0.85, 0.45, (0.35, 0.35, 0.35)),
    "Aluminum":         (0.90, 0.20, (0.77, 0.78, 0.78)),
    "Copper":           (0.95, 0.25, (0.72, 0.45, 0.20)),
    "Brass":            (0.90, 0.25, (0.78, 0.67, 0.35)),
    "Bronze":           (0.85, 0.30, (0.55, 0.35, 0.17)),
    "Chrome":           (1.00, 0.05, (0.55, 0.56, 0.56)),
    "Metal":            (0.85, 0.30, (0.60, 0.60, 0.60)),
    "Metal Alloy":      (0.85, 0.30, (0.58, 0.58, 0.60)),
    # Plastics
    "Plastic":          (0.00, 0.40, (0.60, 0.60, 0.60)),
    "ABS Plastic":      (0.00, 0.35, (0.15, 0.15, 0.15)),
    "Polycarbonate":    (0.00, 0.20, (0.85, 0.85, 0.85)),
    "Polypropylene":    (0.00, 0.50, (0.75, 0.75, 0.75)),
    "PVC":              (0.00, 0.45, (0.70, 0.70, 0.70)),
    "Nylon":            (0.00, 0.55, (0.80, 0.78, 0.75)),
    "Acrylic":          (0.00, 0.10, (0.90, 0.90, 0.90)),
    "Silicone":         (0.00, 0.70, (0.70, 0.68, 0.65)),
    "Resin":            (0.05, 0.30, (0.80, 0.75, 0.65)),
    # Glass/Ceramic
    "Glass":            (0.00, 0.05, (0.90, 0.92, 0.95)),
    "Tempered Glass":   (0.00, 0.05, (0.88, 0.90, 0.93)),
    "Ceramic":          (0.00, 0.40, (0.85, 0.82, 0.78)),
    "Porcelain":        (0.00, 0.15, (0.95, 0.93, 0.90)),
    # Wood
    "Wood":             (0.00, 0.55, (0.55, 0.35, 0.20)),
    "Bamboo":           (0.00, 0.50, (0.70, 0.60, 0.35)),
    "Plywood":          (0.00, 0.60, (0.65, 0.50, 0.30)),
    "MDF":              (0.00, 0.65, (0.60, 0.50, 0.35)),
    "Oak":              (0.00, 0.50, (0.60, 0.40, 0.22)),
    # Fabric/Leather
    "Fabric":           (0.00, 0.80, (0.55, 0.50, 0.45)),
    "Leather":          (0.00, 0.60, (0.30, 0.18, 0.10)),
    "Foam":             (0.00, 0.85, (0.75, 0.72, 0.68)),
    # Rubber
    "Rubber":           (0.00, 0.80, (0.15, 0.15, 0.15)),
    # Stone/Concrete
    "Stone":            (0.00, 0.60, (0.55, 0.52, 0.48)),
    "Marble":           (0.00, 0.20, (0.90, 0.88, 0.85)),
    "Concrete":         (0.00, 0.70, (0.60, 0.58, 0.55)),
    # Special
    "Carbon Fiber":     (0.30, 0.20, (0.15, 0.15, 0.15)),
    "Paper":            (0.00, 0.85, (0.90, 0.88, 0.83)),
}



def get_pbr_for_material(material_name):
    """Look up PBR (metallic, roughness, color) for a material name."""
    if material_name in MATERIAL_PBR:
        return MATERIAL_PBR[material_name]
    mat_lower = material_name.lower()
    for key, props in MATERIAL_PBR.items():
        if key.lower() in mat_lower:
            return props
    return (0.10, 0.45, (0.60, 0.60, 0.60))


def classify_material(material_name):
    """Classify material name into a PBR texture category.
    Returns (category_str, is_composite_bool).
    """
    mat_lower = material_name.lower()
    is_composite = any(sep in mat_lower for sep in
                       [" and ", " with ", " over ", "/", "+",
                        " or ", "-covered", "-coated"])
    if is_composite:
        best_pos, best_cat = len(mat_lower), None
        for keyword, category in MATERIAL_KEYWORD_MAP:
            pos = mat_lower.find(keyword)
            if 0 <= pos < best_pos:
                best_pos = pos
                best_cat = category
        return (best_cat or "plastic"), True
    for keyword, category in MATERIAL_KEYWORD_MAP:
        if keyword in mat_lower:
            return category, False
    return "plastic", False


# ======================================================================
# Overlap map and ShapeNet lookup (cached)
# ======================================================================

_overlap_map = None


def _load_overlap_map():
    """Load PhysXNet -> PartNet overlap mapping (cached)."""
    global _overlap_map
    if _overlap_map is not None:
        return _overlap_map
    if os.path.exists(OVERLAP_MAP_PATH):
        with open(OVERLAP_MAP_PATH) as f:
            _overlap_map = json.load(f)
    else:
        _overlap_map = {}
    return _overlap_map


def get_shapenet_model_dir(obj_id):
    """Find ShapeNet model directory for a PhysXNet/PartNet object.
    Returns path to model dir (containing model_normalized.obj/mtl) or None.
    """
    overlap = _load_overlap_map()
    entry = overlap.get(str(obj_id))
    if not entry or not entry.get("model_id"):
        return None

    model_id = entry["model_id"]
    model_cat = entry.get("model_cat", "")
    synset_id = PARTNET_TO_SYNSET.get(model_cat)
    if not synset_id:
        return None

    synset_dir = os.path.join(SHAPENET_BASE, synset_id)
    if not os.path.isdir(synset_dir):
        # Auto-extract zip if available
        zip_path = synset_dir + ".zip"
        if os.path.exists(zip_path):
            import zipfile
            print(f"  Auto-extracting {zip_path} ...")
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(SHAPENET_BASE)

    model_dir = os.path.join(synset_dir, model_id, "models")
    if os.path.isdir(model_dir):
        return model_dir
    return None


def parse_shapenet_mtl(mtl_path):
    """Parse ShapeNet MTL file. Returns list of material dicts with Kd colors
    and optional texture paths.
    """
    materials = []
    current = None
    mtl_dir = os.path.dirname(mtl_path)

    try:
        with open(mtl_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("newmtl "):
                    if current is not None:
                        materials.append(current)
                    current = {"name": line[7:], "Kd": None, "map_Kd": None, "d": 1.0}
                elif current is not None:
                    if line.startswith("Kd "):
                        parts = line.split()
                        if len(parts) >= 4:
                            current["Kd"] = (float(parts[1]), float(parts[2]), float(parts[3]))
                    elif line.startswith("map_Kd "):
                        tex_ref = line[7:].strip()
                        tex_path = os.path.normpath(os.path.join(mtl_dir, tex_ref))
                        if os.path.exists(tex_path):
                            current["map_Kd"] = tex_path
                    elif line.startswith("d "):
                        try:
                            current["d"] = float(line.split()[1])
                        except (IndexError, ValueError):
                            pass
        if current is not None:
            materials.append(current)
    except (IOError, OSError):
        pass

    return materials


def get_shapenet_colors(obj_id):
    """Get colors from ShapeNet model for a PhysXNet overlap object.
    Returns list of (r, g, b) colors (one per material, excluding transparent)
    or None if not available.
    """
    model_dir = get_shapenet_model_dir(obj_id)
    if model_dir is None:
        return None

    mtl_path = os.path.join(model_dir, "model_normalized.mtl")
    if not os.path.exists(mtl_path):
        return None

    materials = parse_shapenet_mtl(mtl_path)
    if not materials:
        return None

    colors = []
    for mat in materials:
        if mat.get("d", 1.0) < 0.3:
            continue  # skip highly transparent materials
        if mat.get("map_Kd"):
            # Compute average color from texture
            avg = _texture_average_color(mat["map_Kd"])
            if avg:
                colors.append(avg)
                continue
        if mat.get("Kd"):
            kd = mat["Kd"]
            if not (kd[0] < 0.01 and kd[1] < 0.01 and kd[2] < 0.01):
                colors.append(kd)
            elif not (kd[0] > 0.99 and kd[1] > 0.99 and kd[2] > 0.99):
                colors.append(kd)

    return colors if colors else None


def _texture_average_color(tex_path):
    """Compute average RGB color of a texture image. Returns (r, g, b) or None."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(tex_path).convert("RGB")
        arr = np.array(img, dtype=float) / 255.0
        return tuple(arr.mean(axis=(0, 1)))
    except Exception:
        return None


def get_shapenet_textures(obj_id):
    """Get ShapeNet texture file paths for a PhysXNet overlap object.
    Returns dict {material_name: texture_path} or None.
    """
    model_dir = get_shapenet_model_dir(obj_id)
    if model_dir is None:
        return None

    mtl_path = os.path.join(model_dir, "model_normalized.mtl")
    if not os.path.exists(mtl_path):
        return None

    materials = parse_shapenet_mtl(mtl_path)
    textures = {}
    for mat in materials:
        if mat.get("map_Kd"):
            textures[mat["name"]] = mat["map_Kd"]

    return textures if textures else None


def get_partnet_colors(obj_id):
    """Extract colors from PartNet textured_objs MTL files.
    Returns {part_name: (r, g, b)} or None.
    """
    pdir = os.path.join(PARTNET_BASE, str(obj_id))
    tex_dir = os.path.join(pdir, "textured_objs")
    if not os.path.isdir(tex_dir):
        return None

    colors = {}
    for fn in os.listdir(tex_dir):
        if not fn.endswith('.mtl'):
            continue
        mtl_path = os.path.join(tex_dir, fn)
        part_name = fn[:-4]
        try:
            with open(mtl_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("Kd "):
                        vals = line.split()
                        if len(vals) >= 4:
                            r, g, b = float(vals[1]), float(vals[2]), float(vals[3])
                            if part_name not in colors:
                                colors[part_name] = (r, g, b)
                    elif line.startswith("map_Kd ") and part_name not in colors:
                        tex_ref = line.split(None, 1)[1]
                        tex_path = os.path.join(tex_dir, tex_ref)
                        if os.path.exists(tex_path):
                            avg = _texture_average_color(tex_path)
                            if avg:
                                colors[part_name] = avg
        except (IOError, ValueError):
            pass

    return colors if colors else None


# ======================================================================
# ambientCG PBR texture sets (cached discovery)
# ======================================================================

_texture_cache = {}


def _discover_texture_sets():
    """Scan pbr_textures/ directory and build cache of available texture sets."""
    global _texture_cache
    if _texture_cache:
        return _texture_cache
    if not os.path.isdir(PBR_TEXTURES_DIR):
        return {}

    for category in os.listdir(PBR_TEXTURES_DIR):
        cat_dir = os.path.join(PBR_TEXTURES_DIR, category)
        if not os.path.isdir(cat_dir):
            continue
        sets = []
        for tex_id in os.listdir(cat_dir):
            tex_dir = os.path.join(cat_dir, tex_id)
            if not os.path.isdir(tex_dir):
                continue
            tex_set = {"id": tex_id, "dir": tex_dir}
            for fn in os.listdir(tex_dir):
                fn_lower = fn.lower()
                full = os.path.join(tex_dir, fn)
                if "_color" in fn_lower and fn_lower.endswith(".jpg"):
                    tex_set["color"] = full
                elif "_roughness" in fn_lower and fn_lower.endswith(".jpg"):
                    tex_set["roughness"] = full
                elif "_normalgl" in fn_lower and fn_lower.endswith(".jpg"):
                    tex_set["normal"] = full
                elif "_metalness" in fn_lower and fn_lower.endswith(".jpg"):
                    tex_set["metallic"] = full
            if "color" in tex_set:
                sets.append(tex_set)
        if sets:
            _texture_cache[category] = sets

    return _texture_cache


def get_texture_set(category, seed_hash=0):
    """Get an ambientCG PBR texture set for a category. Returns dict or None."""
    cache = _discover_texture_sets()
    sets = cache.get(category, [])
    if not sets:
        return None
    return sets[seed_hash % len(sets)]


# ======================================================================
# Material application (Blender node tree)
# ======================================================================

def apply_textured_material(obj, base_color, metallic, roughness, category,
                            tex_set, part_id, factory_name, seed):
    """Apply PBR material with box-projected ambientCG textures + color tint.

    Creates a proper node tree: TexCoord -> Mapping -> Image Textures -> BSDF.
    """
    mat_name = f"pbr_{category}_{part_id}_{factory_name}_{seed}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nlinks = mat.node_tree.links
    nodes.clear()

    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (800, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (400, 0)
    nlinks.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Texture coordinate -> mapping for box projection
    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-600, 0)
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-400, 0)
    nlinks.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])
    mapping.inputs["Scale"].default_value = (2.0, 2.0, 2.0)

    # Color texture with base_color tint
    if "color" in tex_set:
        color_tex = nodes.new("ShaderNodeTexImage")
        color_tex.location = (-100, 200)
        color_tex.projection = "BOX"
        color_tex.projection_blend = 0.3
        try:
            color_tex.image = bpy.data.images.load(tex_set["color"], check_existing=True)
        except RuntimeError:
            color_tex.image = None
        nlinks.new(mapping.outputs["Vector"], color_tex.inputs["Vector"])

        # Mix texture with base color tint (60% texture, 40% tint)
        mix_color = nodes.new("ShaderNodeMix")
        mix_color.data_type = 'RGBA'
        mix_color.location = (200, 200)
        mix_color.inputs[0].default_value = 0.6
        mix_color.inputs[6].default_value = (*base_color, 1.0)
        nlinks.new(color_tex.outputs["Color"], mix_color.inputs[7])
        nlinks.new(mix_color.outputs[2], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)

    # Roughness texture
    if "roughness" in tex_set:
        rough_tex = nodes.new("ShaderNodeTexImage")
        rough_tex.location = (-100, -100)
        rough_tex.projection = "BOX"
        rough_tex.projection_blend = 0.3
        try:
            rough_tex.image = bpy.data.images.load(tex_set["roughness"], check_existing=True)
            rough_tex.image.colorspace_settings.name = "Non-Color"
        except RuntimeError:
            rough_tex.image = None
        nlinks.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
        nlinks.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = roughness

    # Normal map
    if "normal" in tex_set:
        normal_tex = nodes.new("ShaderNodeTexImage")
        normal_tex.location = (-100, -400)
        normal_tex.projection = "BOX"
        normal_tex.projection_blend = 0.3
        try:
            normal_tex.image = bpy.data.images.load(tex_set["normal"], check_existing=True)
            normal_tex.image.colorspace_settings.name = "Non-Color"
        except RuntimeError:
            normal_tex.image = None
        nlinks.new(mapping.outputs["Vector"], normal_tex.inputs["Vector"])
        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.location = (200, -400)
        normal_map.inputs["Strength"].default_value = 0.5
        nlinks.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
        nlinks.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    # Metallic
    if "metallic" in tex_set:
        metal_tex = nodes.new("ShaderNodeTexImage")
        metal_tex.location = (-100, -700)
        metal_tex.projection = "BOX"
        metal_tex.projection_blend = 0.3
        try:
            metal_tex.image = bpy.data.images.load(tex_set["metallic"], check_existing=True)
            metal_tex.image.colorspace_settings.name = "Non-Color"
        except RuntimeError:
            metal_tex.image = None
        nlinks.new(mapping.outputs["Vector"], metal_tex.inputs["Vector"])
        nlinks.new(metal_tex.outputs["Color"], bsdf.inputs["Metallic"])
    else:
        bsdf.inputs["Metallic"].default_value = metallic

    # Glass special case
    if category == "glass":
        bsdf.inputs["Transmission"].default_value = 0.8
        bsdf.inputs["IOR"].default_value = 1.45
        bsdf.inputs["Roughness"].default_value = 0.05

    obj.data.materials.clear()
    obj.data.materials.append(mat)


# ======================================================================
# Full realistic material pipeline
# ======================================================================

def load_physx_json(metadata):
    """Load PhysXNet/PhysXMobility JSON for material data. Returns None if N/A."""
    factory = metadata.get("factory", "")
    identifier = metadata.get("identifier", "")
    if "PhysXNet" in factory:
        json_path = os.path.join(PHYSXNET_JSON_DIR, f"{identifier}.json")
    elif "PhysXMobility" in factory or "PhysXmobility" in factory:
        json_path = os.path.join(PHYSXMOB_JSON_DIR, f"{identifier}.json")
    else:
        return None
    if os.path.exists(json_path):
        with open(json_path) as f:
            return json.load(f)
    return None


def get_realistic_materials(metadata, links):
    """Get per-link realistic material properties using 3-tier priority:
    1. ShapeNet textures/colors (for overlap PhysXNet objects)
    2. PartNet textured_objs colors (for PhysXMobility and overlap objects)
    3. JSON material name -> classify -> ambientCG PBR texture -> flat PBR fallback

    Returns {link_name: material_info_dict} where material_info_dict has:
        metallic, roughness, color, category, tex_set, source
    """
    result = {}
    json_data = load_physx_json(metadata)
    factory = metadata.get("factory", "")
    identifier = metadata.get("identifier", "")
    # Use hashlib for deterministic seed across processes (Python hash() is randomized)
    import hashlib
    _det_seed = int(hashlib.md5(f"{factory}_{identifier}".encode()).hexdigest(), 16) % (2**31)
    mat_rng = random.Random(_det_seed)

    # Check if textured_objs exist (PhysXMobility native materials)
    scene_dir = metadata.get("scene_dir", "")
    has_textured_objs = os.path.isdir(os.path.join(scene_dir, "textured_objs"))

    if json_data is not None and has_textured_objs:
        # PhysXMobility: native materials from textured_objs OBJ+MTL
        # Just provide PBR enhancement params; actual colors come from imports
        parts_info = json_data.get("parts", [])
        label_to_material = {}
        for p in parts_info:
            label = p.get("label")
            if label is not None:
                label_to_material[label] = p.get("material", "Unknown")

        group_info = json_data.get("group_info", {})
        group_to_labels = {}
        for gid_str, val in group_info.items():
            gid = int(gid_str)
            if isinstance(val, list):
                if (len(val) >= 4 and isinstance(val[-1], str)
                        and val[-1] in ('A', 'B', 'C', 'D', 'CB')):
                    labels = val[0] if isinstance(val[0], list) else [val[0]]
                    group_to_labels[gid] = labels
                else:
                    group_to_labels[gid] = [x for x in val if isinstance(x, int)]

        for link_name, link_info in links.items():
            idx = link_info["part_idx"]
            part_labels = group_to_labels.get(idx, [idx])
            mat_name = "Unknown"
            for lbl in part_labels:
                if lbl in label_to_material:
                    mat_name = label_to_material[lbl]
                    break

            metallic, roughness, _ = get_pbr_for_material(mat_name)
            category, _ = classify_material(mat_name)

            result[link_name] = {
                "metallic": metallic,
                "roughness": roughness,
                "color": None,  # use native OBJ colors
                "category": category,
                "tex_set": None,
                "source": "native",
            }

        if result:
            print(f"  Materials: {len(result)} parts, source={{'native'}}")

    elif json_data is not None:
        # PhysXNet: use ShapeNet/PartNet colors + ambientCG textures
        parts_info = json_data.get("parts", [])
        label_to_material = {}
        for p in parts_info:
            label = p.get("label")
            if label is not None:
                label_to_material[label] = p.get("material", "Unknown")

        group_info = json_data.get("group_info", {})
        group_to_labels = {}
        for gid_str, val in group_info.items():
            gid = int(gid_str)
            if isinstance(val, list):
                if (len(val) >= 4 and isinstance(val[-1], str)
                        and val[-1] in ('A', 'B', 'C', 'D', 'CB')):
                    labels = val[0] if isinstance(val[0], list) else [val[0]]
                    group_to_labels[gid] = labels
                else:
                    group_to_labels[gid] = [x for x in val if isinstance(x, int)]

        # Tier 1: ShapeNet colors (for overlap objects)
        shapenet_colors = get_shapenet_colors(identifier)
        shapenet_avg = None
        if shapenet_colors:
            r_sum = sum(c[0] for c in shapenet_colors)
            g_sum = sum(c[1] for c in shapenet_colors)
            b_sum = sum(c[2] for c in shapenet_colors)
            n = len(shapenet_colors)
            shapenet_avg = (r_sum / n, g_sum / n, b_sum / n)

        # Tier 2: PartNet colors (for overlap objects)
        partnet_colors = get_partnet_colors(identifier)
        partnet_avg = None
        if partnet_colors:
            r_sum = sum(c[0] for c in partnet_colors.values())
            g_sum = sum(c[1] for c in partnet_colors.values())
            b_sum = sum(c[2] for c in partnet_colors.values())
            n = len(partnet_colors)
            partnet_avg = (r_sum / n, g_sum / n, b_sum / n)

        color_tint = shapenet_avg or partnet_avg
        source = "shapenet" if shapenet_avg else ("partnet" if partnet_avg else "pbr")

        for link_name, link_info in links.items():
            idx = link_info["part_idx"]
            part_labels = group_to_labels.get(idx, [idx])
            mat_name = "Unknown"
            for lbl in part_labels:
                if lbl in label_to_material:
                    mat_name = label_to_material[lbl]
                    break

            # Get PBR properties from material name
            metallic, roughness, base_color = get_pbr_for_material(mat_name)

            # Classify for texture selection
            category, _ = classify_material(mat_name)
            defaults = CATEGORY_DEFAULTS.get(category, CATEGORY_DEFAULTS["plastic"])

            # Apply color tint from ShapeNet/PartNet
            if color_tint:
                base_color = tuple(
                    0.5 * t + 0.5 * b for t, b in zip(color_tint, base_color)
                )

            # Per-part variation
            color = tuple(
                max(0, min(1, c + mat_rng.uniform(-0.04, 0.04)))
                for c in base_color
            )
            metallic = max(0, min(1, metallic + mat_rng.uniform(-0.03, 0.03)))
            roughness = max(0, min(1, roughness + mat_rng.uniform(-0.05, 0.05)))

            # Tier 3: ambientCG texture set
            tex_hash = int(hashlib.md5(f"{factory}_{identifier}_{idx}_{category}".encode()).hexdigest(), 16)
            tex_set = get_texture_set(category, abs(tex_hash))

            result[link_name] = {
                "metallic": metallic,
                "roughness": roughness,
                "color": color,
                "category": category,
                "tex_set": tex_set,
                "source": source,
            }

        if result:
            sources = set(v["source"] for v in result.values())
            print(f"  Materials: {len(result)} parts, source={sources}")

    else:
        # IS factory: OBJs exported with fully baked Infinigen procedural shaders
        # (diffuse + roughness + metallic + normal textures via export.bake_object).
        # All PBR channels are already baked — do NOT touch materials at all.
        for link_name in links:
            result[link_name] = {
                "source": "is_baked",
            }

    return result

# 16 fixed hemisphere views: 4 elevations x 4 azimuths
_HEMI_ELEVS = [5, 25, 45, 65]
_HEMI_AZIMS = [-67.5, -22.5, 22.5, 67.5]
HEMI_VIEWS = {}
for _i, _elev in enumerate(_HEMI_ELEVS):
    for _j, _azim in enumerate(_HEMI_AZIMS):
        HEMI_VIEWS[f"hemi_{_i*4+_j:02d}"] = (_elev, _azim)

# 8 orbit views (moving cameras, 180 degree arcs)
ORBIT_VIEWS = {
    "orbit_00": (10, 180, 10, 0),
    "orbit_01": (10, -180, 10, 0),
    "orbit_02": (30, 180, 30, 0),
    "orbit_03": (30, -180, 30, 0),
    "orbit_04": (50, 150, 15, 0),
    "orbit_05": (50, -150, 15, 0),
    "orbit_06": (15, 170, 50, 0),
    "orbit_07": (15, -170, 50, 0),
}

# 8 sweep views (pans, tilts, diagonal sweeps)
SWEEP_VIEWS = {
    "sweep_00": (15, -80, 15, 80),
    "sweep_01": (35, -75, 35, 75),
    "sweep_02": (55, -60, 55, 60),
    "sweep_03": (25, 80, 25, -80),
    "sweep_04": (5, 0, 70, 0),
    "sweep_05": (5, -45, 65, -45),
    "sweep_06": (5, 45, 65, 45),
    "sweep_07": (10, -60, 55, 60),
}

# "fast" view set: Monte Carlo sampled subset (4 hemi + 2 orbit + 2 sweep)
# Chosen to maximize coverage: spread across elevations and azimuth range
FAST_HEMI_VIEWS = {k: HEMI_VIEWS[k] for k in ["hemi_01", "hemi_06", "hemi_09", "hemi_14"]}
FAST_ORBIT_VIEWS = {k: ORBIT_VIEWS[k] for k in ["orbit_01", "orbit_04"]}
FAST_SWEEP_VIEWS = {k: SWEEP_VIEWS[k] for k in ["sweep_00", "sweep_04"]}

# Named convenience views (static)
NAMED_VIEWS = {
    "front": (25, 0),
    "side": (25, 90),
    "back": (25, 180),
    "threequarter": (25, 45),
}


# ======================================================================
# Argument Parsing
# ======================================================================

def parse_args():
    # Blender passes args after "--"
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render animode videos from precompute metadata")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--animode", default="all", help="Animode name or 'all'")
    parser.add_argument("--views", nargs="+", default=["hemi"],
                        help="View groups: hemi, orbit, sweep, all, front, side, etc.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--cam_distance", type=float, default=4.68,
                        help="Camera distance from origin (default: 4.68, was 3.6)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--envmap", default=None, help="Path to envmap HDR/EXR")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--color_mode", default="both",
                        choices=["realistic", "part", "group", "both"],
                        help="realistic: per-part PBR materials from source data; "
                             "part: binary part0/part1 coloring; "
                             "group: each reduced graph node gets unique color; "
                             "both: realistic + group (default)")
    parser.add_argument("--bg_mode", default="nobg",
                        choices=["nobg", "withbg", "both"],
                        help="nobg: transparent background (default); "
                             "withbg: envmap visible as background; "
                             "both: render both nobg and withbg")
    return parser.parse_args(argv)


# ======================================================================
# URDF Parsing (Blender context, uses mathutils)
# ======================================================================

class Joint:
    """URDF joint info for forward kinematics."""
    def __init__(self, name, jtype, parent_link, child_link,
                 axis, origin_xyz, origin_rpy, lower, upper):
        self.name = name
        self.jtype = jtype
        self.parent_link = parent_link
        self.child_link = child_link
        self.axis = Vector(axis).normalized()
        self.origin_xyz = Vector(origin_xyz)
        self.origin_rpy = origin_rpy  # (r, p, y)
        self.lower = lower
        self.upper = upper

    @property
    def motion_range(self):
        if self.jtype == "continuous":
            return 2.0 * math.pi
        return abs(self.upper - self.lower)


def parse_urdf(urdf_path):
    """Parse URDF into links and joints.

    Returns:
        links: dict {name: {"part_idx": int|None, "mesh_files": [(path, vis_xyz)]}}
        joints: list of Joint
        root_link: str
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    urdf_dir = os.path.dirname(urdf_path)

    links = {}
    for link_el in root.findall("link"):
        name = link_el.get("name")
        part_idx = None
        m = re.match(r"l_(\d+)", name)
        if m:
            part_idx = int(m.group(1))
        else:
            m = re.match(r"link_(\d+)", name)
            if m:
                part_idx = int(m.group(1))

        mesh_files = []
        for visual_el in link_el.findall("visual"):
            geom = visual_el.find("geometry")
            if geom is not None:
                mesh_el = geom.find("mesh")
                if mesh_el is not None:
                    filename = mesh_el.get("filename", "")
                    # Resolve relative path
                    if not os.path.isabs(filename):
                        filename = os.path.join(urdf_dir, filename)
                    vis_origin = visual_el.find("origin")
                    vis_xyz = [0.0, 0.0, 0.0]
                    if vis_origin is not None and vis_origin.get("xyz"):
                        vis_xyz = [float(v) for v in vis_origin.get("xyz").split()]
                    mesh_files.append((filename, vis_xyz))

        links[name] = {"name": name, "part_idx": part_idx, "mesh_files": mesh_files}

    joints = []
    children_set = set()
    for joint_el in root.findall("joint"):
        name = joint_el.get("name", "")
        jtype = joint_el.get("type", "fixed")
        parent_link = joint_el.find("parent").get("link")
        child_link = joint_el.find("child").get("link")
        children_set.add(child_link)

        axis_el = joint_el.find("axis")
        axis = [1.0, 0.0, 0.0]
        if axis_el is not None and axis_el.get("xyz"):
            axis = [float(v) for v in axis_el.get("xyz").split()]

        origin_el = joint_el.find("origin")
        origin_xyz = [0.0, 0.0, 0.0]
        origin_rpy = [0.0, 0.0, 0.0]
        if origin_el is not None:
            if origin_el.get("xyz"):
                origin_xyz = [float(v) for v in origin_el.get("xyz").split()]
            if origin_el.get("rpy"):
                origin_rpy = [float(v) for v in origin_el.get("rpy").split()]

        limit_el = joint_el.find("limit")
        lower, upper = 0.0, 0.0
        if limit_el is not None:
            lower = float(limit_el.get("lower", "0"))
            upper = float(limit_el.get("upper", "0"))

        joints.append(Joint(name, jtype, parent_link, child_link,
                            axis, origin_xyz, origin_rpy, lower, upper))

    all_links = set(links.keys())
    roots = all_links - children_set
    root_link = sorted(roots)[0] if roots else sorted(all_links)[0]

    return links, joints, root_link


def build_kinematic_tree(joints):
    """Build parent/children maps."""
    parent_map = {}
    children_map = defaultdict(list)
    for j in joints:
        parent_map[j.child_link] = (j.parent_link, j)
        children_map[j.parent_link].append((j.child_link, j))
    return parent_map, children_map


# ======================================================================
# Static-Skip Detection
# ======================================================================

def _read_skip_file(meta_dir, filename):
    path = os.path.join(meta_dir, filename)
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def is_recorded_static(meta_dir, animode_name):
    """Return True if animode was previously recorded as static (skip rendering)."""
    for fname in (STATIC_SKIP_M1_FILE, STATIC_SKIP_PROBE_FILE):
        if animode_name in _read_skip_file(meta_dir, fname):
            return True
    return False


def check_static_method1(frame_joint_values):
    """Return True if all frames have identical joint values.

    Happens when BVH collision system fully STOPs the active motion
    at frame 1 (e.g., joint blocked by fixed geometry at rest pose).
    """
    if len(frame_joint_values) < 2:
        return False
    frame1 = frame_joint_values[1]
    for f in range(2, max(frame_joint_values) + 1):
        fvals = frame_joint_values.get(f, {})
        for jn, v1 in frame1.items():
            if abs(fvals.get(jn, v1) - v1) > 1e-4:
                return False
    return True


def _record_skip(meta_dir, filename, animode_name, entry):
    data = _read_skip_file(meta_dir, filename)
    data[animode_name] = entry
    with open(os.path.join(meta_dir, filename), "w") as f:
        json.dump(data, f, indent=2)


def record_static_skip_method1(meta_dir, animode_name):
    _record_skip(meta_dir, STATIC_SKIP_M1_FILE, animode_name,
                 {"reason": "joint_values_identical_across_all_frames"})
    print(f"  [STATIC-M1] {animode_name}: joint values identical → skip")


def record_static_skip_probe(meta_dir, animode_name, saved_views):
    """saved_views: {view_name: [frame_path, ...]}"""
    _record_skip(meta_dir, STATIC_SKIP_PROBE_FILE, animode_name,
                 {"reason": "all_hemi_views_probe_static",
                  "saved_views": saved_views})
    print(f"  [STATIC-PROBE] {animode_name}: all hemi probe frames identical → skip")


def _render_frame_to_file(scene, frame_idx, out_path):
    """Render a single frame to out_path (write_still=True).

    Uses scene.render.filepath; Blender writes PNG directly (no frame padding
    when filepath has no # characters and write_still=True is used for a
    single frame rather than animation).
    """
    scene.frame_current = frame_idx
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    return os.path.exists(out_path)


def _file_md5(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def check_animode_static_probe(scene, num_frames, meta_dir, animode_name,
                               cam_center, cam_distance):
    """Probe-render PROBE_FRAME_COUNT evenly-spaced frames per hemi view.

    With use_animated_seed=False and seed=0, Cycles renders are byte-identical
    for frames with the same geometry.  If all frames in every hemi view have
    the same MD5 → animode is visually static.

    Returns:
        is_static (bool)
        saved_views (dict): {view_name: [frame_paths]} for 3 inspection views
                            (populated only when is_static=True)
    """
    import tempfile, shutil

    # Evenly-spaced probe frame indices (1-based, inclusive of first and last)
    probe_frames = []
    for i in range(PROBE_FRAME_COUNT):
        idx = 1 + int(round(i * (num_frames - 1) / max(PROBE_FRAME_COUNT - 1, 1)))
        idx = max(1, min(num_frames, idx))
        if idx not in probe_frames:
            probe_frames.append(idx)
    probe_frames = sorted(probe_frames)

    # Temp directory for probe PNGs (always cleaned up unless static & inspecting)
    tmp_dir = tempfile.mkdtemp(prefix="infinipart_probe_")

    # Save / restore render settings
    saved = {
        "res_x":       scene.render.resolution_x,
        "res_y":       scene.render.resolution_y,
        "samples":     scene.cycles.samples,
        "denoising":   scene.cycles.use_denoising,
        "anim_seed":   scene.cycles.use_animated_seed,
        "seed":        scene.cycles.seed,
        "filepath":    scene.render.filepath,
        "camera":      scene.camera,
        "frame_start": scene.frame_start,
        "frame_end":   scene.frame_end,
    }

    scene.render.resolution_x      = PROBE_RESOLUTION
    scene.render.resolution_y      = PROBE_RESOLUTION
    scene.cycles.samples           = PROBE_SAMPLE_COUNT
    scene.cycles.use_denoising     = False
    scene.cycles.use_animated_seed = False   # same seed every frame → static = same bytes
    scene.cycles.seed              = 0

    # Single reusable probe camera (repositioned per view)
    cam_data = bpy.data.cameras.new("_probe_cam_data")
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(60)
    cam_obj = bpy.data.objects.new("_probe_cam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj

    all_static = True
    # {view_name: [tmp_file_paths]} for inspection views (kept only if static)
    inspection_files = {}

    try:
        for vi, (view_name, (elev, azim)) in enumerate(HEMI_VIEWS.items()):
            # Position probe camera
            er = math.radians(elev)
            ar = math.radians(azim)
            cx, cy, cz = cam_center
            x = cam_distance * math.cos(er) * math.cos(ar)
            y = cam_distance * math.cos(er) * math.sin(ar)
            z = cam_distance * math.sin(er)
            cam_obj.location = (x, y, z)
            direction = Vector((cx - x, cy - y, cz - z))
            cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

            # Render probe frames to temp files
            hashes = []
            view_files = []
            ok = True
            for fidx in probe_frames:
                p = os.path.join(tmp_dir, f"v{vi:02d}_f{fidx:04d}.png")
                if not _render_frame_to_file(scene, fidx, p):
                    ok = False
                    break
                hashes.append(_file_md5(p))
                view_files.append((fidx, p))

            if not ok:
                all_static = False
                break

            # All hashes identical → this view is static
            if len(set(hashes)) != 1:
                all_static = False
                break

            # Keep files for inspection views (in memory reference only)
            if view_name in PROBE_SAVE_VIEW_KEYS:
                inspection_files[view_name] = view_files

        print(f"  [Probe] {'STATIC' if all_static else 'HAS MOTION'} "
              f"({len(HEMI_VIEWS)} hemi views × {len(probe_frames)} frames @ "
              f"{PROBE_RESOLUTION}px, {PROBE_SAMPLE_COUNT} spp)")

    finally:
        # Restore render settings unconditionally
        scene.render.resolution_x      = saved["res_x"]
        scene.render.resolution_y      = saved["res_y"]
        scene.cycles.samples           = saved["samples"]
        scene.cycles.use_denoising     = saved["denoising"]
        scene.cycles.use_animated_seed = saved["anim_seed"]
        scene.cycles.seed              = saved["seed"]
        scene.render.filepath          = saved["filepath"]
        scene.frame_start              = saved["frame_start"]
        scene.frame_end                = saved["frame_end"]
        scene.camera                   = saved["camera"]
        bpy.data.objects.remove(cam_obj, do_unlink=True)
        bpy.data.cameras.remove(cam_data, do_unlink=True)

    # If static: copy inspection frames to permanent location; cleanup tmp
    saved_views = {}
    if all_static and inspection_files:
        probe_dir = os.path.join(meta_dir, animode_name, "probe")
        os.makedirs(probe_dir, exist_ok=True)
        for vname, file_list in inspection_files.items():
            paths = []
            for fidx, tmp_path in file_list:
                dst = os.path.join(probe_dir, f"{vname}_f{fidx:04d}.png")
                shutil.copy2(tmp_path, dst)
                paths.append(dst)
            saved_views[vname] = paths

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return all_static, saved_views


# ======================================================================
# Transform Math (mathutils)
# ======================================================================

def mat_translate(x, y, z):
    m = Matrix.Identity(4)
    m[0][3] = x
    m[1][3] = y
    m[2][3] = z
    return m


def mat_rotate_rpy(r, p, y):
    """Rotation matrix from roll-pitch-yaw (XYZ extrinsic = ZYX intrinsic)."""
    ex = Euler((r, p, y), 'XYZ')
    return ex.to_matrix().to_4x4()


def mat_rotate_axis_angle(axis, angle):
    """4x4 rotation around axis by angle (radians)."""
    ax = axis.normalized()
    return Matrix.Rotation(angle, 4, ax)


def compute_joint_local_transform(joint, q_value):
    """T = T_origin(xyz, rpy) x T_joint(q)."""
    ox, oy, oz = joint.origin_xyz
    T_origin = mat_translate(ox, oy, oz) @ mat_rotate_rpy(*joint.origin_rpy)

    if joint.jtype == "fixed":
        T_joint = Matrix.Identity(4)
    elif joint.jtype in ("revolute", "continuous"):
        T_joint = mat_rotate_axis_angle(joint.axis, q_value)
    elif joint.jtype == "prismatic":
        ax = joint.axis.normalized()
        T_joint = mat_translate(ax.x * q_value, ax.y * q_value, ax.z * q_value)
    else:
        T_joint = Matrix.Identity(4)

    return T_origin @ T_joint


# ======================================================================
# Trajectory Generation
# ======================================================================

def compute_trajectory_value(joint, traj_type, t, q_safe_limit=None):
    """Compute joint parameter q at normalized time t in [0, 1].

    All trajectories start from q=0 (the URDF natural/assembly position) and
    animate toward the farther extreme from zero.  This guarantees the video
    always begins in the physically correct resting state regardless of whether
    the joint range is [0, hi], [lo, 0] (e.g. bag handle), or [-x, +x].

    new_v3: if q_safe_limit is provided, it is used as far_q (overrides joint limits).
    This allows precomputed safe limits to cap the trajectory without render-time STOP.
    """
    lo = joint.lower if joint.jtype != "continuous" else 0.0
    hi = joint.upper if joint.jtype != "continuous" else 2 * math.pi

    # Farther extreme from the natural position (q=0)
    far_q = hi if abs(hi) >= abs(lo) else lo

    # new_v3: use precomputed safe limit if available
    if q_safe_limit is not None:
        far_q = q_safe_limit

    if traj_type == "sinusoidal_oscillation":
        # 0 → far_q → 0  (smooth loop, starts and ends at natural)
        q = far_q * 0.5 * (1 - math.cos(2 * math.pi * t))
    elif traj_type == "one_way_sinusoidal":
        # 0 → far_q  (one-way, starts at natural)
        q = far_q * 0.5 * (1 - math.cos(math.pi * t))
    elif traj_type == "linear":
        # 0 → far_q
        q = far_q * t
    elif traj_type == "linear_oscillation":
        # 0 → far_q → 0
        if t < 0.5:
            q = far_q * (2 * t)
        else:
            q = far_q * (2 * (1 - t))
    else:
        q = far_q * t  # fallback
    return q


# ======================================================================
# Utility
# ======================================================================

def _interp1d(x_pts, y_pts, x):
    """Simple 1D linear interpolation (numpy-free, works inside Blender)."""
    if len(x_pts) == 0:
        return 0.0
    if x <= x_pts[0]:
        return y_pts[0]
    if x >= x_pts[-1]:
        return y_pts[-1]
    for i in range(len(x_pts) - 1):
        if x_pts[i] <= x <= x_pts[i + 1]:
            t = (x - x_pts[i]) / (x_pts[i + 1] - x_pts[i] + 1e-12)
            return y_pts[i] + t * (y_pts[i + 1] - y_pts[i])
    return y_pts[-1]


# ======================================================================
# Forward Kinematics
# ======================================================================

def forward_kinematics(joints, children_map, root_link, joint_values):
    """Compute world transforms for all links given joint parameter values.

    Args:
        joint_values: {joint_name: q_value}

    Returns:
        link_transforms: {link_name: 4x4 Matrix}
    """
    link_transforms = {root_link: Matrix.Identity(4)}
    queue = deque([root_link])
    visited = set()

    while queue:
        link_name = queue.popleft()
        if link_name in visited:
            continue
        visited.add(link_name)

        parent_T = link_transforms.get(link_name, Matrix.Identity(4))
        for child_link, joint in children_map.get(link_name, []):
            q = joint_values.get(joint.name, 0.0)
            T_local = compute_joint_local_transform(joint, q)
            link_transforms[child_link] = parent_T @ T_local
            queue.append(child_link)

    return link_transforms


def compute_frame_joint_values(joints_by_name, split_info, t):
    """Compute joint values at normalized time t for a given animode split.

    Active joints: follow trajectory
    Passive joints: constant pre-opening angle
    Fixed/other joints: 0
    """
    traj_type = split_info["trajectory_type"]
    classification = split_info["joint_classification"]
    pre_opening = split_info.get("pre_opening_angles", {})

    values = {}

    # Custom lid trajectories: active joint value is the lift displacement (URDF metres)
    if traj_type in ("lid_separation", "lid_flip"):
        custom_params = split_info.get("custom_lid_params", {})
        lift_dist = custom_params.get("lift_dist", 0.1)
        flip_start_frac = custom_params.get("flip_start_frac", 0.5)
        for jname, cls in classification.items():
            joint = joints_by_name.get(jname)
            if joint is None:
                continue
            if cls == "active":
                if traj_type == "lid_separation":
                    # Smooth one-way lift: 0 → lift_dist over full duration
                    q = lift_dist * 0.5 * (1 - math.cos(math.pi * t))
                else:  # lid_flip: lift completes at flip_start_frac, then hold
                    lift_t = min(t / max(flip_start_frac, 1e-6), 1.0)
                    q = lift_dist * 0.5 * (1 - math.cos(math.pi * lift_t))
                values[jname] = q
            elif cls == "passive":
                values[jname] = pre_opening.get(jname, 0.0)
            else:
                values[jname] = 0.0
        return values

    for jname, cls in classification.items():
        joint = joints_by_name.get(jname)
        if joint is None:
            continue
        if cls == "active":
            values[jname] = compute_trajectory_value(joint, traj_type, t)
        elif cls == "passive":
            # Animate passive joints from 0 to pre_opening angle
            # using the same trajectory shape as active joints
            target = pre_opening.get(jname, 0.0)
            if abs(target) > 1e-8:
                # Scale t through trajectory shape to get smooth 0->target motion
                # Use one_way_sinusoidal for smooth ease-in/out
                values[jname] = target * 0.5 * (1 - math.cos(math.pi * t))
            else:
                values[jname] = 0.0
        else:  # fixed
            values[jname] = 0.0

    return values


# ======================================================================
# Part Loading
# ======================================================================

def clear_scene():
    """Remove all objects, meshes, cameras, lights from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.cameras:
        bpy.data.cameras.remove(block)
    for block in bpy.data.lights:
        bpy.data.lights.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)


def import_obj(filepath, name=None):
    """Import an OBJ file and return the imported object."""
    before = set(bpy.data.objects.keys())

    # Prefer new importer (Blender 4.x) — correctly wires PBR maps from MTL;
    # fall back to legacy importer for Blender 3.x
    try:
        bpy.ops.wm.obj_import(filepath=filepath, forward_axis='NEGATIVE_Y', up_axis='Z')
    except Exception:
        bpy.ops.import_scene.obj(filepath=filepath, axis_forward='-Y', axis_up='Z')

    after = set(bpy.data.objects.keys())
    new_names = after - before
    if not new_names:
        return None

    # CRITICAL: reset rotation_euler to (0,0,0) after import
    # Blender OBJ importer applies axis conversion as object-level rotation
    # but our OBJs are already Z-up
    imported_objs = [bpy.data.objects[n] for n in new_names]
    for obj in imported_objs:
        obj.rotation_euler = (0, 0, 0)

    # If multiple objects imported, join them
    if len(imported_objs) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for obj in imported_objs:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = imported_objs[0]
        bpy.ops.object.join()
        result = bpy.context.active_object
    else:
        result = imported_objs[0]

    if name:
        result.name = name
    return result


def load_parts_im_format(scene_dir, links, link_to_part_idx):
    """Load parts for IM/PhysXNet format: objs/{idx}/{idx}.obj + origins.json."""
    origins_path = os.path.join(scene_dir, "origins.json")
    with open(origins_path) as f:
        origins = json.load(f)

    parts = {}  # link_name -> blender object

    # Find objs/ directory (may be nested: outputs/{Factory}/{id}/objs/)
    objs_dir = os.path.join(scene_dir, "objs")
    if not os.path.isdir(objs_dir):
        # Search for nested objs/ directory
        for root, dirs, files in os.walk(scene_dir):
            if "objs" in dirs:
                objs_dir = os.path.join(root, "objs")
                break

    for link_name, info in links.items():
        idx = info["part_idx"]
        if idx is None:
            continue

        obj_path = os.path.join(objs_dir, str(idx), f"{idx}.obj")
        if not os.path.exists(obj_path):
            continue

        obj = import_obj(obj_path, name=f"part_{link_name}")
        if obj is None:
            continue

        # Apply origin offset
        origin_key = str(idx)
        if origin_key in origins:
            ox, oy, oz = origins[origin_key]
            obj.location = Vector((ox, oy, oz))
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.transform_apply(location=True)
            obj.select_set(False)

        parts[link_name] = obj

    return parts


def load_parts_urdf_format(scene_dir, links, joints, root_link):
    """Load parts for IS/PartNet format: mesh files referenced in URDF.

    Each link may have multiple visual meshes, each with a visual origin offset.
    We also need to apply kinematic chain transforms (joint origins) to get
    each part into the correct world position.
    """
    # Compute world transforms via BFS (same as split_precompute)
    children_map = defaultdict(list)
    for j in joints:
        children_map[j.parent_link].append((j.child_link, j))

    # BFS to compute link world transforms
    link_T = {root_link: Matrix.Identity(4)}
    queue = deque([root_link])
    visited = set()

    while queue:
        parent = queue.popleft()
        if parent in visited:
            continue
        visited.add(parent)

        parent_T = link_T[parent]
        for child_link, joint in children_map.get(parent, []):
            T_local = compute_joint_local_transform(joint, 0.0)
            link_T[child_link] = parent_T @ T_local
            queue.append(child_link)

    parts = {}
    for link_name, info in links.items():
        if info["part_idx"] is None:
            continue
        if not info["mesh_files"]:
            continue

        link_objs = []
        for mesh_path, vis_xyz in info["mesh_files"]:
            if not os.path.exists(mesh_path):
                print(f"  WARNING: mesh not found: {mesh_path}")
                continue

            obj = import_obj(mesh_path, name=f"part_{link_name}_{len(link_objs)}")
            if obj is None:
                continue

            # Apply visual origin offset
            if any(abs(v) > 1e-8 for v in vis_xyz):
                obj.location = Vector(vis_xyz)
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.transform_apply(location=True)
                obj.select_set(False)

            link_objs.append(obj)

        if not link_objs:
            continue

        # Join multiple visuals into one object
        if len(link_objs) > 1:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in link_objs:
                obj.select_set(True)
            bpy.context.view_layer.objects.active = link_objs[0]
            bpy.ops.object.join()
            result = bpy.context.active_object

            # Remove coplanar/duplicate geometry that causes Z-fighting.
            # Some PartNet objects have tiny degenerate quads coplanar with
            # the main surface; after join they create overlapping faces in
            # the same mesh, causing random per-sample flickering in Cycles.
            bpy.ops.object.mode_set(mode='EDIT')
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.mesh.remove_doubles(threshold=0.0001)
            bpy.ops.mesh.delete_loose()
            bpy.ops.object.mode_set(mode='OBJECT')
        else:
            result = link_objs[0]

        result.name = f"part_{link_name}"

        # Apply kinematic chain transform to get world position
        if link_name in link_T:
            T_world = link_T[link_name]
            # Transform all vertices by T_world
            mesh = result.data
            for vert in mesh.vertices:
                v = T_world @ Vector((*vert.co, 1.0))
                vert.co = v.xyz

        parts[link_name] = result

    return parts


def load_scene_parts(metadata, links, joints, root_link):
    """Auto-detect format and load parts."""
    scene_dir = metadata["scene_dir"]
    origins_path = os.path.join(scene_dir, "origins.json")

    if os.path.exists(origins_path):
        print(f"  Loading parts (IM/PhysXNet format) from {scene_dir}")
        return load_parts_im_format(scene_dir, links, None)
    else:
        print(f"  Loading parts (URDF-direct format) from {scene_dir}")
        return load_parts_urdf_format(scene_dir, links, joints, root_link)


# ======================================================================
# Normalization
# ======================================================================

def _physx_upright_matrix():
    """Rotation matrix for PhysX datasets: Y-up → Z-up + front facing.
    Combined: Rz(+90) @ Rx(+90) → (x,y,z) → (z, x, y)
    """
    from mathutils import Matrix
    return Matrix([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ])


def apply_upright_rotation_blender(parts, factory_name):
    """Apply PhysX coordinate rotation to Blender objects (before normalize)."""
    if "PhysX" not in factory_name:
        return
    R = _physx_upright_matrix()
    for link_name, obj in parts.items():
        mesh = obj.data
        for vert in mesh.vertices:
            v = R @ Vector((*vert.co, 1.0))
            vert.co = v.xyz
        mesh.update()


def build_normalize_matrices(center, scale, factory_name=""):
    """Build S, S_inv 4x4 matrices for normalized-space FK.

    For PhysX datasets, S incorporates the upright rotation (Y-up→Z-up):
      S_R = S_plain @ R,  S_R_inv = R_inv @ S_inv_plain
    so that  S_R @ T_world @ S_R_inv  = S @ R @ T @ R^-1 @ S^-1
    """
    cx, cy, cz = center
    S = Matrix.Identity(4)
    S[0][0] = scale; S[1][1] = scale; S[2][2] = scale
    S[0][3] = -cx * scale; S[1][3] = -cy * scale; S[2][3] = -cz * scale

    S_inv = Matrix.Identity(4)
    S_inv[0][0] = 1.0/scale; S_inv[1][1] = 1.0/scale; S_inv[2][2] = 1.0/scale
    S_inv[0][3] = cx; S_inv[1][3] = cy; S_inv[2][3] = cz

    if "PhysX" in factory_name:
        R = _physx_upright_matrix()
        R_inv = R.inverted()
        S = S @ R
        S_inv = R_inv @ S_inv

    return S, S_inv


def normalize_parts(parts, center, scale):
    """Apply precompute normalization to all loaded parts."""
    center_v = Vector(center)
    for link_name, obj in parts.items():
        mesh = obj.data
        for vert in mesh.vertices:
            v = Vector(vert.co)
            vert.co = ((v - center_v) * scale)[:]
        mesh.update()


# ======================================================================
# Materials
# ======================================================================

def _specular_key(bsdf):
    """Return the Specular input name (changed in Blender 4.0+)."""
    if "Specular IOR Level" in bsdf.inputs:
        return "Specular IOR Level"
    return "Specular"

def assign_material(obj, color, metallic=0.3, roughness=0.5, name=None):
    """Assign a simple PBR material to an object."""
    mat_name = name or f"mat_{obj.name}"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs[_specular_key(bsdf)].default_value = 0.0 if roughness >= 1.0 else 0.5

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def _is_metallic_material(img_nodes, mat_name=""):
    """Check if material is metallic via METAL map or material name."""
    # Check material name for common metal keywords
    name_lower = mat_name.lower()
    if any(kw in name_lower for kw in ('metal', 'steel', 'chrome', 'iron', 'brass',
                                        'copper', 'galvanized', 'aluminum', 'aluminium')):
        return True
    if 'metallic' not in img_nodes:
        return False
    img = img_nodes['metallic'].image
    if img is None:
        return False
    # Sample pixels; use a lower threshold (0.3) since UV-unmapped black
    # background areas dilute the average
    pixels = img.pixels[:]
    if len(pixels) == 0:
        return False
    step = max(1, len(pixels) // (4 * 64))
    samples = [pixels[i * 4] for i in range(0, len(pixels) // 4, step)]
    avg_metallic = sum(samples) / len(samples) if samples else 0
    return avg_metallic > 0.3


def _is_dark_diffuse(bsdf):
    """Check if the diffuse texture connected to Base Color is too dark."""
    bc_input = bsdf.inputs['Base Color']
    if bc_input.links:
        # There's a texture connected; sample it
        from_node = bc_input.links[0].from_node
        if from_node.type == 'TEX_IMAGE' and from_node.image:
            img = from_node.image
            pixels = img.pixels[:]
            if len(pixels) == 0:
                return False
            step = max(1, len(pixels) // (4 * 64))
            # Average luminance of sampled pixels
            lum_sum = 0
            count = 0
            for i in range(0, len(pixels) // 4, step):
                r, g, b = pixels[i*4], pixels[i*4+1], pixels[i*4+2]
                lum_sum += 0.299 * r + 0.587 * g + 0.114 * b
                count += 1
            avg_lum = lum_sum / count if count else 0
            return avg_lum < 0.25
    else:
        # No texture, check default value
        val = bc_input.default_value
        lum = 0.299 * val[0] + 0.587 * val[1] + 0.114 * val[2]
        return lum < 0.25
    return False


def fix_is_baked_materials(obj):
    """Fix IS factory baked PBR materials after OBJ import.

    The IS factory bake pipeline often produces:
    - Very dark DIFFUSE for metallic materials (galvanized_metal etc.)
    - Near-white ROUGHNESS (=1.0, fully rough) for metals
    Both together make metals appear pitch-black.

    This function:
    1. Wires any missing PBR maps (fallback for legacy importer)
    2. For metallic materials with dark diffuse: brightens base color via
       Bright/Contrast node so metal reflections are visible
    3. For metallic materials: clamps roughness to a lower range via
       a Math node so metals have proper sheen
    """
    if not obj.data.materials:
        return
    for mat in obj.data.materials:
        if mat is None or not mat.use_nodes:
            continue
        tree = mat.node_tree
        bsdf = tree.nodes.get("Principled BSDF")
        if not bsdf:
            continue

        # Collect all Image Texture nodes by filename suffix
        img_nodes = {}
        diffuse_node = None
        for node in tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                name = node.image.name.upper()
                if '_METAL' in name:
                    img_nodes['metallic'] = node
                elif '_ROUGHNESS' in name:
                    img_nodes['roughness'] = node
                elif '_NORMAL' in name:
                    img_nodes['normal'] = node
                elif '_DIFFUSE' in name:
                    diffuse_node = node

        # Fix colorspace: data textures must be Non-Color (OBJ importer sets sRGB)
        for key in ('metallic', 'roughness', 'normal'):
            if key in img_nodes:
                img_nodes[key].image.colorspace_settings.name = 'Non-Color'

        # Wire missing PBR maps (for legacy importer fallback)
        if 'metallic' in img_nodes and not bsdf.inputs['Metallic'].links:
            tree.links.new(img_nodes['metallic'].outputs['Color'], bsdf.inputs['Metallic'])

        if 'roughness' in img_nodes and not bsdf.inputs['Roughness'].links:
            tree.links.new(img_nodes['roughness'].outputs['Color'], bsdf.inputs['Roughness'])

        if 'normal' in img_nodes and not bsdf.inputs['Normal'].links:
            normal_map = tree.nodes.new('ShaderNodeNormalMap')
            normal_map.location = (img_nodes['normal'].location[0] + 300,
                                   img_nodes['normal'].location[1])
            tree.links.new(img_nodes['normal'].outputs['Color'], normal_map.inputs['Color'])
            tree.links.new(normal_map.outputs['Normal'], bsdf.inputs['Normal'])

        spec_key = _specular_key(bsdf)
        if not bsdf.inputs[spec_key].links:
            bsdf.inputs[spec_key].default_value = 0.5

        # --- Fix dark metallic materials ---
        is_metal = _is_metallic_material(img_nodes, mat.name)
        if is_metal:
            # Fix 1: Brighten dark diffuse for metallic materials
            # In PBR, metallic=1 means base color = reflection color.
            # Dark base color + metallic = black object. Real metals (chrome,
            # stainless steel) have bright base colors (0.6-0.9).
            if _is_dark_diffuse(bsdf):
                bc_input = bsdf.inputs['Base Color']
                if bc_input.links:
                    # Insert Bright/Contrast node between diffuse tex and BSDF
                    old_link = bc_input.links[0]
                    from_socket = old_link.from_socket
                    tree.links.remove(old_link)

                    bc_node = tree.nodes.new('ShaderNodeBrightContrast')
                    bc_node.location = (bsdf.location[0] - 200, bsdf.location[1])
                    bc_node.inputs['Bright'].default_value = 0.55
                    bc_node.inputs['Contrast'].default_value = 0.3
                    tree.links.new(from_socket, bc_node.inputs['Color'])
                    tree.links.new(bc_node.outputs['Color'], bsdf.inputs['Base Color'])
                else:
                    # No texture, just override with a reasonable metal color
                    bsdf.inputs['Base Color'].default_value = (0.7, 0.7, 0.7, 1.0)



def enhance_existing_materials(obj, metallic=0.10, roughness=0.45):
    """Enhance existing OBJ-imported materials with PBR properties.

    Keeps original colors/textures from the MTL/baked textures, only adjusts
    metallic and roughness on UNCONNECTED inputs (so baked texture maps are
    preserved). Used for PhysXMobility and IS factory objects.
    """
    if not obj.data.materials:
        return
    for mat in obj.data.materials:
        if mat is None:
            continue
        if not mat.use_nodes:
            mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            # Only set default_value on inputs that have NO incoming link
            # (baked textures are connected via image nodes — don't override)
            if not bsdf.inputs["Metallic"].links:
                bsdf.inputs["Metallic"].default_value = metallic
            if not bsdf.inputs["Roughness"].links:
                bsdf.inputs["Roughness"].default_value = roughness
            spec_key = _specular_key(bsdf)
            if not bsdf.inputs[spec_key].links:
                bsdf.inputs[spec_key].default_value = 0.3


# ======================================================================
# Camera
# ======================================================================

def create_camera(name, center, distance, elev_deg, azim_deg, lens=50):
    """Create a static camera at spherical coordinates."""
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.clip_start = 0.001
    cam_data.clip_end = 100
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)

    elev = math.radians(elev_deg)
    azim = math.radians(azim_deg)
    x = distance * math.cos(elev) * math.cos(azim) + center[0]
    y = distance * math.cos(elev) * math.sin(azim) + center[1]
    z = distance * math.sin(elev) + center[2]

    cam.location = (x, y, z)
    direction = Vector(center) - cam.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam.rotation_euler = rot_quat.to_euler()
    return cam


def create_animated_camera(name, center, distance, start_elev, start_azim,
                           end_elev, end_azim, num_frames, lens=50):
    """Create a camera that moves from start to end in spherical coords."""
    cam_data = bpy.data.cameras.new(name)
    cam_data.lens = lens
    cam_data.clip_start = 0.001
    cam_data.clip_end = 100
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.collection.objects.link(cam)

    for frame in range(1, num_frames + 1):
        t = (frame - 1) / max(num_frames - 1, 1)
        elev_deg = start_elev + (end_elev - start_elev) * t
        azim_deg = start_azim + (end_azim - start_azim) * t

        elev = math.radians(elev_deg)
        azim = math.radians(azim_deg)
        x = distance * math.cos(elev) * math.cos(azim) + center[0]
        y = distance * math.cos(elev) * math.sin(azim) + center[1]
        z = distance * math.sin(elev) + center[2]

        cam.location = (x, y, z)
        direction = Vector(center) - cam.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam.rotation_euler = rot_quat.to_euler()

        cam.keyframe_insert(data_path="location", frame=frame)
        cam.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Linear interpolation
    if cam.animation_data and cam.animation_data.action:
        for fc in cam.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                kp.interpolation = 'LINEAR'

    return cam


def remove_camera(cam):
    """Remove a camera object and its data."""
    cam_data = cam.data
    bpy.data.objects.remove(cam, do_unlink=True)
    bpy.data.cameras.remove(cam_data)


# ======================================================================
# Render Setup
# ======================================================================

def setup_render_engine():
    """Set up Cycles with GPU."""
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences

    for backend in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            usable = [d for d in prefs.devices if d.type == backend]
            if usable:
                for dev in prefs.devices:
                    dev.use = (dev.type == backend)
                    if dev.use:
                        print(f"  GPU ({backend}): {dev.name}")
                break
        except Exception:
            continue

    scene.cycles.device = 'GPU'


def setup_render_settings(resolution, fps, num_frames, samples):
    """Configure render settings."""
    scene = bpy.context.scene
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = num_frames

    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    try:
        scene.cycles.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
    except TypeError:
        pass
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01

    scene.render.use_persistent_data = True
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'

    try:
        scene.view_settings.view_transform = 'Filmic'
    except TypeError:
        scene.view_settings.view_transform = 'AgX'
    try:
        scene.view_settings.look = 'Medium Contrast'
    except TypeError:
        scene.view_settings.look = 'None'
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def setup_compositor_dual_output(bg_dir):
    """Set up compositor to output both nobg and withbg in a single render pass.

    nobg RGBA goes through the Composite node (scene.render.filepath).
    withbg composites the Environment pass behind the transparent render
    via Alpha Over, output through a File Output node to bg_dir.
    """
    scene = bpy.context.scene
    scene.render.film_transparent = True
    bpy.context.view_layer.use_pass_environment = True

    scene.use_nodes = True
    tree = scene.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)

    rl = tree.nodes.new('CompositorNodeRLayers')
    rl.location = (0, 0)

    # Composite node -> nobg output
    composite = tree.nodes.new('CompositorNodeComposite')
    composite.location = (600, 0)
    tree.links.new(rl.outputs['Image'], composite.inputs['Image'])

    # Alpha Over: env background + transparent foreground
    alpha_over = tree.nodes.new('CompositorNodeAlphaOver')
    alpha_over.location = (300, 200)
    tree.links.new(rl.outputs['Env'], alpha_over.inputs[1])
    tree.links.new(rl.outputs['Image'], alpha_over.inputs[2])

    # File Output node -> withbg output
    bg_out = tree.nodes.new('CompositorNodeOutputFile')
    bg_out.base_path = bg_dir
    bg_out.format.file_format = 'PNG'
    bg_out.format.color_mode = 'RGBA'
    bg_out.file_slots[0].path = "frame_"
    bg_out.location = (600, 200)
    tree.links.new(alpha_over.outputs['Image'], bg_out.inputs[0])


def cleanup_compositor():
    """Reset compositor to default state."""
    bpy.context.scene.use_nodes = False
    bpy.context.view_layer.use_pass_environment = False


def setup_envmap(envmap_path, strength=1.0):
    """Set up environment map lighting."""
    world = bpy.data.worlds.get("World")
    if world is None:
        world = bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    node_links = world.node_tree.links

    nodes.clear()
    bg_node = nodes.new('ShaderNodeBackground')
    bg_node.inputs['Strength'].default_value = strength
    env_node = nodes.new('ShaderNodeTexEnvironment')
    env_node.image = bpy.data.images.load(envmap_path)
    output_node = nodes.new('ShaderNodeOutputWorld')

    node_links.new(env_node.outputs['Color'], bg_node.inputs['Color'])
    node_links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])

    print(f"  Envmap: {os.path.basename(envmap_path)}")
    return envmap_path


# ======================================================================
# Animation (per-frame FK update)
# ======================================================================

def animate_scene(parts, joints, children_map, root_link,
                  joints_by_name, split_info, num_frames):
    """Set keyframes for all parts across all frames using forward kinematics.

    For each frame, compute joint values, run FK, set part transforms.
    """
    # We need to store the initial (rest pose, normalized) vertex positions
    # and update them per frame. Since Blender keyframes work on object
    # transforms (not vertices), we use matrix_world approach:
    #
    # Strategy: store normalized rest vertices. Per frame, compute FK,
    # then compute delta from rest to animated and apply as matrix_world.
    #
    # Actually simpler: we work at the link level.
    # Each link has a rest transform (at q=0 for all joints).
    # Per frame, we compute the animated transform, then the delta is:
    #   T_animated = FK(q_values)
    #   T_rest = FK(q=0 for all)
    #   delta = T_animated @ T_rest.inv()
    # We apply delta to the part's vertices (already at rest position).
    #
    # But vertex-level animation per frame is expensive. Better approach:
    # store rest vertices, set object matrix_world = delta each frame.
    #
    # Even better: use Blender shape keys or just object transform keyframes.
    #
    # The cleanest approach for articulated rendering:
    # 1. Load parts at rest pose (q=0, already normalized)
    # 2. Per frame: compute FK, get link transforms T_frame[link]
    # 3. Compute rest transforms T_rest[link] (FK at q=0)
    # 4. Delta = T_frame[link] @ T_rest[link].inv()
    # 5. Set part.matrix_world = delta (since vertices are in rest-normalized space)

    # Compute rest transforms (q=0 for all joints)
    rest_values = {j.name: 0.0 for j in joints}
    T_rest = forward_kinematics(joints, children_map, root_link, rest_values)

    # Invert rest transforms per link
    T_rest_inv = {}
    for link_name, T in T_rest.items():
        T_rest_inv[link_name] = T.inverted()

    # For each frame, compute animated transforms and set keyframes
    for frame in range(1, num_frames + 1):
        t = (frame - 1) / max(num_frames - 1, 1)
        joint_values = compute_frame_joint_values(joints_by_name, split_info, t)

        # Run FK
        T_frame = forward_kinematics(joints, children_map, root_link, joint_values)

        # Apply to parts
        for link_name, obj in parts.items():
            if link_name not in T_frame:
                continue
            if link_name not in T_rest_inv:
                continue

            # Delta transform: from rest to animated
            delta = T_frame[link_name] @ T_rest_inv[link_name]

            # Apply normalization: the delta should be computed in normalized space
            # Our parts are already in normalized space. The FK transforms are in
            # original URDF space. We need to convert:
            # delta_norm = S @ delta @ S_inv
            # where S scales by 'scale' and translates by -center.
            # But since center/scale are uniform, and FK origin offsets are small,
            # the rotation part of delta is the same. Only translation needs scaling.
            #
            # Actually, for the delta approach to work correctly, we need to
            # compute FK in normalized space. Let's do that instead.

            obj.matrix_world = delta
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Set linear interpolation for all keyframes
    for link_name, obj in parts.items():
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'


def _get_link_descendants(start_link, children_map):
    """Get all descendant link names of start_link (inclusive)."""
    result = {start_link}
    stack = [start_link]
    while stack:
        link = stack.pop()
        for child_link, _joint in children_map.get(link, []):
            if child_link not in result:
                result.add(child_link)
                stack.append(child_link)
    return result


def compute_collision_safe_animation(parts, joints, children_map, root_link,
                                     joints_by_name, split_info, num_frames,
                                     center, scale, factory_name=""):
    """Compute per-frame joint values with BVH collision avoidance.

    CLAUDE.md spec (Render Simulation Phase):
      - Active vs Passive part → passive joint yields/opens per-frame
      - Active vs Fixed part  → STOP all motion

    Two detection strategies based on rest-state geometry:
      - Parts NOT touching at rest: find_nearest() with adaptive proximity
      - Parts touching at rest (hinge connections): overlap() count delta
        (baseline at rest vs current frame — detects new penetration beyond
         what already exists at the hinge contact area)

    Returns: dict {frame: {joint_name: angle}} for frames 1..num_frames
    """
    import time as _time
    from mathutils.bvhtree import BVHTree

    t0 = _time.time()

    classification = split_info["joint_classification"]
    pre_opening = split_info.get("pre_opening_angles", {})
    traj_type = split_info["trajectory_type"]

    # Custom lid animodes: lid separates/flies off — no collision detection needed
    if traj_type in ("lid_separation", "lid_flip"):
        result = {}
        for frame in range(1, num_frames + 1):
            t = (frame - 1) / max(num_frames - 1, 1)
            result[frame] = compute_frame_joint_values(joints_by_name, split_info, t)
        return result

    active_jnames = [jn for jn, cls in classification.items() if cls == "active"]
    passive_jnames = [jn for jn, cls in classification.items() if cls == "passive"]

    # If no active joints, fall back to simple animation
    if not active_jnames:
        result = {}
        for frame in range(1, num_frames + 1):
            t = (frame - 1) / max(num_frames - 1, 1)
            result[frame] = compute_frame_joint_values(joints_by_name, split_info, t)
        return result

    print(f"\n  Collision avoidance:")
    print(f"    Active joints: {active_jnames}")
    print(f"    Passive joints: {passive_jnames}")

    # -- Normalization matrices (includes PhysX upright rotation if applicable) --
    S, S_inv = build_normalize_matrices(center, scale, factory_name)

    # Rest FK in normalized space
    rest_q = {j.name: 0.0 for j in joints}
    T_rest_w = forward_kinematics(joints, children_map, root_link, rest_q)
    T_rest_n = {k: S @ v @ S_inv for k, v in T_rest_w.items()}
    T_rest_n_inv = {k: v.inverted() for k, v in T_rest_n.items()}

    # Cache mesh data (vertices in normalized rest space)
    mesh_cache = {}
    for lname, obj in parts.items():
        mesh = obj.data
        verts = [v.co.copy() for v in mesh.vertices]
        polys = [tuple(p.vertices) for p in mesh.polygons]
        if verts and polys:
            mesh_cache[lname] = (verts, polys)

    # -- Link groups --
    active_links = set()
    for jn in active_jnames:
        j = joints_by_name[jn]
        active_links |= _get_link_descendants(j.child_link, children_map)

    passive_link_map = {}  # {passive_joint_name: set_of_links}
    for jn in passive_jnames:
        j = joints_by_name[jn]
        plinks = _get_link_descendants(j.child_link, children_map) - active_links
        plinks &= set(parts.keys())  # only loaded parts
        if plinks:
            passive_link_map[jn] = plinks

    all_movable_links = set(active_links)
    for plinks in passive_link_map.values():
        all_movable_links |= plinks
    fixed_links = (set(parts.keys()) - all_movable_links) & set(mesh_cache.keys())

    print(f"    Active links: {sorted(active_links)}")
    print(f"    Fixed links: {sorted(fixed_links)}")
    for pjn, plinks in passive_link_map.items():
        print(f"    Passive {pjn} links: {sorted(plinks)}")

    # -- BVH helpers --
    TOUCH_THRESHOLD = 0.005  # parts closer than this at rest = "touching"

    def _build_bvh(link_names, joint_values):
        """Build combined BVH for a set of links at given joint angles."""
        T_w = forward_kinematics(joints, children_map, root_link, joint_values)
        T_n = {k: S @ v @ S_inv for k, v in T_w.items()}
        all_v, all_p, off = [], [], 0
        for lname in sorted(link_names):  # sorted for deterministic polygon order
            if lname not in mesh_cache or lname not in T_n:
                continue
            verts, polys = mesh_cache[lname]
            delta = T_n[lname] @ T_rest_n_inv[lname]
            tverts = [(delta @ Vector((*v, 1.0))).xyz for v in verts]
            all_v.extend(tverts)
            all_p.extend([tuple(vi + off for vi in p) for p in polys])
            off += len(verts)
        if not all_v or not all_p:
            return None
        try:
            return BVHTree.FromPolygons(all_v, all_p)
        except Exception:
            return None

    def _get_sample_points(link_names, joint_values, max_pts=500):
        """Get subsampled world-space vertex positions for a set of links."""
        T_w = forward_kinematics(joints, children_map, root_link, joint_values)
        T_n = {k: S @ v @ S_inv for k, v in T_w.items()}
        pts = []
        for lname in sorted(link_names):  # sorted for deterministic index order
            if lname not in mesh_cache or lname not in T_n:
                continue
            verts, _ = mesh_cache[lname]
            delta = T_n[lname] @ T_rest_n_inv[lname]
            step = max(1, len(verts) // max_pts)
            for i in range(0, len(verts), step):
                pts.append((delta @ Vector((*verts[i], 1.0))).xyz)
        return pts

    def _measure_min_dist(links_a, links_b, joint_values):
        """Measure minimum distance between two link groups."""
        bvh_b = _build_bvh(links_b, joint_values)
        if bvh_b is None:
            return float('inf')
        pts_a = _get_sample_points(links_a, joint_values)
        min_d = float('inf')
        for pt in pts_a:
            loc, norm, idx, dist = bvh_b.find_nearest(pt)
            if dist is not None and dist < min_d:
                min_d = dist
        return min_d

    def _check_proximity(links_a, links_b, joint_values, proximity):
        """Check if link group A has vertices within proximity of link group B surface."""
        bvh_b = _build_bvh(links_b, joint_values)
        if bvh_b is None:
            return False
        pts_a = _get_sample_points(links_a, joint_values)
        for pt in pts_a:
            loc, norm, idx, dist = bvh_b.find_nearest(pt)
            if dist is not None and dist < proximity:
                return True
        return False

    def _compute_contact_mask(links_a, links_b, contact_radius=0.03):
        """Find sample points of links_a that are close to links_b at rest.

        Returns a set of indices into the _get_sample_points() result.
        These are "contact zone" vertices (hinge area) that should be
        EXCLUDED from proximity checks to avoid false positives.
        """
        bvh_b = _build_bvh(links_b, rest_q)
        if bvh_b is None:
            return set()
        pts_a = _get_sample_points(links_a, rest_q)
        mask = set()
        for i, pt in enumerate(pts_a):
            loc, norm, idx, dist = bvh_b.find_nearest(pt)
            if dist is not None and dist < contact_radius:
                mask.add(i)
        return mask

    def _check_with_contact_mask(links_a, links_b, joint_values, proximity,
                                 contact_mask, min_hits=3, debug=False):
        """Proximity check excluding rest-contact-zone vertices.

        Only checks vertices NOT in the contact mask (i.e., vertices that
        are far from the surface at rest). If these vertices become close to
        the surface during animation, it means actual penetration is occurring
        (not just normal hinge contact).
        """
        bvh_b = _build_bvh(links_b, joint_values)
        if bvh_b is None:
            return (False, 0) if debug else False
        pts_a = _get_sample_points(links_a, joint_values)
        hit_count = 0
        for i, pt in enumerate(pts_a):
            if i in contact_mask:
                continue  # skip hinge-area vertices
            loc, norm, idx, dist = bvh_b.find_nearest(pt)
            if dist is not None and dist < proximity:
                hit_count += 1
                if not debug and hit_count >= min_hits:
                    return True
        if debug:
            return hit_count >= min_hits, hit_count
        return hit_count >= min_hits

    # ================================================================
    # Phase 0: Pre-compute active trajectory + rest-state baselines
    # new_v3: use precomputed q_safe_limits as effective far_q
    # ================================================================
    q_safe_limits = split_info.get("q_safe_limits", {})
    passive_trajectories = split_info.get("passive_trajectories", {})

    active_q = {}
    for frame in range(1, num_frames + 1):
        t = (frame - 1) / max(num_frames - 1, 1)
        active_q[frame] = {}
        for jn in active_jnames:
            j = joints_by_name[jn]
            ql = q_safe_limits.get(jn)  # None if not precomputed
            active_q[frame][jn] = compute_trajectory_value(j, traj_type, t, q_safe_limit=ql)

    # -- Collision detection config per pair --
    # Two strategies based on rest-state geometry:
    # - ("proximity", threshold): for non-touching pairs, simple find_nearest
    # - ("masked", contact_mask, proximity): for touching pairs, proximity check
    #   that EXCLUDES rest-contact-zone vertices (hinge area)

    # Active vs Fixed
    fixed_collision_cfg = None  # None only if no fixed links with mesh
    if fixed_links and (active_links & set(mesh_cache.keys())):
        rest_dist = _measure_min_dist(active_links, fixed_links, rest_q)
        if rest_dist < TOUCH_THRESHOLD:
            # Touching at rest → compute contact mask and use masked proximity
            mask = _compute_contact_mask(active_links, fixed_links)
            n_total = len(_get_sample_points(active_links, rest_q))
            fixed_collision_cfg = ("masked", mask, 0.02)
            print(f"    Active-Fixed: touching at rest (dist={rest_dist:.5f}), "
                  f"masked {len(mask)}/{n_total} contact vertices, prox=0.02")
        else:
            prox = min(rest_dist * 0.20, 0.02)
            fixed_collision_cfg = ("proximity", prox)
            print(f"    Active-Fixed: gap={rest_dist:.5f}, proximity={prox:.5f}")

    # Active vs each passive group
    passive_collision_cfg = {}
    for pjn, plinks in passive_link_map.items():
        if not (plinks & set(mesh_cache.keys())):
            continue
        rest_dist = _measure_min_dist(active_links, plinks, rest_q)
        if rest_dist < TOUCH_THRESHOLD:
            mask = _compute_contact_mask(active_links, plinks)
            n_total = len(_get_sample_points(active_links, rest_q))
            passive_collision_cfg[pjn] = ("masked", mask, 0.02)
            print(f"    Active-{pjn}: touching at rest (dist={rest_dist:.5f}), "
                  f"masked {len(mask)}/{n_total} contact vertices, prox=0.02")
        else:
            prox = min(rest_dist * 0.20, 0.02)
            passive_collision_cfg[pjn] = ("proximity", prox)
            print(f"    Active-{pjn}: gap={rest_dist:.5f}, proximity={prox:.5f}")

    # -- Unified collision checker --
    def _has_collision(links_a, links_b, joint_values, cfg):
        """Check collision using the appropriate strategy for this pair."""
        mode = cfg[0]
        if mode == "proximity":
            return _check_proximity(links_a, links_b, joint_values, cfg[1])
        else:
            # masked mode: proximity check excluding contact-zone vertices
            _, contact_mask, prox = cfg
            return _check_with_contact_mask(
                links_a, links_b, joint_values, prox, contact_mask)

    # ================================================================
    # Phase 1: Active vs Fixed → STOP all motion
    #
    # new_v3: if q_safe_limits are precomputed, q_safe_limit is already applied
    # as far_q in the active trajectory (Phase 0). Skip real-time STOP.
    # Legacy fallback: per-frame scan if no precomputed data.
    # ================================================================
    if q_safe_limits:
        print(f"    Phase 1: using precomputed q_safe_limits, skipping real-time STOP")
        for jn, ql in q_safe_limits.items():
            print(f"      {jn}: q_safe_limit={ql:.4f}")
    elif fixed_collision_cfg is not None:
        # Sample all frames for collision status
        sample_step = max(1, num_frames // 60)
        sample_frames_p1 = list(range(1, num_frames + 1, sample_step))
        if sample_frames_p1[-1] != num_frames:
            sample_frames_p1.append(num_frames)

        frame_collides = {}  # sampled frame → bool
        for frame in sample_frames_p1:
            q_test = dict(rest_q)
            q_test.update(active_q[frame])
            frame_collides[frame] = _has_collision(
                active_links, fixed_links, q_test, fixed_collision_cfg)

        # Interpolate collision status for non-sampled frames
        # (conservative: if either neighbor collides, mark as colliding)
        for frame in range(1, num_frames + 1):
            if frame in frame_collides:
                continue
            prev_s = frame - ((frame - 1) % sample_step)
            next_s = min(prev_s + sample_step, num_frames)
            frame_collides[frame] = frame_collides.get(prev_s, False) or \
                                    frame_collides.get(next_s, False)

        # Clamp: find the first collision frame, then smoothly return to q=0
        # for all remaining frames so the animation loops cleanly.
        rest_active_q = {jn: 0.0 for jn in active_jnames}
        last_safe_q = dict(rest_active_q)
        first_clamp = None
        clamped_count = 0

        for frame in range(1, num_frames + 1):
            if frame_collides[frame]:
                if first_clamp is None:
                    first_clamp = frame  # first collision frame
                    q_stop = dict(last_safe_q)  # q just before collision
                clamped_count += 1
            else:
                if first_clamp is None:
                    last_safe_q = dict(active_q[frame])

        if clamped_count > 0:
            # Override clamped frames with smooth linear return q_stop → 0
            # so the animation ends at q=0 and loops without a jump.
            n_clamped = num_frames - first_clamp + 1
            for i, frame in enumerate(range(first_clamp, num_frames + 1)):
                alpha = (i + 1) / (n_clamped + 1)  # 0 < alpha ≤ 1
                active_q[frame] = {jn: q_stop[jn] * (1.0 - alpha)
                                   for jn in active_jnames}
            print(f"    STOP: active vs fixed collision at frame {first_clamp}, "
                  f"smooth return for {n_clamped}/{num_frames} frames")
        else:
            print(f"    No active-vs-fixed collision detected")

    # ================================================================
    # Phase 1.5: Active vs Passive (at max opening) → STOP if passive
    # joint can't open enough to clear the active trajectory.
    #
    # For each passive joint group, check: does the active trajectory
    # still collide with the passive parts when the passive joint is
    # at its MAXIMUM opening (pre_opening target)? If so, the rack/arm
    # must STOP because the door/lid can't open further.
    # ================================================================
    for pjn, plinks in passive_link_map.items():
        cfg = passive_collision_cfg.get(pjn)
        if cfg is None:
            continue
        pj15 = joints_by_name[pjn]
        phys_max_p15 = pj15.upper if pj15.jtype not in ("continuous", "fixed") else math.pi
        if abs(phys_max_p15) < 1e-8 or not (plinks & set(mesh_cache.keys())):
            continue

        # Sample frames: check if active collides with passive at physical max
        sample_step_p15 = max(1, num_frames // 60)
        sample_frames_p15 = list(range(1, num_frames + 1, sample_step_p15))
        if sample_frames_p15[-1] != num_frames:
            sample_frames_p15.append(num_frames)

        frame_collides_p15 = {}
        for frame in sample_frames_p15:
            q_test = dict(rest_q)
            q_test.update(active_q[frame])
            q_test[pjn] = phys_max_p15  # passive at physical max opening
            frame_collides_p15[frame] = _has_collision(
                active_links, plinks, q_test, cfg)

        # Interpolate (conservative)
        for frame in range(1, num_frames + 1):
            if frame in frame_collides_p15:
                continue
            prev_s = frame - ((frame - 1) % sample_step_p15)
            next_s = min(prev_s + sample_step_p15, num_frames)
            frame_collides_p15[frame] = frame_collides_p15.get(prev_s, False) or \
                                        frame_collides_p15.get(next_s, False)

        # Clamp active trajectory where passive can't resolve collision
        rest_active_q = {jn: 0.0 for jn in active_jnames}
        last_safe_q_p15 = dict(rest_active_q)
        clamped_p15 = 0

        for frame in range(1, num_frames + 1):
            if frame_collides_p15[frame]:
                active_q[frame] = dict(last_safe_q_p15)
                clamped_p15 += 1
            else:
                last_safe_q_p15 = dict(active_q[frame])

        if clamped_p15 > 0:
            print(f"    STOP (passive maxed): {pjn} at phys_max={phys_max_p15:.4f}, "
                  f"{clamped_p15}/{num_frames} active frames clamped")

    # ================================================================
    # Phase 2: Active vs Passive → compute passive angle per frame
    #
    # new_v3: if passive_trajectories are precomputed, use them directly (t-based).
    # Legacy fallback: real-time binary search.
    # ================================================================
    passive_q_raw = {}  # {joint_name: [0.0]*(num_frames+2)} 1-indexed

    for pjn, plinks in passive_link_map.items():
        pj = joints_by_name[pjn]
        q_raw = [0.0] * (num_frames + 2)

        if pjn in passive_trajectories and passive_trajectories[pjn] is not None:
            # new_v3: interpolate precomputed passive_trajectory (time-based)
            # traj = [[t, q_passive], ...] where t ∈ [0, 1]
            traj = passive_trajectories[pjn]
            t_pts = [pt[0] for pt in traj]
            qp_pts = [pt[1] for pt in traj]

            for frame in range(1, num_frames + 1):
                t_frame = (frame - 1) / max(num_frames - 1, 1)
                q_p = float(_interp1d(t_pts, qp_pts, t_frame))
                q_raw[frame] = q_p

            passive_q_raw[pjn] = q_raw
            max_qp = max(abs(v) for v in q_raw[1:num_frames+1])
            print(f"    YIELD (precomputed): {pjn} passive_trajectory "
                  f"{len(traj)} pts, max_qp={max_qp:.4f}")
            continue

        # Legacy: real-time binary search
        phys_max = pj.upper if pj.jtype not in ("continuous", "fixed") else math.pi

        cfg = passive_collision_cfg.get(pjn)
        if cfg is None or abs(phys_max) < 1e-8 or not (plinks & set(mesh_cache.keys())):
            passive_q_raw[pjn] = q_raw
            continue

        q_current = 0.0
        collision_frames = 0
        # Check every few frames (denser than before for accuracy)
        sample_step = max(1, num_frames // 60)
        sample_frames = list(range(1, num_frames + 1, sample_step))
        if sample_frames[-1] != num_frames:
            sample_frames.append(num_frames)

        for frame in sample_frames:
            q_all = dict(rest_q)
            q_all.update(active_q[frame])
            # Include other passive joints at their current estimate
            for other_pjn in passive_q_raw:
                q_all[other_pjn] = passive_q_raw[other_pjn][frame]
            q_all[pjn] = q_current

            if _has_collision(active_links, plinks, q_all, cfg):
                collision_frames += 1
                # Binary search: find minimum passive angle that clears collision
                q_lo, q_hi = q_current, phys_max
                for _ in range(12):
                    q_mid = (q_lo + q_hi) / 2.0
                    q_all[pjn] = q_mid
                    if _has_collision(active_links, plinks, q_all, cfg):
                        q_lo = q_mid  # still colliding, need more opening
                    else:
                        q_hi = q_mid  # safe, can try less opening
                # Add small clearance, clamp to physical max
                q_safe = q_hi + abs(phys_max) * 0.02
                if phys_max >= 0:
                    q_current = min(q_safe, phys_max)
                else:
                    q_current = max(-q_safe, phys_max)

            q_raw[frame] = q_current

        # Interpolate between sample frames
        for i in range(len(sample_frames) - 1):
            f0, f1 = sample_frames[i], sample_frames[i + 1]
            v0, v1 = q_raw[f0], q_raw[f1]
            for f in range(f0 + 1, f1):
                frac = (f - f0) / (f1 - f0)
                q_raw[f] = v0 + (v1 - v0) * frac

        passive_q_raw[pjn] = q_raw
        if collision_frames > 0:
            print(f"    YIELD: {pjn} collided at {collision_frames} samples, "
                  f"opened to {q_current:.4f} (phys_max {phys_max:.4f})")

    # ================================================================
    # Phase 3: Anticipation smoothing for passive joints
    # ================================================================
    LEAD = min(10, num_frames // 4)
    passive_q_smooth = {}
    for pjn, q_raw in passive_q_raw.items():
        q_s = list(q_raw)

        # Find jumps and spread backward
        for f in range(2, num_frames + 1):
            if abs(q_raw[f]) > abs(q_raw[f - 1]) + 1e-6:
                lead_start = max(1, f - LEAD)
                span = max(f - lead_start, 1)
                for ff in range(lead_start, f):
                    frac = (ff - lead_start) / span
                    interp = q_raw[f - 1] + (q_raw[f] - q_raw[f - 1]) * math.sin(frac * math.pi / 2)
                    if abs(interp) > abs(q_s[ff]):
                        q_s[ff] = interp

        # Enforce monotonicity (magnitude only increases)
        for f in range(2, num_frames + 1):
            if abs(q_s[f]) < abs(q_s[f - 1]):
                q_s[f] = q_s[f - 1]

        passive_q_smooth[pjn] = q_s

    # ================================================================
    # Combine into per-frame joint values
    # ================================================================
    result = {}
    for frame in range(1, num_frames + 1):
        q = {}
        q.update(active_q[frame])
        for pjn in passive_jnames:
            if pjn in passive_q_smooth:
                q[pjn] = passive_q_smooth[pjn][frame]
            else:
                # Passive joint with no collision-relevant parts — use simple ramp
                target = pre_opening.get(pjn, 0.0)
                t = (frame - 1) / max(num_frames - 1, 1)
                q[pjn] = target * 0.5 * (1 - math.cos(math.pi * t)) if abs(target) > 1e-8 else 0.0
        for jn in classification:
            if jn not in q:
                q[jn] = 0.0  # fixed joints
        result[frame] = q

    elapsed = _time.time() - t0
    print(f"  Collision avoidance done ({elapsed:.1f}s)")

    return result


def animate_scene_normalized(parts, joints, children_map, root_link,
                             joints_by_name, split_info, num_frames,
                             center, scale,
                             precomputed_frame_joint_values=None,
                             factory_name=""):
    """Animate using normalized-space FK with per-frame collision avoidance.

    Compute FK in URDF space, then convert to normalized space for delta.
    The normalization is: p_norm = (p_world - center) * scale

    For a 4x4 transform T in world space, the normalized version is:
      T_norm = S @ T @ S_inv
    where S = scale_matrix(scale) @ translate(-center)
    For PhysX datasets, S also includes the upright rotation (Y-up→Z-up).

    If precomputed_frame_joint_values is provided (dict {frame: {jname: q}}),
    the collision avoidance step is skipped (values already computed).
    """
    S, S_inv = build_normalize_matrices(center, scale, factory_name)

    # Rest transforms in normalized space
    rest_values = {j.name: 0.0 for j in joints}
    T_rest_world = forward_kinematics(joints, children_map, root_link, rest_values)
    T_rest_norm = {k: S @ v @ S_inv for k, v in T_rest_world.items()}
    T_rest_norm_inv = {k: v.inverted() for k, v in T_rest_norm.items()}

    # Compute collision-safe joint values per frame (or use precomputed)
    if precomputed_frame_joint_values is not None:
        frame_joint_values = precomputed_frame_joint_values
    else:
        frame_joint_values = compute_collision_safe_animation(
            parts, joints, children_map, root_link,
            joints_by_name, split_info, num_frames,
            center, scale, factory_name=factory_name)

    # Precompute lid_flip params (rotation about lid centroid after flip_start_frac)
    traj_type = split_info["trajectory_type"]
    lid_flip_links = set()
    flip_start_frac = 0.5
    flip_angle_total = math.pi
    flip_axis_vec = Vector((1.0, 0.0, 0.0))
    lid_rest_centroids = {}
    if traj_type == "lid_flip":
        custom_params = split_info.get("custom_lid_params", {})
        lid_flip_links = set(custom_params.get("lid_links", []))
        flip_start_frac = custom_params.get("flip_start_frac", 0.5)
        flip_angle_total = custom_params.get("flip_angle", math.pi)
        flip_axis_vec = Vector(custom_params.get("flip_axis", [1.0, 0.0, 0.0])).normalized()
        # Compute rest centroid for each lid link in normalized object space
        for lname in lid_flip_links:
            obj = parts.get(lname)
            if obj and obj.data.vertices:
                verts = [v.co.copy() for v in obj.data.vertices]
                centroid = sum(verts, Vector()) / len(verts)
                lid_rest_centroids[lname] = centroid

    for frame in range(1, num_frames + 1):
        joint_values = frame_joint_values[frame]
        t_norm = (frame - 1) / max(num_frames - 1, 1)

        T_frame_world = forward_kinematics(joints, children_map, root_link, joint_values)
        T_frame_norm = {k: S @ v @ S_inv for k, v in T_frame_world.items()}

        for link_name, obj in parts.items():
            if link_name not in T_frame_norm or link_name not in T_rest_norm_inv:
                continue

            delta = T_frame_norm[link_name] @ T_rest_norm_inv[link_name]

            # lid_flip: apply extra 180° rotation to lid links in second half
            if traj_type == "lid_flip" and link_name in lid_flip_links and t_norm > flip_start_frac:
                flip_t = (t_norm - flip_start_frac) / max(1.0 - flip_start_frac, 1e-6)
                current_angle = flip_angle_total * 0.5 * (1 - math.cos(math.pi * flip_t))
                # Rotate about lid centroid in lifted position
                c_rest = lid_rest_centroids.get(link_name, Vector((0.0, 0.0, 0.0)))
                c_lifted = delta @ c_rest
                R_flip = Matrix.Rotation(current_angle, 4, flip_axis_vec)
                T_to_c = mat_translate(-c_lifted.x, -c_lifted.y, -c_lifted.z)
                T_from_c = mat_translate(c_lifted.x, c_lifted.y, c_lifted.z)
                delta = T_from_c @ R_flip @ T_to_c @ delta

            obj.matrix_world = delta
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Linear interpolation
    for link_name, obj in parts.items():
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = 'LINEAR'


# ======================================================================
# Video Encoding
# ======================================================================

def frames_to_video(frame_dir, output_mp4, fps):
    """Encode PNG sequence to MP4 with ffmpeg."""
    frame_pattern = os.path.join(frame_dir, "frame_%04d.png")
    cmd = [
        FFMPEG_BIN, "-y",
        "-framerate", str(fps),
        "-i", frame_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "fast",
        output_mp4,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr[:500]}")
            return False
        return True
    except Exception as e:
        print(f"  ffmpeg failed: {e}")
        return False


# ======================================================================
# View Resolution
# ======================================================================

def resolve_views(view_args):
    """Resolve view group names to dict of {name: config}.

    Returns:
        static_views: {name: (elev, azim)}
        moving_views: {name: (start_elev, start_azim, end_elev, end_azim)}
    """
    static = {}
    moving = {}

    for vg in view_args:
        vg_lower = vg.lower()
        if vg_lower == "all":
            static.update(HEMI_VIEWS)
            moving.update(ORBIT_VIEWS)
            moving.update(SWEEP_VIEWS)
        elif vg_lower == "fast":
            static.update(FAST_HEMI_VIEWS)
            moving.update(FAST_ORBIT_VIEWS)
            moving.update(FAST_SWEEP_VIEWS)
        elif vg_lower == "hemi":
            static.update(HEMI_VIEWS)
        elif vg_lower == "orbit":
            moving.update(ORBIT_VIEWS)
        elif vg_lower == "sweep":
            moving.update(SWEEP_VIEWS)
        elif vg_lower in NAMED_VIEWS:
            static[vg_lower] = NAMED_VIEWS[vg_lower]
        elif vg_lower in HEMI_VIEWS:
            static[vg_lower] = HEMI_VIEWS[vg_lower]
        elif vg_lower in ORBIT_VIEWS:
            moving[vg_lower] = ORBIT_VIEWS[vg_lower]
        elif vg_lower in SWEEP_VIEWS:
            moving[vg_lower] = SWEEP_VIEWS[vg_lower]
        else:
            print(f"  WARNING: unknown view '{vg}', skipping")

    return static, moving


# ======================================================================
# Main
# ======================================================================

def main():
    args = parse_args()

    # Load metadata
    with open(args.metadata) as f:
        metadata = json.load(f)

    print(f"\n{'='*60}")
    print(f"Rendering {metadata['output_name']}/{metadata['identifier']}")
    print(f"{'='*60}")

    # Check required fields
    if "urdf_path" not in metadata or "scene_dir" not in metadata:
        print("ERROR: metadata.json missing urdf_path/scene_dir. Re-run precompute.")
        sys.exit(1)

    metadata_dir = os.path.dirname(os.path.realpath(args.metadata))
    urdf_path = metadata["urdf_path"]
    scene_dir = metadata["scene_dir"]
    # Resolve relative paths (stored relative to metadata.json location, portable across machines)
    if not os.path.isabs(urdf_path):
        urdf_path = os.path.normpath(os.path.join(metadata_dir, urdf_path))
    if not os.path.isabs(scene_dir):
        scene_dir = os.path.normpath(os.path.join(metadata_dir, scene_dir))
    # Propagate resolved paths back so helper functions (e.g. get_realistic_materials) see them
    metadata["urdf_path"] = urdf_path
    metadata["scene_dir"] = scene_dir
    center = metadata["normalize"]["center"]
    scale = metadata["normalize"]["scale"]

    if not os.path.exists(urdf_path):
        print(f"ERROR: URDF not found: {urdf_path}")
        sys.exit(1)

    # Parse URDF
    links, joints, root_link = parse_urdf(urdf_path)
    parent_map, children_map = build_kinematic_tree(joints)
    joints_by_name = {j.name: j for j in joints}

    # Resolve animodes to render
    all_splits = metadata.get("splits", {})
    if args.animode == "all":
        animodes_to_render = sorted(all_splits.keys())
    else:
        if args.animode not in all_splits:
            print(f"ERROR: animode '{args.animode}' not found. Available: {list(all_splits.keys())}")
            sys.exit(1)
        animodes_to_render = [args.animode]

    # Resolve views
    static_views, moving_views = resolve_views(args.views)
    total_views = len(static_views) + len(moving_views)
    print(f"  Animodes: {animodes_to_render}")
    print(f"  Views: {len(static_views)} static + {len(moving_views)} moving = {total_views}")

    num_frames = int(args.fps * args.duration)
    print(f"  Frames: {num_frames} ({args.duration}s @ {args.fps}fps)")

    # Select envmap
    if args.envmap:
        envmap_path = args.envmap
    else:
        envmaps = [os.path.join(ENVMAP_DIR, f) for f in os.listdir(ENVMAP_DIR)
                    if f.endswith(('.hdr', '.exr'))]
        envmap_path = random.choice(envmaps) if envmaps else None

    # Output directory is the parent of metadata.json
    meta_dir = os.path.dirname(os.path.abspath(args.metadata))

    # Render each animode
    for animode_name in animodes_to_render:
        print(f"\n--- Animode: {animode_name} ---")
        split_info = all_splits[animode_name]
        animode_dir = os.path.join(meta_dir, animode_name)

        # Fast-path: previously recorded as static → skip without any setup
        if is_recorded_static(meta_dir, animode_name):
            print(f"  SKIP (recorded static): {animode_name}")
            continue

        # Set up Blender scene
        clear_scene()
        setup_render_engine()
        setup_render_settings(args.resolution, args.fps, num_frames, args.samples)
        if envmap_path:
            setup_envmap(envmap_path)

        # Load parts
        parts = load_scene_parts(metadata, links, joints, root_link)
        if not parts:
            print(f"  ERROR: No parts loaded, skipping {animode_name}")
            continue

        print(f"  Loaded {len(parts)} parts: {sorted(parts.keys())}")

        # Apply PhysX coordinate rotation (Y-up → Z-up) before normalize
        apply_upright_rotation_blender(parts, metadata.get("factory", ""))

        # Normalize
        normalize_parts(parts, center, scale)

        # Compute collision-safe animation first (needed for static checks)
        factory = metadata.get("factory", "")
        frame_joint_values = compute_collision_safe_animation(
            parts, joints, children_map, root_link,
            joints_by_name, split_info, num_frames, center, scale,
            factory_name=factory)

        # --- Static check Method 1: joint values identical across all frames ---
        if check_static_method1(frame_joint_values):
            record_static_skip_method1(meta_dir, animode_name)
            continue

        # Apply animation keyframes (pass precomputed values — no double computation)
        animate_scene_normalized(parts, joints, children_map, root_link,
                                 joints_by_name, split_info, num_frames,
                                 center, scale,
                                 precomputed_frame_joint_values=frame_joint_values,
                                 factory_name=factory)

        # Camera setup
        cam_center = [0.0, 0.0, 0.0]
        cam_distance = args.cam_distance

        # --- Static check Method 2: probe-render 4 frames per hemi view ---
        is_probe_static, probe_saved = check_animode_static_probe(
            bpy.context.scene, num_frames, meta_dir, animode_name,
            cam_center, cam_distance)
        if is_probe_static:
            record_static_skip_probe(meta_dir, animode_name, probe_saved)
            continue

        # Determine which color passes to render
        two_coloring = split_info.get("two_coloring", {})
        if args.color_mode == "both":
            color_passes = ["realistic", "group"]
        else:
            color_passes = [args.color_mode]

        # Pre-load realistic materials (only if needed)
        realistic_mats = None
        if any(p == "realistic" for p in color_passes):
            realistic_mats = get_realistic_materials(metadata, links)

        GROUP_COLORS = [
            (0.22, 0.46, 0.72),  # blue
            (0.89, 0.35, 0.13),  # orange
            (0.17, 0.63, 0.17),  # green
            (0.84, 0.15, 0.16),  # red
            (0.58, 0.40, 0.74),  # purple
            (0.55, 0.34, 0.29),  # brown
            (0.89, 0.47, 0.76),  # pink
            (0.50, 0.50, 0.50),  # gray
            (0.74, 0.74, 0.13),  # olive
            (0.09, 0.75, 0.81),  # cyan
            (0.98, 0.60, 0.01),  # amber
            (0.40, 0.76, 0.65),  # teal
        ]

        for color_pass in color_passes:
            suffix_map = {"group": "_group", "part": "_part", "realistic": "_nobg"}
            vid_suffix = suffix_map.get(color_pass, "_nobg")
            print(f"  --- Color pass: {color_pass} (suffix: {vid_suffix}) ---")

            # Assign materials for this pass
            if color_pass == "group":
                all_groups = []
                for group in two_coloring.get("part0_groups", []):
                    all_groups.append(group)
                for group in two_coloring.get("part1_groups", []):
                    all_groups.append(group)
                idx_to_group = {}
                for gi, group in enumerate(all_groups):
                    for link_idx in group:
                        idx_to_group[link_idx] = gi
                for link_name, obj in parts.items():
                    idx = links[link_name]["part_idx"]
                    gi = idx_to_group.get(idx, 0)
                    color = GROUP_COLORS[gi % len(GROUP_COLORS)]
                    assign_material(obj, color, metallic=0.0, roughness=1.0)
            elif color_pass == "realistic":
                factory = metadata.get("factory", "")
                identifier = metadata.get("identifier", "")
                for link_name, obj in parts.items():
                    mat_info = realistic_mats.get(link_name, {})
                    source = mat_info.get("source", "pbr")
                    metallic = mat_info.get("metallic", 0.10)
                    roughness = mat_info.get("roughness", 0.45)

                    if source == "is_baked":
                        # IS factory: baked PBR textures — wire METAL/ROUGHNESS/NORMAL maps
                        fix_is_baked_materials(obj)
                    elif source == "native":
                        # PhysXMobility: has Kd colors but no roughness/metallic maps
                        enhance_existing_materials(obj, metallic, roughness)
                    else:
                        color = mat_info.get("color", (0.60, 0.60, 0.60))
                        category = mat_info.get("category", "plastic")
                        tex_set = mat_info.get("tex_set")
                        if tex_set and category != "glass":
                            apply_textured_material(
                                obj, color, metallic, roughness,
                                category, tex_set,
                                links[link_name]["part_idx"],
                                factory, identifier)
                        else:
                            assign_material(obj, color, metallic=metallic,
                                            roughness=roughness)
            else:  # "part" — binary part0/part1 coloring
                part0_link_idxs = set()
                for group in two_coloring.get("part0_groups", []):
                    part0_link_idxs.update(group)
                part1_link_idxs = set()
                for group in two_coloring.get("part1_groups", []):
                    part1_link_idxs.update(group)
                for link_name, obj in parts.items():
                    idx = links[link_name]["part_idx"]
                    if idx in part0_link_idxs:
                        assign_material(obj, (0.45, 0.55, 0.65), metallic=0.0, roughness=1.0)
                    elif idx in part1_link_idxs:
                        assign_material(obj, (0.85, 0.55, 0.25), metallic=0.0, roughness=1.0)
                    else:
                        assign_material(obj, (0.6, 0.6, 0.6), metallic=0.0, roughness=1.0)

            # Determine bg modes for this color pass
            # Only realistic pass supports withbg; group/part always nobg
            if color_pass == "realistic":
                if args.bg_mode == "both":
                    render_nobg, render_withbg = True, True
                elif args.bg_mode == "withbg":
                    render_nobg, render_withbg = False, True
                else:
                    render_nobg, render_withbg = True, False
            else:
                render_nobg, render_withbg = True, False

            # Render static views
            for view_name, (elev, azim) in sorted(static_views.items()):
                nobg_mp4 = os.path.join(animode_dir, f"{view_name}{vid_suffix}.mp4")
                withbg_mp4 = os.path.join(animode_dir, f"{view_name}_withbg.mp4")

                skip_nobg = not render_nobg or (args.skip_existing and os.path.exists(nobg_mp4))
                skip_withbg = not render_withbg or (args.skip_existing and os.path.exists(withbg_mp4))
                if skip_nobg and skip_withbg:
                    print(f"  SKIP {view_name}{vid_suffix} (exists)")
                    continue

                nobg_frame_dir = os.path.join(animode_dir, f"{view_name}{vid_suffix}")
                withbg_frame_dir = os.path.join(animode_dir, f"{view_name}_withbg")

                if render_nobg and render_withbg:
                    os.makedirs(nobg_frame_dir, exist_ok=True)
                    os.makedirs(withbg_frame_dir, exist_ok=True)
                    setup_compositor_dual_output(withbg_frame_dir)
                    bpy.context.scene.render.filepath = os.path.join(nobg_frame_dir, "frame_")
                elif render_withbg:
                    os.makedirs(withbg_frame_dir, exist_ok=True)
                    bpy.context.scene.render.film_transparent = False
                    bpy.context.scene.use_nodes = False
                    bpy.context.scene.render.filepath = os.path.join(withbg_frame_dir, "frame_")
                else:
                    os.makedirs(nobg_frame_dir, exist_ok=True)
                    bpy.context.scene.render.film_transparent = True
                    bpy.context.scene.use_nodes = False
                    bpy.context.scene.render.filepath = os.path.join(nobg_frame_dir, "frame_")

                cam = create_camera(view_name, cam_center, cam_distance, elev, azim)
                bpy.context.scene.camera = cam
                bpy.ops.render.render(animation=True)

                if render_nobg and render_withbg:
                    cleanup_compositor()

                if render_nobg and not skip_nobg:
                    ok = frames_to_video(nobg_frame_dir, nobg_mp4, args.fps)
                    print(f"  {view_name}{vid_suffix}: {'OK' if ok else 'FAIL (ffmpeg)'}")
                if render_withbg and not skip_withbg:
                    ok = frames_to_video(withbg_frame_dir, withbg_mp4, args.fps)
                    print(f"  {view_name}_withbg: {'OK' if ok else 'FAIL (ffmpeg)'}")

                remove_camera(cam)

            # Render moving views
            for view_name, (se, sa, ee, ea) in sorted(moving_views.items()):
                nobg_mp4 = os.path.join(animode_dir, f"{view_name}{vid_suffix}.mp4")
                withbg_mp4 = os.path.join(animode_dir, f"{view_name}_withbg.mp4")

                skip_nobg = not render_nobg or (args.skip_existing and os.path.exists(nobg_mp4))
                skip_withbg = not render_withbg or (args.skip_existing and os.path.exists(withbg_mp4))
                if skip_nobg and skip_withbg:
                    print(f"  SKIP {view_name}{vid_suffix} (exists)")
                    continue

                nobg_frame_dir = os.path.join(animode_dir, f"{view_name}{vid_suffix}")
                withbg_frame_dir = os.path.join(animode_dir, f"{view_name}_withbg")

                if render_nobg and render_withbg:
                    os.makedirs(nobg_frame_dir, exist_ok=True)
                    os.makedirs(withbg_frame_dir, exist_ok=True)
                    setup_compositor_dual_output(withbg_frame_dir)
                    bpy.context.scene.render.filepath = os.path.join(nobg_frame_dir, "frame_")
                elif render_withbg:
                    os.makedirs(withbg_frame_dir, exist_ok=True)
                    bpy.context.scene.render.film_transparent = False
                    bpy.context.scene.use_nodes = False
                    bpy.context.scene.render.filepath = os.path.join(withbg_frame_dir, "frame_")
                else:
                    os.makedirs(nobg_frame_dir, exist_ok=True)
                    bpy.context.scene.render.film_transparent = True
                    bpy.context.scene.use_nodes = False
                    bpy.context.scene.render.filepath = os.path.join(nobg_frame_dir, "frame_")

                cam = create_animated_camera(view_name, cam_center, cam_distance,
                                             se, sa, ee, ea, num_frames)
                bpy.context.scene.camera = cam
                bpy.ops.render.render(animation=True)

                if render_nobg and render_withbg:
                    cleanup_compositor()

                if render_nobg and not skip_nobg:
                    ok = frames_to_video(nobg_frame_dir, nobg_mp4, args.fps)
                    print(f"  {view_name}{vid_suffix}: {'OK' if ok else 'FAIL (ffmpeg)'}")
                if render_withbg and not skip_withbg:
                    ok = frames_to_video(withbg_frame_dir, withbg_mp4, args.fps)
                    print(f"  {view_name}_withbg: {'OK' if ok else 'FAIL (ffmpeg)'}")

                remove_camera(cam)

    # Update metadata with envmap info
    if envmap_path:
        metadata["envmap"] = os.path.basename(envmap_path)
        meta_path = os.path.abspath(args.metadata)
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    print(f"\nDone rendering {metadata['output_name']}/{metadata['identifier']}")


if __name__ == "__main__":
    main()
