"""
PBR Material System for PhysXNet/PhysXMobility rendering.

Three material sources (priority order):
  1. ShapeNet textures (future — placeholder, activated when ShapeNet data available)
  2. PartNet textured_objs colors (for 1,081 overlap objects)
  3. ambientCG PBR texture library with box projection (default for all objects)

Usage in Blender (render_articulation.py):
  from pbr_material_system import apply_pbr_materials
  apply_pbr_materials(factory_name, seed, part_objects, json_data)
"""

import json
import os
import random
from collections import defaultdict

# ======================================================================
# Configuration paths
# ======================================================================

PBR_TEXTURES_DIR = os.path.join(os.path.dirname(__file__), "pbr_textures")
PARTNET_BASE = "/mnt/data/yurh/dataset3D/Partnet"
OVERLAP_MAP_PATH = os.path.join(os.path.dirname(__file__), "physxnet_partnet_overlap.json")

# ShapeNet placeholder — set to actual path when downloaded
SHAPENET_BASE = None  # e.g. "/mnt/data/yurh/ShapeNetCore.v2"


# ======================================================================
# Material keyword → texture category mapping
# ======================================================================

# Maps material keywords (lowercase) to PBR texture category.
# Checked in order; first match wins.
MATERIAL_KEYWORD_MAP = [
    # Metals
    ("stainless", "metal"),
    ("steel", "metal"),
    ("iron", "metal"),
    ("aluminum", "metal"),
    ("aluminium", "metal"),
    ("chrome", "metal"),
    ("copper", "metal"),
    ("brass", "metal"),
    ("bronze", "metal"),
    ("titanium", "metal"),
    ("gold", "metal"),
    ("silver", "metal"),
    ("zinc", "metal"),
    ("nickel", "metal"),
    ("tungsten", "metal"),
    ("metal", "metal"),

    # Glass (no PBR texture needed — use transparent material)
    ("glass", "glass"),

    # Wood
    ("bamboo", "wood"),
    ("plywood", "wood"),
    ("mdf", "wood"),
    ("oak", "wood"),
    ("walnut", "wood"),
    ("laminate", "wood"),
    ("wood", "wood"),
    ("bentwood", "wood"),
    ("wicker", "wood"),

    # Ceramic/Porcelain
    ("ceramic", "ceramic"),
    ("porcelain", "ceramic"),

    # Marble/Stone
    ("marble", "marble"),
    ("granite", "stone"),
    ("slate", "stone"),
    ("stone", "stone"),
    ("concrete", "concrete"),

    # Leather
    ("leather", "leather"),
    ("leatherette", "leather"),

    # Fabric/Textile
    ("fabric", "fabric"),
    ("cotton", "fabric"),
    ("polyester", "fabric"),
    ("nylon", "fabric"),
    ("felt", "fabric"),
    ("silk", "fabric"),
    ("wool", "fabric"),
    ("mesh", "fabric"),
    ("canvas", "fabric"),
    ("upholster", "fabric"),
    ("cushion", "fabric"),
    ("cloth", "fabric"),
    ("woven", "fabric"),
    ("thread", "fabric"),

    # Foam (use fabric texture with adjusted roughness)
    ("foam", "fabric"),

    # Rubber
    ("rubber", "rubber"),
    ("tpu", "rubber"),
    ("silicone", "rubber"),

    # Paper/Cardboard
    ("paper", "paper"),
    ("cardboard", "paper"),
    ("cork", "paper"),

    # Plastic (default for many)
    ("abs", "plastic"),
    ("polycarbonate", "plastic"),
    ("polypropylene", "plastic"),
    ("pvc", "plastic"),
    ("polyethylene", "plastic"),
    ("acrylic", "plastic"),
    ("resin", "plastic"),
    ("plastic", "plastic"),
    ("nylon", "plastic"),
    ("vinyl", "plastic"),
    ("teflon", "plastic"),
    ("fiberglass", "plastic"),
    ("carbon fiber", "plastic"),
]

# Fallback PBR values when no texture file found
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


def classify_material(material_name):
    """Classify a PhysXNet material name into a PBR texture category.

    Returns (category_str, is_composite_bool).
    For composite materials like "Plastic and Metal", uses the keyword that
    appears earliest in the material name string.
    """
    mat_lower = material_name.lower()
    is_composite = any(sep in mat_lower for sep in
                       [" and ", " with ", " over ", "/", "+",
                        " or ", "-covered", "-coated", "-padded"])

    if is_composite:
        # Find the keyword that appears earliest in the name
        best_pos = len(mat_lower)
        best_cat = None
        for keyword, category in MATERIAL_KEYWORD_MAP:
            pos = mat_lower.find(keyword)
            if 0 <= pos < best_pos:
                best_pos = pos
                best_cat = category
        if best_cat:
            return best_cat, True
        return "plastic", True

    # Simple material: first keyword match
    for keyword, category in MATERIAL_KEYWORD_MAP:
        if keyword in mat_lower:
            return category, False

    return "plastic", False  # default fallback


# ======================================================================
# Texture set discovery
# ======================================================================

_texture_cache = {}  # category -> [list of texture_set_dicts]


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

            # Find texture files
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
                elif "_displacement" in fn_lower and fn_lower.endswith(".jpg"):
                    tex_set["displacement"] = full

            if "color" in tex_set:
                sets.append(tex_set)

        if sets:
            _texture_cache[category] = sets

    return _texture_cache


def get_texture_set(category, seed_hash=0):
    """Get a texture set for a given category, with deterministic random selection.

    Args:
        category: PBR texture category (e.g. "wood", "metal")
        seed_hash: integer for deterministic selection

    Returns:
        dict with keys: color, roughness, normal, metallic (file paths) or None
    """
    cache = _discover_texture_sets()
    sets = cache.get(category, [])
    if not sets:
        return None
    idx = seed_hash % len(sets)
    return sets[idx]


# ======================================================================
# PartNet color extraction (方案3)
# ======================================================================

_overlap_map = None


def _load_overlap_map():
    """Load the PhysXNet→PartNet overlap mapping."""
    global _overlap_map
    if _overlap_map is not None:
        return _overlap_map

    if not os.path.exists(OVERLAP_MAP_PATH):
        _overlap_map = {}
        return _overlap_map

    with open(OVERLAP_MAP_PATH) as f:
        _overlap_map = json.load(f)
    return _overlap_map


def get_partnet_colors(obj_id):
    """Extract colors from PartNet textured_objs for an overlapping object.

    Returns dict: {part_name: (r, g, b)} where colors are 0-1 float.
    Tries texture images first (average color), then MTL Kd values.
    """
    overlap = _load_overlap_map()
    if str(obj_id) not in overlap:
        return None

    entry = overlap[str(obj_id)]
    pdir = os.path.join(PARTNET_BASE, str(obj_id))
    tex_dir = os.path.join(pdir, "textured_objs")
    img_dir = os.path.join(pdir, "images")

    colors = {}

    # Method 1: Extract average color from texture images
    if os.path.isdir(img_dir):
        try:
            from PIL import Image
            import numpy as np
            for fn in os.listdir(img_dir):
                if fn.endswith(('.jpg', '.png')):
                    img_path = os.path.join(img_dir, fn)
                    img = Image.open(img_path).convert("RGB")
                    arr = np.array(img, dtype=float) / 255.0
                    avg_color = tuple(arr.mean(axis=(0, 1)))
                    colors[fn] = avg_color
        except ImportError:
            pass

    # Method 2: Extract Kd colors from MTL files
    if os.path.isdir(tex_dir):
        for fn in os.listdir(tex_dir):
            if not fn.endswith('.mtl'):
                continue
            mtl_path = os.path.join(tex_dir, fn)
            part_name = fn.replace('.mtl', '')
            try:
                with open(mtl_path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("Kd "):
                            parts = line.split()
                            if len(parts) >= 4:
                                r, g, b = float(parts[1]), float(parts[2]), float(parts[3])
                                if part_name not in colors:
                                    colors[part_name] = (r, g, b)
                        elif line.startswith("map_Kd ") and part_name not in colors:
                            # Has texture reference — try to load
                            tex_ref = line.split(None, 1)[1]
                            tex_path = os.path.join(tex_dir, tex_ref)
                            if os.path.exists(tex_path):
                                try:
                                    from PIL import Image
                                    import numpy as np
                                    img = Image.open(tex_path).convert("RGB")
                                    arr = np.array(img, dtype=float) / 255.0
                                    colors[part_name] = tuple(arr.mean(axis=(0, 1)))
                                except ImportError:
                                    pass
            except (IOError, ValueError):
                pass

    return colors if colors else None


def get_partnet_avg_color(obj_id):
    """Get a single average color representing the whole PartNet object.

    Useful for tinting the PBR texture.
    Returns (r, g, b) tuple or None.
    """
    colors = get_partnet_colors(obj_id)
    if not colors:
        return None

    r_sum, g_sum, b_sum = 0, 0, 0
    for _, (r, g, b) in colors.items():
        r_sum += r
        g_sum += g
        b_sum += b
    n = len(colors)
    return (r_sum / n, g_sum / n, b_sum / n)


# ======================================================================
# ShapeNet texture lookup (placeholder for future)
# ======================================================================

def get_shapenet_textures(obj_id):
    """Look up ShapeNet textures for an object via PartNet meta.json.

    Currently returns None (ShapeNet not downloaded).
    When ShapeNet is available, this will return texture file paths.

    Flow: obj_id → PartNet meta.json → model_id (ShapeNet hash) → textures
    """
    if SHAPENET_BASE is None or not os.path.isdir(str(SHAPENET_BASE)):
        return None

    overlap = _load_overlap_map()
    entry = overlap.get(str(obj_id))
    if not entry or not entry.get("model_id"):
        return None

    model_id = entry["model_id"]
    model_cat = entry.get("model_cat", "")

    # ShapeNetCore.v2 structure: {synset_id}/{model_id}/models/model_normalized.obj
    # We would need a category → synset_id mapping
    # TODO: implement when ShapeNet is downloaded
    # synset_id = CATEGORY_TO_SYNSET.get(model_cat)
    # if synset_id:
    #     model_dir = os.path.join(SHAPENET_BASE, synset_id, model_id, "models")
    #     if os.path.isdir(model_dir):
    #         return {
    #             "obj": os.path.join(model_dir, "model_normalized.obj"),
    #             "mtl": os.path.join(model_dir, "model_normalized.mtl"),
    #             "textures_dir": model_dir,
    #         }

    return None


# ======================================================================
# Blender material application (called inside Blender context)
# ======================================================================

def apply_pbr_materials(factory_name, seed, part_objects, json_data,
                        group_to_labels=None, label_to_info=None):
    """Apply PBR materials to Blender objects.

    Priority:
      1. ShapeNet textures (if available)
      2. PartNet colors + PBR texture overlay (for overlap objects)
      3. PBR texture library with box projection (default)

    Args:
        factory_name: factory name string
        seed: object seed (= object ID for PhysXNet)
        part_objects: dict {group_idx: bpy.types.Object}
        json_data: loaded PhysXNet/PhysXMobility JSON
        group_to_labels: optional precomputed {group_idx: [part_labels]}
        label_to_info: optional precomputed {label: part_info_dict}
    """
    import bpy

    obj_id = str(seed)

    # Build mappings if not provided
    if label_to_info is None:
        label_to_info = {}
        for p in json_data.get("parts", []):
            label = p.get("label")
            if label is not None:
                label_to_info[label] = p

    if group_to_labels is None:
        group_to_labels = _build_group_to_labels(json_data)

    # Check material sources
    shapenet_tex = get_shapenet_textures(obj_id)
    partnet_colors = get_partnet_colors(obj_id)

    source_used = "pbr_library"
    if shapenet_tex:
        source_used = "shapenet"
    elif partnet_colors:
        source_used = "partnet_colors"

    mat_rng = random.Random(hash((factory_name, seed)))
    enhanced_count = 0

    for group_idx, obj in part_objects.items():
        # Get material name from JSON
        part_labels = group_to_labels.get(group_idx, [group_idx])
        info = {}
        for lbl in part_labels:
            info = label_to_info.get(lbl, {})
            if info:
                break
        material_name = info.get("material", "Unknown")

        # Classify material
        category, is_composite = classify_material(material_name)
        defaults = CATEGORY_DEFAULTS.get(category, CATEGORY_DEFAULTS["plastic"])

        # Select texture set (deterministic per part)
        tex_hash = hash((factory_name, seed, group_idx, category))
        tex_set = get_texture_set(category, abs(tex_hash))

        # Determine color: PartNet override or category default
        base_color = defaults["color"]
        if partnet_colors:
            # Use average PartNet color as tint
            avg = get_partnet_avg_color(obj_id)
            if avg:
                # Blend: 70% PartNet color, 30% category default
                base_color = tuple(
                    0.7 * p + 0.3 * d for p, d in zip(avg, defaults["color"])
                )

        # Add slight per-part variation
        color_var = tuple(
            max(0, min(1, c + mat_rng.uniform(-0.05, 0.05)))
            for c in base_color
        )
        metallic = max(0, min(1, defaults["metallic"] + mat_rng.uniform(-0.05, 0.05)))
        roughness = max(0, min(1, defaults["roughness"] + mat_rng.uniform(-0.08, 0.08)))

        # Apply to Blender object
        if tex_set and category != "glass":
            _apply_pbr_texture_blender(
                obj, tex_set, color_var, metallic, roughness,
                category, group_idx, factory_name, seed
            )
        else:
            _apply_flat_pbr_blender(
                obj, color_var, metallic, roughness,
                category, group_idx
            )
        enhanced_count += 1

    if enhanced_count > 0:
        print(f"  PBR materials applied: {enhanced_count} parts, "
              f"source={source_used}, overlap={'yes' if partnet_colors else 'no'}")


def _build_group_to_labels(json_data):
    """Build group_idx -> [part_labels] mapping from group_info."""
    group_info = json_data.get("group_info", {})
    group_to_labels = {}
    for gid_str, val in group_info.items():
        gid = int(gid_str)
        if (isinstance(val, list) and len(val) >= 4
                and isinstance(val[-1], str)
                and val[-1] in ('A', 'B', 'C', 'D', 'CB')):
            labels = val[0] if isinstance(val[0], list) else [val[0]]
            group_to_labels[gid] = labels
        elif isinstance(val, list):
            group_to_labels[gid] = [x for x in val if isinstance(x, int)]
    return group_to_labels


def _apply_pbr_texture_blender(obj, tex_set, base_color, metallic, roughness,
                                category, group_idx, factory_name, seed):
    """Apply PBR texture set to a Blender object using box projection.

    Creates a proper node tree: Texture Coordinate → Mapping → Image Textures → BSDF
    """
    import bpy

    mat_name = f"pbr_{category}_{group_idx}_{factory_name}_{seed}"

    # Create new material
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create output node
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (800, 0)

    # Create Principled BSDF
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (400, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Texture coordinate node (Object coordinates for box projection)
    tex_coord = nodes.new("ShaderNodeTexCoord")
    tex_coord.location = (-600, 0)

    # Mapping node for scale control
    mapping = nodes.new("ShaderNodeMapping")
    mapping.location = (-400, 0)
    links.new(tex_coord.outputs["Object"], mapping.inputs["Vector"])

    # Scale based on object size for reasonable texture tiling
    # Use a default scale that works for normalized objects
    scale = 2.0
    mapping.inputs["Scale"].default_value = (scale, scale, scale)

    # === Color texture ===
    if "color" in tex_set:
        color_tex = nodes.new("ShaderNodeTexImage")
        color_tex.location = (-100, 200)
        color_tex.projection = "BOX"
        color_tex.projection_blend = 0.3
        try:
            color_tex.image = bpy.data.images.load(tex_set["color"], check_existing=True)
        except RuntimeError:
            color_tex.image = None

        links.new(mapping.outputs["Vector"], color_tex.inputs["Vector"])

        # Mix with base color for tinting
        mix_color = nodes.new("ShaderNodeMix")
        mix_color.data_type = 'RGBA'
        mix_color.location = (200, 200)
        mix_color.inputs[0].default_value = 0.6  # 60% texture, 40% base color
        mix_color.inputs[6].default_value = (*base_color, 1.0)  # A (base)
        links.new(color_tex.outputs["Color"], mix_color.inputs[7])  # B (texture)
        links.new(mix_color.outputs[2], bsdf.inputs["Base Color"])
    else:
        bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)

    # === Roughness texture ===
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
        links.new(mapping.outputs["Vector"], rough_tex.inputs["Vector"])
        links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])
    else:
        bsdf.inputs["Roughness"].default_value = roughness

    # === Normal map ===
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
        links.new(mapping.outputs["Vector"], normal_tex.inputs["Vector"])

        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.location = (200, -400)
        normal_map.inputs["Strength"].default_value = 0.5  # subtle normals
        links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
        links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])

    # === Metallic (from texture or flat value) ===
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
        links.new(mapping.outputs["Vector"], metal_tex.inputs["Vector"])
        links.new(metal_tex.outputs["Color"], bsdf.inputs["Metallic"])
    else:
        bsdf.inputs["Metallic"].default_value = metallic

    # === Glass-specific settings ===
    if category == "glass":
        bsdf.inputs["Transmission"].default_value = 0.8
        bsdf.inputs["IOR"].default_value = 1.45
        bsdf.inputs["Roughness"].default_value = 0.05
        mat.blend_method = 'HASHED'  # for EEVEE transparency

    # Apply material to object
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def _apply_flat_pbr_blender(obj, base_color, metallic, roughness,
                             category, group_idx):
    """Apply flat PBR material (no textures) to a Blender object.

    Used for glass or when texture files are unavailable.
    """
    import bpy

    mat_name = f"pbr_flat_{category}_{group_idx}"

    # Reuse or create material
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is None:
        for node in mat.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                bsdf = node
                break

    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*base_color, 1.0)
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = roughness

        if category == "glass":
            bsdf.inputs["Transmission"].default_value = 0.8
            bsdf.inputs["IOR"].default_value = 1.45
            bsdf.inputs["Roughness"].default_value = 0.05

        if metallic > 0.5:
            bsdf.inputs["Specular"].default_value = 0.5 + metallic * 0.3

    obj.data.materials.clear()
    obj.data.materials.append(mat)


# ======================================================================
# Utility: count usable assets
# ======================================================================

def count_usable_assets():
    """Count and report how many PhysXNet objects have each material source.

    Returns dict with counts and details.
    """
    from physxnet_factory_rules import (
        PHYSXNET_JSON_DIR, PHYSXMOB_JSON_DIR,
        get_all_physxnet_ids, get_all_physxmob_ids,
    )

    overlap = _load_overlap_map()
    tex_cache = _discover_texture_sets()

    physxnet_ids = get_all_physxnet_ids()
    physxmob_ids = get_all_physxmob_ids()

    # Count by material source
    has_shapenet = 0
    has_partnet = len(overlap)
    has_pbr_texture = 0
    has_flat_only = 0

    # Count by texture category coverage
    category_counts = defaultdict(int)
    uncovered_materials = defaultdict(int)

    for oid in physxnet_ids:
        json_path = os.path.join(PHYSXNET_JSON_DIR, f"{oid}.json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        for part in data.get("parts", []):
            mat = part.get("material", "Unknown")
            cat, _ = classify_material(mat)
            category_counts[cat] += 1
            if cat not in tex_cache:
                uncovered_materials[mat] += 1

    results = {
        "total_physxnet": len(physxnet_ids),
        "total_physxmob": len(physxmob_ids),
        "overlap_with_partnet": has_partnet,
        "texture_categories_available": list(tex_cache.keys()),
        "texture_sets_per_category": {k: len(v) for k, v in tex_cache.items()},
        "parts_by_category": dict(category_counts),
        "uncovered_materials": dict(sorted(uncovered_materials.items(),
                                           key=lambda x: -x[1])[:20]),
    }

    return results


# ======================================================================
# CLI: test and report
# ======================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PBR Material System")
    parser.add_argument("--count", action="store_true",
                        help="Count usable assets and report")
    parser.add_argument("--test-classify", type=str,
                        help="Test material classification")
    parser.add_argument("--test-colors", type=str,
                        help="Test PartNet color extraction for object ID")
    args = parser.parse_args()

    if args.count:
        results = count_usable_assets()
        print(json.dumps(results, indent=2))

    if args.test_classify:
        cat, comp = classify_material(args.test_classify)
        print(f"Material: {args.test_classify}")
        print(f"Category: {cat}")
        print(f"Composite: {comp}")
        tex = get_texture_set(cat, hash(args.test_classify))
        if tex:
            print(f"Texture set: {tex['id']}")
            for k, v in tex.items():
                if k not in ("id", "dir"):
                    print(f"  {k}: {os.path.basename(v)}")
        else:
            print("No texture set available")
        defaults = CATEGORY_DEFAULTS.get(cat, CATEGORY_DEFAULTS["plastic"])
        print(f"Defaults: {defaults}")

    if args.test_colors:
        colors = get_partnet_colors(args.test_colors)
        if colors:
            print(f"PartNet colors for {args.test_colors}:")
            for name, (r, g, b) in colors.items():
                print(f"  {name}: ({r:.3f}, {g:.3f}, {b:.3f})")
        else:
            print(f"No PartNet colors for {args.test_colors}")
