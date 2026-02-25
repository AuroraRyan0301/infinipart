#!/usr/bin/env python3
"""
Convert PartNet-Mobility models to Infinite-Mobility output format.

PartNet structure (per model):
  - meta.json: category, anno_id, model_id
  - mobility.urdf: links (base, link_0, link_1, ..., link_X_helper) and joints
  - textured_objs/: OBJ+MTL files referenced by URDF
  - semantics.txt: link_name motion_type semantic_name

IM output structure (per model):
  - origins.json: centroids per link
  - scene.urdf: links named l_0, l_1, ..., mesh paths as outputs/Factory/seed/objs/i/i.obj
  - outputs/Factory/seed/objs/i/i.obj: per-link merged OBJ
  - data_infos_{seed}.json at factory level

Key transforms:
  - PartNet OBJs are in a local frame; the fixed joint from base has
    rpy=(pi/2, 0, -pi/2) which rotates: x_world = -z_local, y_world = -x_local, z_world = y_local
  - Per-link visual origins (xyz offsets) must be baked into vertex positions
  - Multiple OBJs per link are merged (concatenated with index offsets)
  - _helper links become abstract links in IM format
"""

import argparse
import json
import math
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict


# ========== Category mapping ==========

# Categories that exist in both PartNet and IM -> add "Sapien" suffix
OVERLAP_CATEGORIES = {
    "Bottle", "Dishwasher", "Lamp", "Microwave", "Oven", "Toilet", "Window",
    "Door", "Faucet", "StorageFurniture", "Table", "Refrigerator",
    "KitchenPot", "Chair",
}

# Categories unique to PartNet -> direct factory name
NEW_CATEGORIES = {
    "Scissors", "Fan", "Globe", "Safe", "TrashCan", "Laptop", "Box",
    "Bucket", "Cart", "Clock", "CoffeeMachine", "Dispenser", "Display",
    "Eyeglasses", "FoldingChair", "Kettle", "Knife", "Lighter", "Pen",
    "Pliers", "Suitcase", "Stapler", "Switch", "Toaster", "USB",
    "WashingMachine", "Mouse", "Remote", "Printer", "Phone", "Camera",
    "Keyboard",
}


def category_to_factory(cat):
    """Map PartNet category to IM factory name."""
    if cat in OVERLAP_CATEGORIES:
        return f"{cat}SapienFactory"
    else:
        return f"{cat}Factory"


# ========== Rotation helpers ==========

def rpy_to_matrix(roll, pitch, yaw):
    """Convert RPY (roll=Rx, pitch=Ry, yaw=Rz) to 3x3 rotation matrix.
    URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    R = [
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr],
    ]
    return R


def mat_vec_mul(R, v):
    """Multiply 3x3 matrix by 3-vector."""
    return [
        R[0][0]*v[0] + R[0][1]*v[1] + R[0][2]*v[2],
        R[1][0]*v[0] + R[1][1]*v[1] + R[1][2]*v[2],
        R[2][0]*v[0] + R[2][1]*v[1] + R[2][2]*v[2],
    ]


def mat_mat_mul(A, B):
    """Multiply two 3x3 matrices."""
    return [
        [sum(A[i][k]*B[k][j] for k in range(3)) for j in range(3)]
        for i in range(3)
    ]


def mat_transpose(R):
    """Transpose a 3x3 matrix."""
    return [[R[j][i] for j in range(3)] for i in range(3)]


# ========== OBJ I/O ==========

def parse_obj(filepath):
    """Parse an OBJ file. Returns vertices, normals, texcoords, faces, and
    material/group directives.

    Each face element is a list of tuples: (v_idx, vt_idx, vn_idx) where
    indices are 1-based (0 means absent). Face format can be:
      f v v v
      f v//vn v//vn v//vn
      f v/vt v/vt v/vt
      f v/vt/vn v/vt/vn v/vt/vn
    """
    vertices = []   # list of [x, y, z]
    normals = []    # list of [nx, ny, nz]
    texcoords = []  # list of [u, v, ...]
    faces = []      # list of [ [(vi, vti, vni), ...], ...]
    materials = []  # list of (line_type, value, face_index)
    mtllib = None

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if parts[0] == 'v' and len(parts) >= 4:
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vn' and len(parts) >= 4:
                normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif parts[0] == 'vt' and len(parts) >= 3:
                texcoords.append([float(x) for x in parts[1:]])
            elif parts[0] == 'f':
                face = []
                for vert_str in parts[1:]:
                    # Parse v, v/vt, v//vn, v/vt/vn
                    indices = vert_str.split('/')
                    vi = int(indices[0])
                    vti = 0
                    vni = 0
                    if len(indices) >= 2 and indices[1]:
                        vti = int(indices[1])
                    if len(indices) >= 3 and indices[2]:
                        vni = int(indices[2])
                    face.append((vi, vti, vni))
                faces.append(face)
            elif parts[0] == 'mtllib':
                mtllib = ' '.join(parts[1:])
            elif parts[0] in ('usemtl', 'o', 'g', 's'):
                materials.append((parts[0], ' '.join(parts[1:]), len(faces)))

    return {
        'vertices': vertices,
        'normals': normals,
        'texcoords': texcoords,
        'faces': faces,
        'materials': materials,
        'mtllib': mtllib,
    }


def transform_vertices(vertices, offset, rotation=None):
    """Apply offset (translation) and optional rotation to vertices.
    offset: [ox, oy, oz] added to each vertex.
    rotation: 3x3 matrix applied AFTER offset (or None for identity).
    """
    result = []
    for v in vertices:
        vt = [v[0] + offset[0], v[1] + offset[1], v[2] + offset[2]]
        if rotation is not None:
            vt = mat_vec_mul(rotation, vt)
        result.append(vt)
    return result


def transform_normals(normals, rotation):
    """Apply rotation to normals (no translation)."""
    if rotation is None:
        return [n[:] for n in normals]
    result = []
    for n in normals:
        result.append(mat_vec_mul(rotation, n))
    return result


def merge_objs(obj_list):
    """Merge multiple parsed OBJ dicts into one.
    Adjusts face indices to account for concatenation.
    Returns a merged OBJ dict.
    """
    merged = {
        'vertices': [],
        'normals': [],
        'texcoords': [],
        'faces': [],
        'materials': [],
        'mtllib': None,
    }

    v_offset = 0
    vn_offset = 0
    vt_offset = 0

    for obj in obj_list:
        merged['vertices'].extend(obj['vertices'])
        merged['normals'].extend(obj['normals'])
        merged['texcoords'].extend(obj['texcoords'])

        if obj['mtllib'] and merged['mtllib'] is None:
            merged['mtllib'] = obj['mtllib']

        # Adjust material/group references
        for mtype, mval, fidx in obj['materials']:
            merged['materials'].append((mtype, mval, fidx + len(merged['faces'])))

        # Adjust face indices
        for face in obj['faces']:
            new_face = []
            for vi, vti, vni in face:
                new_vi = vi + v_offset if vi > 0 else vi - v_offset  # handle negative indices too
                new_vti = (vti + vt_offset) if vti != 0 else 0
                new_vni = (vni + vn_offset) if vni != 0 else 0
                # Negative indices count from end - only offset positive ones
                if vi > 0:
                    new_vi = vi + v_offset
                else:
                    new_vi = vi  # negative indices are relative, no offset needed
                if vti > 0:
                    new_vti = vti + vt_offset
                elif vti < 0:
                    new_vti = vti
                else:
                    new_vti = 0
                if vni > 0:
                    new_vni = vni + vn_offset
                elif vni < 0:
                    new_vni = vni
                else:
                    new_vni = 0
                new_face.append((new_vi, new_vti, new_vni))
            merged['faces'].append(new_face)

        v_offset += len(obj['vertices'])
        vn_offset += len(obj['normals'])
        vt_offset += len(obj['texcoords'])

    return merged


def write_obj(filepath, obj_data, mtl_filename=None):
    """Write an OBJ file from parsed data."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'w') as f:
        f.write("# Converted from PartNet\n")
        if mtl_filename:
            f.write(f"mtllib {mtl_filename}\n")

        # Write vertices
        for v in obj_data['vertices']:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        # Write texture coords
        for vt in obj_data['texcoords']:
            f.write("vt " + " ".join(f"{x:.6f}" for x in vt) + "\n")

        # Write normals
        for vn in obj_data['normals']:
            f.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")

        # Determine face format
        has_vt = len(obj_data['texcoords']) > 0
        has_vn = len(obj_data['normals']) > 0

        # Write materials and faces
        mat_dict = {}
        for mtype, mval, fidx in obj_data['materials']:
            mat_dict.setdefault(fidx, []).append((mtype, mval))

        for i, face in enumerate(obj_data['faces']):
            if i in mat_dict:
                for mtype, mval in mat_dict[i]:
                    f.write(f"{mtype} {mval}\n")

            face_parts = []
            for vi, vti, vni in face:
                if vti != 0 and vni != 0:
                    face_parts.append(f"{vi}/{vti}/{vni}")
                elif vni != 0:
                    face_parts.append(f"{vi}//{vni}")
                elif vti != 0:
                    face_parts.append(f"{vi}/{vti}")
                else:
                    face_parts.append(f"{vi}")
            f.write("f " + " ".join(face_parts) + "\n")


def compute_centroid(vertices):
    """Compute the centroid of a list of vertices."""
    if not vertices:
        return [0.0, 0.0, 0.0]
    n = len(vertices)
    cx = sum(v[0] for v in vertices) / n
    cy = sum(v[1] for v in vertices) / n
    cz = sum(v[2] for v in vertices) / n
    return [cx, cy, cz]


# ========== MTL handling ==========

def collect_mtl_files(partnet_dir, obj_filenames):
    """Collect all MTL content from the OBJ files' mtllib references.
    Returns merged MTL content as a string, with material names de-duplicated.
    """
    mtl_content = []
    seen_mtl = set()

    for obj_fn in obj_filenames:
        # Derive MTL filename from OBJ filename
        base = os.path.splitext(os.path.basename(obj_fn))[0]
        mtl_path = os.path.join(partnet_dir, "textured_objs", base + ".mtl")
        if os.path.exists(mtl_path) and mtl_path not in seen_mtl:
            seen_mtl.add(mtl_path)
            with open(mtl_path, 'r') as f:
                mtl_content.append(f.read())

    return "\n".join(mtl_content)


# ========== URDF parsing ==========

def parse_partnet_urdf(urdf_path):
    """Parse a PartNet mobility.urdf file.

    Returns:
        links: dict of link_name -> {
            'visuals': [{'origin_xyz': [x,y,z], 'mesh': 'textured_objs/xxx.obj'}, ...],
            'is_helper': bool,
        }
        joints: dict of joint_name -> {
            'type': str,
            'origin_xyz': [x,y,z],
            'origin_rpy': [r,p,y] or None,
            'axis': [ax,ay,az] or None,
            'parent': str,
            'child': str,
            'limit_lower': float or None,
            'limit_upper': float or None,
        }
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for link_elem in root.findall('link'):
        name = link_elem.get('name')
        visuals = []
        for vis in link_elem.findall('visual'):
            origin = vis.find('origin')
            xyz = [0.0, 0.0, 0.0]
            if origin is not None and origin.get('xyz'):
                xyz = [float(x) for x in origin.get('xyz').split()]
            geom = vis.find('geometry')
            mesh_fn = None
            if geom is not None:
                mesh = geom.find('mesh')
                if mesh is not None:
                    mesh_fn = mesh.get('filename')
            if mesh_fn:
                visuals.append({'origin_xyz': xyz, 'mesh': mesh_fn})

        links[name] = {
            'visuals': visuals,
            'is_helper': name.endswith('_helper'),
        }

    joints = {}
    for joint_elem in root.findall('joint'):
        jname = joint_elem.get('name')
        jtype = joint_elem.get('type')

        origin = joint_elem.find('origin')
        xyz = [0.0, 0.0, 0.0]
        rpy = None
        if origin is not None:
            if origin.get('xyz'):
                xyz = [float(x) for x in origin.get('xyz').split()]
            if origin.get('rpy'):
                rpy = [float(x) for x in origin.get('rpy').split()]

        axis_elem = joint_elem.find('axis')
        axis = None
        if axis_elem is not None and axis_elem.get('xyz'):
            axis = [float(x) for x in axis_elem.get('xyz').split()]

        parent = joint_elem.find('parent').get('link')
        child = joint_elem.find('child').get('link')

        limit_lower = None
        limit_upper = None
        limit_elem = joint_elem.find('limit')
        if limit_elem is not None:
            if limit_elem.get('lower') is not None:
                limit_lower = float(limit_elem.get('lower'))
            if limit_elem.get('upper') is not None:
                limit_upper = float(limit_elem.get('upper'))

        joints[jname] = {
            'type': jtype,
            'origin_xyz': xyz,
            'origin_rpy': rpy,
            'axis': axis,
            'parent': parent,
            'child': child,
            'limit_lower': limit_lower,
            'limit_upper': limit_upper,
        }

    return links, joints


def parse_semantics(semantics_path):
    """Parse semantics.txt.
    Returns dict: link_name -> {'motion': str, 'semantic': str}
    """
    result = {}
    if not os.path.exists(semantics_path):
        return result
    with open(semantics_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                result[parts[0]] = {'motion': parts[1], 'semantic': parts[2]}
            elif len(parts) == 2:
                result[parts[0]] = {'motion': parts[1], 'semantic': 'unknown'}
    return result


# ========== Tree traversal ==========

def build_kinematic_tree(links, joints):
    """Build a kinematic tree from parsed URDF.
    Returns:
        children: dict link_name -> [(joint_name, child_link_name), ...]
        parent_joint: dict link_name -> joint_name (the joint connecting to parent)
        root: the root link name (usually 'base')
    """
    children = {name: [] for name in links}
    parent_joint = {}

    for jname, jdata in joints.items():
        parent = jdata['parent']
        child = jdata['child']
        if parent in children:
            children[parent].append((jname, child))
        parent_joint[child] = jname

    # Find root (link with no parent joint)
    root = None
    for name in links:
        if name not in parent_joint:
            root = name
            break

    return children, parent_joint, root


def get_base_rotation(joints):
    """Find the fixed joint from 'base' that carries the coordinate rotation.
    Returns the 3x3 rotation matrix, or None if no such joint exists.
    """
    for jname, jdata in joints.items():
        if jdata['parent'] == 'base' and jdata['type'] == 'fixed':
            if jdata['origin_rpy'] is not None:
                rpy = jdata['origin_rpy']
                # Check if it is a non-trivial rotation
                if any(abs(x) > 1e-6 for x in rpy):
                    return rpy_to_matrix(rpy[0], rpy[1], rpy[2])
    return None


# ========== Main conversion logic ==========

def compute_world_transforms(links, joints, children, parent_joint, root, base_rotation):
    """Compute the world-frame transform for each link by walking the tree.

    For each link, compute:
      - rotation: 3x3 matrix (cumulative from root)
      - translation: [x,y,z] in world frame

    The base_rotation is applied to convert from PartNet's local frame
    (where Y is up) to world frame (where Z is up).

    Returns dict: link_name -> {'rotation': R, 'translation': [x,y,z]}
    """
    transforms = {}
    identity = [[1,0,0],[0,1,0],[0,0,1]]

    # BFS from root
    queue = [(root, identity, [0.0, 0.0, 0.0])]
    while queue:
        link_name, parent_R, parent_t = queue.pop(0)
        transforms[link_name] = {'rotation': parent_R, 'translation': parent_t}

        for jname, child_name in children.get(link_name, []):
            jdata = joints[jname]
            # Joint origin is relative to parent link frame
            j_xyz = jdata['origin_xyz']
            j_rpy = jdata['origin_rpy']

            # Compute child frame position in world
            # child_pos = parent_R * j_xyz + parent_t
            rotated_xyz = mat_vec_mul(parent_R, j_xyz)
            child_t = [parent_t[i] + rotated_xyz[i] for i in range(3)]

            # Compute child frame rotation
            if j_rpy is not None and any(abs(x) > 1e-6 for x in j_rpy):
                j_R = rpy_to_matrix(j_rpy[0], j_rpy[1], j_rpy[2])
                child_R = mat_mat_mul(parent_R, j_R)
            else:
                child_R = [row[:] for row in parent_R]

            queue.append((child_name, child_R, child_t))

    return transforms


def convert_model(partnet_model_dir, output_dir, factory_name, seed):
    """Convert a single PartNet model to IM format.

    Args:
        partnet_model_dir: path to PartNet model directory (e.g., .../Partnet/3380/)
        output_dir: IM outputs root (e.g., .../Infinite-Mobility/outputs/)
        factory_name: IM factory name (e.g., BottleSapienFactory)
        seed: integer seed (model index)

    Returns True on success, False on error.
    """
    anno_id = os.path.basename(partnet_model_dir)
    urdf_path = os.path.join(partnet_model_dir, "mobility.urdf")
    semantics_path = os.path.join(partnet_model_dir, "semantics.txt")

    if not os.path.exists(urdf_path):
        print(f"  [SKIP] No mobility.urdf found for {anno_id}")
        return False

    # Parse inputs
    links, joints = parse_partnet_urdf(urdf_path)
    semantics = parse_semantics(semantics_path)
    children, parent_joint, root = build_kinematic_tree(links, joints)
    base_rotation = get_base_rotation(joints)

    # Compute world transforms for every link
    transforms = compute_world_transforms(links, joints, children, parent_joint, root, base_rotation)

    # Output paths
    model_out = os.path.join(output_dir, factory_name, str(seed))
    objs_out = os.path.join(model_out, "outputs", factory_name, str(seed), "objs")
    os.makedirs(model_out, exist_ok=True)

    # Identify real links (with visuals) and helper links (no visuals, used for compound joints)
    # Also identify the fixed joint's child (the "base body" link in PartNet)
    real_links = []       # links with visual meshes, ordered
    helper_links = []     # helper links (no visuals, compound joint intermediaries)
    base_fixed_child = None

    for jname, jdata in joints.items():
        if jdata['parent'] == 'base' and jdata['type'] == 'fixed':
            base_fixed_child = jdata['child']
            break

    # BFS to get ordered list of links (excluding 'base')
    visited = set()
    bfs_order = []
    queue = [root]
    while queue:
        ln = queue.pop(0)
        if ln in visited:
            continue
        visited.add(ln)
        if ln != 'base':
            bfs_order.append(ln)
        for jname, child_name in children.get(ln, []):
            queue.append(child_name)

    for ln in bfs_order:
        if links[ln]['is_helper']:
            helper_links.append(ln)
        elif links[ln]['visuals']:
            real_links.append(ln)
        # Links with no visuals and not helpers (empty links) are skipped

    # Assign IM link indices: real links get sequential l_0, l_1, ...
    im_link_map = {}  # partnet_link_name -> im_link_name
    im_idx = 0
    for ln in real_links:
        im_link_map[ln] = f"l_{im_idx}"
        im_idx += 1

    # Helper links get abstract names
    abstract_counter = 0
    for ln in helper_links:
        im_link_map[ln] = f"link_abstract_{abstract_counter}"
        abstract_counter += 1

    # ---- Process meshes for each real link ----
    # Load, apply visual origin offset, apply world rotation, merge, compute centroid
    origins = OrderedDict()
    data_infos_parts = []

    for link_idx, ln in enumerate(real_links):
        im_name = im_link_map[ln]
        link_data = links[ln]
        link_transform = transforms.get(ln, {'rotation': [[1,0,0],[0,1,0],[0,0,1]], 'translation': [0,0,0]})
        link_R = link_transform['rotation']
        link_t = link_transform['translation']

        obj_list = []
        obj_filenames = []
        for vis in link_data['visuals']:
            mesh_path = os.path.join(partnet_model_dir, vis['mesh'])
            if not os.path.exists(mesh_path):
                print(f"  [WARN] Missing mesh: {mesh_path}")
                continue

            parsed = parse_obj(mesh_path)
            obj_filenames.append(vis['mesh'])

            # Apply visual origin offset to vertices (in link-local frame)
            origin_xyz = vis['origin_xyz']
            parsed['vertices'] = transform_vertices(parsed['vertices'], origin_xyz)

            # Apply link's world transform (rotation) to vertices
            # The world position of a vertex: R_link * v_local + t_link
            # But for IM format, meshes are stored in world frame (origin at world origin)
            # so we apply: v_world = R_link * v_local + t_link
            new_verts = []
            for v in parsed['vertices']:
                vw = mat_vec_mul(link_R, v)
                vw = [vw[i] + link_t[i] for i in range(3)]
                new_verts.append(vw)
            parsed['vertices'] = new_verts

            # Apply rotation to normals (no translation)
            parsed['normals'] = transform_normals(parsed['normals'], link_R)

            obj_list.append(parsed)

        if not obj_list:
            print(f"  [WARN] No valid meshes for link {ln}")
            continue

        # Merge all OBJs for this link
        merged = merge_objs(obj_list)

        # Compute centroid
        centroid = compute_centroid(merged['vertices'])
        idx_str = str(link_idx)
        origins[idx_str] = centroid

        # Subtract centroid from vertices (IM format stores centroid-subtracted OBJs;
        # render_articulation.py load_scene_parts() adds centroid back via obj.location)
        merged['vertices'] = [[v[i] - centroid[i] for i in range(3)]
                              for v in merged['vertices']]

        # Write merged OBJ
        obj_dir = os.path.join(objs_out, idx_str)
        obj_path = os.path.join(obj_dir, f"{idx_str}.obj")
        mtl_filename = f"{idx_str}.mtl"
        write_obj(obj_path, merged, mtl_filename=mtl_filename)

        # Collect and write merged MTL
        mtl_content = collect_mtl_files(partnet_model_dir, obj_filenames)
        if mtl_content.strip():
            mtl_path = os.path.join(obj_dir, mtl_filename)
            with open(mtl_path, 'w') as f:
                f.write(mtl_content)

        # data_infos entry
        sem = semantics.get(ln, {'semantic': 'unknown'})
        part_name = sem['semantic'] + "_part"
        data_infos_parts.append({
            "part_name": part_name,
            "file_name": f"{idx_str}.obj",
            "file_obj_path": f"outputs/{factory_name}/{seed}/objs/{idx_str}.obj",
        })

    # ---- Copy texture images directory ----
    # MTL files reference textures as "../images/texture_0.jpg" (relative to part dir),
    # so images/ must be at the same level as the numbered part directories.
    src_images_dir = os.path.join(partnet_model_dir, "images")
    if os.path.isdir(src_images_dir):
        dst_images_dir = os.path.join(objs_out, "images")
        shutil.copytree(src_images_dir, dst_images_dir, dirs_exist_ok=True)

    origins["world"] = [0, 0, 0]

    # ---- Generate IM URDF ----
    urdf_str = generate_im_urdf(
        factory_name, seed, real_links, helper_links,
        im_link_map, links, joints, children, parent_joint,
        transforms, base_rotation, origins, base_fixed_child,
    )

    # Write outputs
    with open(os.path.join(model_out, "origins.json"), 'w') as f:
        json.dump(origins, f, indent=2)

    with open(os.path.join(model_out, "scene.urdf"), 'w') as f:
        f.write(urdf_str)

    # Write data_infos at factory level
    data_infos_path = os.path.join(output_dir, factory_name, f"data_infos_{seed}.json")
    with open(data_infos_path, 'w') as f:
        json.dump([{"part": data_infos_parts}], f, indent=2)

    return True


def generate_im_urdf(factory_name, seed, real_links, helper_links,
                     im_link_map, links, joints, children, parent_joint,
                     transforms, base_rotation, origins, base_fixed_child):
    """Generate Infinite-Mobility format URDF string."""

    lines = []
    lines.append("<?xml version='1.0' encoding='UTF-8'?>")
    lines.append('<robot name="scene">')

    # World link
    lines.append('  <link name="l_world">')
    lines.append('    <inertial>')
    lines.append('      <origin xyz="0.0 0.0 0.0" rpy="0.0 -0.0 0.0"/>')
    lines.append('      <mass value="1.0"/>')
    lines.append('      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>')
    lines.append('    </inertial>')
    lines.append('  </link>')

    # Real links with visuals
    for link_idx, ln in enumerate(real_links):
        im_name = im_link_map[ln]
        idx_str = str(link_idx)
        lines.append(f'  <link name="{im_name}">')
        lines.append('    <inertial>')
        lines.append('      <origin xyz="0.0 0.0 0.0" rpy="0.0 -0.0 0.0"/>')
        lines.append('      <mass value="1.0"/>')
        lines.append('      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>')
        lines.append('    </inertial>')
        lines.append('    <visual>')
        lines.append('      <geometry>')
        lines.append(f'        <mesh filename="outputs/{factory_name}/{seed}/objs/{idx_str}/{idx_str}.obj"/>')
        lines.append('      </geometry>')
        lines.append('      <origin xyz="0.0 0.0 0.0" rpy="0.0 -0.0 0.0"/>')
        lines.append('    </visual>')
        lines.append('  </link>')

    # Helper/abstract links (no visuals)
    for ln in helper_links:
        im_name = im_link_map[ln]
        lines.append(f'  <link name="{im_name}">')
        lines.append('    <inertial>')
        lines.append('      <origin xyz="0.0 0.0 0.0" rpy="0.0 -0.0 0.0"/>')
        lines.append('      <mass value="1.0"/>')
        lines.append('      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>')
        lines.append('    </inertial>')
        lines.append('  </link>')

    # Generate joints
    # Strategy:
    # - The fixed joint from base to its child becomes a fixed joint from l_world to the child's IM link
    #   with origin at the child's centroid (in world frame)
    # - All other joints: convert parent/child to IM names, transform origin and axis to world frame
    joint_counter = 0
    for jname, jdata in joints.items():
        pn_parent = jdata['parent']
        pn_child = jdata['child']
        jtype = jdata['type']

        # Skip the base link itself
        if pn_parent == 'base' and jtype == 'fixed':
            # This is the coordinate transform joint. In IM, the child of base
            # connects to l_world with a fixed joint.
            # The child is typically the "body" link.
            child_im = im_link_map.get(pn_child)
            if child_im is None:
                continue

            # Origin: the centroid of the child link in world frame
            child_idx = real_links.index(pn_child) if pn_child in real_links else -1
            if child_idx >= 0 and str(child_idx) in origins:
                centroid = origins[str(child_idx)]
            else:
                centroid = [0.0, 0.0, 0.0]

            lines.append(f'  <joint name="joint_fixed_{joint_counter}" type="fixed">')
            lines.append(f'    <limit effort="2000.0" velocity="2.0" lower="-1.0" upper="1.0"/>')
            lines.append(f'    <parent link="l_world"/>')
            lines.append(f'    <child link="{child_im}"/>')
            lines.append(f'    <axis xyz="1. 0. 0."/>')
            xyz_str = f"{centroid[0]} {centroid[1]} {centroid[2]}"
            lines.append(f'    <origin xyz="{xyz_str}" rpy="0.0 -0.0 0.0"/>')
            lines.append(f'  </joint>')
            joint_counter += 1
            continue

        # Map parent and child to IM names
        im_parent = im_link_map.get(pn_parent)
        im_child = im_link_map.get(pn_child)

        if im_parent is None or im_child is None:
            # Parent or child not in our link map (e.g. 'base')
            # If parent is base, remap to l_world
            if pn_parent == 'base':
                im_parent = 'l_world'
            else:
                continue
            if im_child is None:
                continue

        # Transform joint origin to world frame
        # Joint origin is relative to the parent link frame
        parent_tf = transforms.get(pn_parent, {'rotation': [[1,0,0],[0,1,0],[0,0,1]], 'translation': [0,0,0]})
        parent_R = parent_tf['rotation']
        parent_t = parent_tf['translation']

        # World-frame joint origin
        j_xyz = jdata['origin_xyz']
        j_world = mat_vec_mul(parent_R, j_xyz)
        j_world = [j_world[i] + parent_t[i] for i in range(3)]

        # For the IM URDF, joint origins are relative to the parent link.
        # In IM format, since all meshes are in world frame and link origins are at
        # world origin, the joint origin should express the offset from the parent
        # link's reference point. Since IM links don't have an inherent position
        # (their origin is the world origin), we express it relative to parent centroid.
        parent_centroid = [0.0, 0.0, 0.0]
        if im_parent == "l_world":
            parent_centroid = [0.0, 0.0, 0.0]
        elif pn_parent in real_links:
            pidx = real_links.index(pn_parent)
            pidx_str = str(pidx)
            if pidx_str in origins:
                parent_centroid = origins[pidx_str]

        # Relative joint origin (from parent's centroid in world frame)
        rel_origin = [j_world[i] - parent_centroid[i] for i in range(3)]

        # Transform axis to world frame (rotation only)
        axis = jdata['axis']
        if axis is not None:
            axis_world = mat_vec_mul(parent_R, axis)
            # Normalize
            mag = math.sqrt(sum(a*a for a in axis_world))
            if mag > 1e-10:
                axis_world = [a/mag for a in axis_world]
        else:
            axis_world = [1.0, 0.0, 0.0]

        # Determine IM joint type and name
        im_jtype = jtype
        joint_name_prefix = f"joint_{jtype}"

        # For helper links that form compound joints, use descriptive naming
        if pn_child in helper_links or pn_parent in [h for h in helper_links]:
            # Check if this is part of a compound joint (helper chain)
            # Find what types of joints connect through the helper
            if links.get(pn_child, {}).get('is_helper', False):
                # Joint to a helper: this is the first part of a compound joint
                # Find the second joint (from helper to real link)
                second_joints = []
                for jn2, jd2 in joints.items():
                    if jd2['parent'] == pn_child:
                        second_joints.append((jn2, jd2))
                if second_joints:
                    # Name the compound joint
                    types = sorted([jtype] + [jd2['type'] for _, jd2 in second_joints])
                    joint_name_prefix = "joint_" + "_".join(types)

        # Set limit values
        lower = jdata['limit_lower']
        upper = jdata['limit_upper']
        if lower is None:
            lower = -1.0
        if upper is None:
            upper = 1.0

        # Ensure limits make sense
        if lower > upper:
            lower, upper = upper, lower

        # For prismatic joints, keep original limits; for revolute, also keep
        # For continuous, use default
        if jtype == 'continuous':
            lower = -1.0
            upper = 1.0

        lines.append(f'  <joint name="{joint_name_prefix}_{joint_counter}" type="{im_jtype}">')
        lines.append(f'    <limit effort="2000.0" velocity="2.0" lower="{lower}" upper="{upper}"/>')
        lines.append(f'    <parent link="{im_parent}"/>')
        lines.append(f'    <child link="{im_child}"/>')
        axis_str = f"{axis_world[0]:.6g} {axis_world[1]:.6g} {axis_world[2]:.6g}"
        lines.append(f'    <axis xyz="{axis_str}"/>')
        origin_str = f"{rel_origin[0]} {rel_origin[1]} {rel_origin[2]}"
        lines.append(f'    <origin xyz="{origin_str}" rpy="0.0 -0.0 0.0"/>')
        lines.append(f'  </joint>')
        joint_counter += 1

    lines.append('</robot>')
    return "\n".join(lines) + "\n"


# ========== Main ==========

def discover_models(partnet_dir, category=None):
    """Discover all PartNet models, optionally filtered by category.

    Returns dict: category -> [list of (anno_id, model_dir)]
    """
    result = {}
    for d in sorted(os.listdir(partnet_dir)):
        model_dir = os.path.join(partnet_dir, d)
        meta_path = os.path.join(model_dir, "meta.json")
        if not os.path.isdir(model_dir) or not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        cat = meta.get("model_cat", "")
        if category and cat != category:
            continue
        result.setdefault(cat, []).append((meta.get("anno_id", d), model_dir))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert PartNet-Mobility models to Infinite-Mobility output format."
    )
    parser.add_argument(
        "--partnet_dir", type=str,
        default="/mnt/data/yurh/dataset3D/Partnet",
        help="Root directory of PartNet models",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/mnt/data/yurh/Infinite-Mobility/outputs",
        help="IM outputs root directory",
    )
    parser.add_argument(
        "--category", type=str, default=None,
        help="PartNet category to convert (e.g. 'Bottle'). If not specified, use --all.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Convert all categories",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing conversions (skip check for origins.json)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only list models that would be converted, without doing anything",
    )
    args = parser.parse_args()

    if not args.category and not args.all:
        parser.error("Specify --category <name> or --all")

    # Discover models
    models_by_cat = discover_models(args.partnet_dir, args.category)
    if not models_by_cat:
        print(f"No models found in {args.partnet_dir}" +
              (f" for category '{args.category}'" if args.category else ""))
        return

    all_cats = sorted(models_by_cat.keys())
    total_models = sum(len(v) for v in models_by_cat.values())
    print(f"Found {total_models} models across {len(all_cats)} categories")

    if args.dry_run:
        for cat in all_cats:
            factory = category_to_factory(cat)
            print(f"  {cat} -> {factory}: {len(models_by_cat[cat])} models")
        return

    # Convert
    success_count = 0
    skip_count = 0
    error_count = 0

    for cat in all_cats:
        factory = category_to_factory(cat)
        model_list = models_by_cat[cat]
        print(f"\n{'='*60}")
        print(f"Category: {cat} -> {factory} ({len(model_list)} models)")
        print(f"{'='*60}")

        # Create factory directory
        factory_dir = os.path.join(args.output_dir, factory)
        os.makedirs(factory_dir, exist_ok=True)

        for seed, (anno_id, model_dir) in enumerate(model_list):
            model_out = os.path.join(args.output_dir, factory, str(seed))
            origins_path = os.path.join(model_out, "origins.json")

            # Check if already converted
            if not args.force and os.path.exists(origins_path):
                skip_count += 1
                continue

            print(f"  [{seed+1}/{len(model_list)}] Converting {anno_id} -> {factory}/{seed} ...", end="", flush=True)
            try:
                ok = convert_model(model_dir, args.output_dir, factory, seed)
                if ok:
                    success_count += 1
                    print(" OK")
                else:
                    error_count += 1
                    print(" FAILED")
            except Exception as e:
                error_count += 1
                print(f" ERROR: {e}")

        # Write a mapping file for this factory (anno_id -> seed)
        mapping = {str(seed): anno_id for seed, (anno_id, _) in enumerate(model_list)}
        mapping_path = os.path.join(factory_dir, "partnet_mapping.json")
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done. Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")


if __name__ == "__main__":
    main()
