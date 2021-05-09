import os
import json
import numpy as np
import math
from math import inf
import bpy
from mathutils import Vector

# direction names for minecraft cube face UVs
DIRECTIONS = np.array([
    "north",
    "east",
    "west",
    "south",
    "up",
    "down",
])

# normals for minecraft directions in BLENDER world space
# e.g. blender (-1, 0, 0) is minecraft north (0, 0, -1)
# shape (f,n,v) = (6,6,3)
#   f = 6: number of cuboid faces to test
#   n = 6: number of normal directions
#   v = 3: vector coordinates (x,y,z)
DIRECTION_NORMALS = np.array([
    [-1.,  0.,  0.],
    [ 0.,  1.,  0.],
    [ 0., -1.,  0.],
    [ 1.,  0.,  0.],
    [ 0.,  0.,  1.],
    [ 0.,  0., -1.],
])
DIRECTION_NORMALS = np.tile(DIRECTION_NORMALS[np.newaxis,...], (6,1,1))


def index_of(val, in_list):
    """Return index of value in in_list"""
    try:
        return in_list.index(val)
    except ValueError:
        return -1 


def merge_dict_properties(dict_original, d):
    """Merge inner dict properties"""
    for k in d:
        if k in dict_original and isinstance(dict_original[k], dict):
            dict_original[k].update(d[k])
        else:
            dict_original[k] = d[k]
    
    return dict_original


def get_base_path(filepath_parts, curr_branch=None, new_branch=None):
    """"Typical path formats for texture is like:
            "textures": {
                "skin": "entity/wolf/wolf9"
            },
    Matches the base path before branch, then get base path of other branch
        curr_branch = "shapes"
        new_branch = "textures"
        filepath_parts = ["C:", "vs", "resources", "shapes", "entity", "land", "wolf-male.json"]
                                                       |
                                             Matched branch point
        
        new base path = ["C:", "vs", "resources"] + ["textures"]
    """
    # match base path
    idx_base_path = index_of(curr_branch, filepath_parts)

    if idx_base_path != -1:
        # system agnostic path join
        joined_path = os.path.join(os.sep, filepath_parts[0] + os.sep, *filepath_parts[1:idx_base_path], new_branch)
        return joined_path
    else:
        return "" # failed


def create_textured_principled_bsdf(mat_name, tex_path):
    """Create new material with `mat_name` and texture path `tex_path`
    """
    mat = bpy.data.materials.new(mat_name)
    mat.use_nodes = True
    node_tree = mat.node_tree
    nodes = node_tree.nodes
    bsdf = nodes.get("Principled BSDF") 

    # add texture node
    if bsdf is not None:
        if "Base Color" in bsdf.inputs:
            tex_input = nodes.new(type="ShaderNodeTexImage")
            tex_input.interpolation = "Closest"

            # load image, if fail make a new image with filepath set to tex path
            try:
                img = bpy.data.images.load(tex_path, check_existing=True)
            except:
                print("FAILED TO LOAD IMAGE:", tex_path)
                img = bpy.data.images.new(os.path.split(tex_path)[-1], width=16, height=16)
                img.filepath = tex_path
        
            tex_input.image = img
            node_tree.links.new(tex_input.outputs[0], bsdf.inputs["Base Color"])
        
        # disable shininess
        if "Specular" in bsdf.inputs:
            bsdf.inputs["Specular"].default_value = 0.0
    
    return mat


def parse_element(
    e,
    parent_cube_origin,
    parent_rotation_origin,  # = blender origin
    textures,
    tex_width=16.0,
    tex_height=16.0,
    import_uvs=True,
):
    """Load a single element into a Blender object.
    Note vintage story cube origins are relative to the parent's
    "from" corner, the origin input is the parent cube's from vertex.
                     from     to    (relative to parent_cube_origin)
                       |       |
                       v       v
                       |   x   |   child
                       |_______| 
                |
                |  xp  <------------parent rotation origin
                |.____  parent         (blender origin)
            parent_cube_origin

              .  
            (0,0,0)
    
    New locations in blender space:
        child_blender_origin = parent_cube_origin - parent_rotation_origin + child_rotation_origin
        from_blender_local = from - child_rotation_origin
        to_blender_local = to - child_rotation_origin
    """
    # get cube min/max
    v_min = np.array([e["from"][2], e["from"][0], e["from"][1]])
    v_max = np.array([e["to"][2], e["to"][0], e["to"][1]])

    # get rotation origin
    location = np.array([
        parent_cube_origin[0] - parent_rotation_origin[0],
        parent_cube_origin[1] - parent_rotation_origin[1], 
        parent_cube_origin[2] - parent_rotation_origin[2],
    ])
    if "rotationOrigin" in e: # add rotation origin
        child_rotation_origin = np.array([
            e["rotationOrigin"][2],
            e["rotationOrigin"][0],
            e["rotationOrigin"][1],
        ])
        location = location + child_rotation_origin
    else:
        child_rotation_origin = np.array([0.0, 0.0, 0.0])
    
    # this cube corner origin
    new_cube_origin = parent_cube_origin + v_min
    new_rotation_origin = parent_rotation_origin + location

    # get euler rotation
    rot_euler = np.array([0.0, 0.0, 0.0])
    if "rotationX" in e:
        rot_euler[1] = e["rotationX"] * math.pi / 180.0
    if "rotationY" in e:
        rot_euler[2] = e["rotationY"] * math.pi / 180.0
    if "rotationZ" in e:
        rot_euler[0] = e["rotationZ"] * math.pi / 180.0

    # create cube
    bpy.ops.mesh.primitive_cube_add(location=location, rotation=rot_euler)
    obj = bpy.context.active_object
    mesh = obj.data
    mesh_materials = {} # tex_name => material_index

    # center local mesh coordiantes
    v_min = v_min - child_rotation_origin
    v_max = v_max - child_rotation_origin
    
    # set vertices
    mesh.vertices[0].co[:] = v_min[0], v_min[1], v_min[2]
    mesh.vertices[1].co[:] = v_min[0], v_min[1], v_max[2]
    mesh.vertices[2].co[:] = v_min[0], v_max[1], v_min[2]
    mesh.vertices[3].co[:] = v_min[0], v_max[1], v_max[2]
    mesh.vertices[4].co[:] = v_max[0], v_min[1], v_min[2]
    mesh.vertices[5].co[:] = v_max[0], v_min[1], v_max[2]
    mesh.vertices[6].co[:] = v_max[0], v_max[1], v_min[2]
    mesh.vertices[7].co[:] = v_max[0], v_max[1], v_max[2]

    # set face uvs
    uv = e.get("faces")
    if uv is not None:
        if import_uvs:
            face_normals = np.zeros((6,1,3))
            for i, face in enumerate(mesh.polygons):
                face_normals[i,0,0:3] = face.normal
            
            # map face normal -> face name
            # NOTE: this process may not be necessary since new blender
            # objects are created with the same face normal order,
            # so could directly map index -> minecraft face name.
            # keeping this in case the order changes in future
            face_directions = np.argmax(np.sum(face_normals * DIRECTION_NORMALS, axis=2), axis=1)
            face_directions = DIRECTIONS[face_directions]

            # set uvs face order in blender loop, determined experimentally
            uv_layer = mesh.uv_layers.active.data
            for uv_direction, face in zip(face_directions, mesh.polygons):
                face_uv = uv.get(uv_direction)
                if face_uv is not None:
                    if "uv" in face_uv:
                        # unpack uv coords in minecraft coord space [xmin, ymin, xmax, ymax]
                        # transform from minecraft [0, 16] space +x,-y space to blender [0,1] +x,+y
                        face_uv_coords = face_uv["uv"]
                        xmin = face_uv_coords[0] / tex_width
                        ymin = 1.0 - face_uv_coords[3] / tex_height
                        xmax = face_uv_coords[2] / tex_width
                        ymax = 1.0 - face_uv_coords[1] / tex_height
                    else:
                        xmin = 0.0
                        ymin = 1.0
                        xmax = 1.0
                        ymax = 0.0
                    
                    # write uv coords based on rotation
                    k = face.loop_start
                    if "rotation" not in face_uv or face_uv["rotation"] == 0:
                        uv_layer[k].uv[0:2] = xmax, ymin
                        uv_layer[k+1].uv[0:2] = xmax, ymax
                        uv_layer[k+2].uv[0:2] = xmin, ymax
                        uv_layer[k+3].uv[0:2] = xmin, ymin

                    elif face_uv["rotation"] == 90:
                        uv_layer[k].uv[0:2] = xmax, ymax
                        uv_layer[k+1].uv[0:2] = xmin, ymax
                        uv_layer[k+2].uv[0:2] = xmin, ymin
                        uv_layer[k+3].uv[0:2] = xmax, ymin

                    elif face_uv["rotation"] == 180:
                        uv_layer[k].uv[0:2] = xmin, ymax
                        uv_layer[k+1].uv[0:2] = xmin, ymin
                        uv_layer[k+2].uv[0:2] = xmax, ymin
                        uv_layer[k+3].uv[0:2] = xmax, ymax

                    elif face_uv["rotation"] == 270:
                        uv_layer[k].uv[0:2] = xmin, ymin
                        uv_layer[k+1].uv[0:2] = xmax, ymin
                        uv_layer[k+2].uv[0:2] = xmax, ymax
                        uv_layer[k+3].uv[0:2] = xmin, ymax

                    else: # invalid rotation, should never occur... do default
                        uv_layer[k].uv[0:2] = xmax, ymin
                        uv_layer[k+1].uv[0:2] = xmax, ymax
                        uv_layer[k+2].uv[0:2] = xmin, ymax
                        uv_layer[k+3].uv[0:2] = xmin, ymin

                    # assign material
                    if "texture" in face_uv:
                        tex_name = face_uv["texture"][1:] # remove the "#" in start
                        if tex_name in mesh_materials:
                            face.material_index = mesh_materials[tex_name]
                        elif tex_name in textures: # need new mapping
                            idx = len(obj.data.materials)
                            obj.data.materials.append(textures[tex_name])
                            mesh_materials[tex_name] = idx
                            face.material_index = idx

    # set name (choose whatever is available or "cube" if no name or comment is given)
    obj.name = e.get("name") or "cube"

    return obj, new_cube_origin, new_rotation_origin


def load_element(
    element,
    parent,
    cube_origin,
    rotation_origin,
    all_objects,
    textures,
    tex_width=16.0,
    tex_height=16.0,
    import_uvs=True,
):
    """Recursively load a geometry cuboid"""
    obj, new_cube_origin, new_rotation_origin = parse_element(
        element,
        cube_origin,
        rotation_origin,
        textures,
        tex_width=tex_width,
        tex_height=tex_height,
        import_uvs=import_uvs,
    )
    all_objects.append(obj)
    
    # set parent
    if parent is not None:
        obj.parent = parent
    
    if "children" in element:
        for child in element["children"]:
            load_element(
                child,
                obj,
                new_cube_origin,
                new_rotation_origin,
                all_objects,
                textures,
                tex_width,
                tex_height,
                import_uvs,
            )

    return obj


def load(context,
         filepath,
         import_uvs = True,               # import face uvs
         import_textures = True,          # import textures into materials
         translate_origin_by_8 = False,   # shift model by (-8, -8, -8)
         recenter_to_origin = True,       # recenter model to origin, overrides translate origin
         **kwargs):
    """Main import function"""

    with open(filepath, "r") as f:
        data = json.load(f)
    
    # chunks of import file path, to get base directory
    filepath_parts = filepath.split(os.path.sep)

    # check if groups in .json, not a spec, used by this exporter as additional data to group models together
    if "groups" in data:
        groups = data["groups"]
    else:
        groups = {}
    
    # objects created
    root_objects = []  # root level objects
    all_objects = []   # all objects added

    # vintage story coordinate system origin
    if translate_origin_by_8:
        vintagestory_origin = np.array([8., 8., 8.])
    else:
        # ignore if not translating
        vintagestory_origin = np.array([0., 0., 0.])

    # set scene collection as active
    scene_collection = bpy.context.view_layer.layer_collection
    bpy.context.view_layer.active_layer_collection = scene_collection

    # =============================================
    # import textures, create map of material name => material
    # =============================================
    """Assume two types of texture formats:
        "textures:" {
            "down": "#bottom",               # texture alias to another texture
            "bottom": "block/stone",   # actual texture image
        }

    Loading textures is two pass:
        1. load all actual texture images
        2. map aliases to loaded texture images
    """
    tex_width = data["textureWidth"] if "textureWidth" in data else 16.0
    tex_height = data["textureHeight"] if "textureHeight" in data else 16.0
    textures = {}
    if import_textures and "textures" in data:
        # get textures base path for models
        tex_base_path = get_base_path(filepath_parts, curr_branch="shapes", new_branch="textures")

        # load texture images
        for tex_name, tex_path in data["textures"].items():
            # skip aliases
            if tex_path[0] == "#":
                continue
            
            filepath_tex = os.path.join(tex_base_path, *tex_path.split("/")) + ".png"
            textures[tex_name] = create_textured_principled_bsdf(tex_name, filepath_tex)

        # map texture aliases
        for tex_name, tex_path in data["textures"].items():
            if tex_path[0] == "#":
                tex_path = tex_path[1:]
                if tex_path in textures:
                    textures[tex_name] = textures[tex_path]

    # =============================================
    # recursively import geometry, uvs
    # =============================================
    root_origin = np.array([0.0, 0.0, 0.0])
    root_elements = data["elements"]
    for e in root_elements:
        obj = load_element(
            e,
            None,
            root_origin,
            root_origin,
            all_objects,
            textures,
            tex_width=tex_width,
            tex_height=tex_height,
            import_uvs=True
        )
        root_objects.append(obj)

    # =============================================
    # model post-processing
    # =============================================
    if recenter_to_origin:
        # model bounding box vector
        model_v_min = np.array([inf, inf, inf])
        model_v_max = np.array([-inf, -inf, -inf])
        
        # re-used buffer
        v_world = np.zeros((3, 8))
        
        # get bounding box
        for obj in root_objects:
            mesh = obj.data
            mat_world = obj.matrix_world
            for i, v in enumerate(mesh.vertices):
                v_world[0:3,i] = mat_world @ v.co
            
            model_v_min = np.amin(np.append(v_world, model_v_min[...,np.newaxis], axis=1), axis=1)
            model_v_max = np.amax(np.append(v_world, model_v_max[...,np.newaxis], axis=1), axis=1)

        mean = 0.5 * (model_v_min + model_v_max)
        mean = Vector((mean[0], mean[1], mean[2]))
        for obj in root_objects:
            obj.location = obj.location - mean
    
    # import groups as collections
    for g in groups:
        name = g["name"]
        if name == "Master Collection":
            continue
        
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
        for index in g["children"]:
            col.objects.link(all_objects[index])
            bpy.context.scene.collection.objects.unlink(all_objects[index])
    
    # select newly imported objects
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in all_objects:
        obj.select_set(True)
    
    return {"FINISHED"}
