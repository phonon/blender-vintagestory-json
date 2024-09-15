import os
import json
import numpy as np
import math
import time
from math import inf
import bpy
from mathutils import Vector, Euler, Quaternion
from . import animation

import importlib
importlib.reload(animation)

# convert deg to rad
DEG_TO_RAD = math.pi / 180.0

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
            node_tree.links.new(tex_input.outputs[1], bsdf.inputs["Alpha"]) #  We also want the alpha to be bound to not confuse the end user.
        
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

    Return tuple of:
        obj,                   # new Blender object
        local_cube_origin,     # local space "to" cube corner origin
        new_cube_origin,       # global space cube corner origin
        new_rotation_origin    # local space rotation (Blender) origin
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
        rot_euler[1] = e["rotationX"] * DEG_TO_RAD
    if "rotationY" in e:
        rot_euler[2] = e["rotationY"] * DEG_TO_RAD
    if "rotationZ" in e:
        rot_euler[0] = e["rotationZ"] * DEG_TO_RAD

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

    return obj, v_min, new_cube_origin, new_rotation_origin


def parse_attachpoint(
    e,                      # json element
    parent_cube_origin,     # cube corner origin of parent
):
    """Load attachment point associated with a cube, convert
    into a Blender empty object with special name:
        "attach_AttachPointName"
    where the suffix is the "code": "AttachPointName" in the element.
    This format is used for exporting attachpoints from Blender.

    Location in json is relative to cube origin not rotation origin.
    For some reason json number is a string...wtf?
    """
    px = float(e.get("posX") or 0.0)
    py = float(e.get("posY") or 0.0)
    pz = float(e.get("posZ") or 0.0)

    rx = float(e.get("rotationX") or 0.0)
    ry = float(e.get("rotationY") or 0.0)
    rz = float(e.get("rotationZ") or 0.0)

    # get location, rotation converted to Blender space
    location = np.array([
        pz + parent_cube_origin[0],
        px + parent_cube_origin[1],
        py + parent_cube_origin[2],
    ])
    
    rotation = DEG_TO_RAD * np.array([rz, rx, ry])

    # create object
    bpy.ops.object.empty_add(type="ARROWS", radius=1.0, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.show_in_front = True
    obj.name = "attach_" + (e.get("code") or "attachpoint")

    return obj


def rebuild_hierarchy_with_bones(
    root_objects,
):
    """Wrapper to make armature and replace cubes in hierarchy
    with bones. This is multi step process due to how Blender
    EditBone and PoseBones work
    """
    bpy.ops.object.mode_set(mode="OBJECT") # ensure correct starting context
    bpy.ops.object.add(type="ARMATURE", enter_editmode=True)
    armature = bpy.context.active_object
    armature.show_in_front = True

    for obj in root_objects:
        add_bone_to_armature_from_object(
            obj,
            armature,
            None,
        )
    
    # switch to pose mode, set bone positions from object transform
    bpy.ops.object.mode_set(mode="POSE")
    for obj in root_objects:
        set_bone_pose_from_object(
            obj,
            armature,
        )
    
    # set rest pose, not sure if we want this or not...
    bpy.ops.pose.armature_apply()

    bpy.ops.object.mode_set(mode="OBJECT")

    return armature


def add_bone_to_armature_from_object(
    obj,
    armature,
    parent_bone,
):
    # skip non mesh (e.g. attach points)
    if not isinstance(obj.data, bpy.types.Mesh):
        return
    
    bone = armature.data.edit_bones.new(obj.name)

    # this orients bone to blender XYZ
    bone.head = (0., 0., 0.)
    bone.tail = (0., 1., 0.)

    if parent_bone is not None:
        bone.parent = parent_bone
        bone.use_connect = False
    
    for child in obj.children:
        add_bone_to_armature_from_object(
            child,
            armature,
            bone,
        )


def set_bone_pose_from_object(
    obj,
    armature,
):
    # skip non mesh (e.g. attach points)
    if not isinstance(obj.data, bpy.types.Mesh):
        return
    
    name = obj.name
    pose_bone = armature.pose.bones[name]
    pose_bone.location = (
        obj.location.x,
        obj.location.y,
        obj.location.z,
    )
    pose_bone.rotation_mode = "XYZ"
    pose_bone.rotation_euler = obj.rotation_euler
    
    # now parent obj to bone
    obj.parent = armature
    obj.parent_type = "BONE"
    obj.parent_bone = name
    obj.location = (0., -1.0, 0.)
    obj.rotation_euler = (0., 0., 0.)

    for child in obj.children:
        set_bone_pose_from_object(
            child,
            armature,
        )


def parse_animation(
    e,        # json element
    armature, # armature to associate action with
    stats,    # import stats
):
    def add_keyframe_point(fcu, frame, val):
        """Helper to add keyframe point to fcurve"""
        idx = len(fcu.keyframe_points)
        fcu.keyframe_points.add(1)
        fcu.keyframe_points[idx].interpolation = "LINEAR"
        fcu.keyframe_points[idx].co = frame, val

    name = e["code"] # use code as name instead of name field
    action = bpy.data.actions.new(name=name)
    action.use_fake_user = True # prevents deletion on file save

    # flag to repeat animation (insert duplicate keyframe at end)
    repeat_animation = False

    # add special marker for onActivityStopped and onAnimationEnd
    num_frames = e.get("quantityframes") or 0
    if "onAnimationEnd" in e:
        marker = action.pose_markers.new(name="onAnimationEnd_{}".format(e["onAnimationEnd"]))
        marker.frame = num_frames - 1
        if e["onAnimationEnd"].lower() != "hold": # death animations hold on finish
            repeat_animation = True
    if "onActivityStopped" in e:
        marker = action.pose_markers.new(name="onActivityStopped_{}".format(e["onActivityStopped"]))
        marker.frame = num_frames + 20
    
    # load keyframe data
    animation_adapter = animation.AnimationAdapter(action, name=name)

    # insert first keyframe at end to properly loop
    keyframes = e["keyframes"].copy()
    if repeat_animation and len(keyframes) > 0 and num_frames > 0:
        # make copy of frame 0 and insert at num_frames-1
        keyframe_0_copy = {
            "frame": num_frames - 1,
            "elements": keyframes[0]["elements"],
        }
        keyframes.append(keyframe_0_copy)
    
    for keyframe in keyframes:
        frame = keyframe["frame"]
        for bone_name, data in keyframe["elements"].items():
            fcu_name_prefix = "pose.bones[\"{}\"]".format(bone_name)
            fcu_name_location = fcu_name_prefix + ".location"
            fcu_name_rotation = fcu_name_prefix + ".rotation_euler"

            # add bone => rotation mode
            animation_adapter.set_bone_rotation_mode(bone_name, "rotation_euler")

            # position fcurves
            fcu_px = animation_adapter.get(fcu_name_location, 0)
            fcu_py = animation_adapter.get(fcu_name_location, 1)
            fcu_pz = animation_adapter.get(fcu_name_location, 2)
            
            # euler rotation fcurves
            fcu_rx = animation_adapter.get(fcu_name_rotation, 0)
            fcu_ry = animation_adapter.get(fcu_name_rotation, 1)
            fcu_rz = animation_adapter.get(fcu_name_rotation, 2)

            # add keyframe points (note vintage story ZXY -> XYZ)
            if "offsetX" in data:
                add_keyframe_point(fcu_py, frame, data["offsetX"])
            if "offsetY" in data:
                add_keyframe_point(fcu_pz, frame, data["offsetY"])
            if "offsetZ" in data:
                add_keyframe_point(fcu_px, frame, data["offsetZ"])
            
            bone = armature.data.bones[bone_name]
            bone_rot = bone.matrix_local.copy()
            bone_rot.translation = Vector((0., 0., 0.))

            # _, bone_rot_quat, _ = bone.matrix_local.decompose()
            # bone_rot_euler = bone_rot_quat.to_euler("XYZ")

            rx = data["rotationX"] * DEG_TO_RAD if "rotationX" in data else None
            ry = data["rotationY"] * DEG_TO_RAD if "rotationY" in data else None
            rz = data["rotationZ"] * DEG_TO_RAD if "rotationZ" in data else None
            if rx is not None or ry is not None or rz is not None:
                rx = rx if rx is not None else 0.0
                ry = ry if ry is not None else 0.0
                rz = rz if rz is not None else 0.0
                rot_vs_original = Euler((rz, rx, ry), "XZY")
                rot_vs = Euler((rz, rx, ry), "XZY").to_quaternion().to_euler("XYZ")
                rx = rot_vs.x
                ry = rot_vs.y
                rz = rot_vs.z

                if bone.parent is not None:
                    mat_local = bone.parent.matrix_local.inverted_safe() @ bone.matrix_local
                else:
                    mat_local = bone.matrix_local.copy()
                
                bone_rot_local = mat_local.copy()
                bone_rot_local.translation = Vector((0., 0., 0.))

                _, bone_rot_quat, _ = mat_local.decompose()

                bone_rot_eff = bone_rot_quat.to_euler("XYZ")
                bone_rot_eff.x += rx
                bone_rot_eff.y += ry
                bone_rot_eff.z += rz
                # print(bone.name, "bone_rot_eff", bone_rot_eff, "bone_rot_local", bone_rot_local.to_euler("XYZ"))

                rot_eff = bone_rot_eff.to_matrix().to_4x4()
                rot_mat = rot_eff @ bone_rot_local.inverted_safe()
                rot = rot_mat.to_euler("XYZ")

                # debugging
                # rot_result_mat = rot.to_matrix().to_4x4() @ bone_rot_local
                # rot_result = rot_result_mat.to_euler("XYZ")
                # if bone.name == "tail":
                #     print(bone.name, "rot_vs:", rot_vs, "rot_vs_original:", rot_vs_original)
                #     print(bone.name, "using direct values:", Euler((rz, rx, ry), "XYZ"), "rot:", rot, "target_rot:", bone_rot_eff)
                #     print(bone.name, "ROT RESULT", rot_result, rot_result.x * animation.RAD_TO_DEG, rot_result.y * animation.RAD_TO_DEG, rot_result.z * animation.RAD_TO_DEG)
                #     print("rot_result * rot_local = ", rot_mat @ bone_rot_local)
                #     print("rot_eff = ", rot_eff, rot_eff.to_euler("XYZ"))
                #     print("rot_result = ", rot_result_mat, rot_result_mat.to_euler("XYZ"))

                # transform to bone euler
                ax_angle, theta = rot.to_quaternion().to_axis_angle()
                transformed_ax_angle = bone_rot_local.inverted_safe() @ ax_angle
                rot = Quaternion(transformed_ax_angle, theta).to_euler("XYZ")

                # ax_angle, theta = bone_rot_eff.to_quaternion().to_axis_angle()
                # transformed_ax_angle = bone_rot.inverted_safe() @ ax_angle
                # rot = Quaternion(transformed_ax_angle, theta).to_euler("XYZ")

                # rot = Euler((rz, rx, ry), "XZY")
                # ax_angle, theta = rot.to_quaternion().to_axis_angle()
                # if bone.parent is not None:
                #     parent_bone = bone.parent
                #     parent_bone_rot = parent_bone.matrix_local.copy()
                #     parent_bone_rot.translation = Vector((0., 0., 0.))
                #     transformed_ax_angle = parent_bone_rot @ (bone_rot.inverted_safe() @ ax_angle)
                # else:
                #     transformed_ax_angle = bone_rot.inverted_safe() @ ax_angle
                # rot = Quaternion(transformed_ax_angle, theta).to_euler("XYZ")
                
                # vs: rotation relative to parent axis, need to convert
                add_keyframe_point(fcu_rx, frame, rot.x)
                add_keyframe_point(fcu_ry, frame, rot.y)
                add_keyframe_point(fcu_rz, frame, rot.z)
            
            # vs: rotation relative to parent axis, need to convert
            # if "rotationX" in data:
            #     add_keyframe_point(fcu_ry, frame, data["rotationX"] * DEG_TO_RAD)
            # if "rotationY" in data:
            #     add_keyframe_point(fcu_rz, frame, data["rotationY"] * DEG_TO_RAD)
            # if "rotationZ" in data:
            #     add_keyframe_point(fcu_rx, frame, data["rotationZ"] * DEG_TO_RAD)
    
    # resample animations for blender
    animation_adapter.resample_to_blender()

    # update stats
    stats.animations += 1


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
    stats=None,
):
    """Recursively load a geometry cuboid"""
    obj, local_cube_origin, new_cube_origin, new_rotation_origin = parse_element(
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
    
    # increment stats (debugging)
    if stats:
        stats.cubes += 1

    # parse attach points
    if "attachmentpoints" in element:
        for attachpoint in element["attachmentpoints"]:
            p = parse_attachpoint(
                attachpoint,
                local_cube_origin,
            )
            p.parent = obj
            all_objects.append(p)
            
            # increment stats (debugging)
            if stats:
                stats.attachpoints += 1

    # recursively load children
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
                stats=stats,
            )

    return obj


class ImportStats():
    """Track statistics on imported data"""
    def __init__(self):
        self.cubes = 0
        self.attachpoints = 0
        self.animations = 0
        self.textures = 0


def load(context,
         filepath,
         import_uvs=True,               # import face uvs
         import_textures=True,          # import textures into materials
         import_animations=True,        # load animations
         translate_origin=None,         # origin translate either [x, y, z] or None
         recenter_to_origin=False,      # recenter model to origin, overrides translate origin
         debug_stats=True,              # print statistics on imported models
         **kwargs
):
    """Main import function"""

    # debug
    t_start = time.process_time()
    stats = ImportStats() if debug_stats else None

    with open(filepath, "r") as f:
        s = f.read()
        try:
            data = json.loads(s)
        except Exception as err:
            # sometimes format is in loose json, `name: value` instead of `"name": value`
            # this tries to add quotes to keys without double quotes
            # this simple regex fails if any strings contain colons
            try:
                import re
                s2 = re.sub("(\w+):", r'"\1":',  s)
                data = json.loads(s2)
            # unhandled issue
            except Exception as err:
                raise err

        
        # data = json.load(f)
    
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
    if translate_origin is not None:
        translate_origin = Vector(translate_origin)

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

            # update stats
            if stats:
                stats.textures += 1

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
            import_uvs=True,
            stats=stats,
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
    
    # do raw origin translation
    elif translate_origin is not None:
        for obj in root_objects:
            obj.location = obj.location + translate_origin

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
    
    # import animations
    if import_animations and "animations" in data and len(data["animations"]) > 0:
        # go through objects, rebuild hierarchy using bones instead of direct parenting
        # to support bone based animation
        armature = rebuild_hierarchy_with_bones(root_objects)

        # load animations
        for anim in data["animations"]:
            parse_animation(anim, armature, stats)
    
    # select newly imported objects
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    for obj in all_objects:
        obj.select_set(True)
    
    # print stats
    if debug_stats:
        t_end = time.process_time()
        dt = t_end - t_start
        print("Imported .json in {}s".format(dt))
        print("- Cubes: {}".format(stats.cubes))
        print("- Attach Points: {}".format(stats.attachpoints))
        print("- Textures: {}".format(stats.textures))
        print("- Animations: {}".format(stats.animations))

    return {"FINISHED"}
