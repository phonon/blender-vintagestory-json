import bpy
from bpy import context
from mathutils import Vector, Euler, Quaternion, Matrix
import math
import numpy as np
from math import inf
import posixpath # need "/" separator
import os
import json

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
# shape (f,n) = (6,3)
#   f = 6: number of cuboid faces to test
#   v = 3: vertex coordinates (x,y,z)
DIRECTION_NORMALS = np.array([
    [-1.,  0.,  0.],
    [ 0.,  1.,  0.],
    [ 0., -1.,  0.],
    [ 1.,  0.,  0.],
    [ 0.,  0.,  1.],
    [ 0.,  0., -1.],
])
# DIRECTION_NORMALS = np.tile(DIRECTION_NORMALS[np.newaxis,...], (6,1,1))

# blender counterclockwise uv -> minecraft uv rotation lookup table
# (these values experimentally determined)
# access using [uv_loop_start_index][vert_loop_start_index]
COUNTERCLOCKWISE_UV_ROTATION_LOOKUP = [
    [0, 270, 180, 90],
    [90, 0, 270, 180],
    [180, 90, 0, 270],
    [270, 180, 90, 0],
]

# blender clockwise uv -> minecraft uv rotation lookup table
# (these values experimentally determined)
# access using [uv_loop_start_index][vert_loop_start_index]
# Note: minecraft uv must also be x-flipped
CLOCKWISE_UV_ROTATION_LOOKUP = [
    [90, 0, 270, 180],
    [0, 270, 180, 90],
    [270, 180, 90, 0],
    [180, 90, 0, 270],
]


def filter_root_objects(objects):
    """Get root objects (objects without parents) in
    Blender scene"""
    root_objects = []
    for obj in objects:
        if obj.parent is None:
            root_objects.append(obj)
    return root_objects


def sign(x):
    """Return sign of value as 1 or -1"""
    if x >= 0:
        return 1
    else:
        return -1


def to_vintagestory_axis(ax):
    """Convert blender space to VS space:
    X -> Z
    Y -> X
    Z -> Y
    """
    if "X" in ax:
        return "Z"
    elif "Y" in ax:
        return "X"
    elif "Z" in ax:
        return "Y"


def to_y_up(arr):
    """Convert blender space to VS space:
    X -> Z
    Y -> X
    Z -> Y
    """
    return np.array([arr[1], arr[2], arr[0]])


def to_vintagestory_rotation(arr):
    """Convert blender space rotation to VS space:
    X -> Z
    Y -> X
    Z -> Y
    """
    return np.array([arr[1], arr[2], arr[0]])


def clamp_rotation(r):
    """Clamp a euler angle rotation in numpy array format
    [rx, ry, rz] to within bounds [-360, 360]
    """
    while r[0] > 360.0:
        r[0] -= 360.0
    while r[0] < -360.0:
        r[0] += 360.0
    
    while r[1] > 360.0:
        r[1] -= 360.0
    while r[1] < -360.0:
        r[1] += 360.0
    
    while r[2] > 360.0:
        r[2] -= 360.0
    while r[2] < -360.0:
        r[2] += 360.0
    
    return r


def get_reduced_rotation(rotation):
    """Split rotation into all 90 deg rotations
    and the residual.
    """
    residual_rotation = rotation.copy()
    
    # number of 90 degree steps
    xsteps = 0
    ysteps = 0
    zsteps = 0
    
    # get sign of euler angles
    rot_x_sign = sign(rotation.x)
    rot_y_sign = sign(rotation.y)
    rot_z_sign = sign(rotation.z)
    
    eps = 1e-6 # angle error range for floating point precision issues
    
    # included sign() in loop condition to prevent angle
    # overflowing to other polarity
    while abs(residual_rotation.x) > math.pi/2 - eps and sign(residual_rotation.x) == rot_x_sign:
        angle = rot_x_sign * math.pi/2
        residual_rotation.x -= angle
        xsteps += rot_x_sign
    while abs(residual_rotation.y) > math.pi/2 - eps and sign(residual_rotation.y) == rot_y_sign:
        angle = rot_y_sign * math.pi/2
        residual_rotation.y -= angle
        ysteps += rot_y_sign
    while abs(residual_rotation.z) > math.pi/4 - eps and sign(residual_rotation.z) == rot_z_sign:
        angle = rot_z_sign * math.pi/2
        residual_rotation.z -= angle
        zsteps += rot_z_sign
    
    rotation_90deg = Euler((xsteps * math.pi/2, ysteps * math.pi/2, zsteps * math.pi/2), "XYZ")
    
    return residual_rotation, rotation_90deg


class TextureInfo():
    """Description of a texture, gives image source path and size
    """
    def __init__(self, path, size):
        self.path = path
        self.size = size


def get_material_color(mat):
    """Get material color as tuple (r, g, b, a). Return None if no
    material node has color property.
    Inputs:
    - mat: Material
    Returns:
    - color tuple (r, g, b,a ) if a basic color,
      "texture_path" string if a texture,
      or None if no color/path
    """
    # get first node with valid color
    if mat.node_tree is not None:
        for n in mat.node_tree.nodes:
            # principled BSDF
            if "Base Color" in n.inputs:
                node_color = n.inputs["Base Color"]
                # check if its a texture path
                for link in node_color.links:
                    from_node = link.from_node
                    if isinstance(from_node, bpy.types.ShaderNodeTexImage):
                        if from_node.image is not None and from_node.image.filepath != "":
                            img = from_node.image
                            img_size = [img.size[0], img.size[1]]
                            return TextureInfo(img.filepath, img_size)
                # else, export color tuple
                color = node_color.default_value
                color = (color[0], color[1], color[2], color[3])
                return color
            # most other materials with color
            elif "Color" in n.inputs:
                color = n.inputs["Color"].default_value
                color = (color[0], color[1], color[2], color[3])
                return color
    
    return None


def get_object_color(obj, material_index, default_color = (0.0, 0.0, 0.0, 1.0)):
    """Get obj material color in index as either 
    - tuple (r, g, b, a) if using a default color input
    - texture file name string "path" if using a texture input
    """
    if material_index < len(obj.material_slots):
        slot = obj.material_slots[material_index]
        material = slot.material
        if material is not None:
            color = get_material_color(material)
            if color is not None:
                return color
    
    return default_color


def loop_is_clockwise(coords):
    """Detect if loop of 2d coordinates is clockwise or counterclockwise.
    Inputs:
    - coords: List of 2d array indexed coords, [p0, p1, p2, ... pN]
              where each is array indexed as p[0] = p0.x, p[1] = p0.y
    Returns:
    - True if clockwise, False if counterclockwise
    """
    num_coords = len(coords)
    area = 0
    
    # use polygon winding area to detect if loop is clockwise or counterclockwise
    for i in range(num_coords):
        # next index
        k = i + 1 if i < num_coords - 1 else 0
        area += (coords[k][0] - coords[i][0]) * (coords[k][1] + coords[i][1])
    
    # clockwise if area positive
    return area > 0


def create_color_texture(
    colors,
    min_size = 16,
):
    """Create a packed square texture from list of input colors. Each color
    is a distinct RGB tuple given a 3x3 pixel square in the texture. These
    must be 3x3 pixels so that there is no uv bleeding near the face edges.
    Also includes a tile for a default color for faces with no material.
    This is the next unfilled 3x3 tile.

    Inputs:
    - colors: Iterable of colors. Each color should be indexable like an rgb
              tuple c = (r, g, b), just so that r = c[0], b = c[1], g = c[2].
    - min_size: Minimum size of texture (must be power 2^n). By default
                16 because Minecraft needs min sized 16 textures for 4 mipmap levels.'
    
    Returns:
    - tex_pixels: Flattened array of texture pixels.
    - tex_size: Size of image texture.
    - color_tex_uv_map: Dict map from rgb tuple color to minecraft format uv coords
                        (r, g, b) -> (xmin, ymin, xmax, ymax)
    - default_color_uv: Default uv coords for unmapped materials (xmin, ymin, xmax, ymax).
    """
    # blender interprets (r,g,b,a) in sRGB space
    def linear_to_sRGB(v):
        if v < 0.0031308:
            return v * 12.92
        else:
            return 1.055 * (v ** (1/2.4)) - 0.055
    
    # fit textures into closest (2^n,2^n) sized texture
    # each color takes a (3,3) pixel chunk to avoid color
    # bleeding at UV edges seams
    # -> get smallest n to fit all colors, add +1 for a default color tile
    color_grid_size = math.ceil(math.sqrt(len(colors) + 1)) # colors on each axis
    tex_size = max(min_size, 2 ** math.ceil(math.log2(3 * color_grid_size))) # fit to (2^n, 2^n) image
    
    # composite colors into white RGBA grid
    tex_colors = np.ones((color_grid_size, color_grid_size, 4))
    color_tex_uv_map = {}
    for i, c in enumerate(colors):
        # convert color to sRGB
        c_srgb = (linear_to_sRGB(c[0]), linear_to_sRGB(c[1]), linear_to_sRGB(c[2]), c[3])

        tex_colors[i // color_grid_size, i % color_grid_size, :] = c_srgb
        
        # uvs: [x1, y1, x2, y2], each value from [0, 16] as proportion of image
        # map each color to a uv
        x1 = ( 3*(i % color_grid_size) + 1 ) / tex_size * 16
        x2 = ( 3*(i % color_grid_size) + 2 ) / tex_size * 16
        y1 = ( 3*(i // color_grid_size) + 1 ) / tex_size * 16
        y2 = ( 3*(i // color_grid_size) + 2 ) / tex_size * 16
        color_tex_uv_map[c] = [x1, y1, x2, y2]
    
    # default color uv coord (last coord + 1)
    idx = len(colors)
    default_color_uv = [
        ( 3*(idx % color_grid_size) + 1 ) / tex_size * 16,
        ( 3*(idx // color_grid_size) + 1 ) / tex_size * 16,
        ( 3*(idx % color_grid_size) + 2 ) / tex_size * 16,
        ( 3*(idx // color_grid_size) + 2 ) / tex_size * 16
    ]

    # triple colors into 3x3 pixel chunks
    tex_colors = np.repeat(tex_colors, 3, axis=0)
    tex_colors = np.repeat(tex_colors, 3, axis=1)
    tex_colors = np.flip(tex_colors, axis=0)

    # pixels as flattened array (for blender Image api)
    tex_pixels = np.ones((tex_size, tex_size, 4))
    tex_pixels[-tex_colors.shape[0]:, 0:tex_colors.shape[1], :] = tex_colors
    tex_pixels = tex_pixels.flatten("C")

    return tex_pixels, tex_size, color_tex_uv_map, default_color_uv


def generate_element(
    obj,                           # current object
    groups=None,                   # running dict of collections
    model_colors=None,             # running dict of all model colors
    model_textures=None,           # running dict of all model textures
    parent_cube_origin=None,       # parent cube "from" origin (coords in VintageStory space)
    parent_rotation_origin=None,   # parent object rotation origin (coords in VintageStory space)
    parent_rotation_90deg=None,    # parent 90 degree rotation matrix
    export_uvs=True,               # export uvs
    export_generated_texture=True,
):
    """Recursive function to generate output element from
    Blender object

    See diagram from importer for location transformation.

    VintageStory => Blender space:
        child_blender_origin = parent_cube_origin - parent_rotation_origin + child_rotation_origin
        from_blender_local = from - child_rotation_origin
        to_blender_local = to - child_rotation_origin

    Blender space => VS Space:
        child_rotation_origin = child_blender_origin - parent_cube_origin + parent_rotation_origin
        from = from_blender_local + child_rotation_origin
        to = to_blender_local + child_rotation_origin
    """
    mesh = obj.data
    if not isinstance(mesh, bpy.types.Mesh):
        return None
    
    # count number of vertices, ignore if not cuboid
    num_vertices = len(mesh.vertices)
    if num_vertices != 8:
        return None
    
    # object blender origin
    origin = np.array(obj.location)

    # get local mesh coordinates
    v_local = np.zeros((3, 8))
    for i, v in enumerate(mesh.vertices):
        v_local[0:3,i] = v.co
    
    # ================================
    # first reduce rotation to [-90, 90] by applying all 90 deg
    # rotations directly to vertices
    # ================================
    if parent_rotation_90deg is not None:
        ax_angle, theta = obj.rotation_euler.to_quaternion().to_axis_angle()
        transformed_ax_angle = parent_rotation_90deg @ ax_angle
        obj_rotation = Quaternion(transformed_ax_angle, theta).to_euler("XYZ")
        v_local = parent_rotation_90deg @ v_local
        origin = parent_rotation_90deg @ origin
    else:
        obj_rotation = obj.rotation_euler
    
    residual_rotation, rotation_90deg = get_reduced_rotation(obj_rotation)

    mat_rotate_90deg = np.array(rotation_90deg.to_matrix())
    
    # rotate axis of residual rotation
    ax_angle, theta = residual_rotation.to_quaternion().to_axis_angle()
    transformed_ax_angle = mat_rotate_90deg @ ax_angle
    rotation_transformed = Quaternion(transformed_ax_angle, theta).to_euler("XYZ")

    # get min/max for to/from points
    v_local_transformed = mat_rotate_90deg @ v_local
    v_min = np.amin(v_local_transformed, axis=1)
    v_max = np.amax(v_local_transformed, axis=1)
    
    rotation = np.array([
        rotation_transformed.x * 180.0 / math.pi,
        rotation_transformed.y * 180.0 / math.pi,
        rotation_transformed.z * 180.0 / math.pi,
    ])

    # change axis to vintage story y-up axis
    v_min = to_y_up(v_min)
    v_max = to_y_up(v_max)
    origin = to_y_up(origin)
    rotation = to_vintagestory_rotation(rotation)
    
    # translate to vintage story coord space
    rotation_origin = origin - parent_cube_origin + parent_rotation_origin
    v_from = v_min + rotation_origin
    v_to = v_max + rotation_origin
    cube_origin = v_from
    
    # ================================
    # texture/uv generation
    # 
    # NOTE: BLENDER VS MINECRAFT/VINTAGE STORY UV AXIS
    # - blender: uvs origin is bottom-left (0,0) to top-right (1, 1)
    # - minecraft/vs: uvs origin is top-left (0,0) to bottom-right (16, 16)
    # minecraft uvs: [x1, y1, x2, y2], each value from [0, 16] as proportion of image
    # as well as 0, 90, 180, 270 degree uv rotation

    # uv loop to export depends on:
    # - clockwise/counterclockwise order
    # - uv starting coordinate (determines rotation) relative to face
    #   vertex loop starting coordinate
    # 
    # Assume "natural" index order of face vertices and uvs without
    # any rotations in local mesh space is counterclockwise loop:
    #   3___2      ^ +y
    #   |   |      |
    #   |___|      ---> +x
    #   0   1
    # 
    # uv, vertex starting coordinate is based on this loop.
    # Use the uv rotation lookup tables constants to determine rotation.
    # ================================

    # initialize faces
    faces = {
        "north": {"texture": "#0", "uv": [0, 0, 4, 4], "autoUv": False},
        "east": {"texture": "#0", "uv": [0, 0, 4, 4], "autoUv": False},
        "south": {"texture": "#0", "uv": [0, 0, 4, 4], "autoUv": False},
        "west": {"texture": "#0", "uv": [0, 0, 4, 4], "autoUv": False},
        "up": {"texture": "#0", "uv": [0, 0, 4, 4], "autoUv": False},
        "down": {"texture": "#0", "uv": [0, 0, 4, 4], "autoUv": False},
    }
    
    uv_layer = mesh.uv_layers.active.data

    for i, face in enumerate(mesh.polygons):
        if i > 5: # should be 6 faces only
            print(f"WARNING: {obj} has >6 faces")
            break

        # stack + reshape to (6,3)
        # face_normal = np.array(face.normal)
        # face_normal_stacked = np.transpose(face_normal[..., np.newaxis], (1,0))
        face_normal = face.normal
        face_normal_transformed = mat_rotate_90deg @ face_normal
        face_normal_stacked = np.transpose(face_normal_transformed[..., np.newaxis], (1,0))
        face_normal_stacked = np.tile(face_normal_stacked, (6,1))

        # get face direction string
        face_direction_index = np.argmax(np.sum(face_normal_stacked * DIRECTION_NORMALS, axis=1), axis=0)
        d = DIRECTIONS[face_direction_index]
        
        face_texture = get_object_color(obj, face.material_index)
        
        # solid color tuple
        if isinstance(face_texture, tuple) and export_generated_texture:
            faces[d] = face_texture # replace face with color
            if model_colors is not None:
                model_colors.add(face_texture)
        # texture
        elif isinstance(face_texture, TextureInfo):
            faces[d]["texture"] = face_texture.path
            model_textures[face_texture.path] = face_texture

            tex_width = face_texture.size[0]
            tex_height = face_texture.size[1]

            if export_uvs:                
                # uv loop
                loop_start = face.loop_start
                face_uv_0 = uv_layer[loop_start].uv
                face_uv_1 = uv_layer[loop_start+1].uv
                face_uv_2 = uv_layer[loop_start+2].uv
                face_uv_3 = uv_layer[loop_start+3].uv

                uv_min_x = min(face_uv_0[0], face_uv_2[0])
                uv_max_x = max(face_uv_0[0], face_uv_2[0])
                uv_min_y = min(face_uv_0[1], face_uv_2[1])
                uv_max_y = max(face_uv_0[1], face_uv_2[1])

                uv_clockwise = loop_is_clockwise([face_uv_0, face_uv_1, face_uv_2, face_uv_3])

                # vertices loops
                # project 3d vertex loop onto 2d loop based on face normal,
                # minecraft uv mapping starting corner experimentally determined
                verts = [ v_local_transformed[:,v] for v in face.vertices ]
                
                if face_normal[0] > 0.5: # normal = (1, 0, 0)
                    verts = [ (v[1], v[2]) for v in verts ]
                elif face_normal[0] < -0.5: # normal = (-1, 0, 0)
                    verts = [ (-v[1], v[2]) for v in verts ]
                elif face_normal[1] > 0.5: # normal = (0, 1, 0)
                    verts = [ (-v[0], v[2]) for v in verts ]
                elif face_normal[1] < -0.5: # normal = (0, -1, 0)
                    verts = [ (v[0], v[2]) for v in verts ]
                elif face_normal[2] > 0.5: # normal = (0, 0, 1)
                    verts = [ (v[1], -v [0]) for v in verts ]
                elif face_normal[2] < -0.5: # normal = (0, 0, -1)
                    verts = [ (v[1], v[0]) for v in verts ]
                
                vert_min_x = min(verts[0][0], verts[2][0])
                vert_max_x = max(verts[0][0], verts[2][0])
                vert_min_y = min(verts[0][1], verts[2][1])
                vert_max_y = max(verts[0][1], verts[2][1])

                vert_clockwise = loop_is_clockwise(verts)
                
                # get uv, vert loop starting corner index 0..3 in face loop

                # uv start corner index
                uv_start_x = face_uv_0[0]
                uv_start_y = face_uv_0[1]
                if uv_start_y < uv_max_y:
                    # start coord 0
                    if uv_start_x < uv_max_x:
                        uv_loop_start_index = 0
                    # start coord 1
                    else:
                        uv_loop_start_index = 1
                else:
                    # start coord 2
                    if uv_start_x > uv_min_x:
                        uv_loop_start_index = 2
                    # start coord 3
                    else:
                        uv_loop_start_index = 3
                
                # vert start corner index
                vert_start_x = verts[0][0]
                vert_start_y = verts[0][1]
                if vert_start_y < vert_max_y:
                    # start coord 0
                    if vert_start_x < vert_max_x:
                        vert_loop_start_index = 0
                    # start coord 1
                    else:
                        vert_loop_start_index = 1
                else:
                    # start coord 2
                    if vert_start_x > vert_min_x:
                        vert_loop_start_index = 2
                    # start coord 3
                    else:
                        vert_loop_start_index = 3

                # set uv flip and rotation based on
                # 1. clockwise vs counterclockwise loop
                # 2. relative starting corner difference between vertex loop and uv loop
                # NOTE: if face normals correct, vertices should always be counterclockwise...
                face_uvs = np.zeros((4,))

                if uv_clockwise == False and vert_clockwise == False:
                    face_uvs[0] = uv_min_x
                    face_uvs[1] = uv_max_y
                    face_uvs[2] = uv_max_x
                    face_uvs[3] = uv_min_y
                    face_uv_rotation = COUNTERCLOCKWISE_UV_ROTATION_LOOKUP[uv_loop_start_index][vert_loop_start_index]
                elif uv_clockwise == True and vert_clockwise == False:
                    # invert x face uvs
                    face_uvs[0] = uv_max_x
                    face_uvs[1] = uv_max_y
                    face_uvs[2] = uv_min_x
                    face_uvs[3] = uv_min_y
                    face_uv_rotation = CLOCKWISE_UV_ROTATION_LOOKUP[uv_loop_start_index][vert_loop_start_index]
                elif uv_clockwise == False and vert_clockwise == True:
                    # invert y face uvs, case should not happen
                    face_uvs[0] = uv_max_x
                    face_uvs[1] = uv_max_y
                    face_uvs[2] = uv_min_x
                    face_uvs[3] = uv_min_y
                    face_uv_rotation = CLOCKWISE_UV_ROTATION_LOOKUP[uv_loop_start_index][vert_loop_start_index]
                else: # uv_clockwise == True and vert_clockwise == True:
                    # case should not happen
                    face_uvs[0] = uv_min_x
                    face_uvs[1] = uv_max_y
                    face_uvs[2] = uv_max_x
                    face_uvs[3] = uv_min_y
                    face_uv_rotation = COUNTERCLOCKWISE_UV_ROTATION_LOOKUP[uv_loop_start_index][vert_loop_start_index]
                
                xmin = face_uvs[0] * tex_width
                ymin = (1.0 - face_uvs[1]) * tex_height
                xmax = face_uvs[2] * tex_width
                ymax = (1.0 - face_uvs[3]) * tex_height
                faces[d]["uv"] = [ xmin, ymin, xmax, ymax ]
                
                if face_uv_rotation != 0 and face_uv_rotation != 360:
                    faces[d]["rotation"] = face_uv_rotation if face_uv_rotation >= 0 else 360 + face_uv_rotation
    
    # build children
    children = []
    for child in obj.children:
        child_element = generate_element(
            child,
            groups=groups,
            model_colors=model_colors,
            model_textures=model_textures,
            parent_cube_origin=cube_origin,
            parent_rotation_origin=rotation_origin,
            parent_rotation_90deg=mat_rotate_90deg,
            export_uvs=export_uvs,
        )
        if child_element is not None:
            children.append(child_element)

    # ================================
    # build element
    # ================================
    new_element = {
        "name": obj.name,
        "from": v_from,
        "to": v_to,
        "rotationOrigin": rotation_origin,
    }

    # add rotations
    if rotation[0] != 0.0:
        new_element["rotationX"] = rotation[0]
    if rotation[1] != 0.0:
        new_element["rotationY"] = rotation[1]
    if rotation[2] != 0.0:
        new_element["rotationZ"] = rotation[2]

    # add collection link
    collection = obj.users_collection[0]
    if collection is not None:
        new_element["group"] = collection.name

    # add faces
    new_element["faces"] = faces

    # add children last
    new_element["children"] = children

    return new_element

def save_objects(
    filepath,
    objects,
    recenter_origin = False,
    translate_origin=None,         # origin translate either [x, y, z] or None
    generate_texture=True,
    use_only_exported_object_colors=False,
    texture_folder="",
    texture_filename="",
    export_uvs=True,
    minify=False,
    decimal_precision=-1,
    **kwargs
):
    """Main exporter function. Parses Blender objects into Minecraft
    cuboid format, uvs, and handles texture read and generation.
    Will save .json file to output paths.

    Inputs:
    - filepath: Output file path name.
    - object: Iterable collection of Blender objects
    - recenter_origin: Recenter model so that its center is at specified translate_origin
    - translate_origin: New origin to shift
    - generate_texture: Generate texture from solid material colors. By default, creates
            a color texture from all materials in file (so all groups of
            objects can share the same texture file).
    - use_only_exported_object_colors:
            Generate texture colors from only exported objects instead of default using
            all file materials.
    - texture_folder: Output texture subpath, for typical "item/texture_name" the
            texture folder would be "item".
    - texture_file_name: Name of texture file. TODO: auto-parse textures from materials.
    - export_uvs: Export object uvs.
    - minify: Minimize output file size (write into single line, remove spaces, ...)
    - decimal_precision: Number of digits after decimal to keep in numbers.
            Requires `minify = True`. Set to -1 to disable.
    """
    
    # debug
    print("USE ORIGIN SHIFT:", recenter_origin)
    print("ORIGIN SHIFT:", translate_origin)
    print("")

    # output json model stub
    model_json = {
        "textureWidth": 16.0, # default, will be overridden
        "textureHeight": 16.0, # default, will be overridden
        "textures": {},
        "textureSizes": {},
    }

    # elements at top of hierarchy
    root_elements = []

    # object collections to be exported
    groups = {}
    
    # all material colors tuples from all object faces
    if use_only_exported_object_colors:
        model_colors = set()
    else:
        model_colors = None
    
    # all material texture paths and sizes:
    # tex_name => {
    #   "path": path,
    #   "size": [x, y],
    # }
    model_textures = {}
    
    # parse objects
    for obj in objects:
        element = generate_element(
            obj,
            groups=groups,
            model_colors=model_colors,
            model_textures=model_textures,
            parent_cube_origin=np.array([0., 0., 0.]),     # root cube origin 
            parent_rotation_origin=np.array([0., 0., 0.]), # root rotation origin
            export_uvs=export_uvs,
        )

        if element is not None:
            root_elements.append(element)
    
    # ===========================
    # generate texture images
    # ===========================
    if generate_texture:
        # default, get colors from all materials in file
        if model_colors is None:
            model_colors = set()
            for mat in bpy.data.materials:
                color = get_material_color(mat)
                if isinstance(color, tuple):
                    model_colors.add(color)
        
        tex_pixels, tex_size, color_tex_uv_map, default_color_uv = create_color_texture(model_colors)

        # texture output filepaths
        if texture_filename == "":
            current_dir = os.path.dirname(filepath)
            filepath_name = os.path.splitext(os.path.basename(filepath))[0]
            texture_save_path = os.path.join(current_dir, filepath_name + ".png")
            texture_model_path = posixpath.join(texture_folder, filepath_name)
        else:
            current_dir = os.path.dirname(filepath)
            texture_save_path = os.path.join(current_dir, texture_filename + ".png")
            texture_model_path = posixpath.join(texture_folder, texture_filename)
        
        # create + save texture
        tex = bpy.data.images.new("tex_colors", alpha=True, width=tex_size, height=tex_size)
        tex.file_format = "PNG"
        tex.pixels = tex_pixels
        tex.filepath_raw = texture_save_path
        tex.save()

        # write texture info to output model
        model_json["textureSizes"]["0"] = [tex_size, tex_size]
        model_json["textures"]["0"] = texture_model_path
    
    # if not generating texture, just write texture path to json file
    # TODO: scan materials for textures, then update output size
    elif texture_filename != "":
        model_json["textureSizes"]["0"] = [16, 16]
        model_json["textures"]["0"] = posixpath.join(texture_folder, texture_filename)
    
    # ===========================
    # process face texture paths
    # convert blender path names "//folder\tex.png" -> "item/tex"
    # add textures indices for textures, and create face mappings like "#1"
    # note: #0 id reserved for generated color texture
    # ===========================
    texture_refs = {} # maps blender path name -> #n identifiers
    texture_id = 1    # texture id in "#1" identifier
    for texture_path, texture_info in model_textures.items():
        texture_out_path = texture_path
        if texture_out_path[0:2] == "//":
            texture_out_path = texture_out_path[2:]
        texture_out_path = texture_out_path.replace("\\", "/")
        texture_out_path = os.path.splitext(texture_out_path)[0]
        
        texture_refs[texture_path] = "#" + str(texture_id)
        model_json["textures"][str(texture_id)] = posixpath.join(texture_folder, texture_out_path)
        model_json["textureSizes"][str(texture_id)] = texture_info.size
        texture_id += 1

    # ===========================
    # root object post-processing:
    # 1. recenter with origin shift
    # ===========================
    if translate_origin is not None:
        translate_origin = to_y_up(translate_origin)
        for element in root_elements:
            # re-centering
            element["to"] = translate_origin + element["to"]
            element["from"] = translate_origin + element["from"]
            element["rotationOrigin"] = translate_origin + element["rotationOrigin"]  
    
    # ===========================
    # all object post processing
    # 2. map solid color face uv -> location in generated texture
    # 3. rewrite path textures -> texture name reference
    # ===========================
    def final_element_processing(element):
        # convert numpy to python list
        element["to"] = element["to"].tolist()
        element["from"] = element["from"].tolist()
        element["rotationOrigin"] = element["rotationOrigin"].tolist()

        faces = element["faces"]
        for d, f in faces.items():
            if isinstance(f, tuple):
                color_uv = color_tex_uv_map[f] if f in color_tex_uv_map else default_color_uv
                faces[d] = {
                    "uv": color_uv,
                    "texture": "#0",
                }
            elif isinstance(f, dict):
                face_texture = f["texture"]
                if face_texture in texture_refs:
                    f["texture"] = texture_refs[face_texture]
                else:
                    face_texture = "#0"
        
        for child in element["children"]:
            final_element_processing(child)
        
    for element in root_elements:
        final_element_processing(element)
    
    # ===========================
    # convert groups
    # ===========================
    groups_export = []
    for g in groups:
        groups_export.append({
            "name": g,
            "origin": [0, 0, 0],
            "children": groups[g],
        })

    # save
    model_json["elements"] = root_elements
    model_json["groups"] = groups_export

    # ===========================
    # minification options to reduce .json file size
    # ===========================
    if minify == True:
        # go through json dict and replace all float with rounded strings
        if decimal_precision >= 0:
            def round_float(x):
                return round(x, decimal_precision)
            
            def minify_element(elem):
                elem["from"] = [round_float(x) for x in elem["from"]]
                elem["to"] = [round_float(x) for x in elem["to"]]
                elem["rotationOrigin"] = [round_float(x) for x in elem["rotationOrigin"]]
                for face in elem["faces"].values():
                    face["uv"] = [round_float(x) for x in face["uv"]]
                
                for child in elem["children"]:
                    minify_element(child)

            for elem in model_json["elements"]:
                minify_element(elem)
    
    # save json
    with open(filepath, "w") as f:
        json.dump(model_json, f, separators=(",", ":"), indent=2)


def save(
    context,
    filepath,
    selection_only = False,
    **kwargs,
):
    if selection_only:
        objects = filter_root_objects(bpy.context.selected_objects)
    else:
        objects = filter_root_objects(bpy.context.scene.collection.all_objects[:])

    save_objects(filepath, objects, **kwargs)

    print("SAVED", filepath)

    return {"FINISHED"}