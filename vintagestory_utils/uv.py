import math
import numpy as np
import logging
import bpy

log = logging.getLogger(__name__)

# single pre-defined axes vectors, for convenience
X_AXIS = np.array([1.0, 0.0, 0.0])
X_NEG_AXIS = np.array([-1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Y_NEG_AXIS = np.array([0.0, -1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])
Z_NEG_AXIS = np.array([0.0, 0.0, -1.0])

# pre-defined axes array block
AXIS = np.array([
    [1.0, 0.0, 0.0], # X
    [-1.0, 0.0, 0.0], # -X
    [0.0, 1.0, 0.0], # Y
    [0.0, -1.0, 0.0], # -Y
    [0.0, 0.0, 1.0], # Z
    [0.0, 0.0, -1.0], # -Z
])

# indices into AXIS array
IDX_X_AXIS = 0
IDX_X_NEG_AXIS = 1
IDX_Y_AXIS = 2
IDX_Y_NEG_AXIS = 3
IDX_Z_AXIS = 4
IDX_Z_NEG_AXIS = 5


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


def index_of_vmin(
    face_verts: np.ndarray, # shape (4, 3)
):
    """Determine index and vertex of vmin or "v0" (bottom left) vertex of a
    face in the XY plane as defined in format below:
    
             ^ y axis
             |
             |  v2
      v3..---^^\
        \     / \
         \   x---\---------> x axis
          \       \
           \       \
            \..---^^^ v1
            v0
          = vmin
    
    Note the face is not necessarily axis-aligned which is
    the challenge. Current method:
    1. Find two points with smallest x values. If any points have same x value,
       use the one with smallest y value.
    2. Between those two points, pick the one with smallest y value.
       Use that as the "vmin" point.
    Returns index of point.
    """
    idx_min_x = 0
    idx_2nd_min_x = math.inf
    x_min = face_verts[0,0]
    y_min = face_verts[0,1]
    x_2nd_min = math.inf
    y_2nd_min = math.inf

    for i in range(1, 4):
        x = face_verts[i,0]
        y = face_verts[i,1]
        if x < x_min:
            idx_2nd_min_x = idx_min_x
            x_2nd_min = x_min
            y_2nd_min = y_min
            idx_min_x = i
            x_min = x
            y_min = y
        elif x == x_min:
            if y < y_min:
                idx_2nd_min_x = idx_min_x
                x_2nd_min = x_min
                y_2nd_min = y_min
                idx_min_x = i
                x_min = x
                y_min = y
            else:
                idx_2nd_min_x = i
                x_2nd_min = x
                y_2nd_min = y
        elif x < x_2nd_min:
            idx_2nd_min_x = i
            x_2nd_min = x
            y_2nd_min = y
        elif x == x_2nd_min:
            if y < y_2nd_min:
                idx_2nd_min_x = i
                x_2nd_min = x
                y_2nd_min = y
        else:
            pass # x > x_min and x > x_2nd_min, do nothing
    
    # return x_min or x_2nd_min, whichever has smaller y value
    if y_min <= y_2nd_min:
        return idx_min_x
    else:
        return idx_2nd_min_x

class OpUVCuboidUnwrap(bpy.types.Operator):
    """Specialized VS cuboid UV unwrap"""

    """
    This unwraps cuboid UVs into format (in Blender axes):
             __________ __________
            |    UP    |  DOWN    |
            |   (+z)   |  (-z)    |
     _______|__________|__________|_______
    | LEFT  |  FRONT   | RIGHT |  BACK    |
    | (-x)  |  (-y)    | (+x)  |  (+y)    |
    |_______|__________|_______|__________|

    The notation is the FRONT, BACK, LEFT, RIGHT, UP, DOWN faces.
    Example using "-y" as the FRONT axis is shown above.

    The front-facing side of the model, is generally most important, so
    we want unwrap format to create a strip across the sides of the cuboid
    as these are usually the most viewed location, so we want artist to most
    easily blend together the texture across these sides.
    
    We will always operate on arrays of face sides stored in format:
        faces = [
            LEFT,
            FRONT,
            RIGHT,
            BACK,
            UP,
            DOWN,
        ]
    This is so at the last step
    """
    bl_idname = "vintagestory.uv_cuboid_unwrap"
    bl_label = "Cuboid UV Unwrap (VS)"
    bl_options = {"REGISTER", "UNDO"}

    front_face: bpy.props.EnumProperty(
        items=[ # (identifier, name, description)
            ("-y", "-Y", "Front face is -y"),
            ("+y", "+Y", "Front face is +y"),
            ("-x", "-X", "Front face is -x"),
            ("+x", "+X", "Front face is +x"),
            ("-z", "-Z", "Front face is -z"),
            ("+z", "+Z", "Front face is +z"),
        ],
        default="-y",
        name="Front Face",
        description="Front face of cuboid for unwraping",
    )

    use_local_space: bpy.props.BoolProperty(
        default=False,
        name="Use Local Space",
        description="Use local space vertices instead of world space",
    )

    def execute(self, context):
        args = self.as_keywords()

        # unpack args

        # use local vertices/normals instead of world normals
        use_local_space = args.get("use_local_space", False)

        # map `front_face` string arg to integer axis index
        front_face = args.get("front_face", "-y")
        if front_face == "+x":
            front_axis_index, front_axis = IDX_X_AXIS, X_AXIS
        elif front_face == "-x":
            front_axis_index, front_axis = IDX_X_NEG_AXIS, X_NEG_AXIS
        elif front_face == "+y":
            front_axis_index, front_axis = IDX_Y_AXIS, Y_AXIS
        elif front_face == "-y":
            front_axis_index, front_axis = IDX_Y_NEG_AXIS, Y_NEG_AXIS
        elif front_face == "+z":
            front_axis_index, front_axis = IDX_Z_AXIS, Z_AXIS
        elif front_face == "-z":
            front_axis_index, front_axis = IDX_Z_NEG_AXIS, Z_NEG_AXIS
        else:
            err_msg = f"Invalid front_face: {front_face}, must be one of: +x, -x, +y, -y, +z, -z"
            self.report({"ERROR"}, err_msg)
            raise Exception(err_msg)

        # uv face format indices
        IDX_UV_FACE_LEFT = 0
        IDX_UV_FACE_FRONT = 1
        IDX_UV_FACE_RIGHT = 2
        IDX_UV_FACE_BACK = 3
        IDX_UV_FACE_UP = 4
        IDX_UV_FACE_DOWN = 5

        # need to be in object mode to access context selected objects
        user_mode = context.active_object.mode
        if user_mode != "OBJECT":
            need_to_switch_mode_back = True
            bpy.ops.object.mode_set(mode="OBJECT")
        else:
            need_to_switch_mode_back = False
        
        # only perform on selected objects
        objects = bpy.context.selected_objects
        for obj in objects:
            try:
                mesh = obj.data
                if not isinstance(mesh, bpy.types.Mesh):
                    continue
                
                # skip non cuboid meshes, print warning
                if len(mesh.vertices) != 8:
                    log.warning(f"Skipping UV unwrap of non-cuboid mesh: {obj.name}")
                    continue

                uv_layer = mesh.uv_layers.active.data
                vertices_local = np.ones((4, 8)) # 8 vertices, each as (x,y,z,1) for 4x4 matrix multiplication
                for i, v in enumerate(mesh.vertices):
                    vertices_local[0:3,i] = v.co
                
                # transform vertices to world space
                matrix_world = np.asarray(obj.matrix_world)
                if use_local_space:
                    vertices = vertices_local
                else: # use world space vertices, transform vertices to world space
                    vertices = matrix_world @ vertices_local

                # normal matrix = tranpose of inverse of upper left 3x3 of world matrix
                try:
                    normal_matrix = np.transpose(np.linalg.inv(matrix_world[0:3,0:3]))
                except:
                    log.warning(f"Non-invertible matrix for: {obj.name}, using its world matrix instead")
                    normal_matrix = matrix_world
                
                # Z-axis for cross product
                front_axis = Y_NEG_AXIS

                # First find the "front face" by finding the face normal that
                # is closest matching to the front face normal
                
                # gather original "mesh" face vertices and normals
                mesh_face_uv_loop_start = np.zeros((6,), dtype=int)   # (face,)
                mesh_face_vert_indices = np.zeros((6, 4), dtype=int)  # (face, vert)
                mesh_face_vertices = np.zeros((6, 4, 4), dtype=float) # (face, vert, xyzw)
                mesh_face_normals = np.zeros((6, 3), dtype=float)     # (face, xyz)

                for i, face in enumerate(mesh.polygons):
                    mesh_face_uv_loop_start[i] = face.loop_start

                    # note face vertices contains indices pointers into mesh vertices
                    mesh_face_vertices[i,:,:] = np.stack(
                        [ vertices[:,v] for v in face.vertices ],
                        axis=0,
                    )
                    mesh_face_vert_indices[i,:] = face.vertices
                    mesh_face_normals[i,:] = np.array(face.normal)
                    
                # world space face normals
                mesh_face_normals_world = normal_matrix @ mesh_face_normals.transpose()
                mesh_face_normals_world = mesh_face_normals_world.transpose()

                # determine index of closest matching mesh front face: 
                # detect world space face normal closest to
                # axis-aligned front face normal
                front_index = np.argmax(np.sum(mesh_face_normals_world * front_axis, axis=1), axis=0)

                # Next, determine which faces are adjacent to the front face
                # transform the front face to XY plane and define:
                #
                #          |    up    |
                #     ____v3__________v2_____          ^ +y
                #          |          |                |
                #    left  |   front  | right          +---> +x
                #          |          |
                #    _____v0__________v1_____
                #          |          |
                #          |   down   |
                # 
                # Then check all faces and assign other sides based on
                # which faces contain the same vertices as the front face:
                # - left: v0, v3
                # - right: v1, v2
                # - up: v2, v3
                # - down: v0, v1
                # - back: (none)
                # This method ensures each face is assigned only once
                # (for properly defined cuboids).
                # 
                # NOTATION:
                # - f0, f1, f2, f3 are face local indices in values {0, 1, 2, 3}
                # - v0, v1, v2, v3 are corresponding global mesh vertex indices
                #   these are pointers into the mesh.vertices

                # transform/projection the front face into an XY plane
                # hard-coded based on pre-defined ways for how we should
                # look at each face (which determines projection axes)
                mat_front_face_to_xy = np.array([
                    [1.0, 0.0, 0.0], # x <- x
                    [0.0, 0.0, 1.0], # y <- z
                    [0.0, 0.0, 0.0], # z <- 0
                ])
                mat_back_face_to_xy = np.array([
                    [-1.0, 0.0, 0.0], # x <- -x
                    [0.0, 0.0, 1.0],  # y <- z
                    [0.0, 0.0, 0.0],  # z <- 0
                ])
                mat_left_face_to_xy = np.array([
                    [0.0, -1.0, 0.0], # x <- -y
                    [0.0, 0.0, 1.0],  # y <- z
                    [0.0, 0.0, 0.0],  # z <- 0
                ])
                mat_right_face_to_xy = np.array([
                    [0.0, 1.0, 0.0], # x <- y
                    [0.0, 0.0, 1.0], # y <- z
                    [0.0, 0.0, 0.0], # z <- 0
                ])
                mat_up_face_to_xy = np.array([
                    [1.0, 0.0, 0.0], # x <- x
                    [0.0, 1.0, 0.0], # y <- y
                    [0.0, 0.0, 0.0], # z <- 0
                ])
                mat_down_face_to_xy = np.array([
                    [-1.0, 0.0, 0.0], # x <- -x
                    [0.0, 1.0, 0.0],  # y <- y
                    [0.0, 0.0, 0.0],  # z <- 0
                ])
                
                # map face vertices indices to standard v0, v1, v2, v3 format
                # 1. transform front face coords into an XY plane
                #    (use face normal specific transform)
                # 2. find v0 (bottom left) as "min" vertex
                # 3. determine if loop is clockwise or counterclockwise
                # 4. assign v1, v2, v3 based on loop order
                mesh_face_vertices_xy = mat_front_face_to_xy @ mesh_face_vertices[front_index,:,:3].transpose()
                mesh_face_vertices_xy = mesh_face_vertices_xy.transpose()
                u0 = mesh_face_vertices_xy[0,:3]
                u1 = mesh_face_vertices_xy[1,:3]
                u2 = mesh_face_vertices_xy[2,:3]
                idx_v0 = index_of_vmin(mesh_face_vertices_xy)
                is_cw = loop_is_clockwise([u0, u1, u2])

                # print(f"Mesh front face {front_index}: {mesh_face_vertices[front_index,:,:]}")
                # print(f"Mesh front face XY: {mesh_face_vertices_xy}")
                # print(f"idx_v0 = {idx_v0}")
                # print(f"is_cw = {is_cw}")

                if is_cw:
                    idx_v1 = (idx_v0 + 3) % 4
                    idx_v2 = (idx_v0 + 2) % 4
                    idx_v3 = (idx_v0 + 1) % 4
                else:
                    idx_v1 = (idx_v0 + 1) % 4
                    idx_v2 = (idx_v0 + 2) % 4
                    idx_v3 = (idx_v0 + 3) % 4

                # get mesh global vertex indices for front face
                front_v0 = mesh_face_vert_indices[front_index, idx_v0]
                front_v1 = mesh_face_vert_indices[front_index, idx_v1]
                front_v2 = mesh_face_vert_indices[front_index, idx_v2]
                front_v3 = mesh_face_vert_indices[front_index, idx_v3]

                # print(f"FRONT: idx_v0 = {idx_v0}, idx_v1 = {idx_v1}, idx_v2 = {idx_v2}, idx_v3 = {idx_v3}")
                # print(f"FRONT: v0 = {front_v0}, v1 = {front_v1}, v2 = {front_v2}, v3 = {front_v3}")
                
                mesh_face_f0_f1_f2_f3 = np.zeros((6, 4), dtype=int)
                mesh_face_v0_v1_v2_v3 = np.zeros((6, 4), dtype=int)
                mesh_face_f0_f1_f2_f3[front_index,:] = [idx_v0, idx_v1, idx_v2, idx_v3]
                mesh_face_v0_v1_v2_v3[front_index,:] = [front_v0, front_v1, front_v2, front_v3]

                # detect face directions relative to the front face.
                # do this by matching shared points between faces
                mesh_face_directions = np.full((6,), -1, dtype=int)
                mesh_face_directions[front_index] = IDX_UV_FACE_FRONT
                for i in range(0, 6): # for face in mesh.polygons
                    if i == front_index:
                        continue
                    
                    face_vert_indices = mesh_face_vert_indices[i,:]

                    if front_v0 in face_vert_indices and front_v3 in face_vert_indices:
                        mesh_face_directions[i] = IDX_UV_FACE_LEFT
                        # left f2 is index of front f3
                        f2 = np.argwhere(face_vert_indices == front_v3)[0][0]
                        f0 = (f2 + 2) % 4
                        face_verts_xy = mat_left_face_to_xy @ mesh_face_vertices[i,:,:3].transpose()
                    elif front_v1 in face_vert_indices and front_v2 in face_vert_indices:
                        mesh_face_directions[i] = IDX_UV_FACE_RIGHT
                        # right f0 is index of front f1
                        f0 = np.argwhere(face_vert_indices == front_v1)[0][0]
                        face_verts_xy = mat_right_face_to_xy @ mesh_face_vertices[i,:,:3].transpose()
                    elif front_v2 in face_vert_indices and front_v3 in face_vert_indices:
                        mesh_face_directions[i] = IDX_UV_FACE_UP
                        # up f0 is index of front f3
                        f0 = np.argwhere(face_vert_indices == front_v3)[0][0]
                        face_verts_xy = mat_up_face_to_xy @ mesh_face_vertices[i,:,:3].transpose()
                    elif front_v0 in face_vert_indices and front_v1 in face_vert_indices:
                        mesh_face_directions[i] = IDX_UV_FACE_DOWN
                        # down f2 is index of front f1 
                        f2 = np.argwhere(face_vert_indices == front_v1)[0][0]
                        f0 = (f2 + 2) % 4
                        face_verts_xy = mat_down_face_to_xy @ mesh_face_vertices[i,:,:3].transpose()
                    else:
                        mesh_face_directions[i] = IDX_UV_FACE_BACK
                        face_verts_xy = mat_back_face_to_xy @ mesh_face_vertices[i,:,:3].transpose()
                        f0 = index_of_vmin(face_verts_xy.transpose())
                    
                    is_cw = loop_is_clockwise(face_verts_xy[0:3,0:3].transpose())
                    if is_cw:
                        f3 = (f0 + 1) % 4
                        f2 = (f0 + 2) % 4
                        f1 = (f0 + 3) % 4
                    else:
                        f1 = (f0 + 1) % 4
                        f2 = (f0 + 2) % 4
                        f3 = (f0 + 3) % 4
                    # maps face vertices in f in {0,1,2,3}
                    # to v in global mesh vertices indices
                    v0 = mesh_face_vert_indices[i, f0]
                    v1 = mesh_face_vert_indices[i, f1]
                    v2 = mesh_face_vert_indices[i, f2]
                    v3 = mesh_face_vert_indices[i, f3]
                    mesh_face_f0_f1_f2_f3[i,:] = [f0, f1, f2, f3] # local mesh loop indices {0, 1, 2, 3}
                    mesh_face_v0_v1_v2_v3[i,:] = [v0, v1, v2, v3] # pointers to mesh.vertices

                if -1 in mesh_face_directions:
                    raise Exception("Invalid cuboid mesh, some faces are not properly defined, could not determine face directions")

                # we have assigned all face directions and mapped all
                # face vertices to their v0, v1, v2, v3 face uv order.
                # now, determine face width (v0 -> v1) and height (v1 -> v2)

                # 6 faces, each as 
                face_uv_width_height = np.zeros((6, 2))        # (face, width/height)
                face_uv_xy = np.zeros((6, 4, 2))               # (face, vert, xy)
                face_uv_loop_start = np.zeros((6,), dtype=int) # (face,)

                # creates uvs based on face width/height and 
                # maps faces index to uv index in LEFT-FRONT-RIGHT-BACK-UP-DOWN
                for i in range(0, 6):
                    direction_index = mesh_face_directions[i]
                    f0, f1, f2, f3 = mesh_face_f0_f1_f2_f3[i,:]
                    v0, v1, v2, v3 = mesh_face_v0_v1_v2_v3[i,:]

                    face_width = np.linalg.norm(vertices[:3,v0] - vertices[:3,v1])
                    face_height = np.linalg.norm(vertices[:3,v1] - vertices[:3,v2])
                    
                    face_uv_width_height[direction_index,0] = face_width
                    face_uv_width_height[direction_index,1] = face_height

                    face_uv_xy[direction_index,f0,:] = (0.0, 0.0)
                    face_uv_xy[direction_index,f1,:] = (face_width, 0.0)
                    face_uv_xy[direction_index,f2,:] = (face_width, face_height)
                    face_uv_xy[direction_index,f3,:] = (0.0, face_height)

                    face_uv_loop_start[direction_index] = mesh_face_uv_loop_start[i]
                
                # original uv_xy are local sizes of each face
                # first translate each face to its unwrapped position
                uv_offset = np.zeros((6, 2))
                # LEFT: x=0, y=0, no change
                # FRONT
                uv_offset[1,:] = (face_uv_width_height[0,0], 0.0)
                # RIGHT
                uv_offset[2,:] = (face_uv_width_height[0,0] + face_uv_width_height[1,0], 0.0)
                # BACK
                uv_offset[3,:] = (face_uv_width_height[0,0] + face_uv_width_height[1,0] + face_uv_width_height[2,0], 0.0)
                # UP
                uv_offset[4,:] = (face_uv_width_height[0,0], face_uv_width_height[0,1])
                # DOWN, x = x_left + x_up
                uv_offset[5,:] = (face_uv_width_height[0,0] + face_uv_width_height[4,0], face_uv_width_height[0,1])
                
                # shape broadcasting:
                # (6,4,2)   =     (6,1,2)         +   (6,4,2)
                uv_xy = uv_offset[:,np.newaxis,:] + face_uv_xy
                
                # finally, scale the entire uv map to fit into the (0, 1) square
                uv_scale = 1.0 / uv_xy.max(axis=(0, 1, 2))
                uv_xy_normalized = uv_scale * uv_xy

                uv_x = uv_xy_normalized[:,:,0]
                uv_y = uv_xy_normalized[:,:,1]

                # update face uvs
                for i in range(0, 6):
                    idx = face_uv_loop_start[i]
                    uv_layer[idx].uv = (uv_x[i,0], uv_y[i,0])
                    uv_layer[idx+1].uv = (uv_x[i,1], uv_y[i,1])
                    uv_layer[idx+2].uv = (uv_x[i,2], uv_y[i,2])
                    uv_layer[idx+3].uv = (uv_x[i,3], uv_y[i,3])
            
            except Exception as e:
                import traceback
                traceback.print_exc()
                err_msg = f"Error unwrapping cuboid: {obj.name}, {e}"
                log.error(err_msg)
                self.report({"ERROR"}, err_msg)
            
        if need_to_switch_mode_back:
            bpy.ops.object.mode_set(mode=user_mode)
        
        return {"FINISHED"}


class OpUVPixelUnwrap(bpy.types.Operator):
    """Unwrap all UVs into a single pixel (for single color textures)"""
    bl_idname = "vintagestory.uv_pixel_unwrap"
    bl_label = "Pixel UV Unwrap (VS)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        # args = self.as_keywords()

        # need to be in object mode to access context selected objects
        user_mode = context.active_object.mode
        if user_mode != "OBJECT":
            need_to_switch_mode_back = True
            bpy.ops.object.mode_set(mode="OBJECT")
        else:
            need_to_switch_mode_back = False
        
        # only perform on selected objects
        objects = bpy.context.selected_objects
        for obj in objects:
            mesh = obj.data
            if not isinstance(mesh, bpy.types.Mesh):
                continue
            
            uv_layer = mesh.uv_layers.active.data

            for face in mesh.polygons:
                loop_start = face.loop_start
                # TODO: if object has texture, scale UVs to fit 1 pixel in texture.
                # otherwise, set to unit square (0,0) to (1,1)
                uv_layer[loop_start].uv = (0.0, 0.0)
                uv_layer[loop_start+1].uv = (1.0, 0.0)
                uv_layer[loop_start+2].uv = (1.0, 1.0)
                uv_layer[loop_start+3].uv = (0.0, 1.0)
        
        if need_to_switch_mode_back:
            bpy.ops.object.mode_set(mode=user_mode)
        
        return {"FINISHED"}


class OpUVPackSimpleBoundingBox(bpy.types.Operator):
    """Simple cuboid uv pack that treats all uv faces in a cuboid mesh as a
    connected island. Default Blender uv pack needs connected faces as an
    island, but the cuboid unwraps create disjointed faces."""
    bl_idname = "vintagestory.uv_pack_simple_bounding_box"
    bl_label = "Pixel UV Unwrap (VS)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        self.report({"ERROR"}, "Not implemented")
        return {"FINISHED"}
