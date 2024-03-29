import bpy
import math

class OpPrimitiveAddBlock(bpy.types.Operator):
    """Standard block 16x16, equivalent to a bounding box in VS model creator"""
    bl_idname = "vintagestory.primitive_add_block"
    bl_label = "Add Block"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        bpy.ops.mesh.primitive_cube_add()
        cube = bpy.context.active_object
        bpy.ops.transform.resize(value=(8.0, 8.0, 8.0))
        bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)
        bpy.ops.transform.translate(value=(0, 0, 8.0))
        cube.select_set(True)

        return {"FINISHED"}


class OpPrimitiveAddOctagonal(bpy.types.Operator):
    """Octagonal cylinder (8 sided) built from 4x cuboids
       _
     /   \     height = (1 + sqrt(2)) * edge 
    |     |
     \ _ /
    """
    bl_idname = "vintagestory.primitive_add_octagonal"
    bl_label = "Add Octagonal (4 Cuboids)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        size = 1
        edge = size / (1 + math.sqrt(2))

        # transform for each part in form (scale),[(rotate1), (rotate2), ...]
        transforms = [
            ( (edge, 1, 1), None ),
            ( (1, 1, edge), None ),
            ( (edge, 1, 1), ('Y', math.pi/4) ),
            ( (edge, 1, 1), ('Y', -math.pi/4) ),
        ]

        cubes = []
        for scale, rotation in transforms:
            bpy.ops.mesh.primitive_cube_add()
            cube = bpy.context.active_object
            bpy.ops.transform.resize(value=scale)
            if rotation is not None:
                bpy.ops.transform.rotate(value=rotation[1], orient_axis=rotation[0])
            bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)
            cubes.append(cube)

        # select all
        for c in cubes:
            c.select_set(True)
        
        return {"FINISHED"}


class OpPrimitiveAddOctagonalHollow(bpy.types.Operator):
    """Hollow Octagonal cylinder (8 sided) built from 8x cuboids
       _
     / _ \     height = (1 + sqrt(2)) * edge 
    | |_| |
     \ _ /
    """
    bl_idname = "vintagestory.primitive_add_octagonal_hollow"
    bl_label = "Add Hollow Octagonal (8 Cuboids)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        size_outer_half_extent = 1
        size_inner_half_extent = 0.7
        edge = size_outer_half_extent / (1 + math.sqrt(2))
        thick = 0.5 * (size_outer_half_extent - size_inner_half_extent)

        cubes = []

        for i in range(0,8):
            bpy.ops.mesh.primitive_cube_add()
            cube = bpy.context.active_object
            bpy.ops.transform.resize(value=(edge, 1, thick))
            bpy.ops.transform.translate(value=(0, 0, size_outer_half_extent - thick))
            bpy.ops.object.transform_apply(location=True, scale=True, rotation=False)
            bpy.ops.transform.rotate(value=i * math.pi/4, orient_axis='Y')
            cubes.append(cube)

        # select all
        for c in cubes:
            c.select_set(True)
        
        return {"FINISHED"}


class OpPrimitiveAddHexadecagon(bpy.types.Operator):
    """Solid hexadecagon cylinder (16-sided) built from 8x cuboids
    (height = sin(7pi/16) / sin(pi/16) * edge)"""
    bl_idname = "vintagestory.primitive_add_hexadecagon"
    bl_label = "Add Hexadecagon (8 Cuboids)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        size_outer_half_extent = 1
        thick = size_outer_half_extent * math.sin(math.pi / 16.0) / math.sin(7.0 * math.pi / 16)

        cubes = []

        for i in range(0,16):
            bpy.ops.mesh.primitive_cube_add()
            cube = bpy.context.active_object
            bpy.ops.transform.resize(value=(1, 1, thick))
            bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)
            bpy.ops.transform.rotate(value=i * math.pi/8, orient_axis='Y')
            cubes.append(cube)

        # select all
        for c in cubes:
            c.select_set(True)
        
        return {"FINISHED"}


class OpPrimitiveAddHexadecagonHollow(bpy.types.Operator):
    """Hollow hexadecagon cylinder (16-sided) built from 16x cuboids
    (height = sin(7pi/16) / sin(pi/16) * edge)"""
    bl_idname = "vintagestory.primitive_add_hexadecagon_hollow"
    bl_label = "Add Hollow Hexadecagon (16 Cuboids)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        size_outer_half_extent = 1
        size_inner_half_extent = 0.7
        edge = size_outer_half_extent * math.sin(math.pi / 16.0) / math.sin(7.0 * math.pi / 16)
        thick = 0.5 * (size_outer_half_extent - size_inner_half_extent)

        cubes = []

        for i in range(0,16):
            bpy.ops.mesh.primitive_cube_add()
            cube = bpy.context.active_object
            bpy.ops.transform.resize(value=(edge, 1, thick))
            bpy.ops.transform.translate(value=(0, 0, size_outer_half_extent - thick))
            bpy.ops.object.transform_apply(location=True, scale=True, rotation=False)
            bpy.ops.transform.rotate(value=i * math.pi/8, orient_axis='Y')
            cubes.append(cube)

        # select all
        for c in cubes:
            c.select_set(True)
        
        return {"FINISHED"}


class OpPrimitiveAddOctsphere(bpy.types.Operator):
    """Sphere-estimate based on octagonal on each axis,
    where top part is scaled to be square (edge x edge)
       __      
      /_/ \\
     /|_|\     height = (1 + sqrt(2)) * edge 
    |_|_|_|
     \|_|/
    """
    bl_idname = "vintagestory.primitive_add_octsphere"
    bl_label = "Add Octsphere"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        size = 1
        edge = size / (1 + math.sqrt(2))

        # transform for each part in form (scale),[(rotate1), (rotate2), ...]
        transforms = [
            ( (edge, edge, 1), None ),
            ( (1, edge, edge), None ),
            ( (edge, 1, edge), None ),
            ( (edge, 1, edge), ('X', math.pi/4) ),
            ( (edge, 1, edge), ('X', -math.pi/4) ),
            ( (edge, edge, 1), ('Y', math.pi/4) ),
            ( (edge, edge, 1), ('Y', -math.pi/4) ),
            ( (1, edge, edge), ('Z', math.pi/4) ),
            ( (1, edge, edge), ('Z', -math.pi/4) ),
        ]

        cubes = []
        for scale, rotation in transforms:
            bpy.ops.mesh.primitive_cube_add()
            cube = bpy.context.active_object
            bpy.ops.transform.resize(value=scale)
            if rotation is not None:
                bpy.ops.transform.rotate(value=rotation[1], orient_axis=rotation[0])
            bpy.ops.object.transform_apply(location=False, scale=True, rotation=False)
            cubes.append(cube)

        # select all
        for c in cubes:
            c.select_set(True)
        
        return {"FINISHED"}
