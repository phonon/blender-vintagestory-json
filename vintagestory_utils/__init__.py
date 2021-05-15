import bpy
import math

class VintageStoryPrimitiveAddBlock(bpy.types.Operator):
    """Standard block 16x16, equivalent to a bounding box in
    VS model creator.
    """
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

        return {'FINISHED'}


# solid octagonal:
#    _
#  /   \     height = (1 + sqrt(2)) * edge 
# |     |
#  \ _ /
class VintageStoryPrimitiveAddOctagonal(bpy.types.Operator):
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
        
        return {'FINISHED'}

# hollow octagonal (from 8 cuboids)
#    _
#  / _ \     height = (1 + sqrt(2)) * edge 
# | |_| |
#  \ _ /
class VintageStoryPrimitiveAddOctagonalHollow(bpy.types.Operator):
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
        
        return {'FINISHED'}

# solid hexadecagon (16-sided polygon) (uses 8 cuboids)
# height = sin(7pi/16) / sin(pi/16) * edge
class VintageStoryPrimitiveAddHexadecagon(bpy.types.Operator):
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
        
        return {'FINISHED'}

# hollow hexadecagon (16-sided polygon) (uses 16 cuboids)
# height = sin(7pi/16) / sin(pi/16) * edge
class VintageStoryPrimitiveAddHexadecagonHollow(bpy.types.Operator):
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
        
        return {'FINISHED'}

# Sphere-estimate based on octagonal on each axis
#    __      top part is scaled to be square (edge x edge)
#   /_/ \
#  /|_|\     height = (1 + sqrt(2)) * edge 
# |_|_|_|
#  \|_|/
class VintageStoryPrimitiveAddOctsphere(bpy.types.Operator):
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
        
        return {'FINISHED'}
    
class VIEW3D_MT_vintagestory_submenu(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_vintagestory_submenu"
    bl_label = "VintageStory"
    
    def draw(self, context):
        layout = self.layout
        layout.operator(
            VintageStoryPrimitiveAddBlock.bl_idname,
            text="Block",
            icon="MESH_CUBE")
        layout.operator(
            VintageStoryPrimitiveAddOctagonal.bl_idname,
            text="Octagonal",
            icon="MESH_CYLINDER")
        layout.operator(
            VintageStoryPrimitiveAddOctagonalHollow.bl_idname,
            text="Octagonal (Hollow)",
            icon="MESH_TORUS")
        layout.operator(
            VintageStoryPrimitiveAddHexadecagon.bl_idname,
            text="Hexadecagon",
            icon="MESH_CYLINDER")
        layout.operator(
            VintageStoryPrimitiveAddHexadecagonHollow.bl_idname,
            text="Hexadecagon (Hollow)",
            icon="MESH_TORUS")
        layout.operator(
            VintageStoryPrimitiveAddOctsphere.bl_idname,
            text="Octsphere",
            icon="MESH_UVSPHERE")
    
def add_submenu(self, context):
    self.layout.separator()
    self.layout.menu(VIEW3D_MT_vintagestory_submenu.bl_idname, icon="MESH_CUBE")

# register
classes = [
    VintageStoryPrimitiveAddBlock,
    VintageStoryPrimitiveAddOctagonal,
    VintageStoryPrimitiveAddOctagonalHollow,
    VintageStoryPrimitiveAddHexadecagon,
    VintageStoryPrimitiveAddHexadecagonHollow,
    VintageStoryPrimitiveAddOctsphere,
    VIEW3D_MT_vintagestory_submenu,
]

def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)

    # add minecraft primitives to object add menu
    bpy.types.VIEW3D_MT_add.append(add_submenu)

def unregister():
    # remove minecraft primitives from object add menu
    bpy.types.VIEW3D_MT_add.remove(add_submenu)
    
    from bpy.utils import unregister_class
    for cls in reversed(classes):
        unregister_class(cls)

if __name__ == "__main__":
    register()
