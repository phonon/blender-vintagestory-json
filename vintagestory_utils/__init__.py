import bpy
from . import animation
from . import primitive
from . import uv

# reload imported modules
import importlib
importlib.reload(animation)
importlib.reload(primitive)
importlib.reload(uv)

class VIEW3D_MT_vintagestory_submenu(bpy.types.Menu):
    bl_idname = "VIEW3D_MT_vintagestory_submenu"
    bl_label = "VintageStory"
    
    def draw(self, context):
        layout = self.layout
        layout.operator(
            primitive.OpPrimitiveAddBlock.bl_idname,
            text="Block",
            icon="MESH_CUBE")
        layout.operator(
            primitive.OpPrimitiveAddOctagonal.bl_idname,
            text="Octagonal",
            icon="MESH_CYLINDER")
        layout.operator(
            primitive.OpPrimitiveAddOctagonalHollow.bl_idname,
            text="Octagonal (Hollow)",
            icon="MESH_TORUS")
        layout.operator(
            primitive.OpPrimitiveAddHexadecagon.bl_idname,
            text="Hexadecagon",
            icon="MESH_CYLINDER")
        layout.operator(
            primitive.OpPrimitiveAddHexadecagonHollow.bl_idname,
            text="Hexadecagon (Hollow)",
            icon="MESH_TORUS")
        layout.operator(
            primitive.OpPrimitiveAddOctsphere.bl_idname,
            text="Octsphere",
            icon="MESH_UVSPHERE")

# =============================================================================
# UV tools
# =============================================================================
class VINTAGESTORY_PT_panel_uv_tools(bpy.types.Panel):
    """Vintagestory tools in viewport N-panel:
    Contains UV util tools.
    """
    bl_idname = "VINTAGESTORY_PT_panel_uv_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VintageStory"
    bl_label = "Texture Tools"
 
    def draw(self, context):
        layout = self.layout

        # operator: unwrap selected cuboid objects UVs
        layout.operator(
            operator="vintagestory.uv_cuboid_unwrap",
            icon="UV_DATA",
            text="Cuboid UV Unwrap",
        )
        # operator: unwrap selected objects UVs into single pixel
        layout.operator(
            operator="vintagestory.uv_pixel_unwrap",
            icon="UV_DATA",
            text="Pixel UV Unwrap",
        )
        # operator: simple bounding box uv pack
        layout.operator(
            operator="vintagestory.uv_pack_simple_bounding_box",
            icon="STICKY_UVS_LOC",
            text="UV Pack",
        )

# =============================================================================
# Animation tools
# =============================================================================

class VINTAGESTORY_PT_panel_animation_tools(bpy.types.Panel):
    """Vintagestory tools in viewport N-panel:
    Contains animation tools.
    """
    bl_idname = "VINTAGESTORY_PT_panel_animation_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VintageStory"
    bl_label = "Animation Tools"
 
    def draw(self, context):
        layout = self.layout

        # operator: make selected armature bones rotation mode XZY (default VS mode)
        layout.operator(
            operator="vintagestory.make_bones_xzy",
            icon="GROUP_BONE",
            text="Make Bones XZY",
        )

        # operator: unwrap selected objects UVs into single pixel
        layout.operator(
            operator="vintagestory.assign_bones",
            icon="BONE_DATA",
            text="Auto Assign Bones",
        )

        # operator: assign step parent name to selected objects
        layout.operator(
            operator="vintagestory.assign_step_parent_name",
            icon="CON_CHILDOF",
            text="Step Parent Name",
        )


def add_submenu(self, context):
    self.layout.separator()
    self.layout.menu(VIEW3D_MT_vintagestory_submenu.bl_idname, icon="MESH_CUBE")

# register
classes = [
    # OPERATORS
    primitive.OpPrimitiveAddBlock,
    primitive.OpPrimitiveAddOctagonal,
    primitive.OpPrimitiveAddOctagonalHollow,
    primitive.OpPrimitiveAddHexadecagon,
    primitive.OpPrimitiveAddHexadecagonHollow,
    primitive.OpPrimitiveAddOctsphere,
    animation.OpMakeBonesXZY,
    animation.OpAssignBones,
    animation.OpAssignStepParentName,
    uv.OpUVCuboidUnwrap,
    uv.OpUVPixelUnwrap,
    uv.OpUVPackSimpleBoundingBox,
    # PANELS AND MENUS
    VIEW3D_MT_vintagestory_submenu,
    VINTAGESTORY_PT_panel_uv_tools,
    VINTAGESTORY_PT_panel_animation_tools,
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
