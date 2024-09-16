import bpy
from . import animation
from . import primitive
from . import uv

# reload imported modules
import importlib
importlib.reload(animation)
importlib.reload(primitive)
importlib.reload(uv)

def rna_idprop_context_value(context, context_member, property_type):
    """
    https://github.com/blender/blender/blob/main/scripts/modules/rna_prop_ui.py
    """
    space = context.space_data

    if space is None or isinstance(space, bpy.types.SpaceProperties):
        pin_id = space.pin_id
    else:
        pin_id = None

    if pin_id and isinstance(pin_id, property_type):
        rna_item = pin_id
        context_member = "space_data.pin_id"
    else:
        rna_item = context.path_resolve(context_member)

    return rna_item, context_member

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
def rna_idprop_quote_path(prop):
    return "[\"{:s}\"]".format(bpy.utils.escape_identifier(prop))

def draw_custom_prop(
    layout,
    rna_item,
    context_member,
    property_type,
    value_type,
    key,
) -> bool:
    """Helper to draw custom property into a layout row.
    Return true if successful, false if failed.
    """
    # poll should get this...
    if not rna_item:
        return False
    
    if not isinstance(rna_item, property_type):
        return False
    
    if key not in rna_item or not isinstance(rna_item.get(key), value_type):
        return False
    
    if rna_item.id_data.library is not None:
        use_edit = False
    else:
        use_edit = True
    
    is_lib_override = rna_item.id_data.override_library and rna_item.id_data.override_library.reference

    layout.use_property_decorate = False

    value_row = layout.row()
    value_column = value_row.column(align=True)

    value_column.prop(rna_item, rna_idprop_quote_path(key), text="")

    operator_row = value_row.row(align=True)
    operator_row.alignment = "RIGHT"

    # Do not allow editing of overridden properties (we cannot use a poll function
    # of the operators here since they have no access to the specific property).
    operator_row.enabled = not (is_lib_override and key in rna_item.id_data.override_library.reference)

    if use_edit:
        props = operator_row.operator("wm.properties_remove", text="", icon='X', emboss=False)
        props.data_path = context_member
        props.property_name = key
    
    return True


class VINTAGESTORY_PT_panel_animation_tools(bpy.types.Panel):
    """Vintagestory tools in viewport N-panel:
    Contains animation tools.

    For displaying custom props (e.g. stepParentName, renameObject), see:
    https://github.com/blender/blender/blob/main/scripts/modules/rna_prop_ui.py
    """
    bl_idname = "VINTAGESTORY_PT_panel_animation_tools"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "VintageStory"
    bl_label = "Animation Tools"

    # for viewing object custom properties
    _context_path = "object"
    _property_type = bpy.types.Object

    @classmethod
    def poll(cls, context):
        """'_context_path' MUST be set to access custom properties.
        https://github.com/blender/blender/blob/main/scripts/modules/rna_prop_ui.py
        """
        context_path = "object"
        property_type = bpy.types.Object
        rna_item, _context_member = rna_idprop_context_value(context, context_path, property_type)
        return bool(rna_item)
    
    def draw(self, context):
        layout = self.layout
        rna_item, context_member = rna_idprop_context_value(context, self._context_path, self._property_type)

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
        
        # show step parent name for selected object
        row = layout.row()
        row.label(text="Step Parent Name:")
        
        if not draw_custom_prop(layout, rna_item, context_member, self._property_type, str, "StepParentName"):
            layout.label(text="")

        layout.row().separator()

        # operator: assign rename on export name to selected objects
        layout.operator(
            operator="vintagestory.assign_rename",
            icon="GREASEPENCIL",
            text="Rename on Export",
        )

        
        # rename object property
        row = layout.row()
        row.label(text="Rename on Export:")
        
        if not draw_custom_prop(layout, rna_item, context_member, self._property_type, str, "rename"):
            layout.label(text="")

        
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
    animation.OpAssignRename,
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
