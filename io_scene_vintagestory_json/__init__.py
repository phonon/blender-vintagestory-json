# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import StringProperty, BoolProperty, EnumProperty, IntProperty, FloatProperty
from bpy.types import Operator

from . import export_vintagestory_json
from . import import_vintagestory_json

# reload imported modules
import importlib
importlib.reload(export_vintagestory_json) 
importlib.reload(import_vintagestory_json)

class ImportVintageStoryJSON(Operator, ImportHelper):
    """Import VintageStory .json file"""
    bl_idname = "vintagestory.import_json"
    bl_label = "Import a VintageStory .json model"
    bl_options = {"REGISTER", "UNDO"}

    # ImportHelper mixin class uses this
    filename_ext = ".json"

    filter_glob: StringProperty(
        default="*.json",
        options={"HIDDEN"},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    import_uvs: BoolProperty(
        name="Import UVs",
        description="Import UVs",
        default=True,
    )

    recenter_to_origin: BoolProperty(
        name="Recenter on Origin",
        description="Recenter model center to origin",
        default=False,
    )

    # applies default shift from vintagestory origin
    translate_origin: BoolProperty(
        name="Translate Origin",
        description="Translate model origin after import",
        default=True,
    )
    
    translate_origin_x: FloatProperty(
        name="Translate X",
        description="X export offset (in Blender coordinates)",
        default=-8.,
    )
    translate_origin_y: FloatProperty(
        name="Translate Y",
        description="Y export offset (in Blender coordinates)",
        default=-8.,
    )
    translate_origin_z: FloatProperty(
        name="Translate Z",
        description="Z export offset (in Blender coordinates)",
        default=0.,
    )

    # import animations
    import_animations: BoolProperty(
        name="Import Animations",
        description="Import animations",
        default=True,
    )

    def execute(self, context):
        args = self.as_keywords()
        if args["translate_origin"] == True:
            args["translate_origin"] = [
                args["translate_origin_x"],
                args["translate_origin_y"],
                args["translate_origin_z"],
            ]
        else:
            args["translate_origin"] = None

        return import_vintagestory_json.load(context, **args)


class ExportVintageStoryJSON(Operator, ExportHelper):
    """Exports scene cuboids as VintageStory .json object"""
    bl_idname = "vintagestory.export_json"
    bl_label = "Export as VintageStory .json"

    # ExportHelper mixin class uses this
    filename_ext = ".json"

    filter_glob: StringProperty(
        default="*.json",
        options={"HIDDEN"},
        maxlen=255,  # Max internal buffer length, longer would be clamped.
    )

    selection_only: BoolProperty(
        name="Selection Only",
        description="Export selection",
        default=False,
    )

    # applies default shift from vintagestory origin
    translate_origin: BoolProperty(
        name="Translate Origin",
        description="Translate model origin after export",
        default=True,
    )
    
    translate_origin_x: FloatProperty(
        name="Translate X",
        description="X export offset (in Blender coordinates)",
        default=8.,
    )
    translate_origin_y: FloatProperty(
        name="Translate Y",
        description="Y export offset (in Blender coordinates)",
        default=8.,
    )
    translate_origin_z: FloatProperty(
        name="Translate Z",
        description="Z export offset (in Blender coordinates)",
        default=0.,
    )

    # ================================
    # texture options
    texture_folder: StringProperty(
        name="Texture Subfolder",
        description="Subfolder in resourcepack: assets/vintagestory/textures/[folder]",
        default="",
    )

    color_texture_filename: StringProperty(
        name="Color Texture Name",
        description="Export color texture filename",
        default="",
    )

    export_uvs: BoolProperty(
        name="Export UVs",
        description="Export UVs",
        default=True,
    )
    
    generate_texture: BoolProperty(
        name="Generate Color Texture",
        description="Generate texture image from all material colors",
        default=True,
    )
    
    use_only_exported_object_colors: BoolProperty(
        name="Only Use Exported Object Colors",
        description="Generate texture from material colors only on exported objects",
        default=False,
    )

    # ================================
    # minify options
    minify: BoolProperty(
        name="Minify .json",
        description="Enables minification options to reduce .json file size",
        default=False,
    )

    decimal_precision: IntProperty(
        name="Decimal Precision",
        description="Number of digits after decimal point (use -1 to disable)",
        min=-1,
        max=16,
        soft_min=-1,
        soft_max=16,
        default=8,
    )

    # ================================
    # animation options EXPERIMENTAL
    export_animations: BoolProperty(
        name="Export Animations",
        description="Export bone animations keyframes",
        default=True,
    )

    def execute(self, context):
        args = self.as_keywords()
        if args["translate_origin"] == True:
            args["translate_origin"] = [
                args["translate_origin_x"],
                args["translate_origin_y"],
                args["translate_origin_z"],
            ]
        else:
            args["translate_origin"] = None
        
        return export_vintagestory_json.save(context, **args)
    
    def draw(self, context):
        pass


# export options panel for geometry
class VINTAGESTORY_PT_export_geometry(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "Geometry"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator
        return operator.bl_idname == "VINTAGESTORY_OT_export_json"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        sfile = context.space_data
        operator = sfile.active_operator

        layout.prop(operator, "selection_only")
        layout.prop(operator, "translate_origin")
        layout.prop(operator, "translate_origin_x")
        layout.prop(operator, "translate_origin_y")
        layout.prop(operator, "translate_origin_z")


# export options panel for textures
class VINTAGESTORY_PT_export_textures(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "Textures"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator
        return operator.bl_idname == "VINTAGESTORY_OT_export_json"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        sfile = context.space_data
        operator = sfile.active_operator

        layout.prop(operator, "texture_folder")
        layout.prop(operator, "color_texture_filename")
        layout.prop(operator, "export_uvs")
        layout.prop(operator, "generate_texture")
        layout.prop(operator, "use_only_exported_object_colors")


# export options panel for minifying .json output
class VINTAGESTORY_PT_export_minify(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "Minify"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator
        return operator.bl_idname == "VINTAGESTORY_OT_export_json"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        sfile = context.space_data
        operator = sfile.active_operator

        layout.prop(operator, "minify")
        layout.prop(operator, "decimal_precision")

# export options panel for animation
class VINTAGESTORY_PT_export_animation(bpy.types.Panel):
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "Animation"
    bl_parent_id = "FILE_PT_operator"

    @classmethod
    def poll(cls, context):
        sfile = context.space_data
        operator = sfile.active_operator
        return operator.bl_idname == "VINTAGESTORY_OT_export_json"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.use_property_decorate = False  # No animation.

        sfile = context.space_data
        operator = sfile.active_operator

        layout.prop(operator, "export_animations")

# add io to menu
def menu_func_import(self, context):
    self.layout.operator(ImportVintageStoryJSON.bl_idname, text="VintageStory (.json)")

def menu_func_export(self, context):
    self.layout.operator(ExportVintageStoryJSON.bl_idname, text="VintageStory (.json)")

# register
classes = [
    ImportVintageStoryJSON,
    ExportVintageStoryJSON,
    VINTAGESTORY_PT_export_geometry,
    VINTAGESTORY_PT_export_textures,
    VINTAGESTORY_PT_export_minify,
    VINTAGESTORY_PT_export_animation,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

if __name__ == "__main__":
    register()