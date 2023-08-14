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
    do_translate_origin: BoolProperty(
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
        if args["do_translate_origin"] == True:
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

    # these are inherited + not user settable properties
    skip_save_props = { 
        "filepath",
        "check_existing",
        "initialized",
        "filter_glob",
        "translate_origin",
    }

    # operator initialized flag
    initialized: BoolProperty(
        default=False,
    )

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

    skip_disabled_render: BoolProperty(
        name="Skip Disabled Render",
        description="Skip objects with disabled render",
        default=True,
    )

    # applies default shift from vintagestory origin
    do_translate_origin: BoolProperty(
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

    texture_size_x_override: IntProperty(
        name="Texture Size X Override",
        description="Override texture size X during UV export",
        default=0,
        min=0,
    )

    texture_size_y_override: IntProperty(
        name="Texture Size Y Override",
        description="Override texture size Y during UV export",
        default=0,
        min=0,
    )

    export_uvs: BoolProperty(
        name="Export UVs",
        description="Export UVs",
        default=True,
    )
    
    generate_texture: BoolProperty(
        name="Generate Color Texture",
        description="Generate texture image from all material colors",
        default=False,
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
    # animation options
    export_armature: BoolProperty(
        name="Export Armature",
        description="Export by main armature tree",
        default=True,
    )

    export_animations: BoolProperty(
        name="Export Animations",
        description="Export bone animations keyframes",
        default=True,
    )

    use_main_object_as_bone: BoolProperty(
        name="Use Main Object as Bone",
        description="Use main object with same transform as bone instead of dummy bone",
        default=True,
    )

    # ================================
    # run post-export python script

    # get stored property from scene
    # TODO: should we just store all export properties into scene?

    run_post_export_script: BoolProperty(
        name="Run Post Export Script",
        description="Run post export python script",
        default=False,
    )

    post_export_script: StringProperty(
        name="Post Export Script",
        description="Name of post export python script (in .blend file folder)",
        default="",
    )

    def execute(self, context):
        args = self.as_keywords()
        if args["do_translate_origin"] == True:
            args["translate_origin"] = [
                args["translate_origin_x"],
                args["translate_origin_y"],
                args["translate_origin_z"],
            ]
        else:
            args["translate_origin"] = None
        
        # store executed export config properties to scene
        for prop, val in args.items():
            if prop not in self.skip_save_props:
                bpy.context.scene["vintagestory_export_" + prop] = val

        result = export_vintagestory_json.save(context, **args)

        if self.run_post_export_script:
            import os
            import subprocess
            savefilename= self.filepath
            savefolder = os.path.dirname(savefilename)
            script = os.path.join(savefolder, self.post_export_script)
            if self.post_export_script != "" and os.path.exists(script) and os.path.isfile(script):
                print(f"Running post export script: {script}")
                subprocess.run(["python", script, savefilename], shell=False)
            else:
                self.report({"WARNING"}, f"Post export script not found: {self.post_export_script}")
            
        return result

    def draw(self, context):
        if not self.initialized:
            self.initialized = True

            # first draw: load saved export properties from scene
            args = self.as_keywords()
            for prop in args.keys():
                if prop in self.skip_save_props:
                    continue
                prop_key = "vintagestory_export_" + prop
                if prop_key in bpy.context.scene:
                    setattr(self, prop, bpy.context.scene[prop_key])


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
        layout.prop(operator, "skip_disabled_render")
        layout.prop(operator, "do_translate_origin")
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
        layout.prop(operator, "texture_size_x_override")
        layout.prop(operator, "texture_size_y_override")


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

        layout.prop(operator, "export_armature")
        layout.prop(operator, "export_animations")
        layout.prop(operator, "use_main_object_as_bone")


class VINTAGESTORY_PT_export_scripts(bpy.types.Panel):
    """Export panel for post export script."""
    bl_space_type = "FILE_BROWSER"
    bl_region_type = "TOOL_PROPS"
    bl_label = "Export Python Script"
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

        layout.prop(operator, "run_post_export_script")
        layout.prop(operator, "post_export_script")

# add io to menu
def menu_func_import(self, context):
    self.layout.operator(ImportVintageStoryJSON.bl_idname, text="VintageStory (.json)")

def menu_func_export(self, context):
    self.layout.operator(ExportVintageStoryJSON.bl_idname, text="VintageStory (.json)")

# register
classes = [
    # OPERATORS:
    # main import/export
    ImportVintageStoryJSON,
    ExportVintageStoryJSON,
    # PANELS:
    VINTAGESTORY_PT_export_geometry,
    VINTAGESTORY_PT_export_textures,
    VINTAGESTORY_PT_export_minify,
    VINTAGESTORY_PT_export_animation,
    VINTAGESTORY_PT_export_scripts,
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