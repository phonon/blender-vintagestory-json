# ExportHelper is a helper class, defines filename and
# invoke() function which calls the file selector.
import bpy
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import BoolProperty, FloatProperty, IntProperty, StringProperty
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


def run_export(
    op,
    **args,
):
    """Common internal function to run export and handle post export script.
    Re-usable for different export operators.
    """
    if args.get("do_translate_origin", False):
        args["translate_origin"] = [
            args["translate_origin_x"],
            args["translate_origin_y"],
            args["translate_origin_z"],
        ]
    else:
        args["translate_origin"] = None
    
    # remap texture size overrides value 0 => None
    if "texture_size_x_override" in args:
        if args["texture_size_x_override"] == 0:
            args["texture_size_x_override"] = None
    if "texture_size_y_override" in args:
        if args["texture_size_y_override"] == 0:
            args["texture_size_y_override"] = None
    
    result, msg_type, msg = export_vintagestory_json.save_objects(**args)

    if "FINISHED" in result:
        op.report(msg_type, msg)

        if args.get("run_post_export_script", False):
            import os
            import subprocess
            post_export_script = args.get("post_export_script", "")
            save_filepath = args["filepath"]
            save_dir = os.path.dirname(save_filepath)
            script = os.path.join(save_dir, post_export_script)
            if post_export_script != "" and os.path.exists(script) and os.path.isfile(script):
                print(f"Running post export script: {script}")
                subprocess.run(["python", script, save_filepath], shell=False)
            else:
                op.report({"WARNING"}, f"Post export script not found: {post_export_script}")
    
    return result


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
        default=-1,
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

    generate_animations_file: BoolProperty(
        name="Generate Animations File",
        description="Generate separate animations .json file",
        default=False,
    )

    use_main_object_as_bone: BoolProperty(
        name="Use Main Object as Bone",
        description="Use main object with same transform as bone instead of dummy bone",
        default=True,
    )

    rotate_shortest_distance: BoolProperty(
        name="Rotate Shortest Distance",
        description="Use shortest distance interpolation for rotation keyframes",
        default=False,
    )
    
    animation_version_0: BoolProperty(
        name="Use Animation Version 0",
        description="Use old vintagestory animation format with incompatible transform order",
        default=False,
    )

    # ================================
    # step parent options
    use_step_parent: BoolProperty(
        name="Use Step Parent",
        description="Transform element relative to step parent (for attachments like clothes)",
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

        # store executed export config properties to scene
        for prop, val in args.items():
            if prop not in self.skip_save_props:
                bpy.context.scene["vintagestory_export_" + prop] = val
        
        if self.selection_only:
            args["objects"] = export_vintagestory_json.filter_root_objects(bpy.context.selected_objects)
        else:
            args["objects"] = export_vintagestory_json.filter_root_objects(bpy.context.scene.collection.all_objects[:])
        return run_export(self, **args)

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


class ExportVintageStoryJsonCollection(bpy.types.Operator):
    """Export selected object's associated collection. This finds the first
    selected object's collection in scene root collections and exports all
    objects inside that root collection. This is to make it easy to export
    "skin parts", e.g. vs characters are setup as base model and "skin parts"
    like hair. This enables easily clicking an obj of a skin part (e.g. one
    obj piece of the hair) and then export the whole hair collection, without
    needing to select all hair objects and clicking through export.
    Exported collection is saved as `{collection_name}.json` in the same
    folder as blender file.
    """
    bl_idname = "vintagestory.export_json_collection"
    bl_label = "Export collection as VintageStory .json"
    bl_options = {"REGISTER"}

    def execute(self, context):
        if len(bpy.context.selected_objects) == 0:
            # print warning
            self.report({"WARNING"}, "No objects selected")
            return {"CANCELLED"}
        
        obj = bpy.context.selected_objects[0] # take first selected object
        collection = obj.users_collection[0]
        collection_name = collection.name

        if collection == bpy.context.scene.collection:
            self.report({"WARNING"}, "Cannot export root collection")
            return {"CANCELLED"}

        outer_collections = { collection.name: collection for collection in bpy.context.scene.collection.children }

        # first check if obj is a child of a collection in root collections
        # which is the most common case
        collection_to_export = outer_collections.get(collection_name, None)

        if collection_to_export is None:
            # collection is not directly in an outer collection, must search
            # recursively for which outer collection contains the obj's
            # direct collection
            for outer_collection in outer_collections.values():
                if collection in outer_collection.children_recursive:
                    collection_to_export = outer_collection
                    break
        
        if collection_to_export is None:
            self.report({"ERROR"}, "Could not find collection to export, is it in the scene?")
            return {"CANCELLED"}

        # get blender filepath
        # https://github.com/dfelinto/blender/blob/master/release/scripts/modules/bpy_extras/io_utils.py#L56
        import os
        blend_filepath = context.blend_data.filepath
        save_dir = os.path.dirname(blend_filepath) if blend_filepath is not None else bpy.path.abspath("//")
        save_filepath = os.path.join(save_dir, collection_to_export.name + ".json")

        args = {
            "filepath": save_filepath,
            "objects": export_vintagestory_json.filter_root_objects(collection_to_export.all_objects),
        }
        
        # gather export args stored in scene
        for prop, val in bpy.context.scene.items():
            if "vintagestory_export_" in prop:
                args[prop[20:]] = val

        return run_export(self, **args)


class ExportVintageStoryJsonHighlightedCollections(bpy.types.Operator):
    """Export highlighted collections in Scene Collection as VintageStory
    .json files. Useful to quickly re-export multiple collections when some
    common setting has changed (e.g. material name, texture, export
    property, etc.).
    """
    bl_idname = "vintagestory.export_json_highlighted_collections"
    bl_label = "Export highlighted collections in Scene Collection as VintageStory .json"
    bl_options = {"REGISTER"}

    def execute(self, context):
        def get_highlighted_collections():
            collections = []

            # works on blender 4.0 to get highlighted collections
            screen = bpy.context.screen
            areas = [area for area in screen.areas if area.type == "OUTLINER"]
            regions = [region for region in areas[0].regions if region.type == "WINDOW"]

            with bpy.context.temp_override(area=areas[0], region=regions[0], screen=screen):
                for item in bpy.context.selected_ids:
                    if item.bl_rna.identifier == "Collection":
                        collections.append(item)
            
            return collections
        
        highlighted_collections = get_highlighted_collections()
        print(highlighted_collections)
        if len(highlighted_collections) == 0:
            self.report({"WARNING"}, "No collections highlighted in Scene Collection outliner.")
            return {"CANCELLED"}
        
        # get blender filepath
        # https://github.com/dfelinto/blender/blob/master/release/scripts/modules/bpy_extras/io_utils.py#L56
        import os
        blend_filepath = context.blend_data.filepath
        save_dir = os.path.dirname(blend_filepath) if blend_filepath is not None else bpy.path.abspath("//")
        
        # gather export shared args stored in scene
        args = {}
        for prop, val in bpy.context.scene.items():
            if "vintagestory_export_" in prop:
                args[prop[20:]] = val
        
        # export each highlighted collection
        for collection_to_export in highlighted_collections:
            # collection specific args, re-write for each collection
            args["filepath"] = os.path.join(save_dir, collection_to_export.name + ".json")
            args["objects"] = export_vintagestory_json.filter_root_objects(collection_to_export.all_objects)

            run_export(self, **args)

        return {"FINISHED"}


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
        layout.prop(operator, "use_step_parent")
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
        layout.prop(operator, "generate_animations_file")
        layout.prop(operator, "use_main_object_as_bone")
        layout.prop(operator, "rotate_shortest_distance")
        layout.prop(operator, "animation_version_0")


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
    ExportVintageStoryJsonCollection,
    ExportVintageStoryJsonHighlightedCollections,
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