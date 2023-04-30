import bpy
import math

class OpUVCuboidUnwrap(bpy.types.Operator):
    """Specialized VS cuboid UV unwrap"""
    bl_idname = "vintagestory.uv_cuboid_unwrap"
    bl_label = "Cuboid UV Unwrap (VS)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        args = self.as_keywords()
        print(args)
        return {"FINISHED"}

    def draw(self, context):
        pass


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
    
    def draw(self, context):
        pass