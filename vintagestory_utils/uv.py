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
        args = self.as_keywords()
        print(args)
        return {"FINISHED"}
    
    def draw(self, context):
        pass