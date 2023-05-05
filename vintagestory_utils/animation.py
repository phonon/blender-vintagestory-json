import math
import numpy as np
import logging
import bpy


class OpAssignBones(bpy.types.Operator):
    """TODO: automatically assign cuboids to closest bone"""
    bl_idname = "vintagestory.assign_bones"
    bl_label = "Assign Bones (VS)"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        self.report({"ERROR"}, "Not implemented")
        return {"FINISHED"}
