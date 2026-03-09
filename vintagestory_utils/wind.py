import numpy as np
import bpy
import bmesh

WIND_MODE_INT = {
    "None": 0,
    "ExtraWeakWind": 7,
    "WeakWind": 1,
    "NormalWind": 2,
    "WeakWindInverseBend": 10,
    "WeakWindNoBend": 9,
    "Bend": 4,
    "TallBend": 5,
    "Fruit": 8,
    "Leaves": 3,
    "Water": 6,
    "WaterPlant": 11,
}


class OpAssignFaceWindMode(bpy.types.Operator):
    """Assign face wind data."""
    bl_idname = "vintagestory.assign_wind"
    bl_label = "Wind Mode"
    bl_options = {"REGISTER", "UNDO"}

    # https://apidocs.vintagestory.at/api/Vintagestory.API.Common.EnumWindBitMode.html

    wind_mode: bpy.props.EnumProperty(
        name="Wind Mode",
        description="Wind Mode",
        items=[ # (identifier, name, description, enum int)
            ("None", "None", ""),
            ("ExtraWeakWind", "Extra Weak Wind", ""),
            ("WeakWind", "Weak Wind", ""),
            ("NormalWind", "Normal Wind", ""),
            ("WeakWindInverseBend", "Weak Wind Inverse Bend", ""),
            ("WeakWindNoBend", "Weak Wind No Bend", ""),
            ("Bend", "Bend", ""),
            ("TallBend", "Tall Bend", ""),
            ("Fruit", "Fruit", ""),
            ("Leaves", "Leaves", ""),
            ("Water", "Water", ""),
            ("WaterPlant", "Water Plant", ""),
        ],
        default="WeakWind",
    )

    wind_strength: bpy.props.IntProperty(
        name="Wind Strength",
        description="Value of wind strength in 0 to 255",
        default=1,
        min=0,
        max=255,
        soft_min=0,
        soft_max=255,
    )

    def invoke(self, context, event):
        if bpy.context.mode != "EDIT_MESH":
            self.report({"ERROR"}, "Must be in Edit Mode to assign wind data")
            return {"CANCELLED"}
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        args = self.as_keywords()

        # unpack args
        wind_mode = args.get("wind_mode")
        wind_strength = args.get("wind_strength")

        # get wind mode int value from items list
        wind_mode_int = WIND_MODE_INT[wind_mode]
        
        # get objects
        if len(bpy.context.selected_objects) == 0:
            self.report({"ERROR"}, "No objects selected")
            return {"FINISHED"}

        r_val = wind_mode_int / 255.0
        g_val = wind_strength / 255.0
        select_mode = context.tool_settings.mesh_select_mode  # (vert, edge, face)
        num_verts = 0

        for obj in bpy.context.selected_objects:
            if obj.type != "MESH":
                continue

            mesh = obj.data
            bm = bmesh.from_edit_mesh(mesh)

            # get or create wind float color layer (linear, no sRGB conversion)
            color_layer = bm.loops.layers.float_color.get("wind")
            if color_layer is None:
                color_layer = bm.loops.layers.float_color.new("wind")
                # Initialize all wind colors to 0,0,0
                for face in bm.faces:
                    for loop in face.loops:
                        loop[color_layer] = (0.0, 0.0, 0.0, 1.0)

            if select_mode[2]:  # face mode
                for face in bm.faces:
                    if face.select:
                        for loop in face.loops:
                            c = loop[color_layer]
                            c[0] = r_val
                            c[1] = g_val
                            loop[color_layer] = c
                            num_verts += 1
            elif select_mode[1]:  # edge mode
                target_verts = set()
                for edge in bm.edges:
                    if edge.select:
                        for vert in edge.verts:
                            target_verts.add(vert.index)
                for face in bm.faces:
                    for loop in face.loops:
                        if loop.vert.index in target_verts:
                            c = loop[color_layer]
                            c[0] = r_val
                            c[1] = g_val
                            loop[color_layer] = c
                num_verts += len(target_verts)
            else:  # vertex mode
                target_verts = set()
                for vert in bm.verts:
                    if vert.select:
                        target_verts.add(vert.index)
                for face in bm.faces:
                    for loop in face.loops:
                        if loop.vert.index in target_verts:
                            c = loop[color_layer]
                            c[0] = r_val
                            c[1] = g_val
                            loop[color_layer] = c
                num_verts += len(target_verts)

            bmesh.update_edit_mesh(mesh)

        self.report({"INFO"}, f"Set wind mode {wind_mode} strength {wind_strength} on {num_verts} verts")

        return {"FINISHED"}


class OpClearFaceWindMode(bpy.types.Operator):
    """Clear face wind data. Object mode: remove wind layer. Edit mode: clear selected."""
    bl_idname = "vintagestory.clear_face_wind"
    bl_label = "Clear Face Wind Mode"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if len(bpy.context.selected_objects) == 0:
            self.report({"ERROR"}, "No objects selected")
            return {"FINISHED"}

        mode = bpy.context.mode

        if mode == "OBJECT":
            count = 0
            for obj in bpy.context.selected_objects:
                if obj.type != "MESH":
                    continue
                mesh = obj.data
                wind_attr = mesh.color_attributes.get("wind")
                if wind_attr is not None:
                    mesh.color_attributes.remove(wind_attr)
                    count += 1
            self.report({"INFO"}, f"Removed wind color layer from {count} objects")

        elif mode == "EDIT_MESH":
            select_mode = context.tool_settings.mesh_select_mode
            num_cleared = 0

            for obj in bpy.context.selected_objects:
                if obj.type != "MESH":
                    continue

                mesh = obj.data
                bm = bmesh.from_edit_mesh(mesh)

                color_layer = bm.loops.layers.float_color.get("wind")
                if color_layer is None:
                    continue

                zero = [0.0, 0.0, 0.0, 1.0]

                if select_mode[2]:  # face mode
                    for face in bm.faces:
                        if face.select:
                            for loop in face.loops:
                                loop[color_layer] = zero
                                num_cleared += 1
                elif select_mode[1]:  # edge mode
                    target_verts = set()
                    for edge in bm.edges:
                        if edge.select:
                            for vert in edge.verts:
                                target_verts.add(vert.index)
                    for face in bm.faces:
                        for loop in face.loops:
                            if loop.vert.index in target_verts:
                                loop[color_layer] = zero
                    num_cleared += len(target_verts)
                else:  # vertex mode
                    target_verts = set()
                    for vert in bm.verts:
                        if vert.select:
                            target_verts.add(vert.index)
                    for face in bm.faces:
                        for loop in face.loops:
                            if loop.vert.index in target_verts:
                                loop[color_layer] = zero
                    num_cleared += len(target_verts)

                bmesh.update_edit_mesh(mesh)

            self.report({"INFO"}, f"Cleared wind data on {num_cleared} verts")
            
        else:
            self.report({"ERROR"}, "Must be in Object or Edit mode")
            return {"CANCELLED"}

        return {"FINISHED"}