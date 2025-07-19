import numpy as np
import bpy
from mathutils import Matrix

class OpMakeBonesXZY(bpy.types.Operator):
    """Make bone rotation mode XZY (corresponds to VintageStory rotation order)"""
    bl_idname = "vintagestory.make_bones_xzy"
    bl_label = "Make Bones XZY"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):

        # need to be in object mode to access context selected objects
        user_mode = context.active_object.mode
        if user_mode != "OBJECT":
            need_to_switch_mode_back = True
            bpy.ops.object.mode_set(mode="OBJECT")
        else:
            need_to_switch_mode_back = False

        num_updated = 0

        for obj in bpy.context.selected_objects:
            if not isinstance(obj.data, bpy.types.Armature):
                continue
            
            armature = obj
            bones = armature.pose.bones
            for b in bones:
                b.rotation_mode = "XZY"
            
            num_updated += 1
                
        if need_to_switch_mode_back:
            bpy.ops.object.mode_set(mode=user_mode)
        
        # if no armature, note error for user
        if num_updated > 0:
            self.report({"INFO"}, f"Updated {num_updated} armatures bones to XZY")
        else:
            self.report({"ERROR"}, "No armature in selection")

        return {"FINISHED"}


class OpAssignBones(bpy.types.Operator):
    """Automatically assign meshes to closest bone as an object bone type"""
    bl_idname = "vintagestory.assign_bones"
    bl_label = "Assign Bones"
    bl_options = {"REGISTER", "UNDO"}

    by_centroid: bpy.props.BoolProperty(
        default=True,
        name="By Centroid",
        description="Detect closest bone from cuboid centroid (otherwise use object position)",
    )

    def execute(self, context):
        args = self.as_keywords()

        # unpack args
        closest_from_centroid = args.get("by_centroid", True)

        # need to be in object mode to access context selected objects
        user_mode = context.active_object.mode
        if user_mode != "OBJECT":
            need_to_switch_mode_back = True
            bpy.ops.object.mode_set(mode="OBJECT")
        else:
            need_to_switch_mode_back = False
        
        # only perform on selected objects:
        # 1. find first armature among selected, that will be skeleton
        # 2. find all meshes among selected, those will be objects
        armature = None
        selected = bpy.context.selected_objects
        mesh_objects = []
        for obj in selected:
            if isinstance(obj.data, bpy.types.Armature):
                if armature is None:
                    armature = obj
                else:
                    self.report({"WARNING"}, "Skipping multiple armatures in selection")
            elif isinstance(obj.data, bpy.types.Mesh):
                mesh_objects.append(obj)
        
        # if no armature, error
        if armature is None:
            self.report({"ERROR"}, "No armature in selection")
            return {"FINISHED"}
        
        # print(f"Armature: {armature}")
        # print(f"Bones: {armature.data.bones}")

        armature_mat_world = armature.matrix_world
        bones = armature.data.bones
        num_bones = len(bones)
        bone_positions = np.zeros((num_bones, 3))
        for i, b in enumerate(bones):
            bone_positions[i,:] = armature_mat_world @ b.head_local

        for obj in mesh_objects:
            mesh = obj.data

            # find closest bone and set as parent
            if closest_from_centroid:
                # get mesh centroid = mean of vertices in world space
                num_vertices = len(mesh.vertices)
                vertices_local = np.ones((4, num_vertices)) # (xyzw, verts)
                for i, v in enumerate(mesh.vertices):
                    vertices_local[0:3,i] = v.co
                matrix_world = np.asarray(obj.matrix_world)
                vertices = matrix_world @ vertices_local
                centroid = np.mean(vertices, axis=(1,))[np.newaxis,:3]
                dxdydz = bone_positions - centroid
            else:
                matrix_world = np.asarray(obj.matrix_world)
                location = matrix_world[:3,3]
                dxdydz = bone_positions - location[np.newaxis,:]
            
            # find closest bone to object magnitude squared distance
            dist = np.sum(dxdydz * dxdydz, axis=(1,))
            closest = np.argmin(dist)
            bone_closest = bones[closest]

            # for setting matrix parent inverse, see discussion here on 
            # blender api:
            # https://blender.stackexchange.com/a/112856

            # bone tip translation matrix
            translate_to_tip_mat = Matrix.Translation(bone_closest.tail - bone_closest.head)
            bone_mat_world = armature.matrix_world @ translate_to_tip_mat @ bone_closest.matrix.to_4x4()
            obj_mat_world = obj.matrix_world.copy()

            # keep obj matrix world and matrix basis constant, set parent inverse
            obj_mat_local = bone_mat_world.inverted() @ obj_mat_world
            obj.matrix_parent_inverse = obj_mat_local @ obj.matrix_basis.inverted()
            
            # set parent to closest bone
            obj.parent = armature
            obj.parent_bone = bone_closest.name
            obj.parent_type = "BONE"

            obj.matrix_world = obj_mat_world
        
        if need_to_switch_mode_back:
            bpy.ops.object.mode_set(mode=user_mode)

        return {"FINISHED"}


class OpAssignStepParentName(bpy.types.Operator):
    """Assign StepParentName custom property to object (leave empty to remove)"""
    bl_idname = "vintagestory.assign_step_parent_name"
    bl_label = "Step Parent Name"
    bl_options = {"REGISTER", "UNDO"}

    step_parent_name: bpy.props.StringProperty(
        name="Step Parent Name",
        description="Step parent name to add",
    )

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        args = self.as_keywords()

        # unpack args
        name = args.get("step_parent_name")
        
        if len(bpy.context.selected_objects) == 0:
            self.report({"ERROR"}, "No objects selected")
            return {"FINISHED"}

        if name is None or name == "":
            # remove "StepParentName" custom StringProperty from selected objects
            num_removed = 0
            for obj in bpy.context.selected_objects:
                if "StepParentName" in obj:
                    del obj["StepParentName"]
                    num_removed += 1
            self.report({"INFO"}, f"Removed StepParentName property from {num_removed} objects")
        else:
            # add "StepParentName" custom StringProperty to selected objects
            for obj in bpy.context.selected_objects:
                obj["StepParentName"] = name
        
        # refresh n panel
        for region in context.area.regions:
            if region.type == "UI":
                region.tag_redraw()

        return {"FINISHED"}


class OpAssignStepParentConstraint(bpy.types.Operator):
    """From object's StepParentName property, assign a 'Child Of'
    constraint to the object to set it as virtual child of the step
    parent object.
    """
    bl_idname = "vintagestory.assign_step_parent_constraint"
    bl_label = "Assign Step Parent Constraint"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if len(bpy.context.selected_objects) == 0:
            self.report({"ERROR"}, "No objects selected")
            return {"FINISHED"}
        
        for obj in bpy.context.selected_objects:
            # get step parent name
            step_parent_name = obj.get("StepParentName")
            if step_parent_name is None:
                self.report({"ERROR"}, "No StepParentName property found")
                continue

            if step_parent_name.startswith("b_"):
                found_bone = False
                # find armature with bone name
                for armature in bpy.data.objects:
                    if not isinstance(armature.data, bpy.types.Armature):
                        continue
                    # search for bone with name
                    bone = armature.data.bones.get(step_parent_name[2:])
                    if bone is None:
                        continue
                    found_bone = True
                    # add constraint to make the object a virtual child of the object
                    if "Child Of" in obj.constraints:
                        constraint = obj.constraints["Child Of"]
                    else:
                        constraint = obj.constraints.new("CHILD_OF")
                    # set target of constraint
                    constraint.target = armature
                    constraint.subtarget = bone.name
                    # set inverse matrix to bone
                    # https://blenderartists.org/t/set-inverse-child-of-constraints-via-python/1133914/4
                    constraint.inverse_matrix = armature.matrix_world @ bone.matrix_local.inverted()
                    break
                if not found_bone:
                    self.report({"ERROR"}, f"Step parent bone {step_parent_name[2:]} not found")
            # else, try find scene object similar with step parent name
            elif obj["StepParentName"] in bpy.data.objects:
                # add constraint to make the object a virtual child of the object
                if "Child Of" in obj.constraints:
                    constraint = obj.constraints["Child Of"]
                else:
                    constraint = obj.constraints.new("CHILD_OF")
                # set target of constraint
                target = bpy.data.objects[obj["StepParentName"]]
                # set target and inverse matrix to target
                constraint.target = target
                constraint.inverse_matrix = target.matrix_world.inverted()

        return {"FINISHED"}


class OpRemoveStepParentConstraint(bpy.types.Operator):
    """Wrapper around removing 'Child Of' from selected objects"""
    bl_idname = "vintagestory.remove_step_parent_constraint"
    bl_label = "Remove Step Parent Constraint"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        if len(bpy.context.selected_objects) == 0:
            self.report({"ERROR"}, "No objects selected")
            return {"FINISHED"}
        
        for obj in bpy.context.selected_objects:
            # remove "Child Of" constraint
            if "Child Of" in obj.constraints:
                obj.constraints.remove(obj.constraints["Child Of"])

        return {"FINISHED"}


class OpAssignRename(bpy.types.Operator):
    """Assign rename on export custom property to object (empty to remove)"""
    bl_idname = "vintagestory.assign_rename"
    bl_label = "Rename on Export"
    bl_options = {"REGISTER", "UNDO"}

    rename: bpy.props.StringProperty(
        name="rename",
        description="Rename on export new name",
    )

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        args = self.as_keywords()

        # unpack args
        name = args.get("rename")
        
        if len(bpy.context.selected_objects) == 0:
            self.report({"ERROR"}, "No objects selected")
            return {"FINISHED"}

        if name is None or name == "":
            # remove "StepParentName" custom StringProperty from selected objects
            num_removed = 0
            for obj in bpy.context.selected_objects:
                if "rename" in obj:
                    del obj["rename"]
                    num_removed += 1
            self.report({"INFO"}, f"Removed rename property from {num_removed} objects")
        else:
            # add "StepParentName" custom StringProperty to selected objects
            for obj in bpy.context.selected_objects:
                obj["rename"] = name
        
        # refresh n panel
        for region in context.area.regions:
            if region.type == "UI":
                region.tag_redraw()

        return {"FINISHED"}


class OpActionOnAnimationEnd(bpy.types.Operator):
    """Assign animation action's `on_animation_end` property."""
    bl_idname = "vintagestory.action_on_animation_end"
    bl_label = "On Animation End"
    bl_options = {"REGISTER", "UNDO"}

    on_animation_end: bpy.props.EnumProperty(
        name="Animation End",
        description="Behavior on animation end",
        items=[
            ("EaseOut", "EaseOut", "Ease out animation"),
            ("Repeat", "Repeat", "Repeat animation"),
            ("Stop", "Stop", "Stop animation"),
            ("Hold", "Hold", "Hold animation at last frame"),
        ],
    )

    def execute(self, context):
        args = self.as_keywords()

        # unpack args
        prop = args.get("on_animation_end")

        action = context.active_action
        if not action:
            self.report({"ERROR"}, "No active action")
            return {"CANCELLED"}

        action["on_animation_end"] = prop

        return {"FINISHED"}


class OpActionOnActivityStopped(bpy.types.Operator):
    """Assign animation action's `on_activity_stopped` property."""
    bl_idname = "vintagestory.action_on_activity_stopped"
    bl_label = "On Activity Stopped"
    bl_options = {"REGISTER", "UNDO"}

    on_activity_stopped: bpy.props.EnumProperty(
        name="Activity Stopped",
        description="Behavior on activity stopped",
        items=[
            ("PlayTillEnd", "PlayTillEnd", "Play animation till end"),
            ("EaseOut", "EaseOut", "Ease out animation"),
            ("Stop", "Stop", "Immediately stop animation"),
            ("Rewind", "Rewind", "Rewind animation to start"),
        ],
    )

    def execute(self, context):
        args = self.as_keywords()

        # unpack args
        prop = args.get("on_activity_stopped")

        action = context.active_action
        if not action:
            self.report({"ERROR"}, "No active action")
            return {"CANCELLED"}

        action["on_activity_stopped"] = prop

        return {"FINISHED"}
