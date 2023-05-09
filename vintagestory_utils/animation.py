import numpy as np
import bpy
from mathutils import Matrix

class OpAssignBones(bpy.types.Operator):
    """Automatically assign meshes to closest bone as an object bone type."""
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
