import bpy
import numpy as np
from mathutils import Matrix

class OpDuplicateCollection(bpy.types.Operator):
    """Duplicate collection of currently selected object."""
    bl_idname = "vintagestory.duplicate_collection"
    bl_label = "Duplicate Skin Part"
    bl_options = {"REGISTER", "UNDO"}

    name: bpy.props.StringProperty(
        name="name",
        description="New collection name",
    )

    def invoke(self, context, event):
        wm = context.window_manager
        return wm.invoke_props_dialog(self)

    def execute(self, context):
        args = self.as_keywords()

        # unpack args
        new_name = args.get("name")

        if new_name is None or new_name == "":
            self.report({"ERROR"}, "No name provided")
            return {"FINISHED"}
        
        # if name already exists, error and exit
        if new_name in bpy.context.scene.collection.children.keys():
            self.report({"ERROR"}, f"Collection {new_name} already exists")
            return {"FINISHED"}
        
        if len(bpy.context.selected_objects) == 0:
            self.report({"ERROR"}, "No objects selected")
            return {"FINISHED"}

        # create new collection
        obj = bpy.context.selected_objects[0] # take first selected object
        collection = obj.users_collection[0]
        collection_name = collection.name

        if collection == bpy.context.scene.collection:
            self.report({"WARNING"}, "Cannot export root collection")
            return {"CANCELLED"}

        outer_collections = { collection.name: collection for collection in bpy.context.scene.collection.children }

        # first check if obj is a child of a collection in root collections
        # which is the most common case
        collection_to_duplicate = outer_collections.get(collection_name, None)

        if collection_to_duplicate is None:
            # collection is not directly in an outer collection, must search
            # recursively for which outer collection contains the obj's
            # direct collection
            for outer_collection in outer_collections.values():
                if collection in outer_collection.children_recursive:
                    collection_to_duplicate = outer_collection
                    break
        
        if collection_to_duplicate is None:
            self.report({"ERROR"}, "Could not find collection to export, is it in the scene?")
            return {"CANCELLED"}
        
        # create new collection
        new_part_collection = bpy.data.collections.new(new_name)

        # re-cursively duplicate new objects into new collection
        def duplicate_collection(old_collection, new_collection):
            for obj in old_collection.objects:
                new_obj = obj.copy()
                new_obj.data = obj.data.copy()
                new_obj.name = obj.name
                new_collection.objects.link(new_obj)
            for child_collection in old_collection.children:
                new_child_collection = bpy.data.collections.new(child_collection.name)
                new_collection.children.link(new_child_collection)
                duplicate_collection(child_collection, new_child_collection)
        
        duplicate_collection(collection_to_duplicate, new_part_collection)
        
        # link new collection to scene
        bpy.context.scene.collection.children.link(new_part_collection)

        # de-select original object, select new object
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in new_part_collection.all_objects:
            obj.select_set(True)
        
        return {"FINISHED"}


class OpCleanupRotation(bpy.types.Operator):
    """Cleanup cuboid edit mode rotation, convert to object mode rotation."""
    bl_idname = "vintagestory.cleanup_rotation"
    bl_label = "Cleanup Rotation"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        num_obj_realigned = 0

        for obj in bpy.context.selected_objects:
            if not isinstance(obj.data, bpy.types.Mesh):
                continue
            if len(obj.data.polygons) != 6:
                self.report({"WARN"}, f"{obj.name} is not a cuboid for re-alignment (skipping)")
                continue
            
            # determine if this needs realignment:
            # check if face normals are not aligned with world axis
            obj_is_aligned = True
            for f in obj.data.polygons:
                # sum together abs value of face normal components
                sum_normal_components = sum(abs(c) for c in f.normal)
                # if sum not close to 1, then face normal is not aligned with world axis
                if not np.isclose(sum_normal_components, 1.0, atol=1e-6):
                    obj_is_aligned = False
                    break
            
            if obj_is_aligned:
                continue

            # assumption: object is still a cuboid and face normals
            # are negated with each other
            
            # determine 3 orthogonal vectors from face normals
            # to determine rotation matrix.
            # first find pairs of opposite face normals
            face_normals = [f.normal for f in obj.data.polygons]
            opposite_pairs = set() # set of tuples of opposite face normals
            for i, normal1 in enumerate(face_normals):
                for j, normal2 in enumerate(face_normals):
                    if i == j:
                        continue
                    if np.allclose(normal1, -normal2, atol=1e-3):
                        idx_pair = (i, j) if i < j else (j, i)
                        opposite_pairs.add(idx_pair)
                        break
            
            # if not 3 pairs of opposite face normals, warn and skip
            if len(opposite_pairs) != 3:
                self.report({"ERROR"}, f"{obj.name} does not have 3 pairs of opposite face normals (try re-calculating normals)")
                continue

            # use the first normal of each pair to form the rotation matrix
            rotation_matrix = np.array([face_normals[i] for i, j in opposite_pairs])
            
            # apply rotation to each raw vertex
            v_local = np.array([v.co for v in obj.data.vertices]).T
            v_new = rotation_matrix @ v_local
            for i, v in enumerate(obj.data.vertices):
                v.co = v_new[:,i]
            
            # apply inverse rotation to object transform
            obj.matrix_world = obj.matrix_world @ Matrix(rotation_matrix.T).to_4x4()

            num_obj_realigned += 1

        if num_obj_realigned > 0:
            self.report({"INFO"}, f"Re-aligned {num_obj_realigned} objects")
        else:
            self.report({"INFO"}, "No objects needed realignment")

        return {"FINISHED"}
