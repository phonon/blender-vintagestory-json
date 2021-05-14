from mathutils import Vector, Euler, Quaternion, Matrix

ROTATION_MODE_EULER = 0
ROTATION_MODE_QUATERNION = 1

class FcurveRotationMatrixCache():
    """Cache of rotation matrices at given frames sampled from fcurves
    """
    def __init__(self, rotation_mode, fcu_x, fcu_y, fcu_z, fcu_w):
        self.rotation_mode = ROTATION_MODE_EULER if rotation_mode == "rotation_euler" else ROTATION_MODE_QUATERION
        self.fcu_x = fcu_x
        self.fcu_y = fcu_y
        self.fcu_z = fcu_z
        self.fcu_w = fcu_w
        self.cache = {}       # cache of frame => rotation matrix
    

    def get(self, frame):
        if frame in self.cache:
            return self.cache[frame]
        else:
            if self.rotation_mode == ROTATION_MODE_EULER:
                rot_mat = self.get_rot_from_euler(frame)
            else:
                rot_mat = self.get_rot_from_quaternion(frame)
            self.cache[frame] = rot_mat
            return rot_mat
    

    def get_rot_from_euler(self, frame):
        rx = self.fcu_x.evaluate(frame)
        ry = self.fcu_y.evaluate(frame)
        rz = self.fcu_z.evaluate(frame)
        rot_mat = Euler((rx, ry, rz), "XYZ").to_matrix()
        return rot_mat
    

    def get_rot_from_quaternion(self, frame):
        qx = self.fcu_x.evaluate(frame)
        qy = self.fcu_y.evaluate(frame)
        qz = self.fcu_z.evaluate(frame)
        qw = self.fcu_w.evaluate(frame)
        rot_mat = Quaternion(qw, qx, qy, qz).to_matrix()
        return rot_mat


class AnimationAdapter():
    """Helper to create, cache, store/load fcurves and convert
    between Blender and Vintage Story animation system.

    In Vintage Story animation translation applied first:
        v' = R*T*v
    While in Blender:
        v' = T*R*v

    This will resample between using RT <=> TR for position
    keyframe points. 
    """
    def __init__(self, action, name=None):
        self.name = name     # name, for debugging only
        self.action = action # Blender animation action
        self.storage = {}    # store fcurves by name
        self.bones = {}      # map of bones to rotation mode
    

    def add_bone(self, bone_name, rotation_mode):
        """Map bone_name to rotation_mode.
        """
        self.bones[bone_name] = rotation_mode

    
    def get(self, name, index):
        """Get fcurve from storage, or create new curve from an action
        """
        if name in self.storage:
            if self.storage[name][index] != None:
                return self.storage[name][index]
            else:
                fcu = self.action.fcurves.new(data_path=name, index=index)
                self.storage[name][index] = fcu
                return fcu
        else:
            fcu = self.action.fcurves.new(data_path=name, index=index)
            # each index can be an fcurve, support x,y,z,w coords
            self.storage[name] = [None, None, None, None]
            self.storage[name][index] = fcu
            return fcu


    def resample_to_blender(self):
        """Resample animation keyframe position data, which gives
        world space coordinates w from input position v by:
            VintageStory: w = R(v + u)
            Blender:      w = Rv + u'
        Conversion is u' = R*u, where u is the VintageStory keyframe
        position and u' is effective Blender keyframe position. 
        Go through each coord, transform coordinates by rotation matrix.
        """
        for bone, rotation_mode in self.bones.items():
            fcu_name_prefix = "pose.bones[\"{}\"]".format(bone)
            fcu_name_location = fcu_name_prefix + ".location"
            if rotation_mode == "rotation_euler":
                fcu_name_rotation = fcu_name_prefix + ".rotation_euler"
            else:
                fcu_name_rotation = fcu_name_prefix + ".rotation_quaternion"
            
            if fcu_name_location in self.storage:
                fcu_x = self.storage[fcu_name_location][0]
                fcu_y = self.storage[fcu_name_location][1]
                fcu_z = self.storage[fcu_name_location][2]

                # cache sampled rotation matrix at a frame
                fcu_rotation_cache = FcurveRotationMatrixCache(
                    rotation_mode,
                    self.storage[fcu_name_rotation][0],
                    self.storage[fcu_name_rotation][1],
                    self.storage[fcu_name_rotation][2],
                    self.storage[fcu_name_rotation][3],
                )

                # for now, assume keyframes are all at same frames
                for k in range(0, len(fcu_x.keyframe_points)):
                    frame, vx = fcu_x.keyframe_points[k].co
                    _, vy = fcu_y.keyframe_points[k].co
                    _, vz = fcu_z.keyframe_points[k].co
                    
                    # common for all points to be 0
                    if vx == 0.0 and vy == 0.0 and vz == 0.0:
                        continue
                    
                    rot_mat = fcu_rotation_cache.get(frame)
                    v = rot_mat @ Vector((vx, vy, vz))
                    
                    # debug
                    # print(frame, "OLD => NEW:", vx, vy, vz, "=>", v.x, v.y, v.z)

                    fcu_x.keyframe_points[k].co = frame, v.x
                    fcu_y.keyframe_points[k].co = frame, v.y
                    fcu_z.keyframe_points[k].co = frame, v.z


    def resample_to_vintage_story(self):
        """Resample from position keyframe data from using
            VintageStory: w = R(v + u)
            Blender:      w = Rv + u'
        """
        # TODO, for exporting
        pass