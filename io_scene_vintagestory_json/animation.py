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


class KeyframeAdapter():
    """Intermediate representation of keyframes during export
    to VintageStory. Represent keyframes as map of
        frame # => keyframe_data
    where keyframe_data = {
        "frame": frame #
        "elements": {
            "bone name" => { offsetX, offsetY, offsetZ, rotationX, rotationY, rotationZ }
        }
    }
    """
    def __init__(self):
        self.keyframes = {}


    def tolist(self):
        """Convert keyframes frame # => keyframe_data to flat sorted
        array of keyframes
        """
        keyframes_list = []
        frame_numbers = list(self.keyframes.keys())
        frame_numbers.sort()
        for frame in frame_numbers:
            keyframes_list.append(self.keyframes[frame])
        return keyframes_list
    

    def get_bone_keyframe(self, bone_name, frame):
        """keyframes: map of frame # => keyframe data
        frame: frame #
        Get keyframe export data dict at a frame, or create new
        frame data if does not exist
        """
        if frame not in self.keyframes:
            self.keyframes[frame] = {
                "frame": frame,
                "elements": {},
            }
        
        keyframe = self.keyframes[frame]
        if bone_name not in keyframe["elements"]:
            keyframe["elements"][bone_name] = {
                "offsetX": 0.0,
                "offsetY": 0.0,
                "offsetZ": 0.0,
                "rotationX": 0.0,
                "rotationY": 0.0,
                "rotationZ": 0.0,
            }

        return keyframe["elements"][bone_name]
    

    def add_bone_keyframe_points(self, bone_name, fcurve, field):
        """keyframes: map of frame # => keyframe data
        fcurve: fcurve data
        field: output keyframe field, e.g. "offsetX" or "rotationX"
        """
        for p in fcurve.keyframe_points:
            frame, val = p.co
            frame = int(frame)
            keyframe = self.get_bone_keyframe(bone_name, frame)
            keyframe[field] = val


class AnimationAdapter():
    """Helper to create, cache, store/load fcurves and convert
    between Blender and Vintage Story animation system.

    In Vintage Story animation translation applied first:
        v' = R*T*v
    While in Blender:
        v' = T*R*v

    This will resample between using RT <=> TR for position
    keyframe points.

    Storage format is a dict of :
        fcu.data_path => [fcu.x, fcu.y, fcu.z, fcu.w]
    Each bone name maps to a list of fcurves for each fcurve
    array index. If the slot is None, no fcurve exists for that
    index. e.g.
        pose.bones["Body"].location => [FCurve, FCurve, FCurve, None]
    Maps the Body bone data path to 3 FCurves for x, y, z index
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

    
    def add_fcurve(self, fcurve, name, index):
        """Add existing fcurve to storage for given fcurve name
        """
        if name not in self.storage:
            self.storage[name] = [None, None, None, None] # support x, y, z, w coord
        self.storage[name][index] = fcurve

    
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

    
    def create_vintage_story_keyframes(self):
        """Create list of keyframes for this action in VintageStory format.
        1. Make keyframes list, where each frame has list of elements
            keyframes = [
                {
                    frame: #,
                    bone_name => { bone orientation },
                }
                ...
            ]
        2. Convert positions to VintageStory keyframe data format from:
                VintageStory: w = R(v + u)
                Blender:      w = Rv + u'
            Where u' = R*u -> u = R^-1 * u'
        3. Convert quaternion keyframes into euler keyframes
        """
        # map frame # => keyframe data
        keyframes = KeyframeAdapter()

        for bone_name, rotation_mode in self.bones.items():
            fcu_name_prefix = "pose.bones[\"{}\"]".format(bone_name)
            fcu_name_location = fcu_name_prefix + ".location"
            if rotation_mode == "rotation_euler":
                fcu_name_rotation = fcu_name_prefix + ".rotation_euler"
            else:
                fcu_name_rotation = fcu_name_prefix + ".rotation_quaternion"

            if fcu_name_location in self.storage:
                fcu_x = self.storage[fcu_name_location][0]
                fcu_y = self.storage[fcu_name_location][1]
                fcu_z = self.storage[fcu_name_location][2]

                keyframes.add_bone_keyframe_points(bone_name, fcu_x, "offsetZ")
                keyframes.add_bone_keyframe_points(bone_name, fcu_y, "offsetX")
                keyframes.add_bone_keyframe_points(bone_name, fcu_z, "offsetY")

        # convert keyframes map into a list of keyframes
        keyframes_list = keyframes.tolist()

        return keyframes_list