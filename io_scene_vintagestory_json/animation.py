"""Notes on Vintage Story animation format:
See: https://github.com/anegostudios/vsmodelcreator/blob/master/src/at/vintagestory/modelcreator/model/KeyFrameElement.java
Animations are additive euler rotations:
    public void rotateAxis()
    {
        GL11.glRotated(AnimatedElement.rotationX + getRotationX(), 1, 0, 0);
        GL11.glRotated(AnimatedElement.rotationY + getRotationY(), 0, 1, 0);
        GL11.glRotated(AnimatedElement.rotationZ + getRotationZ(), 0, 0, 1);
    }

In blender, bone animations apply as separate rotation matrices
    R = R_animation * R_bone

Exact animation will not be identical.

Note euler rotation orders and axis angle remappings:
     VS Space    Blender
        X   <--->   Y
        Y   <--->   Z
        Z   <--->   X

Vintage story using ZYX euler order as shown above (v' = Rz * Ry * Rx * v).
Mapping to Blender space, we must use XZY euler order for all transforms. 
"""

import math
from mathutils import Vector, Euler, Quaternion, Matrix
import numpy as np

RAD_TO_DEG = 180.0 / math.pi

ROTATION_MODE_EULER = 0
ROTATION_MODE_QUATERNION = 1

# when u give up and just brute force closest euler alternatives
EULER_ALTERNATIVES = np.array([
    [0.0   , 0.0   , 0.0],
    [0.0   , 0.0   , 360.0],
    [0.0   , 0.0   , -360.0],
    [0.0   , 360.0 , 0.0],
    [0.0   , 360.0 , 360.0],
    [0.0   , 360.0 , -360.0],
    [0.0   , -360.0, 0.0],
    [0.0   , -360.0, 360.0],
    [0.0   , -360.0, -360.0],

    [360.0 , 0.0   , 0.0],
    [360.0 , 0.0   , 360.0],
    [360.0 , 0.0   , -360.0],
    [360.0 , 360.0 , 0.0],
    [360.0 , 360.0 , 360.0],
    [360.0 , 360.0 , -360.0],
    [360.0 , -360.0, 0.0],
    [360.0 , -360.0, 360.0],
    [360.0 , -360.0, -360.0],

    [-360.0, 0.0   , 0.0],
    [-360.0, 0.0   , 360.0],
    [-360.0, 0.0   , -360.0],
    [-360.0, 360.0 , 0.0],
    [-360.0, 360.0 , 360.0],
    [-360.0, 360.0 , -360.0],
    [-360.0, -360.0, 0.0],
    [-360.0, -360.0, 360.0],
    [-360.0, -360.0, -360.0],
])


class FcurveRotationMatrixCache():
    """Cache of rotation matrices at given frames sampled from fcurves
    """
    def __init__(self, rotation_mode, fcu_x, fcu_y, fcu_z, fcu_w):
        self.rotation_mode = ROTATION_MODE_EULER if rotation_mode == "rotation_euler" else ROTATION_MODE_QUATERNION
        self.fcu_x = fcu_x
        self.fcu_y = fcu_y
        self.fcu_z = fcu_z
        self.fcu_w = fcu_w
        self.cache = {}         # cache of frame => rotation matrix
        self.inverse_cache = {} # cahe of frame => inverse rotation matrix
    

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
        rot_mat = Quaternion((qw, qx, qy, qz)).to_matrix()
        return rot_mat

    
    def get_inverse(self, frame):
        """Get inverse matrix, cache inverse for frame
        """
        if frame in self.inverse_cache:
            return self.inverse_cache[frame]
        else:
            rot_mat = self.get(frame)
            inverse_rot_mat = rot_mat.inverted_safe()
            self.inverse_cache[frame] = inverse_rot_mat
            return inverse_rot_mat


class DefaultKeyframeSampler():
    """Fake sampler to replicate Fcurve.evaluate(frame) for
    action tracks that do not exist (e.g. x, y coords but no z).
    Return a constant value, e.g. 
        - Location x, y, z -> 0
        - Euler x, y, z -> 0
        - Quaternion x, y, z -> 0, w -> 1.
    """
    def __init__(self, val):
        self.val = val
    
    def evaluate(self, frame):
        return self.val


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
    Handle conversion from Blender space to VintageStory y-up when
    keyframes are added.
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
        frame data if does not exist.
        NOTE: default keyframe element always adds "rotShortestDistance"
        In vintagestory api ElementPose, rotShortestDistance will make rotations take
        shortest path? Can this be an approximation of quaternion slerp within the animation path?
        (This assumes Blender animation rotation mode is quaternion slerp.)
        https://github.com/anegostudios/vsapi/blob/master/Common/Model/Animation/ElementPose.cs
        """
        if frame not in self.keyframes:
            self.keyframes[frame] = {
                "frame": frame,
                "elements": {},
            }
        
        keyframe = self.keyframes[frame]
        if bone_name not in keyframe["elements"]:
            # create default keyframe element
            keyframe["elements"][bone_name] = {
                "rotShortestDistance": True
            }

        return keyframe["elements"][bone_name]
    

    def add_location_keyframes(self, bone_name, frames, locations):
        """Add location keyframes, do conversion to y-up
        """
        for frame, loc in zip(frames, locations):
            keyframe = self.get_bone_keyframe(bone_name, frame)
            keyframe["offsetX"] = loc.y
            keyframe["offsetY"] = loc.z
            keyframe["offsetZ"] = loc.x


    def add_rotation_keyframes(self, bone_name, frames, rotations, on_animation_end=""):
        """Add rotation keyframes, does conversion to y-up.
        """
        for frame, rot in zip(frames, rotations):
            keyframe = self.get_bone_keyframe(bone_name, frame)
            keyframe["rotationX"] = rot.y * RAD_TO_DEG
            keyframe["rotationY"] = rot.z * RAD_TO_DEG
            keyframe["rotationZ"] = rot.x * RAD_TO_DEG
        
        if on_animation_end == "repeat" and len(frames) > 0:
            # force last frame rotation to be same as first frame
            # otherwise vs will try to interpolate from two far away angles
            # causing a weird "twisting" effect
            keyframe_start = self.get_bone_keyframe(bone_name, frames[0])
            keyframe_end = self.get_bone_keyframe(bone_name, frames[-1])

            keyframe_end["rotationX"] = keyframe_start["rotationX"]
            keyframe_end["rotationY"] = keyframe_start["rotationY"]
            keyframe_end["rotationZ"] = keyframe_start["rotationZ"]

    
    def add_rotation_keyframes_version_0(self, bone_name, frames, rotations, on_animation_end=""):
        """Add rotation keyframes, does conversion to y-up.
        Also does built-in "shortest euler angle tracing" where keyframe
        rotations are modified to try and minimize max euler angle changes
        between frames. This is required because vs directly interpolates
        eulers, and there is a lossy quaternion->euler conversion from
        blender :^( 
        """
        def get_shortest_angle_deg(start, end):
            # Subtract the angles, constraining the value to [0, 360)
            diff = ( end - start ) % 360

            # If we are more than 180 we're taking the long way around.
            # Let's instead go in the shorter, negative direction
            if diff > 180 :
                diff = -(360 - diff)
            return diff
            # shortest_angle = ((((end - start) % 360) + 540) % 360) - 180
            # return shortest_angle

        def get_closer_euler_angle(
            prev_rx,
            prev_ry,
            prev_rz,
            rx,
            ry,
            rz,
        ):
            # try two candidates:
            # find shortest angle deltas between prev_r and r
            delta_rx = get_shortest_angle_deg(prev_rx, rx)
            delta_ry = get_shortest_angle_deg(prev_ry, ry)
            delta_rz = get_shortest_angle_deg(prev_rz, rz)

            r_main_candidate = np.array([
                prev_rx + delta_rx,
                prev_ry + delta_ry,
                prev_rz + delta_rz,
            ])

            r_main_alt = np.array([
                180 + r_main_candidate[0],
                180 - r_main_candidate[1],
                180 + r_main_candidate[2],
            ])
            
            # greedy brute force: test 54 closest candidates and pick angle
            # that minimize sum of square angle delta between keyframes

            # closest euler candidates 
            euler_candidates = np.concatenate([
                r_main_candidate + EULER_ALTERNATIVES,
                r_main_alt + EULER_ALTERNATIVES,
            ], axis=0)

            # square magnitude of delta between prev_r and each candidate
            r_deltas = euler_candidates - np.array([prev_rx, prev_ry, prev_rz])
            r_deltas_sq = np.sum(r_deltas * r_deltas, axis=1)

            # find index candidate with smallest sum of squared angle delta
            i_min_delta = np.argmin(r_deltas_sq)
            r_min_delta = euler_candidates[i_min_delta,:]
            
            return r_min_delta[0], r_min_delta[1], r_min_delta[2]

        prev_rx = 0
        prev_ry = 0
        prev_rz = 0

        for frame, rot in zip(frames, rotations):
            # print("frame", frame)
            keyframe = self.get_bone_keyframe(bone_name, frame)
            rx = rot.y * RAD_TO_DEG
            ry = rot.z * RAD_TO_DEG
            rz = rot.x * RAD_TO_DEG
            
            rx, ry, rz = get_closer_euler_angle(prev_rx, prev_ry, prev_rz, rx, ry, rz)

            keyframe["rotationX"] = rx
            keyframe["rotationY"] = ry
            keyframe["rotationZ"] = rz

            prev_rx = rx
            prev_ry = ry
            prev_rz = rz
        
        if on_animation_end == "repeat" and len(frames) > 0:
            # force last frame rotation to be same as first frame
            # otherwise vs will try to interpolate from two far away angles
            # causing a weird "twisting" effect
            keyframe_start = self.get_bone_keyframe(bone_name, frames[0])
            keyframe_end = self.get_bone_keyframe(bone_name, frames[-1])

            keyframe_end["rotationX"] = keyframe_start["rotationX"]
            keyframe_end["rotationY"] = keyframe_start["rotationY"]
            keyframe_end["rotationZ"] = keyframe_start["rotationZ"]

        # for debugging
        # if bone_name == "b_root":
        #     for frame, rot in zip(frames, rotations):
        #         keyframe = self.get_bone_keyframe(bone_name, frame)
        #         print("[{}] rx={} ry={} rz={}".format(
        #             frame,
        #             keyframe["rotationX"],
        #             keyframe["rotationY"],
        #             keyframe["rotationZ"],
        #         ))

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

    For quaternion rotations "pose.bones["BoneName"].rotation_quaternion",
        index 0 => fcu.w
        index 1 => fcu.x
        index 2 => fcu.y
        index 3 => fcu.z

    For euler rotations "pose.bones["BoneName"].rotation_euler",
        index 0 => fcu.x
        index 1 => fcu.y
        index 2 => fcu.z
    """
    def __init__(
        self,
        action,
        name=None,
        armature=None,
        on_animation_end="repeat",
        animation_version_0=False,
    ):
        # name, for debugging only
        self.name = name

        # Blender animation action
        self.action = action

        # store fcurves by name
        self.storage = {}

        # armature, if exists
        self.armature = armature
        
        # map of bone name => rotation mode
        # ("rotation_euler" or "rotation_quaternion")
        self.bone_rotation_mode = {}

        # vintagestory animation end handling
        self.on_animation_end = on_animation_end.lower()

        # animation version 0: old vintagestory format which uses
        # R*T*v format and additive bone euler rotation in keyframes
        # this causes incompatible transforms/animations with blender,
        # only use version 0 for testing/seeing old animation format
        self.animation_version_0 = animation_version_0
    

    def set_bone_rotation_mode(self, bone_name, rotation_mode):
        """Map bone_name to rotation_mode.
        """
        self.bone_rotation_mode[bone_name] = rotation_mode

    
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
        for bone, rotation_mode in self.bone_rotation_mode.items():
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

    
    def get_all_frames(self, fcurve_name):
        """Return sorted list of all fcurve frame numbers
        taken from all x, y, z, w channels.
        """
        frames = set()
        for fcu in self.storage[fcurve_name]:
            if fcu is not None:
                for p in fcu.keyframe_points:
                    frame, _ = p.co
                    frames.add(int(frame))

        frames = list(frames)
        frames.sort()
        return frames
    

    def create_vintage_story_keyframes(self, bone_hierarchy):
        """Create list of keyframes for this action in VintageStory format.
        Assume keyframes can have holes (not all x, y, z coords) and can be
        either use euler or quaternion rotation.
            loc.x   o---o---o---o---o
            loc.y   o-------o--------   <- holes
            loc.z   -----------------   <- does not exist, use default sampler
            rot.x   o-------------o-o   <- rotation keyframes may not match
            rot.y   o-------------o-o      all location keyframes
            rot.z   o-------------o-o
            rot.w   o-------------o-o
        1. For location, rotation keyframes, gather all frame numbers
           where a keyframe exists from all x, y, z, and/or w fcurves.
        2. At each frame number, generate the keyframe coordinate.
            -> For missing points, use fcurve.evaluate() to sample
            -> For missing fcurve, replace with a dummy fcurve with
               evaluate() that returns default value (0 location/rotation)
        3. For all keyframes, apply 90 deg rotations from objects (TODO)
        4. Convert location keyframe positions to VintageStory format:
                VintageStory: w = R(v + u)
                Blender:      w = Rv + u'
            Where u' = R*u -> u = R^-1 * u'
        5. Convert quaternion keyframes into euler keyframes
        6. Make keyframes list, where each frame has list of elements
            keyframes = [
                {
                    frame: #,
                    bone_name => { location, rotation },
                }
                ...
            ]
        
        `bone_hierarchy` is used to check if a bone inserted a dummy object.
        If so, name output bone as "b_{bone.name}", with prefix to avoid bone
        name conflicting with existing objects in scene.
        """
        # map frame # => keyframe data
        keyframes = KeyframeAdapter()
        
        for bone_name, rotation_mode in self.bone_rotation_mode.items():
            fcu_name_prefix = "pose.bones[\"{}\"]".format(bone_name)
            fcu_name_location = fcu_name_prefix + ".location"
            if rotation_mode == "rotation_euler":
                fcu_name_rotation = fcu_name_prefix + ".rotation_euler"
            else:
                fcu_name_rotation = fcu_name_prefix + ".rotation_quaternion"

            bone = self.armature.bones[bone_name]

            # get output bone name
            if bone_hierarchy[bone_name].creating_dummy_object:
                output_bone_name = "b_" + bone_name
            else:
                output_bone_name = bone_name

            # bone local euler rotation in XZY order (vintagestory euler order)
            bone_rot_local_euler = bone.matrix.to_euler("XZY")

            # for debugging
            # print(f"BONE={bone_name} rot={bone_rot_local_euler}")
            # print(f"bone.matrix_local = {bone.matrix_local}")
            # print(f"bone.matrix = {bone.matrix}")

            # =====================
            # rotation keyframes
            # =====================
            if fcu_name_rotation in self.storage:
                frames = self.get_all_frames(fcu_name_rotation)
                rotation_keyframes = []

                # default keyframe samplers
                fcu_x = DefaultKeyframeSampler(0.0)
                fcu_y = DefaultKeyframeSampler(0.0)
                fcu_z = DefaultKeyframeSampler(0.0)
                fcu_w = DefaultKeyframeSampler(1.0)

                for frame in frames:
                    if rotation_mode == "rotation_euler":
                        # euler strategy:
                        # just directly write euler values to output keyframe
                        fcu_w = None
                        fcu_x = self.storage[fcu_name_rotation][0] or DefaultKeyframeSampler(0.0)
                        fcu_y = self.storage[fcu_name_rotation][1] or DefaultKeyframeSampler(0.0)
                        fcu_z = self.storage[fcu_name_rotation][2] or DefaultKeyframeSampler(0.0)
                        rot_anim = Euler((
                            fcu_x.evaluate(frame),
                            fcu_y.evaluate(frame),
                            fcu_z.evaluate(frame),
                        ), "XZY")
                        keyframe_effective_euler = rot_anim
                    elif rotation_mode == "rotation_quaternion":
                        # quaternion strategy:
                        # need to convert quaternion to euler which is intrinsically ill defined
                        # because eulers are not unique...
                        fcu_w = self.storage[fcu_name_rotation][0] or DefaultKeyframeSampler(1.0)
                        fcu_x = self.storage[fcu_name_rotation][1] or DefaultKeyframeSampler(0.0)
                        fcu_y = self.storage[fcu_name_rotation][2] or DefaultKeyframeSampler(0.0)
                        fcu_z = self.storage[fcu_name_rotation][3] or DefaultKeyframeSampler(0.0)
                        rot_anim = Quaternion((
                            fcu_w.evaluate(frame),
                            fcu_x.evaluate(frame),
                            fcu_y.evaluate(frame),
                            fcu_z.evaluate(frame),
                        ))
                        keyframe_effective_euler = rot_anim.to_euler("XZY")
                    else:
                        raise ValueError(f"Unsupported rotation mode: {rotation_mode}")
                    
                    if self.animation_version_0:
                        # animation version 0: vintage story does rotations using:
                        #   GL11.glRotated(AnimatedElement.rotationX + getRotationX(), 1, 0, 0);
                        # so our keyframe value is actually the delta between the target effective
                        # euler angle and the bone's local euler angle, r = r_total - r_local.
                        keyframe_effective_euler.rotate(bone_rot_local_euler)
                        rx = keyframe_effective_euler.x - bone_rot_local_euler.x
                        ry = keyframe_effective_euler.y - bone_rot_local_euler.y
                        rz = keyframe_effective_euler.z - bone_rot_local_euler.z
                    else:
                        # new format, bone keyframe transform applied onto local transform,
                        # can use same format as blender
                        rx = keyframe_effective_euler.x
                        ry = keyframe_effective_euler.y
                        rz = keyframe_effective_euler.z
                    rot_vs = Euler((rx, ry, rz), "XZY")
                    rotation_keyframes.append(rot_vs)
                
                # handles conversion from radians to degrees and to y-up
                if self.animation_version_0:
                    keyframes.add_rotation_keyframes_version_0(
                        output_bone_name,
                        frames, 
                        rotation_keyframes,
                        self.on_animation_end,
                    )

                    # animation version 0: need to convert between vintagestory
                    #   v' = R*T*v
                    # versus blender:
                    #   v' = T*R*v
                    # must collect rotation matrices for converting location
                    # keyframes in the next step
                    rotation_matrix_cache = FcurveRotationMatrixCache(
                        rotation_mode,
                        fcu_x,
                        fcu_y,
                        fcu_z,
                        fcu_w,
                    )
                else:
                    # new animation format in line with blender
                    keyframes.add_rotation_keyframes(
                        output_bone_name,
                        frames, 
                        rotation_keyframes,
                        self.on_animation_end,
                    )
                    rotation_matrix_cache = None
            else:
                rotation_matrix_cache = None
            
            # =====================
            # location keyframes
            # =====================
            if fcu_name_location in self.storage:
                frames = self.get_all_frames(fcu_name_location)

                fcu_x = self.storage[fcu_name_location][0] or DefaultKeyframeSampler(0.0)
                fcu_y = self.storage[fcu_name_location][1] or DefaultKeyframeSampler(0.0)
                fcu_z = self.storage[fcu_name_location][2] or DefaultKeyframeSampler(0.0)

                location_keyframes = []
                for frame in frames:
                    loc = Vector((
                        fcu_x.evaluate(frame),
                        fcu_y.evaluate(frame),
                        fcu_z.evaluate(frame),
                    ))

                    if self.animation_version_0:
                        # vintagestory old animation version 0
                        # apply inverse rotation matrix to convert
                        #     VintageStory: w = R(v + u)
                        #     Blender:      w = Rv + u'
                        # where u' = R*u -> u = R^-1 * u'
                        if loc.x != 0.0 or loc.y != 0.0 or loc.z != 0.0:
                            if rotation_matrix_cache is not None:
                                rot_mat_inverse = rotation_matrix_cache.get_inverse(frame)
                                loc = rot_mat_inverse @ loc
                    
                    location_keyframes.append(loc)
                
                # handles conversion to y-up
                keyframes.add_location_keyframes(output_bone_name, frames, location_keyframes)

        # convert keyframes map into a list of keyframes
        keyframes_list = keyframes.tolist()

        return keyframes_list