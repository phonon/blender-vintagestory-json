bl_info = {
    "name": "Vintage Story JSON Import/Export",
    "author": "phonon",
    "version": (0, 0, 0),
    "blender": (2, 83, 0),
    "location": "View3D",
    "description": "Vintage Story JSON import/export",
    "warning": "",
    "tracker_url": "https://github.com/phonon/blender-vintagestory-json",
    "category": "Vintage Story",
}

from . import io_scene_vintagestory_json
from . import vintagestory_utils

# reload imported modules
import importlib
importlib.reload(io_scene_vintagestory_json) 
importlib.reload(vintagestory_utils)

def register():
    io_scene_vintagestory_json.register()
    vintagestory_utils.register()

def unregister():
    vintagestory_utils.unregister()
    io_scene_vintagestory_json.unregister()