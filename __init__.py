bl_info = {
    "name": "Vintage Story JSON Import/Export",
    "description": "Vintage Story JSON import/export",
    "author": "phonon",
    "version": (0, 1, 0),
    "blender": (3, 3, 0),
    "location": "File > Import-Export",
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