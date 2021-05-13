Vintage Story JSON Import/Export
=======================================
Import/export cuboid geometry between Blender and Vintage Story .json model format. The Blender model must follow very specific restrictions for the exporter to work (read **Export Guide** below).

Supports import/export uvs. This addon can export solid material colors packed into an auto-generated image texture (alongside normal textures), so you can mix textures and solid face colors on Blender meshes.

Tested on Blender 2.83, 2.92.


Installation
---------------------------------------
1. `git clone` or copy this repository into your `scripts/addons` or custom scripts folder.
2. Enable in **Edit > Preferences > Add-ons** (search for *Vintage Story JSON Import/Export*)


Export Guide 
---------------------------------------
- **Only exports cuboid objects** e.g. Object meshes must be rectangular prisms (8 vertices and 6 faces). The local mesh coordinates must be aligned to global XYZ axis. **Do not rotate the mesh vertices in edit mode.**
- **All cuboids must be separate objects.**

Import Guide/Notes
---------------------------------------
- In VS Model Creator, animations are individual object keyframes. This does not map well to Blender actions.
A Blender action can transform multiple bones but only transform a single object directly.
So instead imported animations are mapped to bone animations. The full import is:
    1. Import and build mesh object hierarchy
    2. Traverse mesh hierarchy and set create a bone for each mesh object. Move object transform
    to the bone (sets mesh object transform to identity). Armature hierarchy replaces
    object hierarchy.
    3. Apply the bone transforms as the rest pose. This applies bone transforms, sets them
    to identity, and returns transform to the mesh objects.
    4. Animations format is relative displacement from bone rest pose. Note the location
    transform is applied in transformed space after bone rotation transform is applied. (TODO)