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
