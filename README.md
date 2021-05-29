Vintage Story JSON Import/Export
=======================================
Import/export cuboid geometry between Blender and Vintage Story .json model format. The Blender model must follow very specific restrictions for the exporter to work (read **Export Guide** below).

Supports import/export uvs. This addon can export solid material colors packed into an auto-generated image texture (alongside normal textures), so you can mix textures and solid face colors on Blender meshes.

Tested on Blender 2.92.


Installation
---------------------------------------
1. `git clone` or copy this repository into your `scripts/addons` or custom scripts folder.
2. Enable in **Edit > Preferences > Add-ons** (search for *Vintage Story JSON Import/Export*)


Export Guide/Notes
---------------------------------------
- **Only exports cuboid objects** e.g. Object meshes must be rectangular prisms (8 vertices and 6 faces). The local mesh coordinates must be aligned to global XYZ axis. **Do not rotate the mesh vertices in edit mode.**
- **All cuboids must be separate objects.**
- **Apply all scale to objects before exporting.** Use `ctrl + A` to bring up Apply menu then hit `Apply > Scale`. (Also found in `Object > Apply > Scale` tab in viewport)
- **Attach points**: Create an "Empty" type object (e.g. **Shift + A > Empty > Arrows**) and name it "attach_{name}", the {name} will become an attachpoint. e.g. "attach_Center" will generate an attachpoint called "Center". Parent it to associated object: select empty, select object parent, then set parent with `Ctrl P > Object`.
- **Animation setup**: Animations must use an armature + bones. Create an armature and bones, then parent objects to armature with setting "Bone":
    1. Select armature, enter edit mode (Tab), then select the bone to be the parent. Exit back into object mode (Tab).
    2. Select objects that you want to be the children.
    3. Then select the armature, parent using `Ctrl P > Bone`. In the right side `Object Properties > Relations` panel the `Parent` should be the armature object, the `Parent Type` should be "Bone" and the `Parent Bone` should be the bone you previously selected.
    4. Create animation action on the armature, and make keyframes using the bones (standard Blender bone animation).
    5. On export, if an object has the same name as its parent bone, it will be the object that acts as the bone, the "bone object". Otherwise, a random child will be chosen. The other children of the bone will be parented to the "bone object" in the exported .json.
- **Animation metadata**: Animation metadata uses Action pose markers. First enable these in the Action Editor from the menu **Marker > Show Pose Markers**
    - **"onAnimationEnd" + "quantityFrames"**: Put a pose marker named "onAnimationEnd_{Action}" at the frame where the animation should end. "quantityFrames" will be that (frame + 1), with the assumption animations start at frame 0. "onAnimationEnd" will be the {Action}, e.g. "onAnimationEnd_Stop" at frame 119 will generate keys:
        - "onAnimationEnd": "Stop"
        - "quantityFrames": 120
    - **"onActivityStopped"**: Put a pose marker anywhere named "onActivityStopped_{Action}". e.g. "onActivityStopped_PlayTillEnd" will generate key:
        - "onActivityStopped": "PlayTillEnd"
- **Generating solid color textures:** By default, the exporter will generate a texture containing all solid material colors. So you can texture using just materials + colors without UV mapping or a texture image. This works alongside using texture images and uv mapping.
- **Recalculate normals if textures are on wrong faces.** Applying negative scales can sometimes flip face normals inside, which causes incorrect auto-generated texture colors. Recalculate normals on all meshes to fix (`Select All` > `Edit Mode` > `Select All` > `ctrl + shift + N` to recalculate and **uncheck inside**).
- **Parenting objects:** You can mix directly parenting object to object (`Ctrl P > Object`) along with armature bones. 


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
    4. Animations format is relative displacement from bone rest pose. Note that Blender and VintageStory location is applied differently (this import/export will handle the conversion between these formats):
        - VintageStory: v' = R\*T\*v (translate first, then rotate)
        - Blender: v' = T\*R\*v (rotate first, then translate)


Export Options
---------------------------------------
|  Option  |  Default   | Description  |
|----------|------------|------------- |
| Selection Only | False | If True, export only selected objects|
| Translate Origin | True | Fixed translation of coordinates by `(x,y,z)` (in Blender coordinates) |
| Translate X | 8.0 | `x` export recenter coordinate |
| Translate Y | 8.0 | `y` export recenter coordinate |
| Translate Z | 0.0 | `z` export recenter coordinate |
| Texture Subfolder  |  | Subfolder for model in textures folder: `/textures/[subfolder]` |
| Color Texture Name |  | Name of color texture generated `[name].png` (blank defaults to output `.json` filename) |
| Export UVs | True | Exports face UVs |
| Generate Color Texture | True | Auto-textures solid material colors and generates a `.png` image texture exported alongside model (overwrites UVs). By default will get colors from all materials in the Blender file. |
| Only Use Exported Object Colors | False | When exporting auto-generated color texture, only use material colors on exported objects (instead of all materials in file). |
| Minify .json | False | Enable options to reduce .json file size |
| Decimal Precision | 8 | Number of digits after decimal point in output .json (-1 to disable) |
| Export Animations | False | Export bone animations (TODO) |


Import Options
---------------------------------------
|  Option  |  Default   | Description  |
|----------|------------|------------- |
| Import UVs | True | Import face UVs |
| Recenter to Origin | False | Re-centers model center on Blender world origin (overrides translate option) |
| Translate Origin | False | Fixed translation of coordinates by `(x,y,z)` (in Blender coordinates) |
| Translate X | -8.0 | `x` import recenter coordinate |
| Translate Y | -8.0 | `y` import recenter coordinate |
| Translate Z | 0.0 | `z` import recenter coordinate |
| Import Animations | True | Import animations, converts object hierachy to bone hierarchy |


Known Issues (TODO):
---------------------------------------
- UV export of a cube converted into a plane by scaling a dimension to 0 (e.g. single plane, such as for hair) does not work, need special case for uv
- Cannot properly export animations after rotating model 90 deg (need to carry and apply 90 deg rotations)
- Will not export bone with no child cubes (need to insert dummy cube)