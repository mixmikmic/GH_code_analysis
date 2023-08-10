import numpy as np
import menpo3d.io as mio

mesh = mio.import_builtin_asset('james.obj')

get_ipython().magic('matplotlib qt')
viewer = mesh.view()

viewer_settings = viewer.renderer_settings

# Let's print the current state so that we can see it!
np.set_printoptions(linewidth=500, precision=1, suppress=True)
for k, v in viewer_settings.items():
    print("{}: ".format(k))
    print(v)

from menpo3d.rasterize import GLRasterizer

# Build a rasterizer configured from the current view
r = GLRasterizer(**viewer_settings)

# Rasterize to produce an RGB image
rgb_img = r.rasterize_mesh(mesh)

get_ipython().magic('matplotlib inline')
rgb_img.view()

rgb_img.mask.view()

# The first output is the RGB image as before, the second is the XYZ information
rgb_img, shape_img = r.rasterize_mesh_with_shape_image(mesh)

# The last channel is the z information in model space coordinates
# Note that this is NOT camera depth
shape_img.view(channels=2)

