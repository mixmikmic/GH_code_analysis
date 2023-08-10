from __future__ import print_function
from neuroglancer.server import global_server_args
# setup the binding so it is visible outside the container
global_server_args['bind_port'] = 8989
global_server_args['bind_address'] = '00.0.0.0' # TODO: the server renames addresses of 0.0.0.0 and they become unusable
import neuroglancer
import numpy as np

a = np.zeros((3, 100, 100, 100), dtype=np.uint8)
ix, iy, iz = np.meshgrid(*[np.linspace(0, 1, n) for n in a.shape[1:]], indexing='ij')
a[0, :, :, :] = np.abs(np.sin(4 * (ix + iy))) * 255
a[1, :, :, :] = np.abs(np.sin(4 * (iy + iz))) * 255
a[2, :, :, :] = np.abs(np.sin(4 * (ix + iz))) * 255

b = np.cast[np.uint32](np.floor(np.sqrt((ix - 0.5)**2 + (iy - 0.5)**2 + (iz - 0.5)**2) * 10))
b = np.pad(b, 1, 'constant')
print(a.shape, b.shape)

neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')

viewer = neuroglancer.Viewer(voxel_size=[10, 10, 10])
viewer.add(a,
           name='a',
           # offset is in nm, not voxels
           offset=(200, 300, 150),
           shader="""
void main() {
  emitRGB(vec3(toNormalized(getDataValue(0)),
               toNormalized(getDataValue(1)),
               toNormalized(getDataValue(2))));
}
""")
viewer.add(b, name='b')

viewer

from IPython.display import HTML
HTML(url = str(viewer))

