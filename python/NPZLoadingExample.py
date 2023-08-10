from __future__ import print_function
from neuroglancer.server import global_server_args
# setup the binding so it is visible outside the container
global_server_args['bind_port'] = 8989
global_server_args['bind_address'] = '00.0.0.0' # TODO: the server renames addresses of 0.0.0.0 and they become unusable
import neuroglancer
import numpy as np

with np.load('test_data.npz', 'r') as npz_file:
    raw_ct_image = npz_file['CT']
    ct_image = ((np.expand_dims(raw_ct_image,0)+1024).clip(0,2048)/2048.0).astype(np.float32) 
    raw_PET_image = npz_file['PET']
    pet_image = ((np.expand_dims(raw_PET_image,0)).clip(0,5)/5.0).astype(np.float32) 
    label_image = np.expand_dims(npz_file['Labels'].astype(np.float32),0)
    label_image /= label_image.max() # normalize labels
    vox_size = npz_file['spacing'] # in mm

neuroglancer.set_static_content_source(url='https://neuroglancer-demo.appspot.com')

viewer = neuroglancer.Viewer(voxel_size=1000*vox_size) # since vox_size
viewer.add(ct_image,
           name='CT Image')
viewer.add(pet_image,
           name='PET Image',
          shader = """
void main () {
  emitRGB(colormapJet(toNormalized(getDataValue())));
}
          """) 
# add as a solid red green with a varying degree of transparency
viewer.add(label_image,
           name='Label Image',
          shader = """
void main () {
  emitRGBA(vec4(0, 1.0, 0, toNormalized(getDataValue())));
}
          """) 

viewer

from IPython.display import HTML
HTML(url = str(viewer))

