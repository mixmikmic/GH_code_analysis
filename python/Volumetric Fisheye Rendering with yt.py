# First let's import yt 
import yt
from yt.visualization.volume_rendering.transfer_function_helper import TransferFunctionHelper
from yt.visualization.volume_rendering.api import Scene, VolumeSource

ds = yt.load("data/DD0043/DD0043")

ds.field_list

yt.SlicePlot(ds, 'z', ['Density','Dark_Matter_Density'])

sc = yt.create_scene(ds,field="Density")

sc

sc.show(sigma_clip=4.0)

print (sc.camera)

tf = sc.get_source(0).transfer_function
tf

tfh = TransferFunctionHelper(ds)
tfh.set_field('Density')
tfh.plot(profile_field='Density')

# Let's make a new custom Transfer Function
import numpy as np
tfh.tf.clear()
tfh.set_log(True)
#def coreBoost(vals, minval, maxval):
#    return((vals-minval)/(maxval-minval))
tfh.tf.map_to_colormap(-1.0,2.5, colormap='inferno',scale=10.0)
tfh.tf.add_gaussian(2.5, width=.4, height=[1.0, 1.0, 0.3, 60.0])
tf.grey_opacity = True
tfh.plot(profile_field='Density')

source = sc.get_source(0)
source.set_transfer_function(tfh.tf)
sc.render()
sc.show(sigma_clip=4.0)

cam = sc.add_camera(ds, lens_type='fisheye')
sc.camera.position=[0.55,0.55,0.9]
print (sc.camera)

sc.render()
sc.show()



