from osgeo import gdal
from tvtk.api import tvtk
from mayavi import mlab
import Image
from matplotlib.colors import LightSource

im1 = Image.open("esophagus.jpg")

im2 = im1.rotate(90)
im2.save("tmp/texture6.jpg")

bmp1 = tvtk.JPEGReader(file_name="tmp/texture6.jpg")

my_texture=tvtk.Texture()
my_texture.interpolate=0

# my_texture.set_input(0,bmp1.get_output())
#tvtk.configure_input(my_texture, bmp1)
my_texture=tvtk.Texture(input_connection=bmp1.output_port, interpolate=0)

surf=mlab.pipeline.surface(mlab.pipeline.open("PLY/ASR.ply"))

import numpy as np
surf.scene.camera.view_transform_matrix.to_array().astype(np.float32)

surf.scene.get_size()[0]

# Create light source object.
ls = LightSource(azdeg=0, altdeg=65)

surf.actor.enable_texture = True
surf.actor.tcoord_generator_mode = 'plane'
surf.actor.actor.texture = my_texture

# Change the visualization parameters.
surf.actor.property.interpolation = 'phong'
surf.actor.property.specular = 0.1
surf.actor.property.specular_power = 5
mlab.show()

surf.scene.camera.view_transform_matrix.to_array().astype(np.float32)

def modelview_matrix(self):
    r"""
    Retrieves the modelview matrix for this scene.

    :type: ``(4, 4)`` `ndarray`
    """
    camera = self.figure.scene.camera
    return camera.view_transform_matrix.to_array().astype(np.float32)

def projection_matrix(self):
    r"""
    Retrieves the projection matrix for this scene.

    :type: ``(4, 4)`` `ndarray`
    """
    scene = self.figure.scene
    camera = scene.camera
    scene_size = tuple(scene.get_size())
    aspect_ratio = float(scene_size[0]) / float(scene_size[1])
    p = camera.get_projection_transform_matrix(
        aspect_ratio, -1, 1).to_array().astype(np.float32)
    return p

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
get_ipython().magic('matplotlib inline')

img=mpimg.imread('esophagus.jpg')

plt.imshow(img)
plt.show()

img.shape

import random
import cv2
sample_array=[]
r=50
center=[img.shape[0]/2,150]
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if (i-center[0])**2+(j-center[1])**2<=100**2:
            sample_array.append(list(img[i,j,:]))
            
#array=np.array(sample_array)
length=len(sample_array)

constructed_image=[]
for i in range(img.shape[0]):
    temp_array=[]
    for j in range(img.shape[1]):
        randnum=random.randint(0,length-1)
        temp_array.append(sample_array[randnum])
    constructed_image.append(temp_array)
constructed_image=np.array(constructed_image)

blur = cv2.GaussianBlur(constructed_image,(7,7),0)
dst = cv2.fastNlMeansDenoisingColored(blur,None,10,10,7,21)

plt.imshow(constructed_image)
plt.show()

plt.imshow(blur)
plt.show()

plt.imshow(dst)
plt.show()

import scipy.misc
scipy.misc.imsave('sampled_texture.jpg', blur)

from osgeo import gdal
from tvtk.api import tvtk
from mayavi import mlab
import Image

im1 = Image.open("sampled_texture.jpg")
im2 = im1.rotate(90)
im2.save("tmp/sampled_texture.jpg")
bmp1 = tvtk.JPEGReader(file_name="tmp/sampled_texture.jpg")

my_texture=tvtk.Texture()
my_texture.interpolate=0
my_texture=tvtk.Texture(input_connection=bmp1.output_port, interpolate=0)

from scipy.spatial import distance
import numpy as np
import codecs, json 

surf=mlab.pipeline.surface(mlab.pipeline.open("PLY/ASR.ply"))

# Change the visualization parameters.
surf.actor.property.interpolation = 'phong'
surf.actor.property.specular = 0.5
surf.actor.property.specular_power = 5

surf.actor.enable_texture = True
surf.actor.tcoord_generator_mode = 'plane'
surf.actor.actor.texture = my_texture

cam1,foc1=mlab.move()
focal_length=distance.euclidean(cam1,foc1)
intrinsic_matrix=[[focal_length,0,surf.scene.get_size()[1]/2],[0,focal_length,surf.scene.get_size()[0]/2],[0,0,1]]

mlab.pitch(10)

mlab.view(80,120,45)
surf.scene.camera.elevation(10)
pose, focs = mlab.move()

poses = []
extrinsic_matrices=[]

# Make an animation:
for i in range(200):
    # change distance
    delta = -(float(i)/float(200))*5
    mlab.view(80,130,45+delta)
    mlab.roll(30+delta)
    mlab.yaw(20-2*delta)
    mlab.pitch(20+2*delta)
    #surf.scene.camera.roll(10+delta)
    #surf.scene.camera.azimuth(10-delta)
    #fig=mlab.figure(bgcolor=(1,1,1))
    
    cam1,foc1=mlab.move()
    roll = mlab.roll()
    #azui = mlab.view()[0]
    #elevation = mlab.view()[1]
    yaw = 20-delta*2
    pitch = 20+delta*2
    temp = list(cam1)+[roll,yaw,pitch]
    #print(temp)
    poses.append(temp)
    surf.scene.save_png('test_images/anim%d.png'%i)
dict_test={}
dict_test_poses={}
for i in range(len(poses)):
    dict_test_poses["anim"+str(i)+"_camera_pos"]=poses[i]
dict_test['camera_poses']=dict_test_poses

file_path = "test_images/camera_pose.json" ## your path variable
json.dump(dict_test, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

mlab.show()

cam1,foc1=mlab.move()
focal_length=distance.euclidean(cam1,foc1)
intrinsic_matrix=[[focal_length,0,surf.scene.get_size()[1]/2],[0,focal_length,surf.scene.get_size()[0]/2],[0,0,1]]

mlab.view(90,120,75)

poses=[]
focs=[]
extrinsic_matrices=[]
# Make an animation:
for i in range(10000):
    # change distance
    delta = -(float(i)/float(10000))*36
    mlab.view(80,120,60+delta)
    #fig=mlab.figure(bgcolor=(1,1,1))
    
    cam1,foc1=mlab.move()
    poses.append(cam1)
    focs.append(foc1)
    matrix=surf.scene.camera.view_transform_matrix.to_array().astype(np.float32)
    extrinsic_matrices.append(matrix)
    
    # Save the scene.
    surf.scene.save_png('test_images/anim%d.png'%i)
    #mlab.savefig('test_images/anim%d.png'%i,figure=fig)

mlab.show()

import json


dict_test={}
dict_test_poses={}
for i in range(len(poses)):
    dict_test_poses["anim"+str(i)+"_camera_pose"]=list(poses[i])
dict_test['camera_poses']=dict_test_poses

dict_test_focal_point={}
for i in range(len(poses)):
    dict_test_focal_point["anim"+str(i)+"_focal_point"]=list(focs[i])
dict_test['focal_points']=dict_test_focal_point

file_path = "test_images/camera_pose.json" ## your path variable
json.dump(dict_test, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)


extrinsic={}
extrinsic_intrinsic={}
for i in range(len(extrinsic_matrices)):
    extrinsic["view"+str(i)+"_extrinsic_matrix"]=extrinsic_matrices[i].tolist()
extrinsic_intrinsic['extrinsic_info']=extrinsic
extrinsic_intrinsic['intrinsic_info']=intrinsic_matrix

file_path = "test_images/extrinsic_intrinsic.json" ## your path variable
json.dump(extrinsic_intrinsic, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

from osgeo import gdal
from tvtk.api import tvtk
from mayavi import mlab
import Image

im1 = Image.open("sampled_texture.jpg")
im2 = im1.rotate(90)
im2.save("tmp/sampled_texture.jpg")
bmp1 = tvtk.JPEGReader(file_name="tmp/sampled_texture.jpg")

my_texture=tvtk.Texture()
my_texture.interpolate=0
my_texture=tvtk.Texture(input_connection=bmp1.output_port, interpolate=0)

surf=mlab.pipeline.surface(mlab.pipeline.open("PLY/ASR.ply"))
surf.actor.enable_texture = True
surf.actor.tcoord_generator_mode = 'plane'
surf.actor.actor.texture = my_texture

mlab.show()

from scipy.spatial import distance
import numpy as np
import codecs, json 

poses=[]
focs=[]
extrinsic_matrices=[]

cam1,foc1=mlab.move()
poses.append(cam1)
focs.append(foc1)
focal_length=distance.euclidean(cam1,foc1)
intrinsic_matrix=[[focal_length,0,surf.scene.get_size()[1]/2],[0,focal_length,surf.scene.get_size()[0]/2],[0,0,1]]

# Make an animation:
for i in range(36):
    # Rotate the camera by 10 degrees.
    surf.scene.camera.azimuth(10)

    # Resets the camera clipping plane so everything fits and then
    # renders.
    surf.scene.reset_zoom()
    
    cam1,foc1=mlab.move()
    poses.append(cam1)
    focs.append(foc1)
    matrix=surf.scene.camera.view_transform_matrix.to_array().astype(np.float32)
    extrinsic_matrices.append(matrix)
    
    # Save the scene.
    surf.scene.save_png('saved_images/anim%d.png'%i)

dict_test={}
dict_test_poses={}
for i in range(len(poses)):
    dict_test_poses["anim"+str(i)+"_camera_pose"]=list(poses[i])
dict_test['camera_poses']=dict_test_poses

dict_test_focal_point={}
for i in range(len(poses)):
    dict_test_focal_point["anim"+str(i)+"_focal_point"]=list(focs[i])
dict_test['focal_points']=dict_test_focal_point

file_path = "saved_images/camera_pose.json" ## your path variable
json.dump(dict_test, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
    
extrinsic={}
extrinsic_intrinsic={}
for i in range(len(extrinsic_matrices)):
    extrinsic["anim"+str(i)+"_extrinsic_matrix"]=extrinsic_matrices[i].tolist()
extrinsic_intrinsic['extrinsic_info']=extrinsic
extrinsic_intrinsic['intrinsic_info']=intrinsic_matrix

file_path = "saved_images/extrinsic_intrinsic.json" ## your path variable
json.dump(extrinsic_intrinsic, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

mlab.show()

surf.scene.camera.view_transform_matrix.to_array().astype(np.float32)

