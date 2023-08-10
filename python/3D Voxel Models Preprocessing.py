import numpy as np
import os

import scipy.io

new_train_dir = '/Users/yuxuanzhang/3D_Voxel_Files/'
train_dir = '/Users/yuxuanzhang/Downloads/3DShapeNets/volumetric_data/chair/30/train/'

if not os.path.exists(new_train_dir):
    os.makedirs(new_train_dir)

train_file_names = [f for f in os.listdir(train_dir) if f.endswith('_1.mat')]
for f in train_file_names:
    voxel_matrix = scipy.io.loadmat(train_dir+f)['instance']
    # add padding to original matrix to make it 32*32*32
    voxel_matrix=np.pad(voxel_matrix,(1,1),'constant',constant_values=(0,0)) 
    voxel_matrix.dump(new_train_dir+f[:-4])

test_dir = '/Users/yuxuanzhang/Downloads/3DShapeNets/volumetric_data/chair/30/test/'
test_file_names = [f for f in os.listdir(test_dir) if f.endswith('_1.mat')] 
for f in test_file_names:
    voxel_matrix = scipy.io.loadmat(test_dir+f)['instance']
    # add padding to original matrix to make it 32*32*32
    voxel_matrix=np.pad(voxel_matrix,(1,1),'constant',constant_values=(0,0)) 
    voxel_matrix.dump(new_train_dir+f[:-4])

voxel_matrix = np.load('/Users/yuxuanzhang/3D_Voxel_Files/chair_000000000_1')

from skimage import measure
from stl import mesh
# Use marching cubes to obtain the surface mesh of these ellipsoids
vertices, faces = measure.marching_cubes(voxel_matrix,0.0)

# Create the mesh and save as STL
chair = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        chair.vectors[i][j] = vertices[f[j],:]

# Write the mesh to STL file 
chair.save('matlab_chair.stl')

# Plot out the meshed object
from mpl_toolkits import mplot3d
from matplotlib import pyplot

figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(chair.vectors))

# Auto scale to the mesh size
scale = chair.points.flatten(-1)
axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
pyplot.show()

from stl import mesh

vertices = [] # placeholder for x,y,z coordinates of the vertices
faces = [] # placeholder for the indices of three vertices that form a trangular face

f = open("chair_0461.off","r")
line = f.readline()
n = 0
while line:
    if n>1:
        line = line.rstrip() # get rid of '\n' at the end
        parsed_list = line.split(" ")
        if len(parsed_list) == 3: 
            vertices.append([float(coordinate) for coordinate in parsed_list])
        else:
            faces.append([int(index) for index in parsed_list[1:]])
    line = f.readline()
    n += 1
f.close()

vertices = np.array(vertices)
faces = np.array(faces)
#print vertices
#print faces

# Create the mesh and save as STL
chair = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        chair.vectors[i][j] = vertices[f[j],:]

# Write the mesh to file "chair.stl"
chair.save('dd.stl')

# visualize a mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot

# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure)

# Load the STL files and add the vectors to the plot
axes.add_collection3d(mplot3d.art3d.Poly3DCollection(chair.vectors))

# Auto scale to the mesh size
scale = chair.points.flatten(-1)
axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
pyplot.show()



