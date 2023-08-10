# Extract all libraries needed here
import numpy as np
import pykitti
import matplotlib.pyplot as plt

from source import parseTrackletXML as xmlParser
from source import dataset_utility as du
import projection as proj
import draw_point_cloud as pc
import ground_plane_segmentation as sg

get_ipython().magic('matplotlib inline')

# Store all parameters here

# Change this to the directory where you store KITTI data
basedir = "/Users/davidwoo/Documents/Projects/self-driving-cars/plane-segmentation/KITTI-Dataset"

# Step 1: Load all dataset

def load_dataset(date, drive, calibrated=False, frame_range=None):
    """
    Loads the dataset with `date` and `drive`.
    
    Parameters
    ----------
    date        : Dataset creation date.
    drive       : Dataset drive.
    calibrated  : Flag indicating if we need to parse calibration data. Defaults to `False`.
    frame_range : Range of frames. Defaults to `None`.

    Returns
    -------
    Loaded dataset of type `raw`.
    """
    dataset = pykitti.raw(basedir, date, drive)

    # Load the data
    if calibrated:
        dataset._load_calib()  # Calibration data are accessible as named tuples

    np.set_printoptions(precision=4, suppress=True)
    print('\nDrive: ' + str(dataset.drive))
    print('\nFrame range: ' + str(dataset.frames))

    if calibrated:
        print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
        print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
        print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

    return dataset

def load_tracklets_for_frames(n_frames, xml_path):
    """
    Loads dataset labels also referred to as tracklets, saving them individually for each frame.

    Parameters
    ----------
    n_frames    : Number of frames in the dataset.
    xml_path    : Path to the tracklets XML.

    Returns
    -------
    Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
    contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
    types as strings.
    """
    #print(xml_path)
    tracklets = xmlParser.parseXML(xml_path)

    frame_tracklets = {}
    frame_tracklets_types = {}
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []

    # loop over tracklets
    for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        
        print(tracklets)
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
            frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [
                tracklet.objectType]

    return (frame_tracklets, frame_tracklets_types)

date = '2011_09_26'
drive = '0048'
dataset = load_dataset(date, drive,calibrated=True)


directory = "/Users/davidwoo/Documents/Projects/self-driving-cars/plane-segmentation/KITTI-Dataset"
tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(directory,date, date, drive))

import sys

def convert_to_rgb(minval, maxval, val, colors):
    
    fi = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    i = int(fi)
    f = fi - i

    (r1, g1, b1), (r2, g2, b2) = colors[1], colors[2]
    return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

# Submissions #1  - 3d cloud point on camera

import cv2
import statistics
import sys

# Get velodyne points
dataset_velo = list(dataset.velo)
# Get camera images or pixel points
dataset_rgb = list(dataset.rgb)

# Set parameters
velodyne_max_x=100  # this scales the x-axis values. maybe replace with range
frame = 0 # the frame that we are interested in
include_z = True # include x-axis for velodyne points. this is mostly used for color coding the velodyne points
radius = 2 # the radius of the circle
calib_dir ="/Users/davidwoo/Documents/Projects/self-driving-cars/plane-segmentation/KITTI-Dataset/2011_09_26/"

# Convert velodyne coordinates to pixel coordinates
velo_data,velo_data_raw_sampled = proj.convert_velo_cord_to_img(dataset_velo, calib_dir)

rgb_img = dataset_rgb[frame][frame]

# Crop velodyne points outside of image size. Velodyne points might cover wider range than image 
corped_velo_data = proj.crop_velo_to_img_size([400,1500,3], velo_data,velo_data_raw_sampled,include_z)

def convert_to_rgb(minval, maxval, val, colors):
    
    fi = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    i = int(fi)
    f = fi - i

    (r1, g1, b1), (r2, g2, b2) = colors[1], colors[2]
    return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

# Overlay velodyne points over image
def overlay_velo_img(img, velo_data,radius = 2):
    (x, y,z) = velo_data
    im = np.zeros(img.shape, dtype=np.float32)
    x_axis = np.floor(x).astype(np.int32)
    y_axis = np.floor(y).astype(np.int32)
    z_axis = np.floor(z).astype(np.int32)

    # below draws circles on the image
    for i in range(0, len(x)):

        if z_axis[i] <= 0:
            color = int(1/velodyne_max_x * 256)
            value = 0
        else:
            color = int((z_axis[i])/velodyne_max_x * 256)
            value = z_axis[i] * 4
        colors_range = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # [BLUE, GREEN, RED]
        r, g, b  = convert_to_rgb(0,256,value,colors_range) 
        
        cv2.circle(img, (x_axis[i], y_axis[i]), radius, [r, g, b],-1)

    fig1 = plt.figure(figsize=(20, 20))
    
    return img

result_img = overlay_velo_img(rgb_img, corped_velo_data,radius)
plt.imshow(result_img)

# Submissions - 2d box projection

# try for one box
frame = 0
dataset_rgb = list(dataset.rgb)
rgb_img = dataset_rgb[frame][frame]
dataset_velo = list(dataset.velo)

#vertices = np.array()

max_iter = 0
frame = 0
tracklet = 0
for rect in tracklet_rects[frame]:
    
    tracklet = tracklet + 1
    
    dataset_tracklets = rect.transpose(1,0)
    
    velo_data_tracklets = proj.convert_velo_cord_to_img(dataset_tracklets, calib_dir,7, 2, 0,True)
    
    #print("print tracklets %",dataset_tracklets)
    #print("print tracklets %",velo_data_tracklets[:, 0])

    corped_velo_data_tracklets = proj.crop_velo_to_img_size([400,1500,3], velo_data_tracklets,dataset_velo[0])

    test = np.array([corped_velo_data_tracklets[0],corped_velo_data_tracklets[1]])
    
    if len(corped_velo_data_tracklets[0]) ==8:
        vertices = np.array([[(test.transpose(1,0)[0]),
                              (test.transpose(1,0)[1]), 
                              (test.transpose(1,0)[2]), 
                              (test.transpose(1,0)[3]),
                              (test.transpose(1,0)[4]),
                              (test.transpose(1,0)[5]),
                              (test.transpose(1,0)[6]),
                              (test.transpose(1,0)[7])]],
                            dtype=np.int32)
    elif len(corped_velo_data_tracklets[0]) ==6:
        vertices = np.array([[(test.transpose(1,0)[0]),
                              (test.transpose(1,0)[1]), 
                              (test.transpose(1,0)[2]), 
                              (test.transpose(1,0)[3]),
                              (test.transpose(1,0)[4]),
                              (test.transpose(1,0)[5])]],
                            dtype=np.int32)
    
    elif len(corped_velo_data_tracklets[0]) ==4:
        vertices = np.array([[(test.transpose(1,0)[0]),
                              (test.transpose(1,0)[1]), 
                              (test.transpose(1,0)[2]), 
                              (test.transpose(1,0)[3])]],
                            dtype=np.int32)

    cv2.rectangle(rgb_img, 
                  (min(vertices[0,:,0]),min(vertices[0,:,1])),(max(vertices[0,:,0]),max(vertices[0,:,1])), 
                  (100,255,100), 3)
    
    tracklet = tracklet + 1
    
    #break

fig1 = plt.figure(figsize=(20, 20))
plt.imshow(rgb_img)

get_ipython().magic('matplotlib inline')

#load the data
frame = 0
dataset_velo = list(dataset.velo)
points = 0.8 # this controls the sampling rate
points_step = int(1. / points)
point_size = 0.01 * (1. / points)

velo_range = range(0, dataset_velo[frame].shape[0], points_step)
velo_frame = dataset_velo[frame][velo_range, :] 
velo_frame_test = velo_frame
velo_frame_test[:, 3]=velo_frame[:, 3]*10
 
velo_frame = velo_frame_test

n = int(velo_frame.shape[0] * 0.1)
max_iterations = 100
goal_inliers = n * 0.5
inlier_threshold = 0.01
print("goal_inliers %",goal_inliers)

reference_vector =[0,0,1]

m, b, adj_velo_frame = sg.run_ransac(velo_frame, estimate, lambda x, y: is_inlier(x, y, 0.01), int(velo_frame.shape[0] * 0.1), goal_inliers, max_iterations,reference_vector)
a, b, c, d = m
xx, yy, zz = sg.plot_plane(a, b, c, d)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim3d(-2, 10)
ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))


print("plotting chart")
plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

colors = {
    'Car': 'b',
    'Tram': 'r',
    'Cyclist': 'g',
    'Van': 'c',
    'Truck': 'm',
    'Pedestrian': 'y',
    'Sitter': 'k'
}
axes_limits = [
    [-20, 80], # X axis range
    [-20, 20], # Y axis range
    [-3, 10]   # Z axis range
]
axes_str = ['X', 'Y', 'Z']

def draw_box(pyplot_axis, vertices, axes=[0, 1, 2], color='black'):
    """
    Draws a bounding 3D box in a pyplot axis.
    
    Parameters
    ----------
    pyplot_axis : Pyplot axis to draw in.
    vertices    : Array 8 box vertices containing x, y, z coordinates.
    axes        : Axes to use. Defaults to `[0, 1, 2]`, e.g. x, y and z axes.
    color       : Drawing color. Defaults to `black`.
    """
    vertices = vertices[axes, :]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Lower plane parallel to Z=0 plane
        [4, 5], [5, 6], [6, 7], [7, 4],  # Upper plane parallel to Z=0 plane
        [0, 4], [1, 5], [2, 6], [3, 7]  # Connections between upper and lower planes
    ]
    for connection in connections:
        pyplot_axis.plot(*vertices[:, connection], c=color, lw=0.5)

def display_frame_statistics(dataset, tracklet_rects, tracklet_types, frame, points=0.2):
    """
    Displays statistics for a single frame. Draws camera data, 3D plot of the lidar point cloud data and point cloud
    projections to various planes.
    
    Parameters
    ----------
    dataset         : `raw` dataset.
    tracklet_rects  : Dictionary with tracklet bounding boxes coordinates.
    tracklet_types  : Dictionary with tracklet types.
    frame           : Absolute number of the frame.
    points          : Fraction of lidar points to use. Defaults to `0.2`, e.g. 20%.
    """
    dataset_gray = list(dataset.gray)
    dataset_rgb = list(dataset.rgb)
    dataset_velo = list(dataset.velo)
    
    print('Frame timestamp: ' + str(dataset.timestamps[frame]))
    # Draw camera data
    f, ax = plt.subplots(2, 2, figsize=(15, 5))
    ax[0, 0].imshow(dataset_gray[frame][0], cmap='gray')
    ax[0, 0].set_title('Left Gray Image (cam0)')
    ax[0, 1].imshow(dataset_gray[frame][1], cmap='gray')
    ax[0, 1].set_title('Right Gray Image (cam1)')
    ax[1, 0].imshow(dataset_rgb[frame][0])
    ax[1, 0].set_title('Left RGB Image (cam2)')
    ax[1, 1].imshow(dataset_rgb[frame][1])
    ax[1, 1].set_title('Right RGB Image (cam3)')
    plt.show()

    points_step = int(1. / points)
    point_size = 0.01 * (1. / points)
    velo_range = range(0, dataset_velo[frame].shape[0], points_step)
    velo_frame = dataset_velo[frame][velo_range, :]      
    def draw_point_cloud(ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
        """
        Convenient method for drawing various point cloud projections as a part of frame statistics.
        """
        ax.scatter(*np.transpose(velo_frame[:, axes]), s=point_size, c=velo_frame[:, 3], cmap='gray')
        ax.set_title(title)
        ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
        ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
        if len(axes) > 2:
            ax.set_xlim3d(*axes_limits[axes[0]])
            ax.set_ylim3d(*axes_limits[axes[1]])
            ax.set_zlim3d(*axes_limits[axes[2]])
            ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
        else:
            ax.set_xlim(*axes_limits[axes[0]])
            ax.set_ylim(*axes_limits[axes[1]])
        # User specified limits
        if xlim3d!=None:
            ax.set_xlim3d(xlim3d)
        if ylim3d!=None:
            ax.set_ylim3d(ylim3d)
        if zlim3d!=None:
            ax.set_zlim3d(zlim3d)
            
        for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
            draw_box(ax, t_rects, axes=axes, color=colors[t_type])
            
    # Draw point cloud data as 3D plot
    f2 = plt.figure(figsize=(15, 8))
    ax2 = f2.add_subplot(111, projection='3d')                    
    draw_point_cloud(ax2, 'Velodyne scan', xlim3d=(-10,30))
    plt.show()
    
    # Draw point cloud data as plane projections
    f, ax3 = plt.subplots(3, 1, figsize=(15, 25))
    draw_point_cloud(
        ax3[0], 
        'Velodyne scan, XZ projection (Y = 0), the car is moving in direction left to right', 
        axes=[0, 2] # X and Z axes
    )
    draw_point_cloud(
        ax3[1], 
        'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right', 
        axes=[0, 1] # X and Y axes
    )
    draw_point_cloud(
        ax3[2], 
        'Velodyne scan, YZ projection (X = 0), the car is moving towards the graph plane', 
        axes=[1, 2] # Y and Z axes
    )
    plt.show()

def draw_point_cloud_seg(velo_frame_input,ax, title, axes=[0, 1, 2], xlim3d=None, ylim3d=None, zlim3d=None):
    """
    Convenient method for drawing various point cloud projections as a part of frame statistics.
    """
    
    point_size = 10
    velo_frame_input
    print(point_size)
    ax.scatter(*np.transpose(velo_frame_input[:, axes]), s=point_size, c=velo_frame_input[:, 3], cmap='terrain')
    ax.set_title(title)
    ax.set_xlabel('{} axis'.format(axes_str[axes[0]]))
    ax.set_ylabel('{} axis'.format(axes_str[axes[1]]))
    #ax.patch.set_facecolor('black')
    if len(axes) > 2:
        ax.set_xlim3d(*axes_limits[axes[0]])
        ax.set_ylim3d(*axes_limits[axes[1]])
        ax.set_zlim3d(*axes_limits[axes[2]])
        ax.set_zlabel('{} axis'.format(axes_str[axes[2]]))
    else:
        ax.set_xlim(*axes_limits[axes[0]])
        ax.set_ylim(*axes_limits[axes[1]])
    # User specified limits
    if xlim3d!=None:
        ax.set_xlim3d(xlim3d)
    if ylim3d!=None:
        ax.set_ylim3d(ylim3d)
    if zlim3d!=None:
        ax.set_zlim3d(zlim3d)

    for t_rects, t_type in zip(tracklet_rects[frame], tracklet_types[frame]):
        draw_box(ax, t_rects, axes=axes, color=colors[t_type])


f = plt.figure(figsize=(15, 8))
ax2 = f.add_subplot(111, projection='3d') 

#f, ax3 = plt.subplots(1, 1, figsize=(15, 12))
draw_point_cloud_seg(adj_velo_frame,ax2,'Velodyne scan, XYZ projection, the car is moving in direction left to right')
plt.show()


f, ax3 = plt.subplots(1, 1, figsize=(15, 12))
draw_point_cloud_seg(adj_velo_frame,
    ax3, 
    'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right', 
    axes=[0, 1] # X and Y axes
)
plt.show()

