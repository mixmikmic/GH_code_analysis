#TODO: create a file for dependencies or libraries needed
#!pip install moviepy
#!pip install numpy
#!pip install pykitti
#!pip install opencv-python

# Extract all libraries needed here
import numpy as np
import pykitti
import matplotlib.pyplot as plt

from source import parseTrackletXML as xmlParser
from source import dataset_utility as du

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

#basedir = "/Users/davidwoo/Documents/Projects/self-driving-cars/plane-segmentation/KITTI-Dataset"
#date = '2011_09_26'
#drive = '0048'
#dataset = pykitti.raw(basedir, date, drive)

#print(dataset._load_calib)

date = '2011_09_26'
drive = '0048'
dataset = load_dataset(date, drive,calibrated=True)


directory = "/Users/davidwoo/Documents/Projects/self-driving-cars/plane-segmentation/KITTI-Dataset"
tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), '{}/{}/{}_drive_{}_sync/tracklet_labels.xml'.format(directory,date, date, drive))

# Step 2: Start of projection of 3d to camera

# Helper function

import numpy as np
def parse_string_variable(str):
    var_name = str.split(':')[0]
    after_colon_index = len(var_name) + 1
    value = str[after_colon_index:]
    return (var_name, value)

def read_lines_to_dict(raw_text):
    var_list = []
    for i, line in enumerate(raw_text):
        var_list.append(line.replace('\n', ''))
    for i, line in enumerate(raw_text):
        var_list[i] = parse_string_variable(line)
    return dict(var_list)

def read_files_by_lines(filename):
    assert type(filename) is str
    with open(filename, 'r') as cam_to_cam:
#         data = cam_to_cam.read().replace('\n', 'r')
        data = cam_to_cam.readlines()
    return read_lines_to_dict(data)

def replace_var_from_dict_with_shape(var_dict, key, shape):
    return np.array(var_dict[key]).reshape(shape)


def loadCalibrationCamToCam(filename, verbose=False):
    assert type(filename) is str
    cam_dict = read_files_by_lines(filename)

    for key, value in cam_dict.items():
        if key == 'calib_time':
            cam_dict[key] = value
        else:
            array = []
            for i, string in enumerate(value.split(' ')[1:]):
                array.append(float(string))
            cam_dict[key] = array

    for i in range(0, 4):
        S_rect_0i = 'S_rect_0' + str(i)
        R_rect_0i = 'R_rect_0' + str(i)
        P_rect_0i = 'P_rect_0' + str(i)
        S_0i = 'S_0' + str(i)
        K_0i = 'K_0' + str(i)
        D_0i = 'D_0' + str(i)
        R_0i = 'R_0' + str(i)
        T_0i = 'T_0' + str(i)

        cam_dict[S_rect_0i] = replace_var_from_dict_with_shape(cam_dict, S_rect_0i, (1, 2))
        cam_dict[R_rect_0i] = replace_var_from_dict_with_shape(cam_dict, R_rect_0i, (3, 3))
        cam_dict[P_rect_0i] = replace_var_from_dict_with_shape(cam_dict, P_rect_0i, (3, 4))
        cam_dict[S_0i] = replace_var_from_dict_with_shape(cam_dict, S_0i, (1, 2))
        cam_dict[K_0i] = replace_var_from_dict_with_shape(cam_dict, K_0i, (3, 3))
        cam_dict[D_0i] = replace_var_from_dict_with_shape(cam_dict, D_0i, (1, 5))
        cam_dict[R_0i] = replace_var_from_dict_with_shape(cam_dict, R_0i, (3, 3))
        cam_dict[T_0i] = replace_var_from_dict_with_shape(cam_dict, T_0i, (3, 1))

    if verbose:
          print(S_rect_0i, cam_dict[S_rect_0i])
          print(R_rect_0i, cam_dict[R_rect_0i])
          print(P_rect_0i, cam_dict[P_rect_0i])
          print(S_0i, cam_dict[S_0i])
          print(K_0i, cam_dict[K_0i])
          print(D_0i, cam_dict[D_0i])
          print(R_0i, cam_dict[R_0i])
          print(T_0i, cam_dict[T_0i])
    return cam_dict

def loadCalibrationRigid(filename, verbose=False):
    assert type(filename) is str
    velo_dict = read_files_by_lines(filename)

    for key, value in velo_dict.items():
        if key == 'calib_time':
            velo_dict[key] = value
        else:
            array = []
            for i, string in enumerate(value.split(' ')[1:]):
                array.append(float(string))
            velo_dict[key] = array

    R = 'R'
    T = 'T'
    velo_dict[R] = replace_var_from_dict_with_shape(velo_dict, R, (3, 3))
    velo_dict[T] = replace_var_from_dict_with_shape(velo_dict, T, (3, 1))
    # Tr = [R, T; 0 0 0 1]
    Tr = np.vstack((np.hstack((velo_dict[R], velo_dict[T])), [0, 0, 0, 1]))
    velo_dict['Tr'] = Tr

    if verbose:
      print(R, velo_dict[R])
      print(T, velo_dict[T])
      print('Tr', velo_dict['Tr'])
    return velo_dict['Tr']

# TODO: Limit to 2D matrix
def project(p_in, T):
#   Dimension of data projection matrix
#    assert type(T) == 'numpy.ndarray'
#    assert type(p_in) == 'numpy.ndarray'
    dim_norm, dim_proj = T.shape

    p_in_row_count = p_in.shape[0]
#   Do transformation in homogenouous coordinates
    p2_in = p_in
    if p2_in.shape[1] < dim_proj:
        col_ones = np.ones(p_in_row_count)
        col_ones.shape = (p_in_row_count, 1)
# matlab:       p2_in[:, dim_proj - 1] = 1
        p2_in = np.hstack((p2_in, col_ones))
#   (T*p2_in')'
    p2_out = np.transpose(np.dot(T, np.transpose(p2_in)))
#   Normalize homogeneous coordinates
    denominator = np.outer(p2_out[:, dim_norm - 1], np.ones(dim_norm - 1))
#   Element wise division
    p_out = p2_out[:, 0: dim_norm-1]/denominator
    return p_out

#main 3d points to camera projection function

get_ipython().magic('matplotlib inline')
l_and = lambda *x: np.logical_and.reduce(x)

def convert_velo_cord_to_img(data_set, calib_dir,num_points = 5, cam=2, frame=0,tracklet = False):
    """
    Demostrates projection of the velodyne points into the image plane
    Parameters
    ----------
    dataset = data_set_velo
    base_dir  : Absolute path to sequence base directory (ends with _sync)
    calib_dir : Absolute path to directory that contains calibration files
    Returns
    -------
    """
    calib = loadCalibrationCamToCam(calib_dir + 'calib_cam_to_cam.txt')
    Tr_velo_to_cam = loadCalibrationRigid(calib_dir + 'calib_velo_to_cam.txt')

#     Compute projection matrix velodyne->image plane
    R_cam_to_rect = np.eye(4, dtype=float)
    R_cam_to_rect[0: 3, 0: 3] = calib['R_rect_00']
    P_velo_to_img = np.dot(np.dot(calib['P_rect_0' + str(cam)], R_cam_to_rect), Tr_velo_to_cam)

    print(frame)

    if tracklet: 
        velo_data = data_set
        velo = velo_data
    else:
        velo_data = data_set[frame]
        #print("velo_data shape before %", velo_data.shape)
        velo = velo_data[0:velo_data.shape[0]:num_points]
        #print("velo_data shape after %", velo.shape)
    
    #img_h, img_w, img_ch = dataset_rgb[frame].right.shape
    img_h, img_w, img_ch = 400,1500,3
    
    img_plane_depth = 5
    x_dir_pts = velo[:, 0]
    filtered_x_dir_indices = l_and((x_dir_pts > img_plane_depth))
#     .flatten to remove extra dimension
    indices = np.argwhere(filtered_x_dir_indices).flatten()
#     Depth (x) limited velodyne points
    velo = velo[indices, :]
#     Project to image plane (exclude luminance/intensity)
    velo_img = project(velo[:, 0:3], P_velo_to_img)
    
    if tracklet:
        return velo_img
    else:
        return velo_img,velo

#main 3d points to camera projection function

def crop_velo_to_img_size(img_shape, calib_velo_data,velo_data_raw,include_z = False):
    """
    Parameters:
    ----------
    img_size: camera image size
    velo_data :calibrated and project transformed lidar to camera data
    """
    img_h = img_shape[0]
    img_w = img_shape[1]
    print("crop_velo velo_data %",calib_velo_data[:,0])
    img_dim_x_pts = calib_velo_data[:, 0]
    img_dim_y_pts = calib_velo_data[:, 1]
        
    x_filt = l_and((img_dim_x_pts < img_w), (img_dim_x_pts >= 0))
    y_filt = l_and((img_dim_y_pts < img_h), (img_dim_y_pts >= 0))
    filtered = l_and(x_filt, y_filt)
    indices = np.argwhere(filtered).flatten()
    
    img_dim_x_pts = img_dim_x_pts[indices]
    img_dim_y_pts = img_dim_y_pts[indices]
    
    if include_z:
        img_dim_z_pts = velo_data_raw
        
        print("indices %",indices.shape)
        
        img_dim_z_pts = img_dim_z_pts[indices]
        #euclidean_distance = np.sqrt(np.add(np.square(img_dim_z_pts[:,0]), np.square(img_dim_z_pts[:,1]),np.square(img_dim_z_pts[:,2])))
        
        distance = img_dim_z_pts[:,0]
        
        #print("euclidean_distance %",euclidean_distance.shape)
        #print("euclidean_distance %",euclidean_distance.shape)
        
        print("using xaxis")
        #return (img_dim_x_pts, img_dim_y_pts,euclidean_distance)
        return (img_dim_x_pts, img_dim_y_pts,distance)
    
    return (img_dim_x_pts, img_dim_y_pts)

import sys
EPSILON = sys.float_info.epsilon  # smallest possible difference

def convert_to_rgb(minval, maxval, val, colors):
    #print("colors range shape %",colors)
    fi = float(val-minval) / float(maxval-minval) * (len(colors)-1)
    i = int(fi)
    f = fi - i

    (r1, g1, b1), (r2, g2, b2) = colors[1], colors[2]
    return int(r1 + f*(r2-r1)), int(g1 + f*(g2-g1)), int(b1 + f*(b2-b1))

# Submissions #1  - 3d cloud point on camera

dataset_velo = list(dataset.velo)
dataset_rgb = list(dataset.rgb)
velodyne_max_x=100
frame = 0
include_z = True
radius = 2
calib_dir ="/Users/davidwoo/Documents/Projects/self-driving-cars/plane-segmentation/KITTI-Dataset/2011_09_26/"

velo_data,velo_data_raw_sampled = convert_velo_cord_to_img(dataset_velo, calib_dir)
print(velo_data.shape)

rgb_img = dataset_rgb[frame][frame]
#corped_velo_data = crop_velo_to_img_size([400,1500,3], velo_data,dataset_velo[0],include_z)
corped_velo_data = crop_velo_to_img_size([400,1500,3], velo_data,velo_data_raw_sampled,include_z)
print("corped_velo_data shape %",type(corped_velo_data))
#corped_velo_data = velo_data


import cv2
import statistics
print(cv2.addWeighted)
def overlay_velo_img(img, velo_data,radius = 2):
    (x, y,z) = velo_data
    im = np.zeros(img.shape, dtype=np.float32)
    x_axis = np.floor(x).astype(np.int32)
    y_axis = np.floor(y).astype(np.int32)
    z_axis = np.floor(z).astype(np.int32)
    print("zaxis %",max(z_axis))
    print("zaxis %",min(z_axis))
    print("zaxis %",type(z_axis))
    print(z_axis)
    color_list = []

    # below draws circles on the image
    for i in range(0, len(x)):
        # img, center, radius, setting array [color,thickness,linetype]
        #print("iteration % %",i, z_axis[i])
        #print(type(int(z_axis[i])))
        #print("zaxis %,",z_axis[i])
        if z_axis[i] <= 0:
            color = int(1/velodyne_max_x * 256)
            value = 0
        else:
            color = int((z_axis[i])/velodyne_max_x * 256)
            value = z_axis[i] * 4
        colors_range = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # [BLUE, GREEN, RED]
        r, g, b  = convert_to_rgb(0,256,value,colors_range) 
        
        cv2.circle(img, (x_axis[i], y_axis[i]), radius, [r, g, b],-1)
        color_list.append(color)
    
    print("color list median % min % max %,",statistics.median(color_list),min(color_list),max(color_list))
    fig1 = plt.figure(figsize=(20, 20))
    
    return img

result_img = overlay_velo_img(rgb_img, corped_velo_data,radius)
print(corped_velo_data)
plt.imshow(result_img)

def convert_list_array(input_list,frame):
    #len(tracklet_rects[0])
    #print(tracklet_rects[0])

    #test = []
    iter_num = 0
    for tracklet in input_list[frame]:
        #print("tracket %",tracklet)
        if iter_num == 0:
            tracklet_rects_array = tracklet.transpose(1,0)
        else:
            #test.append(tracklet.transpose(1,0))
            tracklet_rects_array = np.concatenate((tracklet_rects_array, tracklet.transpose(1,0)), axis=0)

        iter_num = iter_num + 1

    #print(tracklet_rects_array)
    #type(tracklet_rects_array)
    
    return tracklet_rects_array

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
    
    velo_data_tracklets = convert_velo_cord_to_img(dataset_tracklets, calib_dir,7, 2, 0,True)
    
    #print("print tracklets %",dataset_tracklets)
    #print("print tracklets %",velo_data_tracklets[:, 0])

    corped_velo_data_tracklets = crop_velo_to_img_size([400,1500,3], velo_data_tracklets,dataset_velo[0])

    test = np.array([corped_velo_data_tracklets[0],corped_velo_data_tracklets[1]])
    
    #print(test)
    #print("length of corped_velo_data_tracklets %", len(corped_velo_data_tracklets[0]))

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
    #vertices
    #print("vertices %",vertices[:,:])
    #print("vertices %",max(vertices[0,:,1]))
    #print("rec vertices % %",(min(vertices[0,:,0]),min(vertices[0,:,1])),(max(vertices[0,:,0]),max(vertices[0,:,1])))
    #cv2.drawContours(rgb_img, vertices,-1, (0,255,0), 3)
    
    # image, rect points, color, thickness
    cv2.rectangle(rgb_img, 
                  (min(vertices[0,:,0]),min(vertices[0,:,1])),(max(vertices[0,:,0]),max(vertices[0,:,1])), 
                  (100,255,100), 3)
    
    tracklet = tracklet + 1
    
    #break

fig1 = plt.figure(figsize=(20, 20))
plt.imshow(rgb_img)

### Submission #3 This is start of ground plane segmentation

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


#load the data
frame = 0
dataset_velo = list(dataset.velo)
points = 0.8 # this controls the sampling rate
points_step = int(1. / points)
point_size = 0.01 * (1. / points)

velo_range = range(0, dataset_velo[10].shape[0], points_step)
velo_frame = dataset_velo[frame][velo_range, :] 
velo_frame_test = velo_frame
velo_frame_test[:, 3]=velo_frame[:, 3]*10
velo_range = range(0, dataset_velo[frame].shape[0], points_step)
velo_frame = dataset_velo[frame][velo_range, :]  
velo_frame = velo_frame_test

velo_frame[velo_frame[:, 2]>-1.7,3] = 5
velo_frame[velo_frame[:, 2]<-1.7,3] = 0


import random

def run_ransac(data_raw, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    data = data_raw[:,0:3]
    for i in range(max_iterations):
        #s = random.sample(data, int(sample_size))
        s = data[np.random.randint(data.shape[0], size=sample_size), :]
        #print("shape s: %",s.shape)
        m = estimate(s)
        ic = 0
        for j in range(len(data)):
            if is_inlier(m, data[j]):
                ic += 1
                #mark point as inline
                data_raw[j][3] = 0
                
            else:
                data_raw[j][3] = 5

        #print (s)
        print ('estimate:', m)
        print ('# inliers:', ic)

        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print ('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic,data_raw


import numpy as np
from matplotlib import pyplot as plt
#from ransac import *


def augment(xyzs):
    axyz = np.ones((len(xyzs), 4))
    axyz[:, :3] = xyzs
    return axyz


def estimate(xyzs):
    axyz = augment(xyzs[:3])
    return np.linalg.svd(axyz)[-1][-1, :]


def is_inlier(coeffs, xyz, threshold):
	return np.abs(coeffs.dot(augment([xyz]).T)) < threshold
	
	
if __name__ == '__main__':
	from matplotlib import pylab
	from mpl_toolkits import mplot3d
	fig = pylab.figure()
	ax = mplot3d.Axes3D(fig)
	
	
def plot_plane(a, b, c, d):
    #xx, yy = np.mgrid[:10, :10]
    xx, yy = np.mgrid[-20:80, -20:20]
    return xx, yy, (-d - a * xx - b * yy) / c
		
		
#n = 100
n = velo_frame.shape[0] * 0.1
max_iterations = 100
goal_inliers = n * 0.5
print("goal_inliers %",goal_inliers)

get_ipython().magic('matplotlib inline')
#m, b, adj_velo_frame = run_ransac(velo_frame[:,0:3], estimate, lambda x, y: is_inlier(x, y, 0.01), int(velo_frame.shape[0] * 0.1), goal_inliers, max_iterations)
m, b, adj_velo_frame = run_ransac(velo_frame, estimate, lambda x, y: is_inlier(x, y, 0.01), int(velo_frame.shape[0] * 0.1), goal_inliers, max_iterations)
a, b, c, d = m
xx, yy, zz = plot_plane(a, b, c, d)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_zlim3d(-2, 10)
ax.plot_surface(xx, yy, zz, color=(0, 1, 0, 0.5))


print("plotting chart")
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
draw_point_cloud_seg(adj_velo_frame,ax2,'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right')
plt.show()


f, ax3 = plt.subplots(1, 1, figsize=(15, 12))
draw_point_cloud_seg(adj_velo_frame,
    ax3, 
    'Velodyne scan, XY projection (Z = 0), the car is moving in direction left to right', 
    axes=[0, 1] # X and Y axes
)
plt.show()

