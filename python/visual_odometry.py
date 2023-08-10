from base import *

from shapely import geometry

#images_folder_path = './data/fisheye_camera_calibration/'

images_folder_path = '/home/wroscoe/d2/data/tub_11_17-09-16/'
img_list = glob.glob(os.path.join(images_folder_path,'*.jpg'))
print(len(img_list))

#sort list by frame number 
img_list = sorted(img_list, key=lambda x: int(x.split('/')[-1].split('_')[0]))
img_list[:10]

#Pull in the camera properties we calculated in the calibrate_camera notebook
w, h = img_size = default.camera_img_size
mtx = default.camera_mtx

focal_length = fx, fy = mtx[0,0], mtx[1,1]
print('focal length: {}'.format(focal_length))

optical_axis_center = cx, cy = mtx[0,2], mtx[1,2]
print('optical_axis_center: {}'.format(optical_axis_center))

k1, k2, k3, k4 = default.camera_dist 
print('distortion coefficients')
print(default.camera_dist)

def get_key_points(img):
    #first frame
    detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
    prev_points = detector.detect(img)
    points = np.array([x.pt for x in prev_points], dtype=np.float32)
    return points

def track_key_points(prev_img, next_img, prev_points):
    
    lk_params = dict(winSize  = (21, 21),
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_img, 
                                                          next_img, 
                                                          prev_points, 
                                                          None, 
                                                          winSize=(21, 21),
                                                          criteria =(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                                                         )
    status = status.reshape(status.shape[0])
    prev_points = prev_points[status == 1]
    next_points = next_points[status == 1]

    return prev_points, next_points

prev_img = cv2.imread(img_list[0], 0)
prev_points = get_key_points(prev_img)

next_img = cv2.imread(img_list[1], 0)

plot_data = [{'name': 'prev', 'array': prev_img},
             {'name': 'next', 'array': next_img}]
show_imgs(plot_data)

prev_points, next_points = track_key_points(prev_img, next_img, prev_points)
len(prev_points)

def calculate_possition_change(prev_img, next_img, prev_points, next_points, 
                               fx, optical_axis_center):
    """
    Calculate the relative rotaion and translation between images.
    """
    
    #filter points to only those trackable between images
    prev_points, next_points = track_key_points(prev_img, next_img, prev_points)
    
    
    E, mask = cv2.findEssentialMat(prev_points, 
                                   next_points, 
                                   focal=fx, 
                                   pp=optical_axis_center, 
                                   method=cv2.RANSAC, 
                                   prob=0.999, 
                                   threshold=1.0)
    
    _, R, t, mask = cv2.recoverPose(E, 
                                    next_points, 
                                    prev_points, 
                                    focal=fx, 
                                    pp = optical_axis_center)
    
    
    return R, t #relative rotation and translation

R, t = calculate_possition_change(prev_img, next_img, prev_points, next_points,
                                  fx, optical_axis_center)
print('relative rotation matrix:')
print((np.round(R, 3)))

print('relative translation: {}')
print(t)

def update_possition(prev_R, prev_t, R, t):

    #absolute_scale = self.getAbsoluteScale(frame_id)
    absolute_scale = 1
    next_t = prev_t + absolute_scale*prev_R.dot(t) 
    next_R = R.dot(prev_R)
    
    return next_t, next_R

prev_t = np.zeros((3, 1))
#prev_R = np.zeros((3,3))  + 1

t_log = [prev_t]

#first frame
prev_img = cv2.imread(img_list[0])
prev_points = get_key_points(prev_img)


#second_frame
next_img = cv2.imread(img_list[1])
prev_points, next_points = track_key_points(prev_img, next_img, prev_points)

#for the second frame, the relative change is the absolute change
prev_R, prev_t = calculate_possition_change(prev_img, next_img, prev_points, next_points,
                                  fx, optical_axis_center)


#log the translation
t_log.append(prev_t)

#move next to previous
prev_img = next_img
prev_points = next_points


for i, img in enumerate(img_list[2:440]):
    next_img = cv2.imread(img)
    prev_points, next_points = track_key_points(prev_img,next_img, prev_points)
    #print(len(prev_points))
    R, t = calculate_possition_change(prev_img, next_img, prev_points, next_points,
                                  fx, optical_axis_center)

    prev_t, prev_R = update_possition(prev_R, prev_t, R, t)

    t_log.append(prev_t)
    
    if(len(next_points) < 100):
        next_points  = get_key_points(next_img)

    #move next to previous
    prev_img = next_img
    prev_points = next_points

t_arr = np.array(t_log)
t_arr = t_arr.reshape(t_arr.shape[:2])
df = pd.DataFrame(t_arr, columns=['x', 'y', 'z'])
df.head()

coords = [(arr[0], arr[1]) for arr in t_arr]
geometry.LineString(coords)

import shutil

for i, src_path in enumerate(img_list[:470]):
    dest_path = os.path.join('/home/wroscoe/code/dk_notebooks/data/one_lap/', str(i).zfill(4)+'.jpg')
    shutil.copyfile(src_path, dest_path)





