get_ipython().magic('reload_ext autoreload')
get_ipython().magic('autoreload 2')
import cv2
import numpy as np
import augmentation as aug
import os
import math
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation
# Config the matlotlib backend as plotting inline in IPython
get_ipython().magic('matplotlib inline')

#image_folders = ['data/dataset-20160929/center_camera', 'data/dataset-20160929/left_camera', 'data/dataset-20160929/right_camera']
image_folders = ['data/train/1/center', 'data/train/1/left', 'data/train/1/right']
images = os.listdir(image_folders[0])

img = cv2.imread(os.path.join(image_folders[0], np.random.choice(images)))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

print('Horizon y: {}'.format(aug.get_horizon_y(img, min_y=200, max_y=300, draw=True)))

distorted = aug.apply_distortion(img, 0.01, 0.1, crop_y=240, draw=True)

def get_animation(frames, file=None, fps=20, repeat_delay=1000):
    fig = plt.figure()
    ims = []

    for f in frames:
        im = plt.imshow(f, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=(1/fps)*1000, repeat_delay=repeat_delay)
    
    if file is not None:
        ani.save('rotation_shift.mp4')
    
    return ani

angs = np.append(np.linspace(0., -0.01, 20), np.linspace(-0.01, 0., 20))
angs = np.append(angs, np.linspace(0., 0.01, 20))
angs = np.append(angs, np.linspace(0.01, 0., 20))

ani = get_animation([aug.apply_distortion(img, ang, 0) for ang in angs])
HTML(ani.to_html5_video())

shifts = np.append(np.linspace(0, -0.1, 20), np.linspace(-0.1, 0, 20))
shifts = np.append(shifts, np.linspace(0, 0.1, 20))
shifts = np.append(shifts, np.linspace(0.1, 0, 20))

ani = get_animation([aug.apply_distortion(img, 0, shift) for shift in shifts])
HTML(ani.to_html5_video())

ani = get_animation([aug.apply_distortion(img, angs[i], shifts[i]) for i in range(len(shifts))])
HTML(ani.to_html5_video())

fig = plt.figure(figsize=(12, 6))
fig.add_subplot(221, title="Original")
plt.imshow(aug.apply_distortion(img, 0, 0))

for i in range(2,5):
    distorted, rotation, shift = aug.random_distortion(img)
    fig.add_subplot(2, 2, i, title="rotation: {:.3f}; shift: {:.3f}".format(rotation, shift))
    plt.imshow(distorted)

#image_folders = ['data/center_camera', 'data/left_camera', 'data/right_camera']
images = os.listdir(image_folders[0])
choice = np.random.choice(images)
img_center = cv2.imread(os.path.join(image_folders[0], choice))
img_left = cv2.imread(os.path.join(image_folders[1], choice))
img_right = cv2.imread(os.path.join(image_folders[2], choice))
img_center = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)
img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(12, 12))
fig.add_subplot(1, 3, 1)
plt.imshow(apply_distortion(img_left, 0, 100, crop=False))
fig.add_subplot(1, 3, 2)
plt.imshow(img_center)
fig.add_subplot(1, 3, 3)
plt.imshow(apply_distortion(img_right, 0, -60, crop=False))

import pandas as pd

center_steering = pd.read_csv('data/dataset-20160929/center_steering.csv')
left_steering = pd.read_csv('data/dataset-20160929/left_steering.csv')

def imread_rgb(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#i = 100
i = np.random.choice(range(500))
fps = 20

imgs_left = []        
f = 0
images = os.listdir(image_folders[1])
while f <= 2 * fps:    
    img_left = imread_rgb(os.path.join(image_folders[1], images[i + f]))
    imgs_left.append(aug.apply_distortion(img_left, 0, 0))

    f = f + 1

i = i + f
    
shift0 = -.5
rotation0 = 0.
shift = shift0
rotation = rotation0

while abs(shift) > 0.005 or abs(rotation) > 0.001:

    img_left = imread_rgb(os.path.join(image_folders[1], images[i]))
    
    speed = left_steering.iloc[i].speed
    steering_wheel_angle = left_steering.iloc[i].steering_wheel_angle
    rotation, shift, steering_wheel_angle = aug.get_steer_back_angle(steering_wheel_angle, speed, rotation, shift)
    imgs_left.append(aug.apply_distortion(img_left, (rotation - rotation0), (shift - shift0)))
    
    if i % 10 == 0:
        print(shift, rotation, steering_wheel_angle, left_steering.iloc[i].steering_wheel_angle)
        
    i = i + 1
        
ani = get_animation(imgs_left, fps=fps)
HTML(ani.to_html5_video())

ani.save('steering.mp4')

images = os.listdir(image_folders[0])
choice = np.random.choice(range(len(images)))
img_center = cv2.imread(os.path.join(image_folders[0], images[choice]))
image = cv2.cvtColor(img_center, cv2.COLOR_BGR2RGB)

fig = plt.figure(figsize=(12, 7))
fig.add_subplot(221, title="Original")
plt.imshow(aug.apply_distortion(image, 0, 0))

for i in range(2,5):
    speed = left_steering.iloc[choice].speed
    
    if speed == 0:
        continue
        
    original_steering_wheel_angle = left_steering.iloc[choice].steering_wheel_angle
    distorted, steering_wheel_angle, rotation, shift = aug.steer_back_distortion(image, 
                                                                             original_steering_wheel_angle,
                                                                             speed)
    fig.add_subplot(2, 2, i, title="original steering angle: {:.3f}; steering angle: {:.3f}\n rotation: {:.3f}; shift: {:.3f}"
                    .format(original_steering_wheel_angle, steering_wheel_angle, rotation, shift))
    plt.imshow(distorted)



