import sys
import cv2
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.pyplot as plt
# display plots in this notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# filter out the warnings
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '../')

from gaze_detector import extract_features_and_detect_gazes
from features import extract_image_features, draw_detected_features
from gaze import test_faces

img = cv2.imread('photos/frame-36.png')

img, faces, face_features = extract_image_features(img)

image_copy = np.copy(img)

draw_detected_features(image_copy, faces, face_features)

plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))

from lib import crop_image

def set_title_and_hide_axis(title):
    plt.title(title)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)

def render_face_grid(face_grid):
    to_print = np.copy(face_grid)
    result_image = np.copy(to_print).reshape(25, 25).transpose()
    plt.figure()
    set_title_and_hide_axis('Face grid')
#     print(result_image.shape)
    plt.imshow(result_image)

def show_extraction_results(img, faces, face_features):
    plt.figure(figsize=(10,10))
#     set_title_and_hide_axis('Original image and extracted features')
    image_copy = np.copy(img)

    draw_detected_features(image_copy, faces, face_features)
    plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB), interpolation="bicubic")

    for i, face in enumerate(faces):
        print('Face #' + str(i))
        #print('i', face, i)
        eyes, face_grid = face_features[i]
        plt.figure()
        set_title_and_hide_axis('Extracted face image')
        plt.imshow(cv2.cvtColor(crop_image(img, face), cv2.COLOR_BGR2RGB), interpolation="bicubic")
        plt.figure()
        #print('face image after extraction')
        render_face_grid(face_grid)

        for i, eye in enumerate(eyes):
            plt.figure()

            if i == 0:  
                set_title_and_hide_axis('Extracted left eye image')
            else:  
                set_title_and_hide_axis('Extracted right eye image')
                
            plt.imshow(cv2.cvtColor(crop_image(img, eye), cv2.COLOR_BGR2RGB), interpolation="bicubic")


def render_gaze_plot(outputs):
    plt.figure(figsize=(10,10))
    circles = []
    circles.append(plt.Circle((0, 0), 0.1, color='b'))

    for output in outputs:
        circles.append(plt.Circle((output[0], output[1]), 0.5, color='r'))

    fig, ax = plt.subplots()
    for circle in circles:
        ax.add_artist(circle)
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_xlabel("Distance from Camera (cm)")
    ax.set_ylabel("Distance from Camera (cm)")
    
    fig.show()

img = cv2.imread('photos/IMG-1053.JPG')
img, faces, face_features = extract_image_features(img)
outputs = test_faces(img, faces, face_features)
render_gaze_plot(outputs)
show_extraction_results(img, faces, face_features)

caffe.set_device(0)
caffe.set_mode_gpu()

#testing with camera
cap = cv2.VideoCapture(0)
cap.set(3, 1920)
cap.set(4, 1200)

ret, frame = cap.read()
cap.release()

img, faces, face_features = extract_image_features(frame)

outputs = test_faces(frame, faces, face_features)
print(outputs)
render_gaze_plot(outputs)
show_extraction_results(img, faces, face_features)



print(outputs)
circles = []
circles.append(plt.Circle((0, 0), 0.1, color='b'))

for output in outputs:
    circles.append(plt.Circle((output[0], output[1]), 0.5, color='r'))


fig, ax = plt.subplots()
for circle in circles:
    ax.add_artist(circle)
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_xlabel("Distance from Camera (cm)")
ax.set_ylabel("Distance from Camera (cm)")
fig.show()

outputs = test_faces(img, faces, face_features)
print("The outputs:", outputs)



# Test performance on CPU
caffe.set_mode_cpu()
outputs = test_faces(faces, face_features)

get_ipython().run_line_magic('timeit', 'test_faces(faces, face_features)')

# run this if cuda enable and gpu has enough memory
caffe.set_mode_gpu()
caffe.set_device(0)  # if we have multiple GPUs, pick the first one

# run once to upload the network to gpu
outputs = test_faces(faces, face_features)

# then timeit
get_ipython().run_line_magic('timeit', 'test_faces(faces, face_features)')

# units in cm
screen_w = 5.58
screen_h = 10.45
screen_aspect = screen_w / screen_h
camera_l = 2.299
camera_t = 0.91
screen_t = 1.719
screen_l = 0.438
phone_w = 6.727
phone_h = 13.844
screen_from_camera = [screen_t - camera_t, screen_l - camera_l]

camera_coords_percentage = [camera_t / phone_h, camera_l / phone_w]

#iphone 8 screen w and screen height from https://www.paintcodeapp.com/news/ultimate-guide-to-iphone-resolutions
screenW = 375
screenH = 667

phone_w_to_screen = phone_w / screen_w
phone_h_to_screen = phone_h / screen_h

def render_gaze(full_image, camera_center, cm_to_px, output):
    xScreen = output[0]
    yScreen = output[1]
    pixelGaze = [round(camera_center[0] - yScreen * cm_to_px), round(camera_center[1] + xScreen * cm_to_px)]
    
    cv2.circle(full_image,(int(pixelGaze[1]), int(pixelGaze[0])), 30, (0, 0, 255), -1)

    
def render_gazes(img, outputs):
    full_image = np.ones((round(img.shape[0] * 2), round(img.shape[1] * 2), 3), dtype=np.uint8)

    full_image_center = [round(full_image.shape[0] * 0.2), round(full_image.shape[1] *.5)]
    camera_center = full_image_center

    cm_to_px = img.shape[0] * 1. / screen_h

    screen_from_camera_px = [round(screen_from_camera[0] * cm_to_px), round(screen_from_camera[1] * cm_to_px)]

    screen_start = [camera_center[0] + screen_from_camera_px[0], camera_center[1] + screen_from_camera_px[1]]
    
    full_image[screen_start[0]:screen_start[0] + img.shape[0], screen_start[1]:screen_start[1] + img.shape[1], :] = img[:, :, :]

    cv2.circle(full_image,(camera_center[1],camera_center[0]), 30, (255, 0, 0), -1)
    
    for output in outputs:
        render_gaze(full_image, camera_center, cm_to_px, output)

    plt.figure(figsize=(10,10))
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.imshow(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB), interpolation="bicubic")    

render_gazes(img, outputs)

# lets create a reusable function to extract the features, pass through the network, and render output
def test_and_render(image_path, show_details=False):
    img, faces, face_features = extract_image_features(cv2.imread(image_path))
    outputs = test_faces(img, faces, face_features)

    if show_details:        
        show_extraction_results(img, faces, face_features)

    render_gazes(img, outputs)

test_and_render('photos/IMG-1066.JPG')

test_and_render('photos/IMG-1036.JPG')

test_and_render('photos/IMG-1037.JPG')

test_and_render('photos/IMG-1038.JPG')

test_and_render('photos/IMG-1044.JPG')

test_and_render('photos/IMG-1052.JPG')

test_and_render('photos/IMG-1053.JPG')

test_and_render('photos/IMG-1054.JPG')

test_and_render('photos/IMG-1055.JPG')

