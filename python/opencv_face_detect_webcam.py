from pynq import Overlay
Overlay("base.bit").download()

# monitor configuration: 640*480 @ 60Hz
from pynq.drivers import HDMI
from pynq.drivers.video import VMODE_640x480
hdmi_out = HDMI('out', video_mode=VMODE_640x480)
hdmi_out.start()

# monitor (output) frame buffer size
frame_out_w = 1920
frame_out_h = 1080
# camera (input) configuration
frame_in_w = 640
frame_in_h = 480

# initialize camera from OpenCV
from pynq.drivers import Frame
import cv2

videoIn = cv2.VideoCapture(0)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);

print("Capture device is open: " + str(videoIn.isOpened()))

# Capture webcam image
import numpy as np

ret, frame_vga = videoIn.read()

# Display webcam image via HDMI Out
if (ret):
    frame_1080p = np.zeros((1080,1920,3)).astype(np.uint8)       
    frame_1080p[0:480,0:640,:] = frame_vga[0:480,0:640,:]
    hdmi_out.frame_raw(bytearray(frame_1080p.astype(np.int8).tobytes()))
else:
    raise RuntimeError("Failed to read from camera.")

# Output webcam image as JPEG
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import numpy as np
plt.imshow(frame_vga[:,:,[2,1,0]])
plt.show()

import cv2

np_frame = frame_vga

face_cascade = cv2.CascadeClassifier(
                        './data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(
                        './data/haarcascade_eye.xml')

gray = cv2.cvtColor(np_frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(np_frame,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = np_frame[y:y+h, x:x+w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# Output OpenCV results via HDMI
frame_1080p[0:480,0:640,:] = frame_vga[0:480,0:640,:]
hdmi_out.frame_raw(bytearray(frame_1080p.astype(np.int8).tobytes()))

# Output OpenCV results via matplotlib
get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import numpy as np
plt.imshow(np_frame[:,:,[2,1,0]])
plt.show()

videoIn.release()
hdmi_out.stop()
del hdmi_out



