#
# FileName        :    PYNQ_Animoji.py
# Description     :    This file contains multithreaded file I/O implementation 
#                        
# File Author Name:    Bhallaji Venkatesan, Divya Sampath, Mounika Reddy, Sahana Sadagopan 
# Tools used      :    Jupyter Notebooks, 
# References      :    www.pynq.io 
#				  :	   http://docs.opencv.org/2.4/doc/tutorials/tutorials.html
#                 :    https://azure.microsoft.com/en-us/services/cognitive-services/emotion/
#                   
#
#
#/


from pynq.overlays.base import BaseOverlay
from pynq.lib.video import *
from matplotlib import pyplot as plt
import numpy as np
#import Image 
from PIL import Image 
orig_img_path = '/home/xilinx/jupyter_notebooks/base/video/mj2.jpeg'
base = BaseOverlay("base.bit")

import requests 
import json
import os
import re
import numpy as np
url_Microsoft_Cog_Services = 'https://westus.api.cognitive.microsoft.com/emotion/v1.0/recognize?'
Image_url = 'frame.jpg'
#Header for Microsoft Cognitive Services requests
headers = {
    # Request headers #To confirm octet or json
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': 'a6a9555023d74986a23d4a6935274f88',
}

#Parameters for Microsoft Cognitive Services requests
params =  {
}

# monitor configuration: 640*480 @ 60Hz
Mode = VideoMode(640,480,24)
hdmi_out = base.video.hdmi_out

hdmi_out.configure(Mode,PIXEL_BGR)
hdmi_out.start()

# monitor (output) frame buffer size
frame_out_w = 1920
frame_out_h = 1080
# camera (input) configuration
frame_in_w = 640
frame_in_h = 480

# initialize camera from OpenCV
import cv2

videoIn = cv2.VideoCapture(0)
videoIn.set(cv2.CAP_PROP_FRAME_WIDTH, frame_in_w);
videoIn.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_in_h);

print("Capture device is open: " + str(videoIn.isOpened()))

# Capture webcam image
import numpy as np
import PIL
import pyscreenshot as ImageGrab
from pynq.lib.arduino import Grove_Buzzer
from pynq.lib.arduino import ARDUINO_GROVE_G1

# Face Recognition with face demarcation
while(True):
    ret, frame_vga = videoIn.read()
    if (ret):      
        np_frame = frame_vga
        face_cascade = cv2.CascadeClassifier(
            '/home/xilinx/jupyter_notebooks/base/video/data/'
            'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(np_frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(np_frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = np_frame[y:y+h, x:x+w]

        outframe = hdmi_out.newframe()
        outframe[0:480,0:640,:] = np_frame[0:480,0:640,:]
        hdmi_out.writeframe(outframe)
        cv2.imwrite("frame.jpg",frame_vga)
        

#Capturing Image of the requested Item
#Converting the Image into a readable binary file for passing to the Microsoft Cognitive Services
        with open (Image_url, 'rb') as f:
            data2 = f.read()
#Http Post request to the Microsoft Cognitive Services 	
        response= requests.request('post',url_Microsoft_Cog_Services,json =json, data = data2, headers = headers, params = params)
        print(response)
        data = response.json()
        print(data)
#Searching for Pattern of the required items in the image response from cognitive services

        array = np.zeros(8)
        array[0] = data[0]['scores']['anger']
        array[1] = data[0]['scores']['contempt']
        array[2] = data[0]['scores']['disgust']
        array[3] = data[0]['scores']['fear']
        array[4] = data[0]['scores']['happiness']
        array[5] = data[0]['scores']['neutral']
        array[6] = data[0]['scores']['sadness']
        array[7] = data[0]['scores']['surprise']
        max_value = array[0]
        max_index = 0


        for i in range(0,7):
            if array[i]>max_value:
                max_value = array[i]
                max_index = i
            else:
                pass
#Emoji Implementation             
        if(max_index == 0):
            print("THE PERSON IS REALLY REALLY ANGERY")
            print("Probably not using PYNQ")
            get_ipython().run_line_magic('matplotlib', 'inline')
            n_plot = cv2.imread("Angry.jpg")
            from matplotlib import pyplot as plt
            import numpy as np
            plt.imshow(n_plot[:,:,[2,1,0]])
            plt.show()
        if(max_index == 1):
            print("THE PERSON IS CONTEMPT")
        if(max_index == 2):
            print("THE PERSON IS REALLY DISGUSTED")
        if(max_index == 3):
            print("THE PERSON IS SCARED")
        if(max_index == 4):
            # Output webcam image as JPEG
            get_ipython().run_line_magic('matplotlib', 'inline')
            n_plot = cv2.imread("Happy.jpg")
            from matplotlib import pyplot as plt
            import numpy as np
            plt.imshow(n_plot[:,:,[2,1,0]])
            plt.show()
            print("The Person is happy! THEY MUST BE USING PYNQ BOARD")
        if(max_index == 5):
            print("THE PERSON IS NEUTRAL")
            get_ipython().run_line_magic('matplotlib', 'inline')
            n_plot = cv2.imread("Neutral.jpg")
            from matplotlib import pyplot as plt
            import numpy as np
            plt.imshow(n_plot[:,:,[2,1,0]])
            plt.show()
        if(max_index == 6):
            print("THE PERSON IS SAD")
            get_ipython().run_line_magic('matplotlib', 'inline')
            n_plot = cv2.imread("sad.jpg")
            from matplotlib import pyplot as plt
            import numpy as np
            plt.imshow(n_plot[:,:,[2,1,0]])
            plt.show()
        if(max_index ==7):
            print("THE PERSON IS SURPRISED")
            get_ipython().run_line_magic('matplotlib', 'inline')
            n_plot = cv2.imread("Surprised.jpg")
            from matplotlib import pyplot as plt
            import numpy as np
            plt.imshow(n_plot[:,:,[2,1,0]])
            plt.show()
        print(max_index)
        #cv2.waitKey(50)
        
    else:
        raise RuntimeError("Failed to read from camera.")
            

videoIn.release()
hdmi_out.stop()
del hdmi_out

