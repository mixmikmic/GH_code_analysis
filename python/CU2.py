# In this cell, some libraries were imported
import cv2
import sys
import os
from PIL import Image, ImageDraw
import pylab
import time

#from PIL import Image as PIL_Image
from pynq.overlays.base import BaseOverlay
base = BaseOverlay("base.bit")
def capture(destination, image_nbumber):
    orig_img_path = destination + str(image_number)+'.JPG'
    get_ipython().system('fswebcam  --no-banner --save {orig_img_path} -d /dev/video0 2> /dev/null')
    return

image_number = 1
while(base.buttons[0].read()==0):
    capture(filepath, image_number)
    image_number += 1

# Face Detection Function
def detectFaces(image_name):
    print ("Face Detection Start.")
    # Read the image and convert to gray to reduce the data
    img = cv2.imread(image_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#Color => Gray
    
    # The haarcascades classifier is used to train data
    #face_cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)#1.3 and 5are the min and max windows of the treatures
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    print ("Face Detection Complete.")
    return result

from matplotlib import pyplot as plt
def draw(image):
    img_ori = Image.open(image)
    plt.imshow(img_ori),plt.show()

#Crop faces and save them in the same directory
filepath ="/home/xilinx/jupyter_notebooks/OpenCV/Face_Detection/images/"
dir_path ="/home/xilinx/jupyter_notebooks/OpenCV/Face_Detection/"
filecount = len(os.listdir(filepath))-1
image_count = 1#count is the number of images
face_cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml")
for fn in os.listdir(filepath): #fn 表示的是文件名
    start = time.time()
    if image_count <= filecount:
        image_name = str(image_count) + '.JPG'
        image_path = filepath + image_name
        image_new = dir_path + image_name
        #print (image_path)
        #print (image_new)
        os.system('cp '+(image_path)+ (' /home/xilinx/jupyter_notebooks/OpenCV/Face_Detection/'))
        faces = detectFaces(image_name)
        if not faces:
            print ("Error to detect face")
        if faces:
            #All croped face images will be saved in a subdirectory
            face_name = image_name.split('.')[0]
            #os.mkdir(save_dir)
            count = 0
            for (x1,y1,x2,y2) in faces:
                file_name = os.path.join(dir_path,face_name+str(count)+".jpg")
                face_name = os.path.basename(file_name)
                #print (face_name)
                Image.open(image_name).crop((x1,y1,x2,y2)).save(file_name)
                #os.system('rm -rf '+(image_path)+' /home/xilinx/jupyter_notebooks/OpenCV/Face_Detection/')
                count+=1    
                draw(file_name)
            print("The " + str(image_count) +" image were done.")
            print("Congratulation! The total of the " + str(count) + " faces in the " +str(image_count) + " image.")
        os.system('rm -rf '+(image_new))
    end = time.time()
    TimeSpan = end - start
    if image_count <= filecount:
        print ("The time of " + str(image_count) + " image is " +str(TimeSpan) + " s.")
    image_count = image_count + 1   

