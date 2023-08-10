# Import required libraries.

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Allows matplotlib graphic output to be displayed in Jupyter notebooks.
get_ipython().magic('matplotlib inline')

# Helper function to convert pictures to BGR color space.

def convertToRGB(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

test1 = cv2.imread('C:\\Users\\pabailey\\Desktop\\heroes.jpg')
plt.imshow(test1)

gray_img = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img)

plt.imshow(gray_img, cmap='gray')

haar_face_cascade = cv2.CascadeClassifier('C:\\Users\\pabailey\\Desktop\\opencv\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_alt.xml')
haar_eye_cascade = cv2.CascadeClassifier('C:\\Users\\pabailey\\Desktop\\opencv\\opencv-master\\data\\haarcascades\\haarcascade_eye.xml')

img = cv2.imread('C:\\Users\\pabailey\\Desktop\\heroes.jpg') # load image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale

faces = haar_face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),10)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = haar_eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),10)

plt.imshow(convertToRGB(img))

haar_face_cascade = cv2.CascadeClassifier('C:\\Users\\pabailey\\Desktop\\opencv\\opencv-master\\data\\haarcascades\\haarcascade_frontalface_alt.xml')
haar_right_eye_cascade = cv2.CascadeClassifier('C:\\Users\\pabailey\\Desktop\\opencv\\opencv-master\\data\\haarcascades\\haarcascade_righteye_2splits.xml')
haar_left_eye_cascade = cv2.CascadeClassifier('C:\\Users\\pabailey\\Desktop\\opencv\\opencv-master\\data\\haarcascades\\haarcascade_lefteye_2splits.xml')

img = cv2.imread('C:\\Users\\pabailey\\Desktop\\heroes.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = haar_face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),10)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    
    eyes_right = haar_right_eye_cascade.detectMultiScale(roi_gray)
    eyes_left = haar_left_eye_cascade.detectMultiScale(roi_gray)
    
    for (ex,ey,ew,eh) in eyes_right:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),10)
    
    for (ex, ey, ew, eh) in eyes_left:
        cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 10)
    

plt.imshow(convertToRGB(img))

def faceDetect(f_cascade, colored_img, scaleFactor = 1.1):
    img_copy = colored_img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5);

    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x,y), (x+w,y+h), (0,255,0), 15)
    
    return img_copy

# Stress testing -show that you need to change parameter in function
# above to get the algorithm to see our dudes

test2 = cv2.imread('C:\\Users\\pabailey\\coolpeoplehangingouttogether.jpg')
faces_detected_img = faceDetect(haar_face_cascade, test2)
plt.imshow(convertToRGB(faces_detected_img))

lbp_face_cascade = cv2.CascadeClassifier('C:\\Users\\pabailey\\Desktop\\opencv\\opencv-master\\data\\lbpcascades\\lbpcascade_frontalface_improved.xml')
faces_detected_img2 = faceDetect(lbp_face_cascade, test2)
plt.imshow(convertToRGB(faces_detected_img2))

haar_profile_test = cv2.CascadeClassifier('C:\\Users\\pabailey\\Desktop\\opencv\\opencv-master\\data\\haarcascades\\haarcascade_profileface.xml')
faces_detected_3 = faceDetect(haar_profile_test, test2)
plt.imshow(convertToRGB(faces_detected_3))



