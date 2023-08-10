import cv2
import numpy as np
import dlib

# Checking the OpenCV version
print("OpenCV version", cv2.__version__)

# Checking the Numpy version
print("Numpy version", np.__version__)

# Checking the dlib version
print("Dlib version", dlib.__version__)

import os
import dlib
import cv2
import numpy as np
import glob
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

PATH = os.path.join("images","faces","*.jpg")

def drawBB(image, bound):
    image = image.copy() 
    pt1 = ( bound.left(),bound.top())
    pt2 = ( bound.right(), bound.bottom())
    return cv2.rectangle(image, pt1, pt2, (0,255,0), 3)       

detector = dlib.get_frontal_face_detector()

for image in glob.glob(PATH):
   
    im = mpimg.imread(image)
    
    dets = detector(im)

    print("Number of faces detected - {0}".format(len(dets)))
    print(image)
    for k, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                k, d.left(), d.top(), d.right(), d.bottom()))
            im = drawBB(im,d)
    
    plt.imshow(im)
    plt.xticks([]), plt.yticks([])
    plt.show()

# Using dlib to detect the faces in an image

import cv2
import dlib
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

get_ipython().magic('matplotlib inline')

PATH = os.path.join("..","dependencies")
FILE_NAME = "shape_predictor_68_face_landmarks.dat"

IMAGE_PATH = os.path.join("images","faces","Justin_Timberlake.jpg")

predictor = dlib.shape_predictor(os.path.join(PATH,FILE_NAME))

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def draw_bb(image, bound):
    image = image.copy() 
    
    pt1 = ( bound.left(), bound.top())
    pt2 = ( bound.right(), bound.bottom())    
    
    return cv2.rectangle(image, pt1, pt2, (0,255,0), thickness=1, lineType=8, shift=0)       


detector = dlib.get_frontal_face_detector()

im  =  mpimg.imread(IMAGE_PATH) # this is the input image

oim = im.copy()

dets = detector(im)

print("Number of faces detected - {0}".format(len(dets)))

for k, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            k, d.left(), d.top(), d.right(), d.bottom()))
        
        im = draw_bb(im,d)
        
        gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)     
        
        shape = predictor(gray, d)
        
        shape = shape_to_np(shape)
        
        # show the face number
        cv2.putText(im, "Face #{}".format(k + 1), (d.left() - 10, d.top() - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for (x, y) in shape:
                cv2.circle(im, (x, y), 1, (0, 0, 255), -1)
        
fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

plt.subplot(211)
plt.imshow(oim)
plt.xticks([]), plt.yticks([])


plt.subplot(212)
plt.imshow(im)
plt.xticks([]), plt.yticks([])

plt.show()

import os
import dlib
import matplotlib.image as mpimg

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])


def face_feature(face_image):
    """
    Given a image, returns a list of the detected faces features using the dlib compute_face_descriptor function
    :param face_image : Image of the face
    :return : List of image features
    """
    
    features = []
    
    KEY = face_image.split("/")[-2] # Change this line according to your file orgnaization
    
    im = mpimg.imread(image)
    
    dets = detector(im)
    
    for k, d in enumerate(dets):
            
            im = draw_bb(im,d)

            gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)     

            sp = predictor(gray, d)

            shape = shape_to_np(sp)

            face_descriptor = facerec.compute_face_descriptor(im, sp)

            features.append(np.array(face_descriptor))
            
    return features

    
def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.
    :param faces: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


FACE_DEC_MODEL_PATH = os.path.join("..","dependencies","shape_predictor_68_face_landmarks.dat")
FACE_REC_MODEL_PATH = os.path.join("..","dependencies","dlib_face_recognition_resnet_model_v1 2.dat")
IMAGES_PATH = os.path.join("images","lfw","*","*.jpg")

predictor = dlib.shape_predictor(FACE_DEC_MODEL_PATH)
facerec   = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)
detector  = dlib.get_frontal_face_detector()

count = 0

features = []
labels = []

for image in glob.glob(IMAGES_PATH):
    
    KEY = image.split("/")[-2]
    
    print(image)
    
    im = mpimg.imread(image)
    
    dets = detector(im)
    
    for k, d in enumerate(dets):
        
            im = draw_bb(im,d)

            gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)     

            sp = predictor(gray, d)

            shape = shape_to_np(sp)

            face_descriptor = facerec.compute_face_descriptor(im, sp)

            features.append(np.array(face_descriptor))
            
            labels.append(KEY)
        
            count+=1


print("Total Faces Indexed", count)

# Using dlib to recognize the face based on cosine simularity

import cv2
import dlib
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from collections import OrderedDict

get_ipython().magic('matplotlib inline')

IMAGE_PATH = os.path.join("images","faces","Justin_Timberlake.jpg")
FACE_DEC_MODEL_PATH = os.path.join("..","dependencies","shape_predictor_68_face_landmarks.dat")
FACE_REC_MODEL_PATH = os.path.join("..","dependencies","dlib_face_recognition_resnet_model_v1 2.dat")

predictor = dlib.shape_predictor(FACE_DEC_MODEL_PATH)
facerec   = dlib.face_recognition_model_v1(FACE_REC_MODEL_PATH)
detector  = dlib.get_frontal_face_detector()

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 36)),
    ("jaw", (0, 17))
])

def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords


def draw_bb(image, bound):
    image = image.copy() 
    
    pt1 = ( bound.left(), bound.top())
    pt2 = ( bound.right(), bound.bottom())    
    
    return cv2.rectangle(image, pt1, pt2, (0,255,0), 3)       


detector = dlib.get_frontal_face_detector()

im  = mpimg.imread(IMAGE_PATH) # this is the input image

dets = detector(im)

print("Number of faces detected - {0}".format(len(dets)))

for k, d in enumerate(dets):
        im = draw_bb(im,d)
        
        gray = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)     
        
        sp = predictor(gray, d)
        
        shape = shape_to_np(sp)
        
        face_descriptor = np.array(facerec.compute_face_descriptor(im, sp))
    
        LABEL = labels[np.argmin(face_distance(features,face_descriptor), axis=0)]
        
        # show the face number
        cv2.putText(im, LABEL, (d.left() - 10, d.top() - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
fig = plt.figure(num=None, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')

plt.imshow(im)
plt.xticks([]), plt.yticks([])

plt.show()



