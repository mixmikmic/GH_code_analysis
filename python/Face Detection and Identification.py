import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

# Import the loading function of Scikit-Image
from skimage import io

img = io.imread('face.jpg')

plt.figure(figsize=(10,10))
plt.imshow(img)

# Import the dlib library
import dlib
# Load the frontal face detector
detector = dlib.get_frontal_face_detector()

# Apply the detector to the img and return the detections (the '1' is a upsampling factor to get better results, not mandatory)
dets = detector(img, 1)
print("Number of faces detected: {}".format(len(dets)))

# Accessing the first detected face
dets[0]

from matplotlib.patches import Rectangle
def draw_detection(det):
    # gca -> get-current-axis
    current_axis = plt.gca()
    # Add a rectangle on top of the image with the position defined by the detected face
    current_axis.add_patch(
                Rectangle(
                    (det.left(), det.top()),  # x, y
                    det.right() - det.left(), det.bottom() - det.top(),  # w, h
                    edgecolor="red", fill=False))
    
plt.figure(figsize=(10,10))
plt.imshow(img)
for det in dets:
    # For each detected face, draw it
    draw_detection(det)



# Shape predictor to refine the face detection result (find face landmarks like eye corners, mouth, etc...)
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# Face recognition model
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

query_img = io.imread('query.jpg')

query_det = detector(query_img, 1)[0]

plt.figure(figsize=(10,10))
plt.imshow(query_img)
draw_detection(query_det)

# Get the landmarks/parts for the face in box d.
query_shape = sp(query_img, query_det)
# Extract the face descriptor
query_face_descriptor = facerec.compute_face_descriptor(query_img, query_shape)
query_face_descriptor = np.asarray(query_face_descriptor)  # Converting the descriptor to standard numpy array

print(query_face_descriptor.shape)
print(query_face_descriptor)

from glob import glob
from tqdm import tqdm  # For the eye-candy progress-bar

# List all the filenames in the faces directory
face_files = glob('faces/*')
face_files

# For each file
for filename in tqdm(face_files, 'Processing'):
    # Read the image
    img = io.imread(filename)
    # Detect faces in img, and take the first detected result
    det = None  # TODO
    # Refine the detected face by extracting the 
    shape = None  # TODO
    # Extract the face descriptor
    face_descriptor = None  # TODO
    # Compute the euclidean distance between our query face descriptor and the current one
    dist = None  # TODO
    # Plot the detection result
    plt.figure()
    plt.imshow(img)
    draw_detection(det)
    plt.title("{} : distance={:.2f}".format(filename.split('/')[-1], dist))

# For the bored people, you could have a better look at the shape predictor that finds out the landmark on the face
# A good starting point is http://dlib.net/face_landmark_detection.py.html

