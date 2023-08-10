import cv2
import dlib 
import numpy

# Checking the version of cv2
print("OpenCV Version : ",cv2.__version__)

# Checking the version of dlib
print("Dlib Version  : ",dlib.__version__)

# Checking the version of numpy
print("Numpy Version : ",numpy.__version__)

# Using dlib to detect the faces in an image

import cv2
import dlib
import os
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from enum import Enum
from collections import OrderedDict

get_ipython().magic('matplotlib inline')

class Point(Enum):
    X = 0
    Y = 1

PATH = os.path.join("..","..","..","dependencies")
FILE_NAME = "shape_predictor_68_face_landmarks.dat"

IMAGE_PATH = os.path.join("..","..","images","face_ops")

predictor = dlib.shape_predictor(os.path.join(PATH,FILE_NAME))
detector = dlib.get_frontal_face_detector()


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


def get_points(image):
    # Note: this function will only recieve faces that have exactly one face
    
    dets = detector(image)
    for k, d in enumerate(dets):

            gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)     

            shape = shape_to_np(predictor(gray, d))

    return shape # where shape is a 68 points list of tuples 


def do_affine_transform(src, src_T, dst_T, size):
    
    #Getting the transformation matrix
    tMat = cv2.getAffineTransform(np.float32(src_T), np.float32(dst_T))
    
    # Next we need to apply the transform on the source
    tImg = cv2.warpAffine(src, tMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR)

    return tImg

def do_morph_face(img1, img2, img3, t1, t2, t3, coeff):
    
    # Creating a bounded rectangle around each triangle
    br1 = cv2.boundingRect(np.float32([t1])) # triangles of image1
    br2 = cv2.boundingRect(np.float32([t2]))# triangles of image2
    br3 = cv2.boundingRect(np.float32([t3])) # triangles of image3
    
    dst = do_affine_transform(img1, img1, img3, (br1[0],br1[1]))
    
    return dst                          

def do_delaunay_ops(image,points,index):
    
    #getting the image bounds
    image_bound = (0,0, image.shape[1], image.shape[0])
    
    # creating an instance of Subdiv2D , used to generate delaunay as well as voronoi tesselation
    subdiv = cv2.Subdiv2D(image_bound)
    
    subdiv.insert(points)
    
    triplets = subdiv.getTriangleList();
    
    # Holds a triplet of landmark points
    out = []
    
    # we will only take the valid triangles, i.e one's that lie in the face region
    for entry in triplets:
        x1 = entry[0], entry[1]
        x2 = entry[2], entry[3]
        x3 = entry[4], entry[5]
        
        if x1 in index and x2 in index and x3 in index:
            out.append((index[x1],index[x2],index[x3]))
            
    return out

    
# co-eff
coeff = 0.5 # taking mean of the two images
    
img1 = mpimg.imread(os.path.join(IMAGE_PATH,"selena.jpg"))
img2 = mpimg.imread(os.path.join(IMAGE_PATH,"kate.jpg"))

# Making sure they have the same dimensions
img1 = cv2.resize(img1, (400,600))
img2 = cv2.resize(img2, (400,600))

plt.imshow(img1)
plt.xticks([]),plt.yticks([])
plt.show()

plt.imshow(img2)
plt.xticks([]),plt.yticks([])
plt.show()

pts1 = get_points(img1)
pts2 = get_points(img2)

# Since we will be morphing the faces, we will not only do weigted blending
# we will also do weighting landmark calculation between the two images

pts3 = []
for idx in range(len(pts1)):
    x3 = (1-coeff)*pts1[idx][0] + coeff*pts2[idx][0] # weighted landmark x in morphed image
    y3 = (1-coeff)*pts1[idx][1] + coeff*pts2[idx][1]
    pts3.append((x3,y3))
    
# Creating an empty image for the morphed image using np.zeros, with shape == img1
img3 = np.zeros(img1.shape)

# Next since we will be performming delaunay triangulation using the average points (pts3)
#Steps:

# 1. Perform delaunay triangulation using the average points we obtained as in pts3
# 2. We get a triplet of length 68
# 3. And this lead to 3 unique triangles in each image, including img3
# 4. We then perform 2 affine transforms:
#                     img1 -> img3
#                     img2 -> img3
# 5. And then we apply alpha blending with the above blending factor to merge the two triangles to form im3
# 6. repeat 5 for all the triangles that were formed
# 7. we have a swapped image

# Indexing the mean landmark points
# In python, tuples can be keys - wow!
pts_idx = {}
pts_idx_rev = {} # this is the reverse index of pts_idx

# Index the pixel locations against landmark points
for idx,pt in enumerate(pts3):
    pts_idx[pt] = idx+1
    pts_idx_rev[idx+1] = pt
    
triplets = do_delaunay_ops(img3, pts3, pts_idx)
    
# Plotting the triangles-meshes on a face, not we don't care about duplicates
mesh_img1 = img1.copy()
mesh_img2 = img2.copy()

for t in triplets:
    p1 = tuple([int(x)-17 for x in pts_idx_rev[t[0]]])
    p2 = tuple([int(x)-17 for x in pts_idx_rev[t[1]]])
    p3 = tuple([int(x)-17 for x in pts_idx_rev[t[2]]])
    
    
    # Image 1
    cv2.line(mesh_img1, p1, p2, (255, 255, 255) , 1)
    cv2.line(mesh_img1, p2, p3, (255, 255, 255) , 1)
    cv2.line(mesh_img1, p1, p3, (255, 255, 255) , 1)
    
    # Image 2
    cv2.line(mesh_img2, p1, p2, (255, 255, 255) , 1)
    cv2.line(mesh_img2, p2, p3, (255, 255, 255) , 1)
    cv2.line(mesh_img2, p1, p3, (255, 255, 255) , 1)

# Plotting the meshed faces

plt.imshow(mesh_img1)
plt.xticks([]),plt.yticks([])
plt.show()

plt.imshow(mesh_img2)
plt.xticks([]),plt.yticks([])
plt.show()    


# Now we will do the morphing
for t in triplets:
    x, y, z = t
    
    t1 = [pts1[x-1], pts1[y-1], pts1[z-1]]  # The triangles in img1
    t2 = [pts2[x-1], pts2[y-1], pts2[z-1]]  # The triangles in img2
    t3 = [pts3[x-1], pts3[y-1], pts3[z-1]]  # The triangles in img3
 

img1 = mpimg.imread(os.path.join(IMAGE_PATH,"kate.jpg"))
img2 = mpimg.imread(os.path.join(IMAGE_PATH,"selena.jpg"))

img1_org = img1.copy()
img2_org = img2.copy()

pts1 = get_points(img1)
pts2 = get_points(img2)

hull = cv2.convexHull(pts1)
hull2 = cv2.convexHull(pts2)

hull = np.array([[x[0][0],x[0][1]] for x in hull],np.int32)
hull = hull.reshape((-1,1,2))
cv2.polylines(img1,[hull],True,(255,255,255))

hull2 = np.array([[x[0][0],x[0][1]] for x in hull2],np.int32)
hull2 = hull2.reshape((-1,1,2))
cv2.polylines(img2,[hull2],True,(255,255,255))

mask1 = np.ones(img1.shape[:-1],dtype=np.uint8)
cv2.fillConvexPoly(mask1, np.int32(hull), (0,0,0))

mask2 = np.ones(img2.shape[:-1],dtype=np.uint8)
cv2.fillConvexPoly(mask2, np.int32(hull2), (0,0,0))


print("Sample Image 1")
plt.imshow(img1_org)
plt.gray()
plt.xticks([]),plt.yticks([])
plt.show() 

print("Sample Image 2")
plt.imshow(img2_org)
plt.gray()
plt.xticks([]),plt.yticks([])
plt.show() 

# convert to 0-255 range
mask1 *= 255
mask1 = cv2.bitwise_not(mask1)

mask2 *= 255
mask2 = cv2.bitwise_not(mask2)


# perform thresholding
ret,thresh = cv2.threshold(mask1,127,255,0)
_,contours1,hierarchy = cv2.findContours(thresh, 1, 2)

ret,thresh = cv2.threshold(mask2,127,255,0)
_,contours2,hierarchy = cv2.findContours(thresh, 1, 2)


# getting the contours
cnt1 = contours1[0]
rect1 = cv2.minAreaRect(cnt1)
box1 = cv2.boxPoints(rect1)
box1 = np.int0(box1)

cnt2 = contours2[0]
rect2 = cv2.minAreaRect(cnt2)
box2 = cv2.boxPoints(rect2)
box2 = np.int0(box2)

# box surrounding the faces - 1 & 2
#print(box1)
#print(box2)

# getting the max and min x and y values for face 1
x1_max,x1_min = np.max(box1[:,0]), np.min(box1[:,0])
y1_max,y1_min = np.max(box1[:,1]), np.min(box1[:,1])

x2_max,x2_min = np.max(box2[:,0]), np.min(box2[:,0])
y2_max,y2_min = np.max(box2[:,1]), np.min(box2[:,1])

# Displaying the mask applied on the image 1
print("Mask 1")
plt.imshow(mask1)
plt.xticks([]),plt.yticks([])
plt.show()

print("Mask 2")
plt.imshow(mask2)
plt.xticks([]),plt.yticks([])
plt.show()

# Bitwise and using the original image and the mask
masked_image1 = cv2.bitwise_and(img1_org,img1_org,mask = mask1)
plt.imshow(masked_image1)
plt.xticks([]),plt.yticks([])
plt.show()

masked_image2 = cv2.bitwise_and(img2_org,img2_org,mask = mask2)
plt.imshow(masked_image2)
plt.xticks([]),plt.yticks([])
plt.show()

#Cropping masked_image1 by box1
cropped_image1 = masked_image1[y1_min:y1_max,x1_min:x1_max]

#Cropping masked_image2 by box2
cropped_image2 = masked_image2[y2_min:y2_max,x2_min:x2_max]

# Cropped Image1
plt.imshow(cropped_image1)
plt.xticks([]),plt.yticks([])
plt.show()

# Cropped Image2
plt.imshow(cropped_image2)
plt.xticks([]),plt.yticks([])
plt.show()

#Next Steps: Resize the cropped images of the size of the opposite boxes, and apply and operation
resized_cimage1 = cv2.resize(cropped_image1, (y2_max-y2_min, x2_max - x2_min))
resized_cimage2 = cv2.resize(cropped_image2, (y1_max-y1_min, x1_max - x1_min))

# Resized Cropped Image1 of Box 2
plt.imshow(resized_cimage1)
plt.xticks([]),plt.yticks([])
plt.show()

# Resized Cropped Image2 of Box 1
plt.imshow(resized_cimage2)
plt.xticks([]),plt.yticks([])
plt.show()

# Getting the outputs - Cheap way of doing it using 

#1
out_image1 = np.zeros(img2_org.shape,dtype=np.uint8)
out_image1 = out_image1*255
out_image1[x2_min:x2_max,y2_min:y2_max,:] = resized_cimage1

morphed_image1 = cv2.addWeighted(img2_org,0.4,out_image1,0.6,0)
plt.imshow(morphed_image1)
plt.xticks([]),plt.yticks([])
plt.show()

#2
out_image2 = np.zeros(img1_org.shape,dtype=np.uint8)
out_image2 = out_image2*255
out_image2[x1_min:x1_max,y1_min:y1_max,:] = resized_cimage2

morphed_image2 = cv2.addWeighted(img1_org,0.4,out_image2,0.6,0)
plt.imshow(morphed_image2)
plt.xticks([]),plt.yticks([])
plt.show()

# Image 1
_im = cv2.cvtColor(out_image2,cv2.COLOR_BGR2GRAY)
res,alpha = cv2.threshold(_im,5,255,cv2.THRESH_BINARY)
inverse_alpha = cv2.bitwise_not(alpha)

new_face = cv2.merge((out_image2[:,:,0],out_image2[:,:,1],out_image2[:,:,2],alpha))

_im2 = cv2.cvtColor(img1_org,cv2.COLOR_BGR2GRAY);
res1,alpha1 = cv2.threshold(_im2,10,255,cv2.THRESH_BINARY);

h,w,_ = img1_org.shape
apc = numpy.ones((h,w),dtype=np.uint8)*255
img2_alpha_org = numpy.dstack(( img1_org, apc))

img2_alpha_org = cv2.bitwise_and(img2_alpha_org,img2_alpha_org,mask=inverse_alpha)
x_im_1 = cv2.add(img2_alpha_org,new_face)

# Image 2

_im = cv2.cvtColor(out_image1,cv2.COLOR_BGR2GRAY)
res,alpha = cv2.threshold(_im,5,255,cv2.THRESH_BINARY)
inverse_alpha = cv2.bitwise_not(alpha)

new_face = cv2.merge((out_image1[:,:,0],out_image1[:,:,1],out_image1[:,:,2],alpha))

_im2 = cv2.cvtColor(img2_org,cv2.COLOR_BGR2GRAY);
res1,alpha1 = cv2.threshold(_im2,10,255,cv2.THRESH_BINARY);

h,w,_ = img2_org.shape
apc = numpy.ones((h,w),dtype=np.uint8)*255
img2_alpha_org = numpy.dstack(( img2_org, apc))

img2_alpha_org = cv2.bitwise_and(img2_alpha_org,img2_alpha_org,mask=inverse_alpha)
x_im_2 = cv2.add(img2_alpha_org,new_face)


fig = plt.figure("Face Swapping",(10,10))

plt.subplot("221")
plt.xlabel("Before")
plt.xticks([]), plt.yticks([])
plt.imshow(img1_org)

plt.subplot("222")
plt.xlabel("After")
plt.xticks([]), plt.yticks([])
plt.imshow(x_im_1)


plt.subplot("223")
plt.xlabel("Before")
plt.xticks([]), plt.yticks([])
plt.imshow(img2_org)


plt.subplot("224")
plt.xlabel("After")
plt.xticks([]), plt.yticks([])
plt.imshow(x_im_2)

plt.show()



