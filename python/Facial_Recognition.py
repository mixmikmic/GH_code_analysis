get_ipython().magic('matplotlib inline')

import numpy as np
from scipy import linalg as la
from os import walk
from scipy.ndimage import imread
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import random
from random import sample

def getFaces(path='./faces94'):
    """Traverse the directory specified by 'path' and return an array containing
    one column vector per subdirectory.
    """
    # Traverse the directory and get one image per subdirectory
    faces = []
    for (dirpath, dirnames, filenames) in walk(path):
        for f in filenames:
            if f[-3:]=="jpg": # only get jpg images
                # load image, convert to grayscale, flatten into vector
                face = imread(dirpath+"/"+f).mean(axis=2).ravel()
                faces.append(face)
                break
    # put all the face vectors column-wise into a matrix
    return np.array(faces).T

def show(im, w=200, h=180):
    """Plot the flattened grayscale image 'im' of width 'w' and height 'h'."""
    plt.imshow(im.reshape((w,h)), cmap=cm.Greys_r)
    plt.show()
    
def show2(test_image, result, input_image="Inputed Image", match="Closest match", w=200, h=180):
    """Convenience function for plotting two flattened grayscale images of
    the specified width and height side by side
    """
    plt.subplot(121)
    plt.title(input_image)
    plt.imshow(test_image.reshape((w,h)), cmap=cm.Greys_r)
    plt.axis("off")
    plt.subplot(122)
    plt.title(match)
    plt.imshow(result.reshape((w,h)), cmap=cm.Greys_r)
    plt.axis("off")
    plt.show()

def meanShift(face_matrix):
    """Takes in a nxm np.array comprised of flattened images along the columns and returns a nx1
    vector of the average face.
    """
    mu = np.mean(face_matrix, axis=1)
    return mu

face_matrix = getFaces()
mu = meanShift(face_matrix)
plt.title("The Average Face", fontsize=20, y=1.1)
plt.axis("off")
show(mu)

def faceDifferences(face_matrix, mu):
        Fbar = face_matrix - np.vstack(mu)
        return Fbar

Fbar = faceDifferences(face_matrix, mu)
show2(face_matrix[:,28], Fbar[:,28], "Original Image", "Mean-Shifted Image")

def eigenFaces(Fbar):
    U = la.svd(Fbar, full_matrices=False)[0]
    return U

U = eigenFaces(Fbar)
show2(face_matrix[:,28], U[:,28], "Original Image", "EigenFace Image")

def basisProject(A, U, s=38):
    Ut = U[:,:s].T
    return Ut.dot(A)

x = 1
e_number = [5, 25, 50, 75, 200, 500]
for s in e_number: 
    # Project face onto subspace spaned by s eigenvalues
    face_in_eigen_basis = basisProject(face_matrix[:,2], U, s)
    # Project face back to R_mn
    face_projected_back = U[:,:s].dot(np.vstack(face_in_eigen_basis))
    # Add the mean back
    reconstructed_face = face_projected_back + np.vstack(mu)
    plt.subplot(2,3,x)
    plt.suptitle("Image Reconstruction With Different Eigen Vectors",fontsize=20, y = 1.12)
    plt.title("%s EigenFaces" % s)
    plt.axis("off")
    plt.imshow(reconstructed_face.reshape((200,180)), cmap=cm.Greys_r)
    x += 1
plt.show()
    

class FacialRec:
##########Members##########
# F, mu, Fbar, and U
###########################
    def __init__(self,path):
        self.initFaces(path)
        self.initMeanImage()
        self.initDifferences()
        self.initEigenfaces()

    def initFaces(self, path):
        self.F = getFaces(path)
        
    def initMeanImage(self):
        self.mu = np.mean(self.F, axis=1)
             
    def initDifferences(self):
        self.Fbar = self.F - np.vstack(self.mu)
    
    def initEigenfaces(self):
        self.U = la.svd(self.Fbar, full_matrices=False)[0]
    
    def show(self, face_num):
        show(self.F[:,face_num])
        
    def project(self, A, s=38):
        c = self.U[:,:s]
        return c.T.dot(A)      
    
    def findNearest(self, image, s=38):
        Fhat = self.project(self.Fbar, s)
        ghat = self.project((np.vstack(image) - np.vstack(self.mu)), s)
        index = np.linalg.norm(Fhat - np.vstack(ghat), axis=0).argmin()
        return index

def sampleFaces(n_tests, path="./faces94"):
    """Return an array containing a sample of n_tests images contained
    in the path as flattened images in the columns of the output
    """
    files = []
    for (dirpath, dirnames, filenames) in walk(path):
        for f in filenames:
            if f[-3:]=="jpg": # only get jpg images
                files.append(dirpath+"/"+f)

    #Get a sample of the images
    test_files = random.sample(files, n_tests)
    #Flatten and average the pixel values
    images = np.array([imread(f).mean(axis=2).ravel() for f in test_files]).T
    return images

recognizer = FacialRec("./faces94")
test_faces = sampleFaces(5)
for x in xrange(5):
    match_index = recognizer.findNearest(test_faces[:,x], 100)
    show2(face_matrix[:,match_index], test_faces[:,x])

