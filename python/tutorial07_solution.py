# Import libraries

# math library
import numpy as np

# remove warning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

# computational time
import time

# import mat data
import scipy.io

# dynamic 3D rotations:
get_ipython().magic('matplotlib notebook')
# no 3D rotations but cleaner images:
#%matplotlib inline    
import matplotlib.pyplot as plt

# 3D visualization
import pylab
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot

# high definition picture
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png2x','pdf')

# visualize 2D images
import scipy.ndimage

# Training set
data = np.loadtxt('data/pca01.txt', delimiter=',')

#YOUR CODE HERE

# plot
x1 = data[:,0] # feature 1
x2 = data[:,1] # feature 2

plt.figure(1,figsize=(6,6))
plt.scatter(x1, x2, s=60, c='r', marker='+') 
plt.title('Data points')
plt.show()

X = data
print('mean before=',np.mean(X,axis=0))
print('std before=',np.std(X,axis=0))

#YOUR CODE HERE
X -= X.mean(axis=0)
X /= np.std(X,axis=0)
print('mean after=',np.mean(X,axis=0))
print('std after=',np.std(X,axis=0))

plt.figure(2,figsize=(6,6))
plt.scatter(X[:,0], X[:,1], s=60, c='r', marker='+') 
plt.title('Data normalized by z-scoring')
plt.show()

#YOUR CODE HERE
Sigma = X.T.dot(X) / X.shape[0]
print(Sigma.shape)
print(Sigma)

def EVD(X):
    s, U = np.linalg.eig(X)
    idx = s.argsort()[::-1] # decreasing order
    return s[idx], U[:,idx]

#YOUR CODE HERE

s, U = EVD(Sigma)
print(s)
print(U)


plt.figure(3)
size_vertex_plot = 10
plt.scatter(X[:,0], X[:,1], s=60, c='r', marker='+') 
k=0; p=s[k]* U[:,k]
plt.quiver(0.0, 0.0, p[0], p[1], scale=1., units='xy', color='b') 
k=1; p=s[k]* U[:,k]
plt.quiver(0.0, 0.0, p[0], p[1], scale=1., units='xy', color='g') 
plt.title('Principal directions')
plt.show()

#YOUR CODE HERE
k = 0
var = 0
tot_var = np.sum(s)
while var < 0.85:
    k += 1
    var = np.sum(s[:k])/ tot_var   

print('k=',k)
print('captured variance=',var)

# Yale Faces
mat = scipy.io.loadmat('data/pca02_yale_faces.mat')
X = mat['X']
print(X.shape)

n = X.shape[0]
d = X.shape[1]
Nx = 55
Ny = 63

# Plot some images
plt.figure(10)
rotated_img = scipy.ndimage.rotate(np.reshape(X[0,:],[Nx,Ny]), -90)
plt.subplot(131).imshow(rotated_img, interpolation='nearest', cmap='Greys_r')
plt.axis('equal')
plt.axis('off')
rotated_img = scipy.ndimage.rotate(np.reshape(X[10,:],[Nx,Ny]), -90)
plt.subplot(132).imshow(rotated_img, interpolation='nearest', cmap='Greys_r')
plt.axis('equal')
plt.axis('off')
rotated_img = scipy.ndimage.rotate(np.reshape(X[20,:],[Nx,Ny]), -90)
plt.subplot(133).imshow(rotated_img, interpolation='nearest', cmap='Greys_r')
plt.axis('equal')
plt.axis('off')
plt.show()

#YOUR CODE HERE
X -= X.mean(axis=0)
X /= np.std(X,axis=0)
Sigma = X.T.dot(X) / X.shape[0]
s, U = EVD(Sigma)
s, U = np.real(s), np.real(U)

# Plot
plt.figure(11)
plt.plot(s[:16]/np.sum(s))
plt.title('Percentage of data variances captured by the first 15 principal directions')

#YOUR CODE HERE
k = 0
var = 0
tot_var = np.sum(s)
while var < 0.95:
    k += 1
    var = np.sum(s[:k])/ tot_var

print('k=',k)
print('captured variance=',var)
    

# Indicator vector of three classes of faces
C = mat['Cgt'].squeeze()
print(C.shape)

# Principal components
PC = X.dot(U)
print(X.shape,U.shape,PC.shape)


# 2D Plot
plt.figure(12)
plt.scatter(PC[:,0], PC[:,1], s=60*np.ones(27), c=C ) 
plt.show()

# 3D Plot
fig = pylab.figure(14)
ax = Axes3D(fig)
size_vertex_plot = 100
ax.scatter(PC[:,0], PC[:,1], PC[:,2], s=60*np.ones(27), c=C)
pyplot.show()

# Generate new faces
X = scipy.io.loadmat('data/pca02_yale_faces.mat')['X']
meanX = np.mean(X,axis=0) 
PD = U.T

plt.figure(15)
k = 0; new_face = meanX - 0.025* s[k]* PD[k,:]
rotated_img = scipy.ndimage.rotate(np.reshape(new_face,[Nx,Ny]), -90)
plt.subplot(331).imshow(rotated_img, interpolation='nearest', cmap='Greys_r')
plt.title('Mean Face - 0.025* PD(1)')
plt.axis('off')
k = 0; new_face = meanX - 0.0* s[k]* PD[k,:]
rotated_img = scipy.ndimage.rotate(np.reshape(new_face,[Nx,Ny]), -90)
plt.subplot(332).imshow(rotated_img, interpolation='nearest', cmap='Greys_r')
plt.title('Mean Face')
plt.axis('off')
k = 0; new_face = meanX + 0.025* s[k]* PD[k,:]
rotated_img = scipy.ndimage.rotate(np.reshape(new_face,[Nx,Ny]), -90)
plt.subplot(333).imshow(rotated_img, interpolation='nearest', cmap='Greys_r')
plt.title('Mean Face + 0.025* PD(1)')
plt.axis('off')
k = 1; new_face = meanX - 0.025* s[k]* PD[k,:]
rotated_img = scipy.ndimage.rotate(np.reshape(new_face,[Nx,Ny]), -90)
plt.subplot(334).imshow(rotated_img, interpolation='nearest', cmap='Greys_r')
plt.title('Mean Face - 0.025* PD(2)')
plt.axis('off')
k = 1; new_face = meanX - 0.0* s[k]* PD[k,:]
rotated_img = scipy.ndimage.rotate(np.reshape(new_face,[Nx,Ny]), -90)
plt.subplot(335).imshow(rotated_img, interpolation='nearest', cmap='Greys_r')
plt.title('Mean Face')
plt.axis('off')
k = 1; new_face = meanX + 0.025* s[k]* PD[k,:]
rotated_img = scipy.ndimage.rotate(np.reshape(new_face,[Nx,Ny]), -90)
plt.subplot(336).imshow(rotated_img, interpolation='nearest', cmap='Greys_r')
plt.title('Mean Face + 0.025* PD(2)')
plt.axis('off')
k = 2; new_face = meanX - 0.025* s[k]* PD[k,:]
rotated_img = scipy.ndimage.rotate(np.reshape(new_face,[Nx,Ny]), -90)
plt.subplot(337).imshow(rotated_img, interpolation='nearest', cmap='Greys_r')
plt.title('Mean Face - 0.025* PD(3)')
plt.axis('off')
k = 2; new_face = meanX - 0.0* s[k]* PD[k,:]
rotated_img = scipy.ndimage.rotate(np.reshape(new_face,[Nx,Ny]), -90)
plt.subplot(338).imshow(rotated_img, interpolation='nearest', cmap='Greys_r')
plt.title('Mean Face')
plt.axis('off')
k = 2; new_face = meanX + 0.025* s[k]* PD[k,:]
rotated_img = scipy.ndimage.rotate(np.reshape(new_face,[Nx,Ny]), -90)
plt.subplot(339).imshow(rotated_img, interpolation='nearest', cmap='Greys_r')
plt.title('Mean Face + 0.025* PD(3)')
plt.axis('off')



