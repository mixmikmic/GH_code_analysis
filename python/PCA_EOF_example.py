get_ipython().magic('matplotlib inline')
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import csv

# read the data from the csv file
data = np.genfromtxt('m80.csv', delimiter='')
data_mean =  np.mean(data,0)

# and plot out a few profiles and the mean depth.
plt.figure()
rows = [ 9,59,99]
labels = [ 'slow','medium','fast']
for i,row in enumerate(rows):
    plt.plot(data[row,:],label=labels[i])
    plt.hold(True)
plt.plot(data_mean,'k--',label='mean')
plt.xlabel('Distance across axis (km)')
plt.ylabel('Relative Elevation (m)')
plt.legend(loc='best')
plt.title('Example cross-axis topography of mid-ocean ridges')
plt.show()

plt.figure()
X = data - data_mean
plt.imshow(X)
plt.xlabel('Distance across axis (Km)')
plt.ylabel('Relative Spreading Rate')
plt.colorbar()
plt.show()

# now calculate the SVD of the de-meaned data matrix
U,S,Vt = la.svd(X,full_matrices=False)

# plot the singular values
plt.figure()
plt.semilogy(S,'bo')
plt.grid()
plt.title('Singular Values')
plt.show()

# and cumulative percent of variance
g = np.cumsum(S*S)/np.sum(S*S)
plt.figure()
plt.plot(g,'bx-')
plt.title('% cumulative percent variance explained')
plt.grid()
plt.show()

plt.figure()
num_EOFs=3
for row in range(num_EOFs):
    plt.plot(Vt[row,:],label='EOF{}'.format(row+1))
plt.grid()
plt.xlabel('Distance (km)')
plt.title('First {} EOFs '.format(num_EOFs))
plt.legend(loc='best')
plt.show()

# recontruct the data using the first 5 EOF's
k=5
Ck = np.dot(U[:,:k],np.diag(S[:k]))
Vtk = Vt[:k,:]
data_k = data_mean + np.dot(Ck,Vtk)

plt.figure()
plt.imshow(data_k)
plt.colorbar()
plt.title('reconstructed data')
plt.show()

# show the original 3 profiles and their recontructed values using the first k EOF's
for i,row in enumerate(rows):
    plt.figure()
    plt.plot(data_k[row,:],label='k={}'.format(k))
    plt.hold(True)
    plt.plot(data[row,:],label='original data')
    Cstring = [ '{:3.0f},  '.format(Ck[row,i]) for i in range(k) ]
    plt.title('Reconstruction profile {}:\n C_{}='.format(row,k)+''.join(Cstring))
    plt.legend(loc='best')
    plt.show()
    

# plot the data in the plane defined by the first two principal components
plt.figure()
plt.scatter(Ck[:,0],Ck[:,1])
plt.xlabel('$V_1$')
plt.ylabel('$V_2$')
plt.grid()
plt.title('Projection onto the first two principal components')
plt.show()

# Or consider the degree of assymetry (EOF 3) as a function of spreading rate
plt.figure()
plt.plot(Ck[:,2],'bo')
plt.xlabel('Spreading rate')
plt.ylabel('$C_3$')
plt.grid()
plt.title('Degree of assymetry')
plt.show()



