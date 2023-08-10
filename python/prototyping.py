from __future__ import division
import numpy as np
from scipy import stats
from scipy import spatial

# Import training data

train = np.genfromtxt('data/trn_data.csv', delimiter=',',skip_header=True)
print train[:5], train.shape

stats.describe(train[:,-1])

test = np.genfromtxt('data/tst_locations.csv', delimiter=',',skip_header=True)
print test[:5], test.shape

# Matrix of coordinates
X = train[:,:-1]
print X[:5]

# Matrix of measured values
Y = train[:,-1:]
print Y[:5]

train[:5,:2]

test[:2]

d = spatial.distance_matrix(train[:5,:2], test[:2])
print d

l = .4
np.exp(-(d**2) / 2*l**2)

def covariance(x, y, l):     
    d = spatial.distance_matrix(x,y)
    K = np.exp(-(d**2) / (2*l*l))
    return K

covariance(X[:5], X[:5], 5)

K = covariance(X,X,5)

K.shape

covariance(X,test,5).shape

test.shape

inv = np.linalg.inv(K+np.var(Y)*np.eye(len(X)))
print inv

K_test = covariance(test, X, 5)
print K_test.dot(inv)

print K_test.dot(inv).dot(Y)

def predictive_mean(x, x_test,y,l,indices=False):
    
    K_xtest_x = covariance(x_test, x, l)

    K = covariance(x, x, l)
    
    sigma_sq_I = np.var(y)*np.eye(len(x))
    inv = np.linalg.inv(K+sigma_sq_I)
    
    predictions = K_xtest_x.dot(inv).dot(y)

    if indices:
        return np.concatenate([x_test, predictions], axis=1)
    else:
        return predictions

predictions = predictive_mean(X, test, Y, 5)

predictions[:5]

test[:5]

np.concatenate([test, predictions], axis=1)

np.genfromtxt('data/grid.csv', delimiter=',')

def make_grid(bounding_box, ncell):
    xmax, xmin, ymax, ymin = bounding_box
    xgrid = np.linspace(xmin, xmax, ncell)
    ygrid = np.linspace(ymin, ymax, ncell)
    mX, mY = np.meshgrid(xgrid, ygrid)
    ngridX = mX.reshape(ncell*ncell, 1)
    ngridY = mY.reshape(ncell*ncell, 1)
    return np.concatenate((ngridX, ngridY), axis=1)

bounding_box = [38.3, 39.3, -120.0, -121.0]

xmax, xmin, ymax, ymin = bounding_box

ncell=3
xgrid = np.linspace(xmin, xmax, ncell)
ygrid = np.linspace(ymin, ymax, ncell)

mX, mY = np.meshgrid(xgrid, ygrid)

ngridY = mY.reshape(ncell*ncell, 1)

ngridY

grid = make_grid(bounding_box=bounding_box, ncell=5)

predictive_mean(x=X, x_test=grid, y=Y, l=1.3, indices=True)

results = []
for i in np.arange(1,100):
    i = i/10
    r = predictive_mean(x=X, x_test=grid, y=Y, l=i, indices=True)
    results.append(r[5,-1])    

results

print train[(train[:,0] < 39.3) & (train[:,0] > 38.3) & (train[:,1] > -121.0) & (train[:,1] < -120.0)]



