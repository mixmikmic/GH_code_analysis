# vim: set filetype=python:

# test using scikit-learn unit tests for linear classifier
from skbayes.linear_models import EBLogisticRegression,VBLogisticRegression
from sklearn.utils.estimator_checks import check_estimator
check_estimator(EBLogisticRegression)
check_estimator(VBLogisticRegression)
print 'Passed all tests'

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import cm
get_ipython().magic('matplotlib inline')

# create data set 
np.random.seed(0)
n_samples  = 500
x          = np.random.randn(n_samples,2)
x[0:250,0] = x[0:250,0] + 3
x[0:250,1] = x[0:250,1] - 3
y          = -1*np.ones(500)
y[0:250]   = 1
eblr = EBLogisticRegression(tol_solver = 1e-3)
vblr = VBLogisticRegression()   
eblr.fit(x,y)
vblr.fit(x,y)

# create grid for heatmap
n_grid = 500
max_x      = np.max(x,axis = 0)
min_x      = np.min(x,axis = 0)
X1         = np.linspace(min_x[0],max_x[0],n_grid)
X2         = np.linspace(min_x[1],max_x[1],n_grid)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(x2,(n_grid**2,))

eblr_grid = eblr.predict_proba(Xgrid)[:,1]
vblr_grid = vblr.predict_proba(Xgrid)[:,1]
grids = [eblr_grid, vblr_grid]
lev   = np.linspace(0,1,11)  
titles = ['Type II Bayesian Logistic Regression', 'Variational Logistic Regression']
for title, grid in zip(titles, grids):
    plt.figure(figsize=(8,6))
    plt.contourf(X1,X2,np.reshape(grid,(n_grid,n_grid)),
                 levels = lev,cmap=cm.coolwarm)
    plt.plot(x[y==-1,0],x[y==-1,1],"bo", markersize = 3)
    plt.plot(x[y==1,0],x[y==1,1],"ro", markersize = 3)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    
    

from sklearn.datasets import make_blobs
from sklearn.metrics import classification_report
from matplotlib import cm
from sklearn.cross_validation import train_test_split
centers = [(-3, -3), (-3, 3), (3, 3)]
n_samples = 600

# create training & test set
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)
X, x, Y, y = train_test_split(X, y, test_size=0.5, random_state=42)

# fit rvc & svc
vblr = VBLogisticRegression()
eblr = EBLogisticRegression()
vblr.fit(X,Y)
eblr.fit(X,Y)

# create grid
n_grid = 100
max_x      = np.max(x,axis = 0)
min_x      = np.min(x,axis = 0)
X1         = np.linspace(min_x[0],max_x[0],n_grid)
X2         = np.linspace(min_x[1],max_x[1],n_grid)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(x2,(n_grid**2,))


eb_grid = eblr.predict_proba(Xgrid)
vb_grid = vblr.predict_proba(Xgrid)
grids   = [eb_grid, vb_grid]
names   = ['EBLogisticRegression','VBLogisticRegression']
classes = np.unique(y)

# plot heatmaps
for grid,name in zip(grids,names):
    fig, axarr = plt.subplots(nrows=1, ncols=3, figsize = (20,8))
    for ax,cl,model in zip(axarr,classes,grid.T):
        ax.contourf(x1,x2,np.reshape(model,(n_grid,n_grid)),cmap=cm.coolwarm)
        ax.plot(x[y==cl,0],x[y==cl,1],"ro", markersize = 5)
        ax.plot(x[y!=cl,0],x[y!=cl,1],"bo", markersize = 5)
    plt.suptitle(' '.join(['Decision boundary for',name,'OVR multiclass classification']))
    plt.show()
    
print "\n === EBLogisticRegression ==="
print classification_report(y,eblr.predict(x))
print "\n === VBLogisticRegression ==="
print classification_report(y,vblr.predict(x))

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

# create data set 
np.random.seed(0)
n_samples  = 500
n_features = 400
# create training & test set
x           = np.random.randn(n_samples,n_features)
centers     = [[0,-1],[2,1]]
x[:,0:2], y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=0)
x,x_test,y,y_test = train_test_split(x,y,test_size = 0.5,
                                     random_state = 0)

eblr = EBLogisticRegression(tol_solver = 1e-2)
vblr = VBLogisticRegression()   
lr = LogisticRegression(C = 1e+3) # LR without regularization
eblr.fit(x,y)
vblr.fit(x,y)
lr.fit(x,y)

# create grid for heatmap
n_grid = 500
max_x      = np.max(x,axis = 0)
min_x      = np.min(x,axis = 0)
X1         = np.linspace(min_x[0],max_x[0],n_grid)
X2         = np.linspace(min_x[1],max_x[1],n_grid)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(x2,(n_grid**2,))
Xg         = np.random.randn(Xgrid.shape[0],n_features)
Xg[:,0:2]  = Xgrid

eblr_grid = eblr.predict_proba(Xg)[:,1]
vblr_grid = vblr.predict_proba(Xg)[:,1]
lr_grid   = lr.predict_proba(Xg)[:,1]
grids = [eblr_grid, vblr_grid,lr_grid]
lev   = np.linspace(0,1,11)  

titles = ['Type II Bayesian Logistic Regression', 'Variational Logistic Regression',
          'Logistic Regression without regularization']

for title, grid in zip(titles, grids):
    plt.figure(figsize=(8,6))
    plt.contourf(X1,X2,np.reshape(grid,(n_grid,n_grid)),
                 levels = lev,cmap=cm.coolwarm)
    plt.plot(x[y==0,0],x[y==0,1],"bo", markersize = 3)
    plt.plot(x[y==1,0],x[y==1,1],"ro", markersize = 3)
    plt.colorbar()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

from sklearn.metrics import classification_report
print "\n === EBLogisticRegression ==="
print classification_report(y_test,eblr.predict(x_test))
print "\n === VBLogisticRegression ==="
print classification_report(y_test,vblr.predict(x_test))
print "\n === LogisticRegression without regularization ==="
print classification_report(y_test,lr.predict(x_test))



