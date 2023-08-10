# vim: set filetype=python:

from skbayes.rvm_ard_models import ClassificationARD
from sklearn.utils.estimator_checks import check_estimator
check_estimator(ClassificationARD)

import sklearn
from matplotlib import cm
import matplotlib
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.grid_search import GridSearchCV
from scipy.stats import multivariate_normal as mvn

def generate_dataset(n_samples = 500, n_features = 100,
                     cov_class_1 = [[0.9,0.1],[1.5,.2]],
                     cov_class_2 = [[0.9,0.1],[1.5,.2]],
                     mean_class_1 = (-1,0.4),
                     mean_class_2 = (-1,-0.4)):
    ''' Generate binary classification problem with two relevant features'''
    X   = np.random.randn(n_samples, n_features)
    Y   = np.ones(n_samples)
    sep = int(n_samples/2)
    Y[0:sep]     = 0
    X[0:sep,0:2] = np.random.multivariate_normal(mean = mean_class_1, 
                   cov = cov_class_1, size = sep)
    X[sep:n_samples,0:2] = np.random.multivariate_normal(mean = mean_class_2,
                        cov = cov_class_2, size = n_samples - sep)
    return X,Y

X,Y = generate_dataset()
plt.figure(figsize = (8,6))
plt.plot(X[Y==0,0],X[Y==0,1],"bo", markersize = 3)
plt.plot(X[Y==1,0],X[Y==1,1],"ro", markersize = 3)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title("Example of dataset")
plt.show()

n_samples  = 500
n_features = 500
np.random.seed(1)

X,Y = generate_dataset(n_samples,n_features)

# training & test data
X,x,Y,y = train_test_split(X,Y, test_size = 0.4)

# fit & predict using ARD
clard = ClassificationARD()
clard.fit(X,Y)
        
# fit & predict using cross validated Logistic Regression
# with L1 / L2 penalty
lrcv_l2 = LogisticRegressionCV(penalty = 'l2')
lrcv_l2.fit(X,Y)
lrcv_l1 = LogisticRegressionCV(penalty = 'l1', solver = 'liblinear')
lrcv_l1.fit(X,Y)
    
# construct grid    
n_grid = 100
max_x      = np.max(x[:,0:2],axis = 0)
min_x      = np.min(x[:,0:2],axis = 0)
X1         = np.linspace(min_x[0],max_x[0],n_grid)
X2         = np.linspace(min_x[1],max_x[1],n_grid)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(x2,(n_grid**2,))
Xg         = np.random.randn(n_grid**2,n_features)
Xg[:,0]    = Xgrid[:,0]
Xg[:,1]    = Xgrid[:,1]

# estimate probabilities for grid data points
clard_grid    = clard.predict_proba(Xg)[:,1]
lrcv_l2_grid  = lrcv_l2.predict_proba(Xg)[:,1]
lrcv_l1_grid  = lrcv_l1.predict_proba(Xg)[:,1]

# plot data
titles = ["ClassificationARD","L2 Logistic Regression",
          "L1 Logistic Regression"]
models  = [clard_grid,lrcv_l2_grid,lrcv_l1_grid]

fig, axarr = plt.subplots(nrows=1, ncols=3, figsize = (24,8))

for ax,title,model in zip(axarr,titles,models):
    ax.contourf(X1,X2,np.reshape(model,(n_grid,n_grid)),cmap=cm.coolwarm)
    ax.plot(x[y==0,0],x[y==0,1],"bo", markersize = 5)
    ax.plot(x[y==1,0],x[y==1,1],"ro", markersize = 5)
    ax.set_title(title)
plt.show()

print('Number of relevant features \n '
      ' \n -- ClassificationARD : {0} \n '
      ' \n -- LRCV L1 : {1} \n '
      ' \n -- LRCV L2 : {2} \n ').format(np.sum(clard.coef_!=0),
                                     np.sum(lrcv_l1.coef_!=0),
                                     np.sum(lrcv_l2.coef_!=0))

from sklearn.datasets import make_blobs
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC as SVC
from sklearn.metrics import classification_report
cs = 5 # blob center scaler
centers = [(0, -cs), (-cs, 0), (cs, 0), (0,cs)]
n_samples  = 500
n_features = 500 

np.random.seed(0)
X           = np.random.random([n_samples,n_features])*cs 
X[:,0:2], y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=1)
X, x, Y, y = train_test_split(X, y, test_size=0.2, random_state=42)

# fit RVC
rvm = ClassificationARD()
rvm.fit(X,Y)

# fit LinearSVC
svc = GridSearchCV(SVC(), param_grid = {"C":np.logspace(-3,3,7)},
                   cv = 5)
svc.fit(X,Y)

# fit Logistic Regression with L1 penalization
lone = LogisticRegressionCV(penalty = 'l1', solver = 'liblinear')
lone.fit(X,Y)

# fit Logistic Regression with L2 penalization
ltwo = LogisticRegressionCV(penalty = 'l2')
ltwo.fit(X,Y)

# create grid
n_grid     = 100
max_x      = np.max(X,axis = 0)
min_x      = np.min(X,axis = 0)
X1         = np.linspace(min_x[0],max_x[0],n_grid)
X2         = np.linspace(min_x[1],max_x[1],n_grid)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(x2,(n_grid**2,))
Xg         = np.random.randn(n_grid**2,n_features)
Xg[:,0]    = Xgrid[:,0]
Xg[:,1]    = Xgrid[:,1]

# predictions
rv_grid = rvm.predict_proba(Xg)
sv_grid = svc.decision_function(Xg) 
lone_grid = lone.predict_proba(Xg)
ltwo_grid = ltwo.predict_proba(Xg)
grids = [rv_grid,sv_grid,lone_grid,ltwo_grid]
names = ['RVC','LinearSVC','L1 LogisticRegressionCV','L2 LogisticRegressionCV']

# plot data
colors = ['ro','go','ko','co']
classes = np.unique(y)
plt.figure(figsize = (12,6))
for cl,col in zip(classes,colors):
    plt.plot(x[y==cl,0],x[y==cl,1],col)   
plt.xlabel('relevant feature 1')
plt.ylabel('relevant feature 2')
plt.title('Data for multiclass classification problem')
plt.show()

# plot heatmaps
for grid,name in zip(grids,names):
    fig, axarr = plt.subplots(nrows=1, ncols=4, figsize = (12,4))
    for ax,cl,model in zip(axarr,classes,grid.T):
        ax.contourf(x1,x2,np.reshape(model,(n_grid,n_grid)),cmap=cm.coolwarm)
        ax.plot(x[y==cl,0],x[y==cl,1],"ro", markersize = 5)
        ax.plot(x[y!=cl,0],x[y!=cl,1],"bo", markersize = 5)
    plt.suptitle(' '.join([name,'OVR Multiclass']))
    plt.show()

print "          =======   Relevance Vector Classifier   ======= \n"
print classification_report(y,rvm.predict(x))
print "          =======   Support Vector Classifier   ======= \n"
print classification_report(y,svc.predict(x))
print "          =======   L1 Logistic Regression   ======= \n"
print classification_report(y,rvm.predict(x))
print "          =======   L2 Logistic Regression   ======= \n"
print classification_report(y,svc.predict(x))

