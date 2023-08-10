import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from SIR import SIR
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import seaborn as sns
get_ipython().magic('matplotlib inline')

dataset = pd.read_csv('death_rate.txt',delim_whitespace=True,header =None)
dnp = dataset.drop(0,axis = 1).values
Y = dnp[:,-1]
X = dnp[:,:-1]

sir_ = SIR()
sir_.fit(X,Y)

transformed_vals = np.real(sir_.transform(X))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(transformed_vals[:,0],transformed_vals[:,1],Y)



x_1 = []
x_2 = []
x_3 = []
x_4 = []
x_5 = []
epsilon = []

for i in range(10000):
    epsilon.append(np.random.normal())
    x_1.append(np.random.normal(0,1))
    x_2.append(np.random.normal(0,1))
    x_3.append(np.random.normal(0,1))
    x_4.append(np.random.normal(0,1))
    x_5.append(np.random.normal(0,1))
    
x_1 = np.array(x_1)
x_2 = np.array(x_2)
x_3 = np.array(x_3)
x_4 = np.array(x_4)
x_5 = np.array(x_5)


X = list(zip(x_1,x_2,x_3,x_4,x_5))
X = np.array(X)
epsilon = np.array(epsilon)
y = (x_1/(6+(x_2-4)**2)) + 0.05*epsilon

sir_ = SIR(K=2)
sir_.fit(X,y)

sir_.eigenvalues

sir_.beta

5/ np.sqrt(2**2 + 5**2 +1)

1/ np.sqrt(2**2 + 5**2 +1)

new = sir_.transform(X)
plt.title('Linear Model Example with K = 1 (SIR)')
plt.xlabel('X projected onto e.d.r space')
plt.ylabel('y')
plt.scatter(new,y)
plt.savefig('linexample.jpg')

pca = PCA(n_components =1)
pca.fit(X)
new = pca.transform(X)
plt.title('Linear Model Example with K = 1 (PCA)')
plt.xlabel('X projected onto principal components space')
plt.ylabel('y')
plt.scatter(new,y)
plt.savefig('linexamplepca.jpg')

new = sir_.transform(X)
plt.title('Quadratic Model Example with K = 1 (SIR)')
plt.xlabel('X projected onto e.d.r space')
plt.ylabel('y')
plt.scatter(new,y)
plt.savefig('quadexample.jpg')

pca = PCA(n_components =1)
pca.fit(X)
new = pca.transform(X)
plt.title('Quadratic Model Example with K = 1 (PCA)')
plt.xlabel('X projected onto principal components space')
plt.ylabel('y')
plt.scatter(new,y)
plt.savefig('quadexamplepca.jpg')

new = sir_.transform(X)
plt.title('Rational Model Example with K = 2 (SIR)')
plt.xlabel('X projected onto e.d.r space [first e.d.r direction]')
plt.ylabel('y')
plt.scatter(new[:,0],y)
plt.savefig('ratexample')

pca = PCA(n_components =2)
pca.fit(X)
new = pca.transform(X)
plt.title('Rational Model Example with K = 1 (PCA)')
plt.xlabel('X projected onto principal components space [first principal component]')
plt.ylabel('y')
plt.scatter(new[:,0],y)
plt.savefig('ratexamplepca.jpg')

from sklearn import preprocessing
dataset = pd.read_csv('death_rate.txt',delim_whitespace=True,header =None)
dnp = dataset.drop(0,axis = 1).values
Y = dnp[:,-1]
X = dnp[:,:-1]
X = preprocessing.scale(X)

sir_ = SIR(K = 2)
sir_.fit(X,Y)
pca_ = PCA(n_components = 2)
pca_.fit(X,Y)

X.shape

new1 = pca_.transform(X)
new2 = sir_.transform(X)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

kf = KFold(n_splits = 5)

scores_sir = []
scores_pca = []
scores_ = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    sir_ = SIR(K = 2)
    sir_.fit(X_train,Y_train)
    pca_ = PCA(n_components = 2)
    pca_.fit(X_train,Y_train)
    new1 = pca_.transform(X_train)
    new2 = sir_.transform(X_train)
    
    test1 = pca_.transform(X_test)
    test2 = sir_.transform(X_test)
    
    #pca
    lr1 = LinearRegression()
    lr1.fit(new1,Y_train)
    scores_pca.append(lr1.score(test1,Y_test))
    
    #sir
    lr2 = LinearRegression()
    lr2.fit(new2,Y_train)
    scores_sir.append(lr2.score(test2, Y_test))
    
    #regular
    lr3 = LinearRegression()
    lr3.fit(X_train,Y_train)
    scores_.append(lr3.score(X_test,Y_test))

np.round(scores_sir,3)

np.mean(scores_pca)

np.round(scores_,3)

sir_.beta

plt.scatter(range(1,len(sir_.eigenvalues)+1),sir_.eigenvalues)
plt.xlabel("Sorted Index of Eigenvalue")
plt.ylabel("Eigenvalue")
plt.title("Sorted Eigenvalues from SIR")
plt.savefig("Eigvalues")



