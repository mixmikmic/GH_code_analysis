import pandas as pd
import numpy as np
from sklearn import model_selection
train = pd.read_csv("train.csv")
features = train.columns[1:]
X = train[features]
X= X.as_matrix()
y = train['label']
y = y.as_matrix()
X, x, Y, y = model_selection.train_test_split(X,y,test_size=0.1,random_state=0)
print X.shape

N = X.shape[0]
d = X.shape[1]+1

data = X
data = np.hstack((np.ones([data.shape[0],1]),data))

for i in range (0, len(Y)):
    if Y[i] != 0:
        Y[i] = 1
        
y = Y

import copy
from itertools import izip
from math import sqrt
import numpy as np
from operator import mul
import matplotlib.mlab as mlab
from scipy.stats import bernoulli, norm
from scipy import optimize, stats
import matplotlib.pyplot as plt
from matplotlib.mlab import bivariate_normal
from scipy import integrate
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.utils import shuffle
from sklearn import linear_model, datasets
import time
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target
X = X[0:100]
Y = Y[0:100]
h = .02  # step size in the mesh

logreg = linear_model.LogisticRegression(C=1e5)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X, Y)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xticks(())
plt.yticks(())

plt.show()


print logreg.multi_class
print logreg.intercept_
print logreg.coef_

d = 3
N = 100
tol = 10**4

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

def ll(x, y, mu, sig):
    if sig < 0:
        sig = -sqrt(-sig)
    else:
        sig = sqrt(sig)
        
    et = np.exp(x)
    z = np.log(et / (1 + et))
    e = y*z + (1-y)*np.log(1-et / (1 + et)) + np.log(gaussian(x, mu, (sig)))
    return e

iris = datasets.load_iris()
X = iris.data[:, 0:1]  # we only take the first two features.
Y = iris.target
X = X[0:100]
Y = Y[0:100]
data = X
y = Y
data.sort()
print y
print data

other_label = 0
def generate_class_data():
    # create data set 
    np.random.seed(0)
    n_samples = 100

    # Gaussian mean=0, variance=1
    data = np.random.randn(n_samples,1)

    # seperate shifting them 1 away
    data[0:50] = data[0:50] + 1
    #data[0:250] = data[0:250] - 2

    # define labels
    y = other_label*np.ones(100)
    y[0:50] = 1
    return data, y


data, y = generate_class_data()

# iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target
#X = X[0:100]
#Y = Y[0:100]

for i in range(0, len(Y)):
    if i>99:
        Y[i] = 0
    else: 
        Y[i] = 1
data = X
y = Y
#print X
print Y
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(data, y)
w_true = logreg.coef_[0]
w0_true = logreg.intercept_[0]
print w_true
print w0_true
plt.plot(data[y==other_label,0],data[y==other_label,1],"bo", markersize = 3)
plt.plot(data[y==1,0],data[y==1,1],"ro", markersize = 3)
#plt.plot(data[y==other_label],data[y==other_label],"bo", markersize = 3)
#plt.plot(data[y==1],data[y==1],"ro", markersize = 3)
plt.title("Generated data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

data = np.hstack((np.ones([data.shape[0],1]),data))
data, y = shuffle(data, y, random_state=0)
#print y
#print data
d = 3
N = 100

def project_subspace(Xi, mu, sig, Mi, Vi):
    
    XT = Xi[:,None] #Xi.T
    X = Xi[None,:] #Xi
    
    # project cavity on 1D subspace
    scalar_moment = np.linalg.inv(X.dot(sig).dot(XT))
    scalar_moment2 = X.dot(mu)
    
    
    V_cavity = 1/(scalar_moment - 1/Vi)
    M_cavity = V_cavity * (scalar_moment * scalar_moment2 - Mi/Vi)

    return M_cavity[0][0], V_cavity[0][0]


def compute_cavity1(Xi, r_i, A_i, r, A):
    

    XT = Xi[:,None] #Xi.T
    X = Xi[None,:] #Xi
    
    scalar_moment = np.linalg.inv(X.dot(A).dot(XT))
    scalar_moment2 = X.dot(np.linalg.inv(A).dot(r))
    
    Mi = X.dot( np.linalg.inv(A_i).dot(r_i) )
    Vi = X.dot((np.linalg.inv(A_i))).dot(XT)

    V_cavity = np.linalg.inv( scalar_moment - 1/Vi )
    
    M_cavity = V_cavity * ( scalar_moment*scalar_moment2 -  Mi/Vi)
    
    return M_cavity[0][0], V_cavity[0][0]

Xi = np.array([1,1])
presmui = np.array([2,1])
presi = np.array([[1, 0.5], [1,1]])
pres_mu = np.array([0.5,1.2])
pres = np.array([[1.3, 1.5], [1.7,1.3]])


cavity_M, cavity_V = compute_cavity1(Xi, presmui, presi, pres_mu, pres)
print cavity_M, cavity_V

def logit_i(x):
    trunc = 8.
    exponent = np.clip(x, -trunc, trunc)
    exponent = np.float64(exponent)
    return  1 / (1 + pow(np.e, -exponent))  
    

def compute_moments(y, M_cavity, V_cavity):
    
    if V_cavity < 0:
        sd = -sqrt(-V_cavity)
    else:
        sd = sqrt(V_cavity)

    lower_bound = M_cavity - 10*sqrt(np.abs(V_cavity))
    upper_bound = M_cavity + 10*sqrt(np.abs(V_cavity))
    
    f = lambda x: gaussian(x, M_cavity, sd) *                 pow(logit_i(x), y) * pow((1-logit_i(x)), (1 - y))
    E0 = integrate.quad(f, lower_bound, upper_bound)[0]

    f = lambda x: gaussian(x, M_cavity, sd) *                 pow(logit_i(x), y) * pow((1-logit_i(x)), (1 - y)) * x
    E1 = integrate.quad(f, lower_bound, upper_bound)[0] 
    
    f = lambda x: gaussian(x, M_cavity, sd) *                 pow(logit_i(x), y) * pow((1-logit_i(x)), (1 - y)) * (x**2)
    E2 = integrate.quad(f, lower_bound, upper_bound)[0] 
    
    if E0 == 0:
        print "WARNING E0 was 0"
        E0 = 0.00000001
        
    M_new = E1 / E0
    V_new = E2 / E0 - (E1 / E0)**2 


    return M_new, V_new



def ll2log(x, y, mu, sig):
    if sig < 0:
        sig = -sqrt(-sig)
    else:
        sig = sqrt(sig)
        
    et = np.exp(x)
    z = np.log(et / (1 + et))
    e = y*z + (1-y)*np.log(1-et / (1 + et)) + np.log(gaussian(x, mu, (sig)))
    return e

#max_x = optimize.minimize(lambda x: -ll2log(x, y[i], M_cavity, V_cavity),  M_cavity)
#M_new = max_x.x[0]
#V_new = max_x.hess_inv[0][0]

def transform_back(Xi, Mi, Vi):
    
    V_inv = 1 / Vi
    
    r = Xi.T * Mi/Vi
    
    XT = Xi[:,None] #Xi.T
    X = Xi[None,:] #Xi
    
    A = V_inv*(XT).dot(X) # legit 
    #A = np.identity(2) * (Xi.T * V_inv).dot(Xi) # spherical
    
    return r, A

transform_back(np.array([1,2.5]), 2, 0.5)

def compute_cavity(r_i, A_i, r, A):

    # remove factor
    r_cavity = r - r_i  
    A_cavity = A - A_i

    return r_cavity, A_cavity

def update_post(r_i, A_i, cavity_r, cavity_A):

    r = cavity_r + r_i  
    
    A = cavity_A + A_i
    
    return r, A

def ss_to_cov(r, A):
    if is_invertible(A):
        sig = np.linalg.inv(A)
    else:
        # handle it
        sig = np.linalg.pinv(A)
    #sig = np.linalg.inv(A)
    mu = sig.dot(r)
    
    return mu, sig

def is_invertible(a):
     return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def plot_projected(M_new, V_new, M_cavity, V_cavity, y):
    
    
    if V_new < 0:
        sd1 = -sqrt(-V_new)
    else:
        sd1 = sqrt(V_new)
    if V_cavity < 0:
        sd2 = -sqrt(-V_cavity)
    else:
        sd2 = sqrt(V_cavity)
        
    x = np.linspace(-10,10,100)    
    
    plt.plot(x, mlab.normpdf(x, M_new, sd1), label="New Aprrox")
    plt.plot(x, mlab.normpdf(x, M_cavity, sd2), label="Cavity/Prior")
    
    
    # parts of approximations
    if V_cavity < 0:
        sd = -sqrt(-V_cavity)
    else:
        sd = sqrt(V_cavity)
    f = lambda x: gaussian(x, M_cavity, sd) * pow(logit_i(x), y) * pow((1-logit_i(x)), (1 - y))
    plt.plot(x, map(f, x), label="Tilted distribution")
    
    f = lambda x: pow(logit_i(x), y) * pow((1-logit_i(x)), (1 - y))
    plt.plot(x, map(f, x), label="Likelihood")
    
    plt.legend()
    plt.show()
    
    
def plot_posterior(m_x, v_x):
    N = 80
    limit = 12
    X = np.linspace(-limit, limit, N)
    Y = np.linspace(-limit, limit, N)
    X, Y = np.meshgrid(X, Y)

    #v_x = -0.5*v_x
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y

    z = multivariate_gaussian(pos, m_x, v_x)

    limits = 3
    plt.imshow(z, extent=(-limit, limit, limit, -limit))
    plt.colorbar()
    plt.show()
    
def plot_posterior_1d(r_old, A_old, r_new, A_new, ri, Ai):
    
    
    mu_old, sig_old = ss_to_cov(r_old, A_old)
    mu_new, sig_new = ss_to_cov(r_new, A_new)
    mu_i, sig_i = ss_to_cov(ri, Ai)
    

    x = np.linspace(-150,150,300)
    
    if sig_i[0][0] < 0:
        sdi = -sqrt(-sig_i[0][0])
    else:
        sdi = sqrt(sig_i[0][0])
        
    if sig_i[1][1] < 0:
        sdi2 = -sqrt(-sig_i[1][1])
    else:
        sdi2 = sqrt(sig_i[1][1])
        
    
    plt.plot((w_true, w_true), (0, 1.5), label="target mean")
    plt.plot((w0_true, w0_true), (0, 1.5), label="target mean")
    plt.plot(x, gaussian(x, mu_old[0], sqrt(sig_old[0][0])), label="Old posterior")
    plt.plot(x, gaussian(x, mu_new[0], sqrt(sig_new[0][0])), label="New posterior")
    plt.plot(x, mlab.normpdf(x, mu_i[0], sdi), label="Updated factor")
    
    plt.plot(x, gaussian(x, mu_old[1], sqrt(sig_old[1][1])), label="Old posterior 2")
    plt.plot(x, gaussian(x, mu_new[1], sqrt(sig_new[1][1])), label="New posterior 2")
    plt.plot(x, mlab.normpdf(x, mu_i[1], sdi2), label="Updated factor 2")
    
    plt.legend()
    plt.show()
    
def sanity_check(M_new, V_new, Xi):
    XT = Xi[:,None] #Xi.T
    X = Xi[None,:] #Xi
    
    A = (XT * 1/V_new).dot(X) 
    r = M_new/V_new * Xi
    
    sig = np.linalg.inv(A)
    mu = sig.dot(r)
    
    return mu, sig

def subtract_moments(M, V, M_cavity, V_cavity):
    
    Vi = 1/(1/V - 1/V_cavity)
    Mi = Vi * (M/V - M_cavity/V_cavity)
    
    return Mi, Vi


def project_factor_subspace(Xi, r, A):
    XT = Xi[:,None] #Xi.T
    X = Xi[None,:] #Xi
    
    # invertible check
    if is_invertible(A):
        sig = np.linalg.inv(A)
    else:
        # handle it
        sig = np.linalg.pinv(A)
    #sig = np.linalg.inv(A)
    mu = sig.dot(r)
    
    Mi = X.dot(mu)
    Vi = X.dot(sig).dot(XT)
    
    return Mi, Vi

def plot_posterior_1d_multi(r_old, A_old, r_new, A_new, ri, Ai):
    global d
    
    mu_old, sig_old = ss_to_cov(r_old, A_old)
    mu_new, sig_new = ss_to_cov(r_new, A_new)

    x = np.linspace(-200,200,400)
    match_colors = ["r", "b", "g"]
    for i in range(0, d):
        if i<(d-1):
            plt.plot((w_true[i], w_true[i]), (0, 1.5), color=match_colors[i+1])
        plt.plot(x, gaussian(x, mu_new[i], sqrt(sig_new[i][i])), label=i, color=match_colors[i])
    plt.plot((w0_true, w0_true), (0, 1.5), color=match_colors[0])

    plt.legend()
    plt.show()

# initialise

prior_post = 5000
prior_factor = 10000000
r_new = np.zeros(d)
A_new = np.linalg.inv(np.identity(d) * prior_post) 
  
    

r = [] 
A = []
for i in range(N):
    A.append(1. / prior_factor)
    r.append(0)
    
    
r_old = r_new
A_old = A_new

# max iterations
max_iter = 10

def alt_project(Xi, r, A, ri, Ai):
    
    mu, sig = ss_to_cov(r, A)
    
    A = Xi.dot(sig).dot(Xi.T)
    B = Xi.dot(mu)

    if Ai != A:
        V_i = 1./((1./A)-(Ai))
    else:
        V_i = 1./(1./A)


    if B/A == ri:
        M_i = V_i*(B/A)
    else:
        M_i = V_i * ((1.*B/A) - (1.*ri))
    return M_i, V_i

def project_up(Xi, ri, V_inv):
    
    r = Xi.T * ri
    
    XT = Xi[:,None] #Xi.T
    X = Xi[None,:] #Xi
    
    A = V_inv*(XT).dot(X) # legit 

    return r, A

# Outer loop
mu, sig = ss_to_cov(r_new, A_new)
#plot_posterior(mu, sig)
tm1 = time.time()
for iteration in xrange(3): # max iterations
    print "Iteration ", iteration

    for i in xrange(N):

        
        ######################################################################################
        
        # 1 & 2 Try alternative projection with dimension reduction
        
        M_cavity, V_cavity = alt_project(data[i], r_old, A_old, r[i],A[i])      

        
        ######################################################################################

        # 3 compute moments
        
        M_new, V_new = compute_moments(y[i], M_cavity, V_cavity)
        #max_x = optimize.minimize(lambda x: -ll(x, y[i], M_cavity, V_cavity),  M_cavity)
        #M_new = max_x.x[0]
        #V_new = max_x.hess_inv[0][0]
        
        if V_new == 0:
            V_new = np.infty
            print " WARNING MATCHED VARIANCE IS 0\n"
            #break
            
        ######################################################################################

        # 4 remove updated factor from matched moments
        
        Mi_approx, Vi_approx = subtract_moments(M_new, V_new, M_cavity, V_cavity)
        
        deltar =  Mi_approx / Vi_approx - r[i]
        deltaA = 1./ Vi_approx - A[i]
        
        ######################################################################################

        # 5 store updated factor in 1 dimension
        
        r[i] = Mi_approx / Vi_approx
        A[i] = 1./ Vi_approx

        ######################################################################################

        # 6 project change in factor up
        
        deltar, deltaA = project_up(data[i], deltar, deltaA)

        ######################################################################################

        # 7 update posterior
        r_new = r_old + deltar
        A_new = A_old + deltaA


        ###################################################################################### 
        

        plot_posterior_1d_multi(r_old, A_old, r_new, A_new, r[i], A[i])
 
        r_old = r_new
        A_old = A_new

tm2 = time.time()
print "Time"
print tm2 - tm1

print "New posterior"
post_mu,post_sig = ss_to_cov(r_new, A_new)
print pd.DataFrame(post_mu)
print pd.DataFrame(post_sig)  

n_samples = 100
x = np.random.multivariate_normal(post_mu,post_sig , n_samples).T

intercepts = []
slopes = []
for i in range (0, n_samples):
    w0 = x[0][i]
    w1 = x[1][i]
    w2 = x[2][i]
    intercepts.append(-w0/w2)
    slopes.append(-w1/w2)
                      
x = np.linspace(0,10,100)

# get lines on x axis
intercepts = np.array(intercepts)
slopes = np.array(slopes)
abline_values = [slopes * i + intercepts for i in x]

# Plot the best fit line over the actual values
plt.plot(x, abline_values, 'g')

# iris
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target
#X = X[0:100]
#Y = Y[0:100]

for i in range(0, len(Y)):
    if i>99:
        Y[i] = 0
    else: 
        Y[i] = 1
data = X
y = Y
#print X
print Y
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(data, y)
w_true = logreg.coef_[0]
w0_true = logreg.intercept_[0]
print w_true
print w0_true
plt.plot(data[y==other_label,0],data[y==other_label,1],"bo", markersize = 3)
plt.plot(data[y==1,0],data[y==1,1],"ro", markersize = 3)
#plt.plot(data[y==other_label],data[y==other_label],"bo", markersize = 3)
#plt.plot(data[y==1],data[y==1],"ro", markersize = 3)
plt.title("Generated data")
plt.xlabel("x")
plt.ylabel("y")

plt.ylim(0, 6)

plt.show()

from scipy.special import expit
def predict_probit(X, mu, sigma):

    # probit approximation to predictive distribution
    ks = 1. / ( 1. + np.pi*sigma**2 / 8)**0.5
    prob = expit(mu*ks)

    return prob


#print y
# create grid for heatmap
x = data
n_grid = 50
max_x      = np.max(x,axis = 1)+3
min_x      = np.min(x,axis = 1)-3
X1         = np.linspace(min_x[0],max_x[0],n_grid)
X2         = np.linspace(min_x[1],max_x[1],n_grid)
x1,x2      = np.meshgrid(X1,X2)
Xgrid      = np.zeros([n_grid**2,2])
Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
Xgrid[:,1] = np.reshape(x2,(n_grid**2,))

print Xgrid.shape
#eblr_grid = eblr.predict_proba(Xgrid)[:,1]
eblr_grid = []
for i in range (0,len(Xgrid)):
    
    # change basis
    #test_vec = np.array([1, Xgrid[i][0], Xgrid[i][1], Xgrid[i][0]*Xgrid[i][1], Xgrid[i][0]**2, Xgrid[i][1]**2])
    test_vec = np.array([1, Xgrid[i][0], Xgrid[i][1]])
    M_test, V_test = project_factor_subspace(test_vec, r_new, A_new)
    eblr_grid.append(predict_probit(test_vec, M_test, V_test))
    
eblr_grid = np.array(eblr_grid)


lev   = np.linspace(0,1,11)  
plt.figure(figsize=(8,6))
plt.contourf(X1,X2,np.reshape(eblr_grid,(n_grid,n_grid)),
             levels = lev,cmap=cm.coolwarm)
plt.plot(data[y==0,0],data[y==0,1],"bo", markersize = 3)
plt.plot(data[y==1,0],data[y==1,1],"ro", markersize = 3)
plt.colorbar()
plt.xlabel("x")
plt.ylabel("y")
plt.show()

iris = datasets.load_iris()
X = iris.data[:, 0:2]  # we only take the first two features.
Y = iris.target
X = X[0:100]
Y = Y[0:100]

data, x_test, y, y_test = model_selection.train_test_split(X,Y,test_size=0.2,random_state=0)
data = X
data = np.hstack((np.ones([data.shape[0],1]),data))
y = y
data.sort()
#print y
#print data

x_test = np.hstack((np.ones([x_test.shape[0],1]),x_test))

from scipy.special import expit
def predict_probit(X, mu, sigma):
    
    # probit approximation to predictive distribution
    ks = 1. / ( 1. + np.pi*sigma**2 / 8)**0.5
    prob = expit(mu*ks)

    return prob

# run test
acc = 0

for i in range(0, len(x_test)):
    test_vec = x_test[i]
    M_test, V_test = project_factor_subspace(test_vec, r_new, A_new)
    print predict_probit(test_vec, M_test, V_test)
    print y_test[i]
    if predict_probit(test_vec, M_test, V_test) > 0.5 and y_test[i]==1:
        acc = acc+1
        
print 1.*acc/len(x_test)
print acc
print len(x_test)



