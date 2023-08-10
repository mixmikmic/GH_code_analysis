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
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn import cross_validation
import time
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')

import pandas as pd

train = pd.read_csv("train.csv")
#train = train.sort_values(by='label', ascending=1)
train = train[train['label'].between(0, 2, inclusive=True)]

features = train.columns[1:]
X = train[features]
X= X.as_matrix()
y = train['label']
y = y.as_matrix()
#X, x, Y, y = cross_validation.train_test_split(X,y,test_size=0.1,random_state=0)
print X.shape
print y[0:10]

d = X.shape[1] + 1
N = X.shape[0] 
print N, d

def sigmoid(x):
    trunc = 8.
    exponent = np.clip(x, -trunc, trunc)
    exponent = np.float64(exponent)
    return 1 / (1 + np.exp(-exponent))

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def multivariate_gaussian(pos, mu, Sigma):

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
    eps = 0.0001
    trunc = 8.
    exponent = np.clip(x, -trunc, trunc)
    exponent = np.float64(exponent)
        
    et = np.exp(exponent)
    z = np.log(1 / (1 + et))
    e = y*z + (1-y)*np.log(1-(1 / (1 + et))) + np.log(gaussian(x, mu, sig)/(1/np.sqrt(2*np.pi*sig**2)))
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
    
    sig = np.linalg.inv(A)
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

def alt_project_first_iteration(Xi, r, A):
    
    mu, sig = ss_to_cov(r, A)
    
    A = Xi.dot(sig).dot(Xi.T)
    B = Xi.dot(mu)

    V_i = 1./(1./A)

    M_i = V_i*(B/A)

    return M_i, V_i

def project_up(Xi, ri, V_inv):
    
    r = Xi.T * ri
    
    XT = Xi[:,None] #Xi.T
    X = Xi[None,:] #Xi
    
    A = V_inv*(XT).dot(X) # legit 

    return r, A

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target
#X = X[0:100]
#Y = Y[0:100]

#for i in range(0, len(Y)):
#    if i>50 and i<99:
#        Y[i] = 0
#    else: 
#        Y[i] = 1
data = X
data = np.hstack((np.ones([data.shape[0],1]),data))
y = Y
d = data.shape[1] 
N = data.shape[0] 
print N, d

train = pd.read_csv("train.csv")
train = train[train['label'].between(0, 2, inclusive=True)]

features = train.columns[1:]
X = train[features]
X = X.as_matrix()
y = train['label']
y = y.as_matrix()

min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)


df = pd.DataFrame(data)
train = pd.read_csv("train.csv")
#train = train[train['label'].between(0, 2, inclusive=True)]

zeros = train[train['label'].between(2, 4, inclusive=False)]
ones = train[train['label'].between(4, 6, inclusive=False)]
twos = train[train['label'].between(7, 9, inclusive=False)]
zeros = zeros.as_matrix()
ones = ones.as_matrix()
twos = twos.as_matrix()
print zeros.shape
print ones.shape
print twos.shape

N = 500*3
data = np.zeros([N, 256])
y = np.zeros(N, dtype=int)

from scipy.misc import imresize
for i in range(0, N/3):
    test = np.reshape(zeros[i][0:-1], (28,28))
    img = imresize(test, (16, 16))
    img = img.flatten()
    data[i] = img

for i in range(0, N/3):
    test = np.reshape(ones[i][0:-1], (28,28))
    img = imresize(test, (16, 16))
    img = img.flatten()
    data[i+N/3] = img
    y[i+N/3] = 1
    
for i in range(0, N/3):
    test = np.reshape(twos[i][0:-1], (28,28))
    img = imresize(test, (16, 16))
    img = img.flatten()
    data[i+2*N/3] = img
    y[i+2*N/3] = 2
    



min_max_scaler = preprocessing.MinMaxScaler()
data = min_max_scaler.fit_transform(data)
data = np.hstack((np.ones([data.shape[0],1]),data))
#data, y = shuffle(data, y, random_state=0)
data, test_dat, y, test_y = cross_validation.train_test_split(data, y, test_size=0.5, random_state=0)
d = data.shape[1] 
N = data.shape[0] 
print N, d
print y

# initialise
prior_post = 5000
prior_factor = 10000
r_new = np.zeros(d)
A_new = np.linalg.inv(np.identity(d) * prior_post) 
 
r_new2 = np.zeros(d)
A_new2 = np.linalg.inv(np.identity(d) * prior_post) 
 
r_new3 = np.zeros(d)
A_new3 = np.linalg.inv(np.identity(d) * prior_post) 
 
r = 0.0
r = np.repeat(r, N, axis=0)
A = 1./prior_factor
A = np.repeat(A, N, axis=0)

r2 = 0.0
r2 = np.repeat(r2, N, axis=0)
A2 = 1./prior_factor
A2 = np.repeat(A2, N, axis=0)

r3 = 0.0
r3 = np.repeat(r3, N, axis=0)
A3 = 1./prior_factor
A3 = np.repeat(A3, N, axis=0)

r_old = r_new
A_old = A_new

r_old2 = r_new2
A_old2 = A_new2

r_old3 = r_new3
A_old3 = A_new3


# max iterations
max_iter = 2

# Outer loop
tm1 = time.time()
for iteration in xrange(max_iter): # max iterations
    print "Iteration ", iteration

    for i in xrange(N):
        if i % 100 == 0:
            print "Factor ", i
        # 1 get cavity distribution 

        if iteration == 0:
            M_cavity, V_cavity = alt_project_first_iteration(data[i], r_old, A_old)
            M_cavity2, V_cavity2 = alt_project_first_iteration(data[i], r_old2, A_old2)
            M_cavity3, V_cavity3 = alt_project_first_iteration(data[i], r_old3, A_old3)
        else:
            M_cavity, V_cavity = alt_project(data[i], r_old, A_old, r[i],A[i])
            M_cavity2, V_cavity2 = alt_project(data[i], r_old2, A_old2, r2[i], A2[i])
            M_cavity3, V_cavity3 = alt_project(data[i], r_old3, A_old3, r3[i], A3[i])
            

        ######################################################################################

        # 3 compute moments
        if y[i] == 0:
            M_new, V_new = compute_moments(1, M_cavity, V_cavity)
            M_new2, V_new2 = compute_moments(0, M_cavity2, V_cavity2)
            M_new3, V_new3 = compute_moments(0, M_cavity3, V_cavity3)
            
            #max_x = optimize.minimize(lambda x: -ll2log(x, 1, M_cavity, V_cavity),  M_cavity)
            #M_new = max_x.x[0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #V_new = max_x.hess_inv[0][0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #max_x = optimize.minimize(lambda x: -ll2log(x, 0, M_cavity2, V_cavity2),  M_cavity2)
            #M_new2 = max_x.x[0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #V_new2 = max_x.hess_inv[0][0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #max_x = optimize.minimize(lambda x: -ll2log(x, 0, M_cavity3, V_cavity3),  M_cavity3)
            #M_new3 = max_x.x[0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #V_new3 = max_x.hess_inv[0][0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
        if y[i] == 1:
            M_new, V_new = compute_moments(0, M_cavity, V_cavity)
            M_new2, V_new2 = compute_moments(1, M_cavity2, V_cavity2)
            M_new3, V_new3 = compute_moments(0, M_cavity3, V_cavity3)
            
            #max_x = optimize.minimize(lambda x: -ll2log(x, 0, M_cavity, V_cavity),  M_cavity)
            #M_new = max_x.x[0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #V_new = max_x.hess_inv[0][0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #max_x = optimize.minimize(lambda x: -ll2log(x, 1, M_cavity2, V_cavity2),  M_cavity2)
            #M_new2 = max_x.x[0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #V_new2 = max_x.hess_inv[0][0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #max_x = optimize.minimize(lambda x: -ll2log(x, 0, M_cavity3, V_cavity3),  M_cavity3)
            #M_new3 = max_x.x[0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #V_new3 = max_x.hess_inv[0][0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
        if y[i] == 2:
            M_new, V_new = compute_moments(0, M_cavity, V_cavity)
            M_new2, V_new2 = compute_moments(0, M_cavity2, V_cavity2)
            M_new3, V_new3 = compute_moments(1, M_cavity3, V_cavity3)
            
            #max_x = optimize.minimize(lambda x: -ll2log(x, 0, M_cavity, V_cavity),  M_cavity)
            #M_new = max_x.x[0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #V_new = max_x.hess_inv[0][0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #max_x = optimize.minimize(lambda x: -ll2log(x, 0, M_cavity2, V_cavity2),  M_cavity2)
            #M_new2 = max_x.x[0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #V_new2 = max_x.hess_inv[0][0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #max_x = optimize.minimize(lambda x: -ll2log(x, 1, M_cavity3, V_cavity3),  M_cavity3)
            #M_new3 = max_x.x[0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
            #V_new3 = max_x.hess_inv[0][0]/(1/np.sqrt(2*np.pi*max_x.hess_inv[0][0]**2))
        
         
        # 4 subtract cavity to get moments of updated approximating factor
        
        Mi_approx, Vi_approx = subtract_moments(M_new, V_new, M_cavity, V_cavity)
        Mi_approx2, Vi_approx2 = subtract_moments(M_new2, V_new2, M_cavity2, V_cavity2)
        Mi_approx3, Vi_approx3 = subtract_moments(M_new3, V_new3, M_cavity3, V_cavity3)
        

        # calculate the change in factor
        
        if iteration == 0:
            deltar =  Mi_approx / Vi_approx - r[i]
            deltaA = 1./ Vi_approx
            r[i] = 1.*Mi_approx / Vi_approx
            A[i] = deltaA
            
            deltar2 =  Mi_approx2 / Vi_approx2 - r2[i]
            deltaA2 = 1./ Vi_approx2 
            r2[i] = 1.*Mi_approx2 / Vi_approx2
            A2[i] = deltaA2

            deltar3 =  Mi_approx3 / Vi_approx3 - r3[i]
            deltaA3 = 1./ Vi_approx3
            r3[i] = 1.*Mi_approx3 / Vi_approx3
            A3[i] = deltaA3
        else: 
            deltar =  Mi_approx / Vi_approx - r[i]
            deltaA = 1./ Vi_approx - A[i]
            r[i] = 1.*Mi_approx / Vi_approx
            A[i] = 1./ Vi_approx

            deltar2 =  Mi_approx2 / Vi_approx2 - r2[i]
            deltaA2 = 1./ Vi_approx2 - A2[i]
            r2[i] = 1.*Mi_approx2 / Vi_approx2
            A2[i] = 1./ Vi_approx2

            deltar3 =  Mi_approx3 / Vi_approx3 - r3[i]
            deltaA3 = 1./ Vi_approx3 - A3[i]
            r3[i] = 1.*Mi_approx3 / Vi_approx3
            A3[i] = 1./ Vi_approx3
        
        # 5 Project the delta change in posterior up to the full space
        
        deltar, deltaA = project_up(data[i], deltar, deltaA)
        deltar2, deltaA2 = project_up(data[i], deltar2, deltaA2)
        deltar3, deltaA3 = project_up(data[i], deltar3, deltaA3)

        # 6 combine updated gi with cavity 
        r_new = r_old + deltar
        A_new = A_old + deltaA
        r_new2 = r_old2 + deltar2
        A_new2 = A_old2 + deltaA2
        r_new3 = r_old3 + deltar3
        A_new3 = A_old3 + deltaA3
      
        r_old = r_new
        A_old = A_new
        
        r_old2 = r_new2
        A_old2 = A_new2
        
        r_old3 = r_new3
        A_old3 = A_new3
        
tm2 = time.time()
print "time"
print tm2-tm1
        

#print "New posterior"
post_mu,post_sig = ss_to_cov(r_new, A_new)
print pd.DataFrame(post_mu)
print pd.DataFrame(post_sig)  

#print "New posterior"
post_mu2,post_sig2 = ss_to_cov(r_new2, A_new2)
print pd.DataFrame(post_mu2)
print pd.DataFrame(post_sig2)  

#print "New posterior"
post_mu3,post_sig3 = ss_to_cov(r_new3, A_new3)
print pd.DataFrame(post_mu3)
print pd.DataFrame(post_sig3)  

from sklearn import cross_validation
y=test_y

score = 0
acc = 0
for j in range (0,len(y)):
    pred = []
    
    test_vec = test_dat[j]
    n_samples = 10000
    x = np.random.multivariate_normal(post_mu,post_sig , n_samples)
    avg = 0
    for i in range(0,len(x)):
        fr = np.array([x[i][:]])
        avg = avg + sigmoid(fr.dot(test_vec))
    avg = avg/len(x)
    pred.append(round(avg,2))
    
    
    n_samples = 10000
    x = np.random.multivariate_normal(post_mu2,post_sig2 , n_samples)
    avg = 0
    for i in range(0,len(x)):
        fr = np.array([x[i][:]])
        avg = avg + sigmoid(fr.dot(test_vec))
    avg = avg/len(x)
    pred.append(round(avg,2))
    
    
    n_samples = 10000
    x = np.random.multivariate_normal(post_mu3,post_sig3 , n_samples)
    avg = 0
    for i in range(0,len(x)):
        fr = np.array([x[i][:]])
        avg = avg + sigmoid(fr.dot(test_vec))
    avg = avg/len(x)
    pred.append(round(avg,2))
    
    ans = 0
    if max(pred)==pred[2]:
        ans = 2
    if max(pred)==pred[1]:
        ans = 1
    if max(pred)==pred[0]:
        ans = 0

    
    if ans == y[j]:
        acc = acc+1
    print y[j], " ", ans
print acc
print 1.*acc/len(y)

totacc = 0
for b in range (0,100):
    # data
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    Y = iris.target
    X, x, Y, y = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=b)
    data = X
    data = np.hstack((np.ones([data.shape[0],1]),data))
    test_dat = x
    test_y = y
    y = Y
    N = len(data)
    # initialise
    prior_post = 5000
    prior_factor = 1000
    r_new = np.zeros(d)
    A_new = np.linalg.inv(np.identity(d) * prior_post) 

    r_new2 = np.zeros(d)
    A_new2 = np.linalg.inv(np.identity(d) * prior_post) 

    r_new3 = np.zeros(d)
    A_new3 = np.linalg.inv(np.identity(d) * prior_post) 

    r = [] 
    A = []
    for i in range(N):
        A.append(np.linalg.inv(np.identity(d) * prior_factor))
        r.append(np.zeros(d))

    r2 = [] 
    A2 = []
    for i in range(N):
        A2.append(np.linalg.inv(np.identity(d) * prior_factor))
        r2.append(np.zeros(d))

    r3 = [] 
    A3 = []
    for i in range(N):
        A3.append(np.linalg.inv(np.identity(d) * prior_factor))
        r3.append(np.zeros(d))


    r_old = r_new
    A_old = A_new

    r_old2 = r_new2
    A_old2 = A_new2

    r_old3 = r_new3
    A_old3 = A_new3


    # max iterations
    max_iter = 4

    # Outer loop
    for iteration in xrange(max_iter): # max iterations

        for i in xrange(N):

            # 1 get cavity distribution 

            r_cavity, A_cavity = compute_cavity(r[i], A[i], r_old, A_old)
            if iteration == 0:
                r_cavity = r_old
                A_cavity = A_old

            r_cavity2, A_cavity2 = compute_cavity(r2[i], A2[i], r_old2, A_old2)
            if iteration == 0:
                r_cavity2 = r_old2
                A_cavity2 = A_old2

            r_cavity3, A_cavity3 = compute_cavity(r3[i], A3[i], r_old3, A_old3)
            if iteration == 0:
                r_cavity3 = r_old3
                A_cavity3 = A_old3

            ######################################################################################

            # 2 project the cavity distribution onto 1d subspace

            # *** maybe a mistake with cavity and i factor
            Mi, Vi = project_factor_subspace(data[i], r[i],A[i])
            mu, sig = ss_to_cov(r_new, A_new)
            M_cavity, V_cavity = project_subspace(data[i], mu, sig, Mi, Vi)

            Mi2, Vi2 = project_factor_subspace(data[i], r2[i],A2[i])
            mu2, sig2 = ss_to_cov(r_new2, A_new2)
            M_cavity2, V_cavity2 = project_subspace(data[i], mu2, sig2, Mi2, Vi2)

            Mi3, Vi3 = project_factor_subspace(data[i], r3[i],A3[i])
            mu3, sig3 = ss_to_cov(r_new3, A_new3)
            M_cavity3, V_cavity3 = project_subspace(data[i], mu3, sig3, Mi3, Vi3)


            ######################################################################################

            # 3 compute moments
            if y[i] == 0:
                M_new, V_new = compute_moments(1, M_cavity, V_cavity)
                M_new2, V_new2 = compute_moments(0, M_cavity2, V_cavity2)
                M_new3, V_new3 = compute_moments(0, M_cavity3, V_cavity3)
                #max_x = optimize.minimize(lambda x: -ll(x, y[i], M_cavity, V_cavity),  M_cavity)
                #M_new = max_x.x[0]
                #V_new = max_x.hess_inv[0][0]
            if y[i] == 1:
                M_new, V_new = compute_moments(0, M_cavity, V_cavity)
                M_new2, V_new2 = compute_moments(1, M_cavity2, V_cavity2)
                M_new3, V_new3 = compute_moments(0, M_cavity3, V_cavity3)
            if y[i] == 2:
                M_new, V_new = compute_moments(0, M_cavity, V_cavity)
                M_new2, V_new2 = compute_moments(0, M_cavity2, V_cavity2)
                M_new3, V_new3 = compute_moments(1, M_cavity3, V_cavity3)


            ###################################################################################### 

            # 4 subtract cavity to get moments of updated approximating factor

            Mi_approx, Vi_approx = subtract_moments(M_new, V_new, M_cavity, V_cavity)
            Mi_approx2, Vi_approx2 = subtract_moments(M_new2, V_new2, M_cavity2, V_cavity2)
            Mi_approx3, Vi_approx3 = subtract_moments(M_new3, V_new3, M_cavity3, V_cavity3)


            # 5 Transform updated factor back 

            r[i], A[i] = transform_back(data[i], Mi_approx, Vi_approx)
            r2[i], A2[i] = transform_back(data[i], Mi_approx2, Vi_approx2)
            r3[i], A3[i] = transform_back(data[i], Mi_approx3, Vi_approx3)


            # 6 combine updated gi with cavity 
            r_new, A_new = update_post(r[i], A[i], r_cavity, A_cavity)
            r_new2, A_new2 = update_post(r2[i], A2[i], r_cavity2, A_cavity2)
            r_new3, A_new3 = update_post(r3[i], A3[i], r_cavity3, A_cavity3)


            r_old = r_new
            A_old = A_new

            r_old2 = r_new2
            A_old2 = A_new2

            r_old3 = r_new3
            A_old3 = A_new3


    post_mu,post_sig = ss_to_cov(r_new, A_new)
    post_mu2,post_sig2 = ss_to_cov(r_new2, A_new2)
    post_mu3,post_sig3 = ss_to_cov(r_new3, A_new3)



    acc = 0
    for j in range (0,len(test_dat)):
        pred = []
        test_vec = np.array([1, test_dat[j][0], test_dat[j][1]])

        n_samples = 100000
        x = np.random.multivariate_normal(post_mu,post_sig , n_samples).T
        avg = 0
        for i in range(0,len(x)):
            fr = np.array([x[0][i], x[1][i], x[2][i]])
            avg = avg + sigmoid(fr.dot(test_vec))
        avg = avg/len(x)
        pred.append(round(avg,2))


        n_samples = 100000
        x = np.random.multivariate_normal(post_mu2,post_sig2 , n_samples).T
        avg = 0
        for i in range(0,len(x)):
            fr = np.array([x[0][i], x[1][i], x[2][i]])
            avg = avg + sigmoid(fr.dot(test_vec))
        avg = avg/len(x)
        pred.append(round(avg,2))


        n_samples = 100000
        x = np.random.multivariate_normal(post_mu3,post_sig3 , n_samples).T
        avg = 0
        for i in range(0,len(x)):
            fr = np.array([x[0][i], x[1][i], x[2][i]])
            avg = avg + sigmoid(fr.dot(test_vec))
        avg = avg/len(x)
        pred.append(round(avg,2))

        ans = 0
        if max(pred)==pred[2]:
            ans = 2
        if max(pred)==pred[1]:
            ans = 1
        if max(pred)==pred[0]:
            ans = 0


        if ans == test_y[j]:
            acc = acc+1
    totacc = totacc + 1.*acc/len(test_y)
    print acc
    print 1.*acc/len(test_y)

print totacc



