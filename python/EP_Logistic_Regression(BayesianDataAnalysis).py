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
get_ipython().run_line_magic('matplotlib', 'inline')

d = 1
N = 100
w = 0.5
clutter_var = 10
prior_var = 100
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
logreg = linear_model.LogisticRegression(C=1e5)
logreg.fit(data, y)
w_true = logreg.coef_[0][0]
w0_true = logreg.intercept_[0]
print w_true
print w0_true

data = np.hstack((np.ones([data.shape[0],1]),data))
data, y = shuffle(data, y, random_state=0)

plt.plot(data[y==other_label],data[y==other_label],"bo", markersize = 3)
plt.plot(data[y==1],data[y==1],"ro", markersize = 3)
plt.title("Generated data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
print y
#print data


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
    return  pow(np.e, exponent) / (1 + pow(np.e, exponent))  
    

def compute_moments(y, M_cavity, V_cavity):
    
    cavity_m = M_cavity
    
    if V_cavity < 0:
        sd = -sqrt(-V_cavity)
    else:
        sd = sqrt(V_cavity)

    lower_bound = M_cavity - 10*sqrt(np.abs(V_cavity))
    upper_bound = M_cavity + 10*sqrt(np.abs(V_cavity))
    
    f = lambda x: gaussian(x, cavity_m, sd) *                 pow(logit_i(x), y) * pow((1-logit_i(x)), (1 - y))
    E0 = integrate.quad(f, lower_bound, upper_bound)[0]

    f = lambda x: gaussian(x, cavity_m, sd) *                 pow(logit_i(x), y) * pow((1-logit_i(x)), (1 - y)) * x
    E1 = integrate.quad(f, lower_bound, upper_bound)[0] 
    
    f = lambda x: gaussian(x, cavity_m, sd) *                 pow(logit_i(x), y) * pow((1-logit_i(x)), (1 - y)) * (x**2)
    E2 = integrate.quad(f, lower_bound, upper_bound)[0] 
    
    if E0 == 0:
        print "WARNING E0 was 0"
        E0 = 0.0001
        
    M_new = E1 / E0
    V_new = E2 / E0 - (E1 / E0)**2 


    return M_new, V_new


def ll2log(x, y, mu, sig):
    if sig < 0:
        sig = -sqrt(-sig)
    else:
        sig = sqrt(sig)
    trunc = 7.
    exponent = np.clip(x, -trunc, trunc)
    exponent = np.float64(exponent)
    et = np.exp(exponent)

    z = np.log(et / (1 + et))
    e = y*z + (1-y)*np.log(1-et / (1 + et)) + np.log(gaussian(x, mu, (sig)))
    return e

def ll(x, y, mu, sig):
    if sig < 0:
        sig = -sqrt(-sig)
    else:
        sig = sqrt(sig)
        
    et = np.exp(x)
    z = np.log(et / (1 + et))
    e = y*z + (1-y)*np.log(1-et / (1 + et)) + np.log(gaussian(x, mu, (sig)))
    return e
'''
max_x = optimize.minimize(lambda x: -ll2log(x, y[i], M_cavity, V_cavity),  M_cavity)
M_new = max_x.x[0]
V_new = max_x.hess_inv[0][0]
print M_new
print V_new
max_x = optimize.minimize(lambda x: -ll(x, y[i], M_cavity, V_cavity),  M_cavity)
M_new = max_x.x[0]
V_new = max_x.hess_inv[0][0]
print M_new
print V_new
'''

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

# initialise
prior_post = 100
prior_factor = 100000
r_new = np.array([0,0])
A_new = np.linalg.inv(np.identity(2) * prior_post) 
                         
r = [] 

A = []
for i in range(N):
    A.append(np.linalg.inv(np.identity(2) * prior_factor))
    r.append(np.array([0, 0]))
    
    
r_old = r_new
A_old = A_new


def plot_projected(M_new, V_new, M_cavity, V_cavity, y):
    
    
    if V_new < 0:
        sd1 = -sqrt(-V_new)
    else:
        sd1 = sqrt(V_new)
    if V_cavity < 0:
        sd2 = -sqrt(-V_cavity)
    else:
        sd2 = sqrt(V_cavity)
        
    x = np.linspace(-8,8,100)    
    
    plt.plot(x, gaussian(x, M_new, sd1), label="New Aprrox")
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
    limit = 8
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
    

    x = np.linspace(-10,10,100)
    
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
    
    sig = np.linalg.inv(A)
    mu = sig.dot(r)
    
    Mi = X.dot(mu)
    Vi = X.dot(sig).dot(XT)
    
    return Mi, Vi

M_cavity = 2
V_cavity = 10
M_new, V_new = compute_moments(1, M_cavity, V_cavity)
print M_new
plot_projected(M_new, V_new, M_cavity, V_cavity, 1)
max_x = optimize.minimize(lambda x: -ll(x, y[i], M_cavity, V_cavity),  M_cavity)
M_new = max_x.x[0]
V_new = max_x.hess_inv[0][0]
plot_projected(M_new, V_new, M_cavity, V_cavity, 1)
print M_new

# Outer loop
mu, sig = ss_to_cov(r_new, A_new)
plot_posterior(mu, sig)
for iteration in xrange(3): # max iterations
    print "Iteration ", iteration

    for i in xrange(N):
        
        ######################################################################################
        
        print "factor ", i
        print "Data"
        print data[i]
        print y[i]
        print
        print "Current factor"
        a,b = ss_to_cov(r[i], A[i])
        print pd.DataFrame(a)
        print pd.DataFrame(b)
        print
        
        ######################################################################################
        
        # 1 get cavity distribution 
        
        r_cavity, A_cavity = compute_cavity(r[i], A[i], r_old, A_old)
        print "removal"
        print r[i], A[i], r_old, A_old
        print r_cavity
        print A_cavity
        # verbose
        print "Cavity"
        a,b = ss_to_cov(r_cavity, A_cavity)
        print pd.DataFrame(a)
        print pd.DataFrame(b)
        print
        
        ######################################################################################
        
        # 2 project the cavity distribution onto 1d subspace
       
        Mi, Vi = project_factor_subspace(data[i], r[i],A[i])
        mu, sig = ss_to_cov(r_new, A_new)
        M_cavity, V_cavity = project_subspace(data[i], mu, sig, Mi, Vi)
        
        # verbose
        print "Projected cavity"
        print M_cavity, V_cavity
        print
        
        
        ######################################################################################

        # 3 compute moments
        
        # Numerical integration
        
        M_new, V_new = compute_moments(y[i], M_cavity, V_cavity)
        
        # alternatively match moments with Laplace approximation
        # Laplace approximation
        
        #max_x = optimize.minimize(lambda x: -ll(x, y[i], M_cavity, V_cavity),  M_cavity)
        #M_new = max_x.x[0]
        #V_new = max_x.hess_inv[0][0]
        
        # verbose
        print "Matched moments "
        print M_new, V_new
        print
        
        if V_new == 0:
            V_new = np.infty
            print " WARNING MATCHED VARIANCE IS 0\n"
            #break


        ###################################################################################### 
         
        # 4 subtract cavity to get moments of updated approximating factor
        
        Mi_approx, Vi_approx = subtract_moments(M_new, V_new, M_cavity, V_cavity)
        
        # verbose
        print "Updated factor "
        print Mi_approx, Vi_approx
        print
        
        ######################################################################################
        
        # 5 Transform updated factor back 
        
        r[i], A[i] = transform_back(data[i], Mi_approx, Vi_approx)
        
        # verbose
        print "Updated factor"
        a, b = ss_to_cov(r[i], A[i])
        print pd.DataFrame(a)
        print pd.DataFrame(b)
        print
        
        ######################################################################################    

        # 6 combine updated gi with cavity 
        r_new, A_new = update_post(r[i], A[i], r_cavity, A_cavity)
        
        # verbose
        print "New posterior"
        a,b = ss_to_cov(r_new, A_new)
        print pd.DataFrame(a)
        print pd.DataFrame(b)  

        ###################################################################################### 
        
        mu, sig = ss_to_cov(r_new, A_new)
        
        if sig[0][0] >= 0 and sig[1][1] >= 0:
            
            # plot
            plot_projected(M_new, V_new, M_cavity, V_cavity, y[i])
            plot_posterior_1d(r_old, A_old, r_new, A_new, r[i], A[i])
            plot_posterior(mu, sig)
            
            r_old = r_new
            A_old = A_new
            
        else:
            print "WARNING NEG VAR POSTERIOR"
        

print np.linalg.inv(np.array([[-0.48206052 , 0.  ],[ 0.,1.16445785]]))

plot_projected(M_new, V_new, M_cavity, V_cavity, y[i])
plot_posterior_1d(r_old, A_old, r_new, A_new, r[i], A[i])
plot_posterior(mu, sig)





