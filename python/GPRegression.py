import numpy as np
import matplotlib as mpl
from scipy.optimize import minimize
from scipy.linalg import eigh
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rc("savefig",dpi=200)

# inputs
# A : rank-2 array, with shape (Na, d), where a_i \in \mathbb{R}^{d} (i = 1,2, \dots, Na) OR 1D array with length Na (which corresponds to d=1)
# B : rank-2 array, with shape (Nb, d), where b_i \in \mathbb{R}^{d} (i = 1,2, \dots, Nb) OR 1D array with length Nb (which corresponds to d=1)
# thts : 1D array with four elements
# output
# K : Na \times Nb matrix, with 
#    K_{i,j} = k(a_i,b_j) = \theta_0 \exp\left( -\frac{\theta_1}{2} \| a_i- b_j \|^2 \right) + \theta_2 + \theta_3 a_{i}^{T} b_j
def genK(A,B,thts):
    # make sure that A and B are matrices
    if len(np.shape(A)) == 1:
        A = np.reshape(A, (len(A),1))
    if len(np.shape(B)) == 1:
        B = np.reshape(B, (len(B),1))

    tmp = np.reshape(np.sum(A**2,axis=1),(len(A),1)) + np.sum(B**2,axis=1)  -2*(A @ B.T)
    return thts[0]*np.exp(-thts[1]/2*tmp) + thts[2] + thts[3]*(A @ B.T)

## 1 dimensional example
xx = np.linspace(-1,1,101)
Thts = np.array([[1.0, 4.0, 0.0, 0.0],                 [9.0, 4.0, 0.0, 0.0],                 [1.0, 64.0, 0.0,0.0],                 [1.0,0.25, 0.0, 0.0],                 [1.0, 4.0, 10.0, 0.0],                 [1.0, 4.0, 0.0, 5.0]                ])

num  = 5
cnt = 0
fig = plt.figure(figsize=(18,10))
while cnt < len(Thts):
    K = genK(xx,xx,Thts[cnt])
    yy = np.random.multivariate_normal(np.zeros(len(xx)), K, num)
    ax = fig.add_subplot(2,3,cnt+1)
    for y in yy:
        ax.plot(xx,y)
    ax.set_title("(%s, %s, %s, %s)"%(Thts[cnt][0],Thts[cnt][1],Thts[cnt][2],Thts[cnt][3]))
    cnt += 1

def pred(xtest,xtrain,ttrain,beta,thts):
    CN = genK(xtrain,xtrain,thts) + 1.0/beta*np.eye(len(xtrain))
    kmat = genK(xtrain,xtest,thts)
    cmat = genK(xtest,xtest,thts) + 1.0/beta*np.eye(len(xtest))
    mvec = (kmat.T) @ np.linalg.inv(CN) @ ttrain
    svec = np.sqrt( np.diag(  cmat -  kmat.T @ np.linalg.inv(CN)@ kmat  )  )
    return mvec,svec

def CAndGradC(X,thts,beta):
    if len(np.shape(X)) == 1:
        X = np.reshape(X, (len(X),1))
    N = len(X)
    
    tmp = np.reshape(np.sum(X**2,axis=1),(len(X),1)) + np.sum(X**2,axis=1)  -2*(X @ X.T)
    
    C = thts[0]*np.exp(-thts[1]/2*tmp) + thts[2] + thts[3]*(X @ (X.T)) + 1.0/beta*np.eye(N)
    gradC0 = np.exp(-thts[1]/2*tmp)
    gradC1 = -thts[0]/2*tmp*np.exp(-thts[1]/2*tmp)
    gradC2 = np.ones((N,N))
    gradC3 = X @ (X.T)
    
    return C, np.array([gradC0, gradC1, gradC2, gradC3])

def neg_logPandGrad(hparams,xtrain,ttrain):
    thts = hparams[:4]
    beta = hparams[4]
    C,gradC = CAndGradC(xtrain,thts,beta)
    Cinv = np.linalg.inv(C)
    logDetC = np.sum(np.log(eigh(C)[0]))
    logP = -0.5*logDetC - 0.5*(ttrain @ Cinv @ttrain)
    gradLogP = np.zeros(5)
    cnt = 0
    while cnt < 4:
        gradLogP[cnt] = -0.5*np.trace(Cinv @ gradC[cnt])  + 0.5*(ttrain @ Cinv @ gradC[cnt] @Cinv @ ttrain)
        cnt += 1
    gradLogP[4] =  0.5*np.trace(Cinv)/(beta*beta)  - 0.5/(beta*beta)* (ttrain @ Cinv @ Cinv @ ttrain)
    return -logP,-gradLogP

def EmpB(xtrain,ttrain,hparams0):
    ans = minimize(neg_logPandGrad,hparams0,args=(xtrain,ttrain),jac=True,                   constraints=({'type':'ineq','fun':lambda x: x[0]},                                {'type':'ineq','fun':lambda x: x[1]},                                {'type':'ineq','fun':lambda x: x[2]},                                {'type':'ineq','fun':lambda x: x[3]},                                {'type':'ineq','fun':lambda x: x[4]}                               )                  )
    print(ans['message'])
    thts = ans['x'][0:4]
    beta = ans['x'][4]
    return thts,beta

def truef(x):
    return np.sin(2*x)  + 0.2*np.sin(x) + 0.1*x

numdat = 50
xdat = np.random.uniform(-3,3,numdat)
ep = 0.3*np.random.randn(numdat)
ydat = truef(xdat) + ep

xcont = np.linspace(np.min(xdat),np.max(xdat),200)

plt.plot(xdat,ydat,'.')
plt.plot(xcont,truef(xcont))
plt.xlabel(r'$x$')
plt.show()

xdat = np.reshape(xdat,(len(xdat),1))

def plotPred(beta,thts,ax):
    xpred = np.linspace(-3,3,101)
    m,sig = pred(xpred,xdat,ydat,beta,thts)

    ax.plot(xdat,ydat,'.',label='data')
    ax.plot(xpred,m,label='predictive mean')
    ax.plot(xpred,truef(xpred),':',label='true')
    ax.fill_between(xpred,m+sig,m-sig,alpha=0.2)
    ax.legend()
    ax.set_title(r'$\beta=%s,\theta = (%s,%s,%s,%s) $'%(beta,thts[0],thts[1],thts[2],thts[3]))

beta = 3.0

Thts = np.array([[1.0, 4.0, 0.0, 0.0],                 [9.0, 4.0, 0.0, 0.0],                 [1.0, 64.0, 0.0,0.0],                 [1.0,0.25, 0.0, 0.0],                 [1.0, 4.0, 10.0, 0.0],                 [1.0, 4.0, 0.0, 5.0]                ])

cnt = 0
fig = plt.figure(figsize=(18,10))
while cnt < len(Thts):
    ax = fig.add_subplot(2,3,cnt+1)
    plotPred(beta,Thts[cnt],ax)
    cnt += 1
plt.show()

hparams0 = np.array([1.0,0.5, 0.0, 0.0,4.0])
thts,beta = EmpB(xdat,ydat,hparams0)

print(thts)
print(beta)

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(1,1,1)
plotPred(beta,thts,ax)
plt.show()

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

kernel = 1.0 * RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1e2))     + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 3e+1))
gp = GaussianProcessRegressor(kernel=kernel,alpha=0.0)

# fitting 
gp.fit(xdat,ydat)

# prediction
y_mean, y_cov = gp.predict(np.reshape(xcont,(len(xcont),1)), return_cov=True)

# plotting
fig = plt.figure(figsize=(8,6))
plt.plot(xdat,ydat,'.',label='data')
plt.plot(xcont, y_mean,label='predictive mean')
plt.plot(xcont, truef(xcont),':',label='true')
plt.fill_between(xcont, y_mean - np.sqrt(np.diag(y_cov)),
                 y_mean + np.sqrt(np.diag(y_cov)),
                 alpha=0.2)
plt.title(f"Initial: {kernel}\nOptimum: {gp.kernel_}\nLog-Marginal-Likelihood: {gp.log_marginal_likelihood(gp.kernel_.theta)}")
plt.show()



