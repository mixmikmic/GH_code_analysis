import numpy as np
from scipy.stats import norm

def LogLikeProbit(betas, y, x):
    """
    Probit Log Likelihood function
    Very slow naive Python version
    Input:
        betas is a np.array of parameters
        y is a one dimensional np.array of endogenous data
        x is a 2 dimensional np.array of exogenous data
            First vertical colmn of X is assumed to be constant term,
            corresponding to betas[0]
    returns:
        negative of log likehood value (scalar)
    """
    result = 0
    #Sum operation
    for i in range(0, len(y)):
        #Get X_i * Beta value
        xb = np.dot(x[i], betas)
        
        #compute both binary probabilities from xb     
        #Add to total log likelihood
        llf = y[i]*np.log(norm.cdf(xb)) + (1-y[i])*np.log(1 - norm.cdf(xb))
        result += llf
    return -result

######################
#ARTIFICIAL DATA
######################

#sample size
n = 1000

#random generators
z1 = np.random.randn(n)
z2 = np.random.randn(n)

#create artificial exogenous variables 
x1 = 0.8*z1 + 0.2*z2
x2 = 0.2*z1 + 0.8*z2
#create error term
u = 2*np.random.randn(n)

#create endogenous variable from x1, x2 and u
ystar = 0.5 + 0.75*x1 - 0.75*x2 + u

#create latent binary variable from ystar
def create_dummy(data, cutoff):
    result = np.zeros(len(data))
    for i in range(0, len(data)):
        if data[i] >= cutoff:
            result[i] = 1
        else:
            result[i] = 0
    return result

#get latent LHS variable
y = create_dummy(ystar, 0.5)

#prepend vector of ones to RHS variables matrix
#for constant term
const = np.ones(n)
x = np.column_stack((const, np.column_stack((x1, x2))))


from scipy.optimize import minimize

#create beta hat vector to maximize on
#will store the values of maximum likelihood beta parameters
#Arbitrarily initialized to all zeros
bhat = np.zeros(len(x[0]))

#unvectorized MLE estimation
probit_est = minimize(LogLikeProbit, bhat, args=(y,x), method='nelder-mead')

#print vector of maximized betahats
probit_est['x']

import statsmodels.tools.numdiff as smt
import scipy as sc

#Get inverse hessian for Cramer Rao lower bound
b_estimates = probit_est['x']
Hessian = smt.approx_hess3(b_estimates, LogLikeProbit, args=(y,x))
invHessian = np.linalg.inv(Hessian)

#Standard Errors from C-R LB
#from diagonal elements of invHessian
SE = np.zeros(len(invHessian))
for i in range(0, len(invHessian)):
    SE[i] =  np.sqrt(invHessian[i,i])
    
#t and p values
t_statistics = (b_estimates/SE)
pval = (sc.stats.t.sf(np.abs(t_statistics), 999)*2)

print("Beta Hats: ", b_estimates)
print("SE: ", SE)
print("t stat: ", t_statistics)
print("P value: ", pval)

for i in range(0, len(y)):
        xb = np.dot(X[i], betas)
        llf = y[i]*np.log(norm.cdf(xb)) + (1-y[i])*np.log(1 - norm.cdf(xb))
        result += llf

xb = np.dot(x, betas)
result = np.sum(    
        (y==1)*np.log(1 - norm.cdf(xb)) + 
        (y==0)*np.log(norm.cdf(xb))
        )

xb = np.dot(x, betas)
result = np.sum(np.log(  
        (y==1)*(1 - norm.cdf(xb)) + 
        (y==0)*(norm.cdf(xb))
        ))

def VectorizedProbitLL(betas, y, x):
    xb = np.dot(x, betas)
    result = np.sum(np.log(  
        (y==0)*(1 - norm.cdf(xb)) + 
        (y==1)*(norm.cdf(xb))
        ))
    return -result

import timeit

get_ipython().magic("timeit minimize(VectorizedProbitLL, bhat, args=(y,x), method='nelder-mead')")

get_ipython().magic("timeit minimize(LogLikeProbit, bhat, args=(y,x), method='nelder-mead')")

get_ipython().magic("timeit minimize(VectorizedProbitLL, bhat, args=(y,x), method='bfgs')")

