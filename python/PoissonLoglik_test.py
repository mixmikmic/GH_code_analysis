import numpy as np
import pandas as pd
import patsy
df = pd.read_csv('DebTrivedi.csv',index_col = [0])
sel = np.array([1, 6, 7, 8, 13, 15, 18])-1
df = df.iloc[:,sel]
# produce design matrices from R-style formula
formula = 'ofp ~ hosp + health + numchron + gender + school + privins'
y, X = patsy.dmatrices(formula, df, return_type='dataframe')

X.head()

# health as Z variables
Z = X.iloc[:,[1,2]]

Y.shape

Y = np.squeeze(y.values)
## convenience variables
n = len(Y)
kx = X.shape[1] # Number of columns in X
kz = Z.shape[1]
Y0 = Y <= 0
Y1 = Y > 0

offsetx = np.zeros(n)
offsetz = np.zeros(n)
weights = np.ones(n)

type(weights)

weights.shape

np.ones(10)

import statsmodels.api as sm

model_count = sm.Poisson(endog = Y, exog = X).fit()

count = model_count.params

#model_zero = sm.Logit(endog = Y0, exog = Z).fit()
#zero = model_zero.params

model_zero = sm.GLM(Y0, exog = Z, family=sm.families.Binomial()).fit()
zero = model_zero.params

model_zero.predict()

coef = {'zero':zero, 'count':count}

coef['zero']

aloha= np.hstack((count.values,zero.values))

aloha

parms = aloha

import scipy as sp
sp.special.expit(Z);

X.shape

kx-1

parms[[np.arange(kx)]]

np.exp(np.dot(X,parms[np.arange(kx)]) + offsetx)

import scipy as sp
linkinv = sp.special.expit

linkinv(Z);

linkinv(np.dot(Z, parms[np.arange(kx,kx+kz)]) + offsetz)

def ziPoisson(parms):
    ## count mean
    mu = np.exp(np.dot(X,parms[np.arange(kx)]) + offsetx)
    ## binary mean
    phi = linkinv(np.dot(Z, parms[np.arange(kx,kx+kz)]) + offsetz)
    ## log-likelihood for y = 0 and y >= 1
    loglik0 = np.log( phi + np.exp( np.log(1-phi) - mu ) ) ## -mu = dpois(0, lambda = mu, log = TRUE)
    loglik1 = np.log(1-phi) + sp.stats.poisson.logpmf(Y, mu)
    ## collect and return
    loglik = np.dot(weights[Y0],loglik0[Y0])+np.dot(weights[Y1],loglik1[Y1])
    return loglik

mu = np.exp(np.dot(X,parms[np.arange(kx)]) + offsetx)
## binary mean
phi = linkinv(np.dot(Z, parms[np.arange(kx,kx+kz)]) + offsetz)
## log-likelihood for y = 0 and y >= 1
loglik0 = np.log( phi + np.exp( np.log(1-phi) - mu ) ) ## -mu = dpois(0, lambda = mu, log = TRUE)
loglik1 = np.log(1-phi) + sp.stats.poisson.logpmf(Y, mu)
loglik = np.dot(weights[Y0],loglik0[Y0])+np.dot(weights[Y1],loglik1[Y1])

np.dot(weights[Y0],loglik0[Y0])

mu = np.exp(np.dot(X,parms[np.arange(kx)]) + offsetx)

(np.dot(X,parms[np.arange(kx)])+offsetx).shape

mu = np.expand_dims(mu,axis=1)

mu.shape

phi = linkinv(np.dot(Z, parms[np.arange(kx,kx+kz)]) + offsetz)

phi.shape

loglik1 = np.log(1-phi) + sp.stats.poisson.logpmf(Y, mu)

loglik1.shape

loglik0 = np.log( phi + np.exp( np.log(1-phi) - mu ) )

loglik0.shape

loglik = np.dot(weights[Y0],loglik0[Y0])+np.dot(weights[Y1],loglik1[Y1])

np.dot(weights[Y0],loglik0[Y0])

np.dot(weights[Y1],loglik1[Y1])

loglik

weights[Y0].shape

loglik0[Y0].shape

mu.shape

loglik1[Y1].shape

phi.shape

Y.shape

sp.stats.poisson.logpmf(Y, mu).shape

np.dot(weights[Y0],loglik0[Y0])+np.dot(weights[Y1],loglik1[Y1])

np.dot(weights[Y1],loglik1[Y1])

loglik0 = np.log( phi + np.exp( np.log(1-phi) - mu ) ) ## -mu = dpois(0, lambda = mu, log = TRUE)
loglik1 = np.log(1-phi) + sp.stats.poisson.logpmf(Y, mu)



mu = np.exp(np.dot(X,parms[np.arange(kx)]) + offsetx)

mu.shape

(np.dot(X,parms[np.arange(kx)])+ offsetx).shape

np.sum(weights[Y0] * loglik0[Y0]) + np.sum(weights[Y1] * loglik1[Y1])

weights[Y0] * loglik0[Y0]

Y0.shape

weights[Y0].shape

3723+683

weights[Y0]

phi = linkinv(np.dot(Z, parms[np.arange(kx,kx+kz)]) + offsetz)

mu = np.exp(np.dot(X,parms[np.arange(kx)]) + offsetx)

dpois = sp.stats.poisson(mu)

np.log(1-phi) + dpois.logpmf(Y)

np.log(1-phi) + sp.stats.poisson.logpmf(Y, mu)

np.arange(kx,kx+kz)

rep(0,10)

np.zeros(10)



