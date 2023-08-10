fit = sp.optimize.minimize(loglikfun, args=(-1.0,), x0 = np.hstack((start['count'].values,start['zero'].values)),            method='BFGS', jac=gradfun, options={'disp': False, 'maxiter': 10000}, tol = reltol)

sm.Logit(1)

## coefficients and covariances
coefc = pd.Series(data = fit.x[0:kx], index = X.columns.values)
coefz = pd.Series(data = fit.x[kx:kx+kz], index = Z.columns.values)
vc = pd.DataFrame(data = -fit.hess_inv, index = np.append(X.columns.values, Z.columns.values),                 columns = np.append(X.columns.values, Z.columns.values))

## fitted and residuals
mu = np.exp(np.dot(X,coefc)+offsetx)
phi = linkinv(np.dot(Z,coefz)+offsetz)
Yhat = (1-phi) * mu
res = np.sqrt(weights) * (Y - Yhat)

vc

mu

phi

Yhat

res

coefc

coefz



np.exp(np.dot(X,coefc)+offsetx).shape

y = np.append(X.columns.values, Z.columns.values)

X.columns.values



(fit.hess_inv).shape



np.hstack((start['count'].values,start['zero'].values)).shape

## coefficients and covariances
coefc = fit.x[0:kx]
#names(coefc) <- names(start$count) <- colnames(X)
#coefz <- fit$par[(kx+1):(kx+kz)]
#names(coefz) <- names(start$zero) <- colnames(Z)

type(start['count'])

import numpy as np
import pandas as pd
import patsy
df = pd.read_csv('DebTrivedi.csv',index_col = [0])
sel = np.array([1, 6, 7, 8, 13, 15, 18])-1
df = df.iloc[:,sel]
# produce design matrices from R-style formula
X_formula = 'ofp ~ hosp + health + numchron + gender + school + privins'
y, X = patsy.dmatrices(X_formula, df, return_type='dataframe')
Z_formula = 'ofp ~ health'
Z = patsy.dmatrices(Z_formula, df, return_type='dataframe')[1]

## convenience variables
Y = np.squeeze(y.values)
n = len(Y)
kx = X.shape[1] # Number of columns in X
kz = Z.shape[1]
Y0 = Y <= 0
Y1 = Y > 0

offsetx = np.zeros(n)
offsetz = np.zeros(n)
weights = np.ones(n)

import statsmodels.api as sm
import scipy as sp
linkinv = sp.special.expit
linkobj = sp.special.logit

def ziPoisson(parms, sign = 1.0):
    ## count mean
    mu = np.exp(np.dot(X,parms[np.arange(kx)]) + offsetx)
    ## binary mean
    phi = linkinv(np.dot(Z, parms[np.arange(kx,kx+kz)]) + offsetz)
    ## log-likelihood for y = 0 and y >= 1
    loglik0 = np.log( phi + np.exp( np.log(1-phi) - mu ) ) ## -mu = dpois(0, lambda = mu, log = TRUE)
    loglik1 = np.log(1-phi) + sp.stats.poisson.logpmf(Y, mu)
    ## collect and return
    loglik = np.dot(weights[Y0],loglik0[Y0])+np.dot(weights[Y1],loglik1[Y1])
    return sign*loglik

def gradPoisson(parms, sign = 1.0):
    ## count mean
    eta = np.dot(X,parms[np.arange(kx)]) + offsetx
    mu = np.exp(eta)
    ## binary mean
    etaz = np.dot(Z, parms[np.arange(kx,kx+kz)]) + offsetz
    muz = linkinv(etaz)
    ## densities at 0
    clogdens0 = -mu
    dens0 = muz*(1-Y1.astype(float)) + np.exp(np.log(1 - muz) + clogdens0)
    ## working residuals  
    wres_count = np.where(Y1,Y-mu,-np.exp(-np.log(dens0) + 
                                          np.log(1 - muz) + clogdens0 + np.log(mu))) 
    link_etaz = np.exp(etaz)/(1+np.exp(etaz))**2
    wres_zero  = np.where(Y1,-1/(1-muz) * link_etaz,                           (link_etaz - np.exp(clogdens0) * link_etaz)/dens0)
    
    
    return sign*(np.hstack((np.expand_dims(wres_count*weights,axis=1)*X,                 np.expand_dims(wres_zero*weights,axis=1)*Z))).sum(axis=0)

loglikfun = ziPoisson
gradfun = gradPoisson
reltol =  (np.finfo(float).eps)**(1/1.6)

model_count = sm.GLM(endog = Y, exog = X, family = sm.families.Poisson(),                                  offset = offsetx , freq_weights = weights).fit()
model_zero = sm.GLM(Y0.astype(int), exog = Z, family=sm.families.Binomial(),                    offset = offsetz , freq_weights = weights).fit()
start = {'zero':model_zero.params, 'count':model_count.params}

mui = model_count.predict()
probi = model_zero.predict()
probi = probi/(probi + (1-probi)*sp.stats.poisson.pmf(0, mui))
probi[Y1] = 0

ll_new = loglikfun(np.hstack((start['count'].values,start['zero'].values)))
ll_old = 2 * ll_new

while np.absolute((ll_old - ll_new)/ll_old) > reltol :
    ll_old = ll_new
    model_count = poisson_mod = sm.GLM(endog = Y, exog = X, family = sm.families.Poisson(),                                  offset = offsetx , freq_weights = weights*(1-probi),                                       start_params = start['count']).fit()
    model_zero = sm.GLM(probi, exog = Z, family=sm.families.Binomial(),                        offset = offsetz, freq_weights = weights,                         start_params = start['zero']).fit()
    start = {'zero':model_zero.params, 'count':model_count.params}

    mui = model_count.predict()
    probi = model_zero.predict()
    probi = probi/(probi + (1-probi)*sp.stats.poisson.pmf(0, mui))
    probi[Y1] = 0

    ll_new = loglikfun(np.hstack((start['count'].values,start['zero'].values)))

sm.GLM(endog = Y, exog = X, family = sm.families.Poisson(),                                  offset = offsetx , freq_weights = weights).fit().summary()

start['count']

from statsmodels.genmod.generalized_estimating_equations import GEE

get_ipython().run_line_magic('pinfo', 'GEE')



