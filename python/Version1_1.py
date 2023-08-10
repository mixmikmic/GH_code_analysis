### Lines below will not go inside the function
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

class Logit(object):
    def __init__(self):
        self.linkclass = sm.genmod.families.links.logit
    def link(self, mu):
        return mu/(1.0 + mu)
    def link_inv(self, eta):
        thresh = 30.0
        eta = np.minimum(np.maximum(eta,-thresh), thresh)
        exp_eta = np.exp(eta)
        return exp_eta/(1+exp_eta)
    def link_inv_deriv(self, eta):
        thresh = 30.0
        eta[abs(eta) > thresh] = FLOAT_EPS
        return np.exp(eta)/(1+np.exp(eta))**2

class Probit(object):
    def __init__(self):
        self.linkclass = sm.genmod.families.links.probit
    def link(self, mu):
        return st.norm.ppf(mu)
    def link_inv(self, eta):
        thresh = -st.norm.ppf(FLOAT_EPS)
        eta = np.minimum(np.maximum(eta,-thresh),thresh)
        return st.norm.cdf(eta)
    def link_inv_deriv(self, eta):
        return np.maximum(st.norm.pdf(eta),FLOAT_EPS)
    
class CLogLog(object):
    def __init__(self):
        self.linkclass = sm.genmod.families.links.cloglog
    def link(self, mu):
        return np.log(-np.log(1 - mu))
    def link_inv(self, eta):
        return np.maximum(np.minimum(-np.expm1(-np.exp(eta)),1-FLOAT_EPS),FLOAT_EPS)
    def link_inv_deriv(self, eta):
        eta = np.minimum(eta,700)
        return np.maximum(np.exp(eta)*np.exp(-np.exp(eta)),FLOAT_EPS)
    
class Cauchit(object):
    def __init__(self):
        self.linkclass = sm.genmod.families.links.cauchy
    def link(self, mu):
        return st.cauchy.ppf(mu)
    def link_inv(self, eta):
        thresh = -st.cauchy.ppf(FLOAT_EPS)
        eta = np.minimum(np.maximum(eta,-thresh),thresh)
        return st.cauchy.cdf(eta)
    def link_inv_deriv(self, eta):
        return nnp.maximum(st.cauchy.pdf(eta),FLOAT_EPS)
    
class Log(object):
    def __init__(self):
        self.linkclass = sm.genmod.families.links.log
    def link(self, mu):
        return np.log(mu)
    def link_inv(self, eta):
        return np.maximum(np.exp(eta), FLOAT_EPS)
    def link_inv_deriv(self, eta):
        return np.maximum(np.exp(eta), FLOAT_EPS)

def setLinkClass(argument):
    Link = {
        'logit': Logit(),
        'probit': Probit(),
        'cloglog': CLogLog(),
        'cauchit': Cauchit(),
        'log': Log(),
    }
    return Link.get(argument, Logit)

## Function starts
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy as sp
import scipy.stats as st
import sys
import warnings

FLOAT_EPS = np.finfo(float).eps

## sanity checks
if len(Y) < 1:
    sys.exit("empty model")
if np.all(Y > 0):
    sys.exit("invalid dependent variable, minimum count is not zero")  
if np.array_equal(np.asarray(Y), (np.round(Y + 0.001)).astype(int)) is False:
    sys.exit("invalid dependent variable, non-integer values")
Y = (np.round(Y + 0.001)).astype(int)
if np.any(Y < 0):
    sys.exit("invalid dependent variable, negative counts")
    
    
## convenience variables
Y = np.squeeze(y.values)
n = len(Y)
kx = X.shape[1] # Number of columns in X
kz = Z.shape[1]
Y0 = Y <= 0
Y1 = Y > 0

## weights and offset

if weights is None:
    weights = 1.0
weights = np.ndarray.flatten(np.array(weights))
if weights.size == 1:
    weights = np.repeat(weights,n)
weights = pd.Series(data = weights, index = X.index)

if offsetx is None:
    offsetx = 0.0
offsetx = np.ndarray.flatten(np.array(offsetx))
if offsetx.size == 1:
    offsetx = np.repeat(offsetx,n)

if offsetz is None:
    offsetz = 0.0
offsetz = np.ndarray.flatten(np.array(offsetz))
if offsetz.size == 1:
    offsetz = np.repeat(offsetz,n)
    
## binary link processing
linkstr = control['link']
linkList = ['logit','probit','cauchit','cloglog','log']
if linkstr not in linkList:
    sys.exit(linkstr +" link not valid. Available links are: " + str(linkList))
link = setLinkClass(linkstr)

def ziPoisson(parms, sign = 1.0):
    ## count mean
    mu = np.exp(np.dot(X,parms[np.arange(kx)]) + offsetx)
    ## binary mean
    phi = link.link_inv(np.dot(Z, parms[np.arange(kx,kx+kz)]) + offsetz)
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
    muz = link.link_inv(etaz)
    ## densities at 0
    clogdens0 = -mu
    dens0 = muz*(1-Y1.astype(float)) + np.exp(np.log(1 - muz) + clogdens0)
    ## working residuals  
    wres_count = np.where(Y1,Y-mu,-np.exp(-np.log(dens0) + 
                                          np.log(1 - muz) + clogdens0 + np.log(mu))) 
    link_etaz = link.link_inv_deriv(etaz)
    wres_zero  = np.where(Y1,-1/(1-muz) * link_etaz,                           (link_etaz - np.exp(clogdens0) * link_etaz)/dens0)
    
    
    return sign*(np.hstack((np.expand_dims(wres_count*weights,axis=1)*X,                 np.expand_dims(wres_zero*weights,axis=1)*Z))).sum(axis=0)


## Parameters: mention these in class definition
##-----------------------------------------------

reltol =  (np.finfo(float).eps)**(1/1.6)
method = 'BFGS'
dist = 'Poisson'
##-----------------------------------------------
reltol = control['tol']
if reltol is None:
    reltol =  (np.finfo(float).eps)**(1/1.6)
method = control['method']
dist = control['dist']
if dist not in ['Poisson']:#,'NegBin','Geom']:
    sys.exit(dist+" method not yet implemented")
if dist is 'Poisson':
    loglikfun = ziPoisson
    gradfun = gradPoisson
options = control['options']
if options is None:
    options = {'disp': False, 'maxiter': 10000}
start = control['start']

# starting values
if start is not None:
    valid = True
    if ('count' in start) is False:
        valid = False
        warnings.warn("invalid starting values, count model coefficients not specified")
        start['count'] = pd.Series(np.repeat(0,kx), index = X.columns.values)
    if ('zero' in start) is False:
        valid = False
        warnings.warn("invalid starting values, zero model coefficients not specified")
        start['zero'] = pd.Series(np.repeat(0,kz), index = Z.columns.values)
    if(len(start['count']) != kx):
        valid = False
        warning("invalid starting values, wrong number of count model coefficients")
    if(len(start['zero']) != kz):
        valid = False
        warning("invalid starting values, wrong number of zero model coefficients")
    
    start = {'zero':start['zero'], 'count':start['count']}
    if valid is False:
        start = None

if start is None:
## EM estimation of starting values
    if (control['EM'] is True) and (dist is 'Poisson'):
        model_count = sm.GLM(endog = Y, exog = X, family = sm.families.Poisson(),                                  offset = offsetx , freq_weights = weights).fit()
        model_zero = sm.GLM(Y0.astype(int), exog = Z, family=sm.families.Binomial(link = link.linkclass),                    offset = offsetz , freq_weights = weights).fit()
        start = {'zero':model_zero.params, 'count':model_count.params}

        mui = model_count.predict()
        probi = model_zero.predict()
        probi = probi/(probi + (1-probi)*sp.stats.poisson.pmf(0, mui))
        probi[Y1] = 0
        probi
        ll_new = loglikfun(np.hstack((start['count'].values,start['zero'].values)))
        ll_old = 2 * ll_new
    
        while np.absolute((ll_old - ll_new)/ll_old) > reltol :
            ll_old = ll_new
            model_count = poisson_mod = sm.GLM(endog = Y, exog = X, family = sm.families.Poisson(),                                  offset = offsetx , freq_weights = weights*(1-probi),                                       start_params = start['count']).fit()
            model_zero = sm.GLM(probi, exog = Z, family=sm.families.Binomial(link = link.linkclass),                        offset = offsetz, freq_weights = weights,                         start_params = start['zero']).fit()
            start = {'zero':model_zero.params, 'count':model_count.params}

            mui = model_count.predict()
            probi = model_zero.predict()
            probi = probi/(probi + (1-probi)*sp.stats.poisson.pmf(0, mui))
            probi[Y1] = 0

            ll_new = loglikfun(np.hstack((start['count'].values,start['zero'].values)))
    
## ML Estimation
fit = sp.optimize.minimize(loglikfun, args=(-1.0/2,), x0 = np.hstack((start['count'].values,start['zero'].values)),            method=method, jac=gradfun, options=options, tol = reltol)

## coefficients and covariances
coefc = pd.Series(data = fit.x[0:kx], index = X.columns.values)
coefz = pd.Series(data = fit.x[kx:kx+kz], index = Z.columns.values)
vc = pd.DataFrame(data = -fit.hess_inv, index = np.append(X.columns.values, Z.columns.values),                 columns = np.append(X.columns.values, Z.columns.values))

## fitted and residuals
mu = np.exp(np.dot(X,coefc)+offsetx)
phi = link.link_inv(np.dot(Z,coefz)+offsetz)
Yhat = (1-phi) * mu
res = np.sqrt(weights) * (Y - Yhat)

## effective observations
nobs = np.sum(weights > 0)

start['count'] = pd.Series(np.repeat(0,kx), index = X.columns.values)

len(start['count']) != kx

options = {'disp': False, 'maxiter': 10000}

model_zero = sm.GLM(Y0.astype(int), exog = Z, family=sm.families.Binomial(link = link.linkclass),                    offset = offsetz , freq_weights = weights).fit()

y = {'options':options,'zzz':2}

probi = model_zero.predict()
probi = probi/(probi + (1-probi)*sp.stats.poisson.pmf(0, mui))

dist = 'Poisson'

dist not in ['Po','NegBin','Geom']

linklist =  ['Poisso','NegBin','Geom']
sys.exit(dist+"method not yet implemented"+str(linklist))
 



