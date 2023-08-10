get_ipython().magic('matplotlib inline')

from __future__ import division, print_function
import os
from os.path import join as pjoin
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from Utilities.metutils import convert

import numpy as np
import scipy.stats as stats

import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.nonparametric.api as smnp
from six import string_types
    
from statsmodels.tools.tools import ECDF

from lmfit import Model, Minimizer, fit_report, conf_interval, printfuncs, report_fit
import corner

import seaborn as sns
from seaborn.utils import _kde_support
sns.set_style("darkgrid")
sns.set_context("poster")

def convertLatLon(strval):
    """
    Convert a string representing lat/lon values from '140S to -14.0, etc.
    
    :param str strval: string containing the latitude or longitude.
    
    :returns: Latitude/longitude as a float value.
    
    """
    hemi = strval[-1].upper()
    fval = float(strval[:-1]) / 10.
    if (hemi == 'S') | (hemi == 'W'): 
        fval *= -1
    if (hemi == 'E') | (hemi == 'W'):
        fval = fval % 360
    return fval
            

COLNAMES = ['BASIN','Number', 'Datetime','TECHNUM', 'TECH','TAU', 'Latitude', 'Longitude', 'Windspeed','Pressure',
            'Status', 'RAD', 'WINDCODE','RAD1', 'RAD2','RAD3', 'RAD4','Poci', 'Roci','rMax', 'GUSTS','EYE',
            'SUBREGION','MAXSEAS', 'INITIALS','DIR', 'SPEED','STORMNAME', 'DEPTH','SEAS',
            'SEASCODE','SEAS1', 'SEAS2','SEAS3', 'SEAS4'] 

COLTYPES = ['|S2', 'i', datetime, 'i', '|S4', 'i', 'f', 'f', 'f', 'f', 
            '|S4', 'f', '|S3', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f', 'f',
            '|S1', 'f', '|S3', 'f', 'f', '|S10', '|S1', 'f', 
            '|S3', 'f', 'f', 'f', 'f']
COLUNITS = ['', '', '', '', '', '', '', '', 'kts', 'hPa', 
            '', 'nm', '', 'nm', 'nm', 'nm', 'nm', 'hPa', 'nm', 'nm', 'kts', 'nm',
            '', '', '', 'degrees', 'kts', '', '', '',
            '', '', '', '', '']
DATEFORMAT = "%Y%m%d%H"
dtype = np.dtype({'names':COLNAMES, 'formats':COLTYPES})
converters = {
    1: lambda s: s.strip(' ,'),
    2: lambda s: datetime.strptime(s.strip(' ,'), DATEFORMAT),
    6: lambda s: float(convertLatLon(s.strip(' ,'))),
    7: lambda s: float(convertLatLon(s.strip(' ,'))),
    8: lambda s: s.strip(' ,'),
    9: lambda s: s.strip(' ,'),
    10: lambda s: s.strip(' ,'),
    11: lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[11], 'km'),
    12: lambda s: s.strip(' ,'),
    13: lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[13], 'km'),
    14: lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[14], 'km'),
    15: lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[15], 'km'),
    16: lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[16], 'km'),
    17: lambda s: float(s.strip(' ,')),
    18: lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[18], 'km'),
#    19: lambda s: float(s.strip(' ,'))
    19: lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[19], 'km')
}
delimiter = (3,4,12,4,6,5,7,7,5,6,4,5,5,6,6,6,6,6,6,5,5,5,5)
skip_header = 0
usecols = tuple(range(23))
missing_value = ""
filling_values = 0

def loadData(filename):
    try:
        data = np.genfromtxt(filename, dtype, delimiter=delimiter, skip_header=skip_header, 
                             converters=converters, missing_values=missing_value, 
                             filling_values=filling_values, usecols=usecols, autostrip=True, invalid_raise=False)
    except IndexError:
        try:
            data = np.genfromtxt(filename, dtype, delimiter=delimiter, skip_header=skip_header, 
                             converters=converters, missing_values=missing_value, 
                             filling_values=filling_values, usecols=tuple(range(18)), autostrip=True, invalid_raise=False)
        except IndexError:
            data = np.genfromtxt(filename, dtype, delimiter=[3,4,12,4,6,5,7,7,5], skip_header=skip_header, 
                             converters=converters, missing_values=missing_value, 
                             filling_values=filling_values, usecols=tuple(range(9)), autostrip=True, invalid_raise=False)
    return data

def filterData(data):
    datetimes, idx = np.unique(data['Datetime'], True)
    filter1 = (data['Status'][idx] == 'TS') | (data['Status'][idx] == 'TY')
    filter2 = (data['Longitude'][idx] >= 60.) & (data['Longitude'][idx] <= 180.)
    filter3 = (data['rMax'][idx] >= 0.1)
    filter4 = (data['Poci'][idx] > 0.1)
    subsidx = np.nonzero(filter1 & filter2 & filter3 & filter4)
    return data[subsidx]

def processfiles(path, basin):
    rmax = np.array([])
    prs = np.array([])
    lat = np.array([])
    poci = np.array([])
    for root, dirs, files in os.walk(path):
        if root.endswith(basin):
            for file in files:
                data = loadData(pjoin(root, file))
                if 'Status' in data.dtype.names:
                    data = filterData(data)
                    if 'rMax' in data.dtype.names:
                        rmax = np.append(rmax, data['rMax'])
                        prs = np.append(prs, data['Pressure'])
                        poci = np.append(poci, data['Poci'])
                        lat = np.append(lat, data['Latitude'])
    return rmax, prs, poci, lat

inputPath = "C:\\WorkSpace\\data\\Raw\\best_tracks"
rmax, prs, poci, lat = processfiles(inputPath, 'sh')
#outputFile = pjoin(inputPath, "rmax-sh.csv")
#np.savetxt(outputFile, np.column_stack((rmax, prs, poci, lat)), delimiter=',', fmt='%6.1f')

print("Parameter estimates:       Shape; Location (fixed);    Scale;    Mean")
fig, ax = plt.subplots(1,1)
sns.distplot(rmax, bins=np.arange(0, 101, 10),
             kde_kws={'clip':(0, 100), 'label':"KDE"}, ax=ax)

shape, loc, scale = stats.lognorm.fit(rmax, scale=np.mean(rmax), floc=0)
print("Southern hemisphere basin: ", shape, loc, scale, np.mean(rmax))
x = np.arange(1, 201)
v = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
fcdf = stats.lognorm.cdf(np.sort(rmax), shape, loc=loc, scale=scale)

ax.plot(x, v, label="Lognormal fit")
ax.legend(loc=0)
ax.set_xlabel(r'$R_{max}$ (km)')
ax.set_ylabel('Probability')
ax.set_xlim((0, 100))
ax.set_title("Southern hemisphere (2002-2014)")


fig.tight_layout()
sns.despine()

ecdf = ECDF(rmax, side='left')

plt.plot(np.sort(rmax), ecdf.y[1:])
plt.plot(np.sort(rmax), fcdf, 'r' )
rsq = stats.pearsonr(np.sort(rmax), fcdf)[0]**2
plt.text( 10, 0.9, r"$R^{2}$ = %f"%rsq)

def filterPoci(field, poci):
    filter1 = (poci >= 0.1)
    subsidx = np.nonzero(filter1)
    return field[subsidx]

rmax = filterPoci(rmax, poci)
dp = filterPoci(poci, poci) - filterPoci(prs, poci)
dp = np.extract(np.nonzero(rmax), dp)
dpsq = dp*dp
expdp = np.exp(-dp)
expdpsq = np.exp(-dpsq)
lat = filterPoci(lat, poci)
lat = np.extract(np.nonzero(rmax), lat)
rmax = np.extract(np.nonzero(rmax), rmax)

latsq = lat*lat

X = np.column_stack((dp, lat))
y = np.log(rmax)

def exp_dpsq(x, gamma, delta):
    dp = x[:,0]
    return gamma*np.exp(-delta*dp*dp)

def lin_dp(x, alpha, beta):
    dp = x[:,0]
    return alpha + beta*dp

def lin_lat(x, zeta):
    lat = np.abs(x[:,1])
    return zeta*lat

rmod = Model(lin_dp) + Model(exp_dpsq) + Model(lin_lat)
params = rmod.make_params(alpha=1., beta=-0.001, gamma=.1, delta=.001, zeta=.001)
def resid(p):
    return p['alpha'] + p['beta']*X[:,0] + p['gamma']*np.exp(-p['delta']*X[:,0]*X[:,0]) + p['zeta']*np.abs(X[:,1]) - y

mini = Minimizer(resid, params)
result = mini.minimize()
print(fit_report(result.params))
ci = conf_interval(mini, result)
printfuncs.report_ci(ci)

rr = mini.emcee(burn=500)

ll = [r'$\{0}$'.format(v) for v in rr.var_names]
with sns.plotting_context("notebook"):
    corner.corner(rr.flatchain, labels=ll, truths=list(rr.params.valuesdict().values()),
              no_fill_contours=True, fill_contours=False, plot_density=False,
              quantiles=[0.05, 0.5, 0.95],
                 data_kwargs=dict(color='r', alpha=0.01),
             contour_kwargs=dict(color='g'))

print(report_fit(rr.params))

print(rr.params)

print(r'alpha = {0}'.format(rr.params['alpha'].value))
print(r'beta = {0}'.format(rr.params['beta'].value))
print(r'gamma = {0}'.format(rr.params['gamma'].value))
print(r'delta = {0}'.format(rr.params['delta'].value))
print(r'zeta = {0}'.format(rr.params['zeta'].value))

result = rmod.fit(y, x=X, params=params)
print(result.fit_report())

result.params = rr.params

plt.plot(X[:,0], y,         'bo')
#plt.plot(X[:,0], result.init_fit, 'k--')
plt.plot(X[:,0], result.best_fit, 'r.')
plt.show()

comps = result.eval_components()
print(comps)

fig, (ax0, ax1) = plt.subplots(1, 2)

ax = sns.distplot(result.residual, kde_kws={'label':'Residuals', 'linestyle':'--'}, ax=ax0, norm_hist=True)
pp = sm.ProbPlot(result.residual, stats.norm, fit=True)
pp.qqplot('Normal', 'Residuals', line='45', ax=ax1, color='gray',alpha=0.5)
fig.tight_layout()
x = np.linspace(-2, 2, 1000)

ax0.legend(loc=0)

fp = stats.norm.fit(result.residual)
ax0.plot(x, stats.norm.pdf(x, fp[0], fp[1]), label='Normal', color='r')
print(fp)
print(stats.mstats.normaltest(result.residual))
ax0.legend()

deltap = np.linspace(0, 100, 100)
lats = np.arange(-30, -1, 4)
#lats = np.arange(2, 31, 4)
fig, ax = plt.subplots(1,1)
sns.set_palette("RdBu", 10)
for l in lats:
    xx = np.column_stack((deltap, l*np.ones(len(deltap))))
    yy = result.eval(x=xx)
    ax.plot(deltap, np.exp(yy), label="%d"%l)
ax.set_ylabel(r"$R_{max}$ (km)")
ax.set_xlabel(r"$\Delta p$ (hPa)")
ax.legend(loc=1)

nx = len(dp)
ind = np.random.choice(np.arange(nx), nx, replace=True)
dp0 = dp[ind]
l0 = lat[ind]

xx = np.column_stack((dp0, l0))
yy = result.eval(x=xx) + np.random.normal(scale=0.33, size=nx)


rm = np.exp(yy)
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].scatter(dp0, rm, c=np.abs(l0), cmap=sns.light_palette('blue', as_cmap=True), s=40, label='Model', alpha=0.5)
ax[0].scatter(dp, rmax, c='w', edgecolor='r', s=50, marker='+', label='Observations')
ax[0].set_xlim(0, 100)
ax[0].set_xlabel(r"$\Delta p$ (hPa)")
ax[0].set_ylabel(r"$R_{max}$ (km)")
ax[0].set_ylim(0, 200)
ax[0].legend(loc=1)
ax[0].grid(True)

ax[1].scatter(l0, rm, c=np.abs(l0), cmap=sns.light_palette('blue', as_cmap=True), s=40, label='Model', alpha=0.5)
ax[1].scatter(lat, rmax, c='w', edgecolor='r', s=50, marker='+', label='Observations')
ax[1].set_xlim(-30, 0)
ax[1].set_xlabel(r"Latitude")
ax[1].set_ylim(0, 200)
ax[1].legend(loc=1)
ax[1].grid(True)

fig.tight_layout()

def bivariate_kde(x, y, bw='scott', gridsize=100, cut=3, clip=None):
    if isinstance(bw, string_types):
        bw_func = getattr(smnp.bandwidths, "bw_" + bw)
        x_bw = bw_func(x)
        y_bw = bw_func(y)
        bw = [x_bw, y_bw]
    elif np.isscalar(bw):
        bw = [bw, bw]

    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    kde = smnp.KDEMultivariate([x, y], "cc", bw)
    x_support = _kde_support(x, kde.bw[0], gridsize, cut, [x.min(), x.max()])# clip[0])
    y_support = _kde_support(y, kde.bw[1], gridsize, cut, [y.min(), y.max()])#clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z

def l2score(obs, model):
    return np.linalg.norm(obs - model, np.inf)

xx, yy, odp_rmax = bivariate_kde(dp,  rmax, bw='scott')
xx, yy, mdp_rmax = bivariate_kde(dp0, rm, bw='scott')

xx, yy, olat_rmax = bivariate_kde(lat,  rmax, bw='scott')
xx, yy, mlat_rmax = bivariate_kde(l0, rm, bw='scott')


l2rmdp = l2score(odp_rmax, mdp_rmax)
l2rmlat = l2score(olat_rmax, mlat_rmax)


fig, ax = plt.subplots(1, 1)
levs = np.arange(0.01, 0.11, 0.01)
ax = sns.kdeplot(dp, rmax, cmap='Reds', kwargs={'levels':levs}, shade=True, shade_lowest=False)
ax = sns.kdeplot(dp0, rm, cmap='Blues', kwargs={'levels':levs})
ax.set_xlim(0, 100)
ax.set_xlabel(r"$\Delta p$ (hPa)")
ax.set_ylabel(r"$R_{max}$ (nm)")
ax.set_ylim(0, 100)
ax.grid(True)

red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(80, 90, "Observed", color=red)
ax.text(80, 80, "Model", color=blue)
ax.text(80, 70, r"$l_2=${0:.3f}".format(l2rmdp))

fig, ax = plt.subplots(1, 1)
ax = sns.kdeplot(lat, rmax, cmap='Reds', kwargs={'levels':levs}, shade=True, shade_lowest=False)
ax = sns.kdeplot(l0, rm, cmap='Blues', kwargs={'levels':levs})
ax.set_xlim(-30, 0)
ax.set_xlabel("Latitude")
ax.set_ylabel(r"$R_{max}$ (nm)")
ax.set_ylim(0, 100)
ax.grid(True)



ax.text(-5, 90, "Observed", color=red)
ax.text(-5, 80, "Model", color=blue)
ax.text(-5, 70, r"$l_2=${0:.3f}".format(l2rmlat))

x = np.arange(1, 101)
v = stats.lognorm.pdf(x, shape, loc=loc, scale=scale)
fig, ax = plt.subplots(1, 1)
sns.distplot(rm, bins=np.arange(0, 101, 5),
             kde_kws={'clip':(0, 100), 'label':"Model data (KDE)"},)
ax.plot(x, v, label="Lognormal fit from observations", color='r')
ax.legend(loc=0)
ax.set_xlabel(r'$R_{max}$ (km)')
ax.set_xlim((0, 100))

fig, ax = plt.subplots(1,1)
sns.distplot(rmax, bins=np.arange(0, 151, 10),
             kde_kws={'clip':(0, 150), 'label':"Observations"}, ax=ax, 
             hist_kws={ "linewidth":3})
sns.distplot(rm, bins=np.arange(0, 151, 10),
             kde_kws={'clip':(0, 150), 'label':"Model"}, ax=ax, color='r',
             hist_kws={ "linewidth":3})
ax.set_ylabel("Probability")
ax.set_xlabel(r"$R_{max}$ (km)")
ax.set_xlim((0, 120))

dparray = np.arange(10, 51, 5)
latarray = np.arange(-23, -5, 2)
testinput = np.column_stack((dparray, latarray))
np.random.seed(10)
print(result.params)
yy = result.eval(x=testinput) #+ np.random.normal(scale=0.335)
print(np.exp(yy))

eps = 0#np.random.normal(0, scale=0.335)
rmw = np.exp(result.eval(x=np.column_stack((np.array([25]), np.array([-15]))))+eps)
print(rmw)

eps = 0#np.random.normal(0, scale=0.335)
newparams = rmod.make_params(alpha=3.5, beta=-0.004, gamma=.7, delta=.002, zeta=.001)
yy = result.eval(params=newparams, x=testinput)+eps
rmw = np.exp(yy)
print(rmw)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(dp, np.abs(lat), rmax, c=rmax, cmap=sns.light_palette('blue', as_cmap=True))
ax.set_xlabel(r"$\Delta p$ (hPa)")
ax.set_ylabel(r"$|\lambda|$ (degrees)")
ax.set_zlabel(r"$R_{max}$ (km)")

xbins = np.arange(0, 101, 1)
ybins = np.arange(0, 30, 1)
(count, xedge, yedge, img) = plt.hist2d(dp, np.abs(lat), bins=[xbins, ybins], weights=rmax, normed=True)
plt.xlabel(r"$\Delta p$ (hPa)")
plt.ylabel(r"$|\lambda|$ (degrees)")
plt.colorbar(label=r"$R_{max}$")

from thinkbayes import Pmf, Suite

#rm = filterPoci(rmax, poci)
print(len(rmax))
pmf = Pmf()
for r in rmax:
    pmf.Incr(r, 1)
    
pmf.Normalize()
for x,y in pmf.Items():
    print( x, y)

def log_prior(theta):
    alpha, beta, gamma, zeta, sigma = theta
    if sigma < 0:
        return -np.inf  # log(0)
    else:
        return -1.5 * np.log(1 + beta ** 2 + gamma ** 2 + zeta ** 2) - np.log(sigma)

def log_likelihood(theta, x1, x2, x3, y):
    alpha, beta, gamma, zeta, sigma = theta
    y_model = alpha + beta * x1 + gamma * x2 + zeta * x3
    return -0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + (y - y_model) ** 2 / sigma ** 2)

def log_posterior(theta, x1, x2, x3, y):
    return log_prior(theta) + log_likelihood(theta, x1, x2, x3, y)

ndim=5
nwalkers=50
nburn=1000
nsteps=2000
np.random.seed(0)
starting_guesses=np.random.random((nwalkers, ndim))

import emcee

ydata = rm
xdata1 = dp
xdata2 = dp * dp
xdata3 = np.abs(lat)**2
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[xdata1, xdata2, xdata3, ydata])

sampler.run_mcmc(starting_guesses, nsteps)

emcee_trace = sampler.chain[:, nburn:, :].reshape(-1, ndim).T

def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    
    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


def plot_MCMC_trace(ax, xdata1, xdata2, xdata3, ydata, trace, scatter=False, **kwargs):
    """Plot traces and contours"""
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace[0], trace[1], ',k', alpha=0.1)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    
    
def plot_MCMC_model(ax, xdata1, xdata2, xdata3, ydata, trace):
    """Plot the linear model and 2sigma contours"""
    ax.plot(xdata1, ydata, 'ok')

    alpha, beta, gamma, zeta = trace[:4]
    xfit = np.linspace(-20, 120, 10)
    yfit = alpha[:, None] + beta[:, None] * xfit
    mu = yfit.mean(0)
    sig = 2 * yfit.std(0)

    ax.plot(xfit, mu, '-k')
    ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')

    ax.set_xlabel(r'$\Delta p$ (hPa)')
    ax.set_ylabel(r'$R_{max}$ (km)')

def plot_MCMC_results(xdata1, xdata2, xdata3, ydata, trace, colors='k'):
    """Plot both the trace and the model together"""
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    plot_MCMC_trace(ax[0], xdata1, xdata2, xdata3, ydata, trace, True, colors=colors)
    plot_MCMC_model(ax[1], xdata1, xdata2, xdata3, ydata, trace)
    fig.tight_layout()

plot_MCMC_results(xdata1, xdata2, xdata3, ydata, emcee_trace)

import corner
fig = corner.corner(emcee_trace.T, labels=[r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\zeta$", r"$\ln\,f$"], 
                    plot_density=False, no_fill_contours=True, data_kwargs={'color':'r'})

import pymc
print(pymc.__version__)

alpha = pymc.Normal('alpha', 0, 1)

@pymc.stochastic(observed=False)
def beta(value=0):
    return -1.5*np.log(1+value**2)

@pymc.stochastic(observed=False)
def gamma(value=0):
    return -1.5*np.log(1+value**2)

@pymc.stochastic(observed=False)
def zeta(value=0):
    return -1.5*np.log(1+value**2)

@pymc.stochastic(observed=False)
def sigma(value=1):
    return -np.log(abs(value))

@pymc.deterministic
def y_model(x1=xdata1, x2=xdata2, x3=xdata3, alpha=alpha, beta=beta, gamma=gamma, zeta=zeta):
    return alpha + beta * x1 + gamma * x2 + zeta * x3

y = pymc.Normal('y', mu=y_model, tau=1. / sigma ** 2, observed=True, value=ydata)

model1 = dict(alpha=alpha, beta=beta, gamma=gamma, zeta=zeta, sigma=sigma, y_model=y_model, y=y)

S = pymc.MCMC(model1)
S.sample(iter=10000,burn=5000)

pymc_trace = [S.trace('alpha')[:],
              S.trace('beta')[:],
              S.trace('gamma')[:],
              S.trace('zeta')[:],
              S.trace('sigma')[:]]
plot_MCMC_results(xdata1, xdata2, xdata3, ydata, pymc_trace)

fig = corner.corner(np.array(pymc_trace).T, labels=[r"$\alpha$", r"$\beta$", r"$\gamma$", r"$\zeta$", r"$\ln\,f$"], 
                    plot_density=False, no_fill_contours=True, data_kwargs={'color':'r'})

from scipy.optimize import curve_fit
def func(x, a, b, c, d, f):
    dp = x[:,0]
    lat = x[:,1]
    return a + b*dp + c*np.exp(-d*dp*dp) + f*np.abs(lat)

xx = np.column_stack((dp, lat))

popt, pcov = curve_fit(func, xx, np.log(rmax))
perr = np.sqrt(np.diag(pcov))
print(popt)
print(perr)

nx = len(dp)
ind = np.random.choice(np.arange(nx), nx, replace=True)
dp0 = dp[ind]
l0 = lat[ind]
xx = np.column_stack((dp0, l0))
yy = func(xx, *popt) + np.random.normal(scale=0.3, size=nx)
rm = np.exp(yy)

fig, ax = plt.subplots(1, 1)
ax.scatter(dp0, rm, c=np.abs(l0), cmap=sns.light_palette('blue', as_cmap=True), s=40, label='Model', alpha=0.5)
ax.scatter(dp, rmax, c='w', edgecolor='r', s=50, marker='+', label='Observations')
ax.set_xlim(0, 100)
ax.set_xlabel(r"$\Delta p$ (hPa)")
ax.set_ylabel(r"$R_{max}$ (km)")
ax.set_ylim(0, 100)
ax.legend(loc=1)
ax.grid(True)

deltap=np.linspace(0, 200, 200)
lats = np.arange(-30, -1, 4)

fig, ax = plt.subplots(1,1)
sns.set_palette(sns.color_palette("coolwarm", 8))
for l in lats:
    xx = np.column_stack((deltap, l*np.ones(len(deltap))))
    yy = func(xx, *popt)
    ax.plot(deltap, np.exp(yy), label="%d"%l)
    
ax.set_ylabel(r"$R_{max}$ (km)")
ax.set_xlabel(r"$\Delta p$ (hPa)")
ax.legend(loc=1)

xx = np.column_stack((dp, lat))
yy = func(xx, *popt)
#rm = filterPoci(rmax, poci)

resid = np.log(rmax) - yy
print(yy)
print(np.log(rmax))

fig, (ax0, ax1) = plt.subplots(1, 2)

ax = sns.distplot(resid, kde_kws={'label':'Residuals', 'linestyle':'--'}, ax=ax0, norm_hist=True)
pp = sm.ProbPlot(resid, stats.norm, fit=True)
pp.qqplot('Normal', 'Residuals', line='45', ax=ax1, color='gray',alpha=0.5)
fig.tight_layout()

#ppfit = pp.fit_params

x = np.linspace(-2, 2, 1000)

ax0.legend(loc=0)

fp = stats.norm.fit(resid)
ax0.plot(x, stats.norm.pdf(x, fp[0], fp[1]), label='Normal', color='r')
print(fp)
print(stats.mstats.normaltest(resid))
ax0.legend()
#p = list(results.params)



