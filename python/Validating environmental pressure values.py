get_ipython().magic('matplotlib inline')

import os
from os.path import join as pjoin
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

from Utilities.metutils import convert
from Utilities.interp3d import interp3d
from Utilities.nctools import ncLoadFile, ncGetData

import numpy as np
import scipy.stats as stats

import pandas as pd
import statsmodels.api as sm
import statsmodels.nonparametric.api as smnp
from six import string_types

import seaborn as sns
sns.set_style("ticks")
sns.set_context("talk")
from seaborn.utils import _kde_support

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
    17: lambda s: float(s.strip(',')),
    18: lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[18], 'km'),
    19: lambda s: convert(float(s.strip(' ,') or 0), COLUNITS[19], 'km'),
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
    filter2 = (data['Longitude'][idx] >= 90.) & (data['Longitude'][idx] <= 180.)
    filter3 = (data['rMax'][idx] >= 0.1)
    subsidx = np.nonzero(filter1 & filter2 & filter3)
    return data[subsidx]

def julianDays(datetime):
    jdays = np.array([float(dt.strftime("%j")) + dt.hour/24. for dt in datetime])
    return jdays

def processFiles(path, basin):
    lon = np.array([])
    lat = np.array([])
    prs = np.array([])
    poci = np.array([])
    day = np.array([])
    for root, dirs, files in os.walk(path):
        if root.endswith(basin):
            for file in files:
                data = loadData(pjoin(root, file))
                if 'Status' in data.dtype.names:
                    data = filterData(data)
                    if 'Poci' in data.dtype.names:
                        poci = np.append(poci, data['Poci'])
                        prs = np.append(prs, data['Pressure'])
                        lat = np.append(lat, data['Latitude'])
                        lon = np.append(lon, data['Longitude'])
                        day = np.append(day, julianDays(data['Datetime']))
    return poci, prs, lon, lat, day

inputPath = "C:\\WorkSpace\\data\\Raw\\best_tracks"
spoci, sprs, slon, slat, sdays = processFiles(inputPath, 'sh')

scoords = np.array([sdays, slat, slon])
ncfile = "C:\\WorkSpace\\tcrm\\MSLP\\slp.day.ltm.nc"
ncobj = ncLoadFile(ncfile)
slpunits = getattr(ncobj.variables['slp'], 'units')
slpdata = ncGetData(ncobj, 'slp')
spenv = interp3d(slpdata, scoords, scale=[365., 180., 360.], offset=[0., -90., 0.])
spenv = convert(spenv, slpunits, 'hPa')

sjp = sns.jointplot(spenv.compress(spoci!=0), spoci.compress(spoci!=0), kind='hex')

sjp.set_axis_labels(r'$P_{ltm }$', r'$P_{oci}$')

poci = spoci.compress(spoci!=0)
penv = spenv.compress(spoci!=0)
cp = sprs.compress(spoci!=0)
dp = penv - cp
lat = slat.compress(spoci!=0)
jday = sdays.compress(spoci!=0)
print(len(poci))

ax = sns.distplot(penv-poci, label=r"$p_{ltm} - p_{oci}$", kde_kws={"label":"KDE"}, 
                  fit=stats.lognorm, fit_kws={"label":"Fitted lognormal",
                                              "color":"0.5", "linestyle":"--"})
ax.set_xlabel(r"$p_{ltm} - p_{oci}$ (hPa)")
ax.set_ylabel("Probability")
ax.legend()
sns.despine()

X = np.column_stack((penv, cp, cp*cp, lat*lat, np.cos(np.pi*2*jday/365)))
X = sm.add_constant(X)
model = sm.OLS(poci, X)
results = model.fit()
print(results.summary())
print('Parameters: ', results.params)
print('P-value: ', results.pvalues)
print('R-squared: ', results.rsquared)
print('T-values: ', results.tvalues)

fig, (ax0, ax1) = plt.subplots(1,2)
ax = sns.distplot(results.resid, label="Resiuals", kde_kws={'label':'KDE of residuals', 'linestyle':'--'}, ax=ax0)
pp = sm.ProbPlot(results.resid, stats.norm, fit=True)
pp.qqplot('Normal', 'Residuals', line='45', ax=ax1, color='gray', alpha=0.5)
fig.tight_layout()

fp = stats.norm.fit(results.resid,)#shape=np.mean(results.resid),scale=np.std(results.resid))


x = np.linspace(-10, 10, 1000)
print(fp)
print(stats.mstats.normaltest(results.resid))
print(stats.shapiro(results.resid))
ax.plot(x, stats.norm.pdf(x, fp[0], fp[1]), label='Normal')
ax.legend(loc=2)
p = list(results.params)
p.append(fp[1])
print(p)

nx = len(poci)
ind = np.random.choice(np.arange(nx), 10000, replace=True)
penv0 = penv[ind]
cp0 = cp[ind]
lat0 = lat[ind]
jday0 = jday[ind]

poci_model = p[0] + p[1]*penv0 + p[2]*cp0 +p[3]*cp0*cp0 + p[4]*lat0*lat0 +     p[5]*np.sin(np.pi*2*jday0/365) + np.random.normal(scale=p[6], size=10000)

fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True)

ax0.scatter(penv0, poci_model, c=np.abs(lat0), 
           cmap=sns.light_palette('blue', as_cmap=True), 
           s=40, label='Model', alpha=0.25)
ax0.scatter(penv, poci, c='r', edgecolor='r', marker='+', 
           s=50, label="Observations")
#ax.set_xlim(1005, 1020)
ax0.set_xlabel(r"$P_{ltm }$ (hPa)")
ax0.set_ylabel(r"$P_{oci}$ (hPa)")
#ax.set_ylim(990, 1015)
ax0.legend(loc=3, frameon=True)
ax0.grid(True)

ax1.scatter(cp0, poci_model, c=np.abs(lat0), 
           cmap=sns.light_palette('blue', as_cmap=True), 
           s=40, label='Model', alpha=0.25)
ax1.scatter(cp, poci, c='r', edgecolor='r', marker='+', 
           s=50, label="Observations")

#ax1.set_xlim(1005, 1020)
ax1.set_xlabel(r"$P_{centre}$ (hPa)")
#ax1.set_ylim(980, 1015)
ax1.legend(loc=3, frameon=True)
ax1.grid(True)
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
    return np.linalg.norm(obs - model)

sns.set_style("darkgrid")
fig, (axes) = plt.subplots(2, 2, sharey=True)

ax0, ax1, ax2, ax3 = axes.flatten()
levs=np.arange(0.01, 0.11, 0.1)
ax = sns.kdeplot(penv, poci, cmap="Reds", ax=ax0, kwargs={'levels':levs}, shade=True, shade_lowest=False)
ax = sns.kdeplot(penv0, poci_model, cmap="Blues", ax=ax0, kwargs={'levels':levs})
xx, yy, ope_poci = bivariate_kde(penv, poci)
xx, yy, mpe_poci = bivariate_kde(penv0, poci_model)
l2pe = l2score(ope_poci, mpe_poci)

ax = sns.kdeplot(cp, poci, cmap="Reds", ax=ax1, kwargs={'levels':levs}, shade=True, shade_lowest=False)
ax = sns.kdeplot(cp0, poci_model, cmap="Blues", ax=ax1, kwargs={'levels':levs})
xx, yy, ocp_poci = bivariate_kde(cp, poci)
xx, yy, mcp_poci = bivariate_kde(cp0, poci_model)
l2cp = l2score(ocp_poci, mcp_poci)

ax = sns.kdeplot(lat, poci, cmap="Reds", ax=ax2, kwargs={'levels':levs}, shade=True, shade_lowest=False)
ax = sns.kdeplot(lat0, poci_model, cmap="Blues", ax=ax2, kwargs={'levels':levs})
xx, yy, olat_poci = bivariate_kde(lat, poci)
xx, yy, mlat_poci = bivariate_kde(lat0, poci_model)
l2lat = l2score(olat_poci, mlat_poci)

ax = sns.kdeplot(jday, poci, cmap="Reds", ax=ax3, kwargs={'levels':levs}, shade=True, shade_lowest=False)
ax = sns.kdeplot(jday0, poci_model, cmap="Blues", ax=ax3, kwargs={'levels':levs})
xx, yy, ojday_poci = bivariate_kde(jday, poci)
xx, yy, mjday_poci = bivariate_kde(jday0, poci_model)
l2jday = l2score(ojday_poci, mjday_poci)

red = sns.color_palette("Reds")[-1]
blue = sns.color_palette("Blues")[-1]
ax0.text(0.1, 0.1, "Observed", color=red, transform=ax0.transAxes)
ax0.text(0.1, 0.05, "Model", color=blue, transform=ax0.transAxes)
ax3.text(0.8, 0.05, r"$l_2=${0:.3f}".format(l2pe), transform=ax0.transAxes)
ax1.text(0.1, 0.1, "Observed", color=red, transform=ax1.transAxes)
ax1.text(0.1, 0.05, "Model", color=blue, transform=ax1.transAxes)
ax3.text(0.8, 0.05, r"$l_2=${0:.3f}".format(l2cp), transform=ax1.transAxes)
ax2.text(0.1, 0.1, "Observed", color=red, transform=ax2.transAxes)
ax2.text(0.1, 0.05, "Model", color=blue, transform=ax2.transAxes)
ax3.text(0.8, 0.05, r"$l_2=${0:.3f}".format(l2lat), transform=ax2.transAxes)
ax3.text(0.1, 0.1, "Observed", color=red, transform=ax3.transAxes)
ax3.text(0.1, 0.05, "Model", color=blue, transform=ax3.transAxes)
ax3.text(0.8, 0.05, r"$l_2=${0:.3f}".format(l2jday), transform=ax3.transAxes)

ax0.set_ylabel(r"$P_{oci}$ (hPa)")
ax0.set_xlabel(r"$P_{ltm}$ (hPa)")
ax1.set_xlabel(r"$P_{centre}$ (hPa)")
ax2.set_ylabel(r"$P_{oci}$ (hPa)")
ax2.set_xlabel("Latitude")
ax3.set_xlabel("Day of year")
ax3.set_xlim((0, 365))
ax0.grid(True)
ax1.grid(True)
ax2.grid(True)
ax3.grid(True)

ax0.text(0.1, 0.9, "(a)", ha='center', va='center', transform=ax0.transAxes)
ax1.text(0.1, 0.9, "(b)", ha='center', va='center', transform=ax1.transAxes)
ax2.text(0.1, 0.9, "(c)", ha='center', va='center', transform=ax2.transAxes)
ax3.text(0.1, 0.9, "(d)", ha='center', va='center', transform=ax3.transAxes)
fig.tight_layout()

def getPoci(penv, pcentre, lat, jdays, eps,
            coeffs=[2324.1564738613392, -0.6539853183796136,
                    -1.3984456535888878, 0.00074072928008818927,
                    0.0044469231429346088, -1.4337623534206905]):
    """
    Calculate a modified pressure for the outermost closed isobar, based
    on a model of daily long-term mean SLP values, central pressure,
    latitude and day of year.

    :param penv: environmental pressure estimate (from long term mean pressure
                 dataset, hPa).
    :param pcentre: Central pressure of storm (hPa).
    :param lat: Latitude of storm (degrees).
    :param jdays: Julian day (day of year).
    :param eps: random variate. Retained as a constant for a single storm.

    :returns: Revised estimate for the pressure of outermost closed isobar.
    """

    if len(coeffs) < 6:
        LOG.warn("Insufficient coefficients for poci calculation")
        LOG.warn("Using default values")
        coeffs=[2324.1564738613392, -0.6539853183796136,
                -1.3984456535888878, 0.00074072928008818927,
                0.0044469231429346088, -1.4337623534206905]

    if isinstance(penv, (np.ndarray, list)) and       isinstance(pcentre, (np.ndarray, list)) and       isinstance(lat, (np.ndarray, list)) and       isinstance(jdays, (np.ndarray, list)):
      assert len(penv) == len(pcentre)
      assert len(penv) == len(lat)
      assert len(penv) == len(jdays)
      
    poci_model = coeffs[0] + coeffs[1]*penv + coeffs[2]*pcentre       + coeffs[3]*pcentre*pcentre + coeffs[4]*lat*lat +         coeffs[5]*np.sin(np.pi*2*jdays/365) + eps

    nvidx = np.where(pcentre == sys.maxint)
    poci_model[nvidx] = sys.maxint
    return poci_model
    
import sys
test_pcentre = np.array([920, 920, sys.maxint, 920, 1010])
test_penv = np.array([1000, 1000, 1000, 1000, 1000])
test_lat = np.array([-10, -9, -8, -7, -7])
test_jday = np.array([311,312,313,314, 314])
eps = np.random.normal(0,scale=2.5717)

pp = getPoci(test_penv, test_pcentre, test_lat, test_jday, eps)
print(pp)





