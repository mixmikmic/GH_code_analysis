get_ipython().magic('matplotlib inline')
from __future__ import division
import seaborn as sns
import pandas as pd
import numpy as np

from functools import wraps
import time
import os
from os.path import join as pjoin

from datetime import datetime
import matplotlib.pyplot as plt

from scipy.stats import genpareto
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.distributions.empirical_distribution import ECDF

from return_period import returnLevels, empiricalReturnPeriod, returnPeriodUncertainty
from distributions import fittedPDF

from IPython.html.widgets import interact, fixed
from IPython.html import widgets

sns.set_context("poster")
sns.set_style("ticks")

def parse(yr, month, day, time):
    """
    Parse year, month and day as strings and return a datetime.
    
    Handles the case of a missing time string (Pandas returns nan 
    if the field is empty).
    """
    if time is np.nan:
        time='0000'
    timestr = '{0}-{1}-{2} {3}'.format(yr, month, day, time)
    
    return datetime.strptime(timestr, '%Y-%m-%d %H%M')

def timer(func):
    """
    Decorator to report execution time of a function/script.
    """
    @wraps(func)
    def wrap(*args, **kwargs):
        t1 = time.time()
        res = func(*args, **kwargs)

        tottime = time.time() - t1
        msg = "%02d:%02d:%02d " %           reduce(lambda ll, b : divmod(ll[0], b) + ll[1:],
                        [(tottime,), 60, 60])

        print "Time for {0}: {1}".format(func.func_name, msg) 
        return res

    return wrap

def find_nearest_index(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def calculateShape(mu, data):
    """
    :param float mu: threshold parameter for the GPD distribution.
    :param data: :class:`numpy.ndarray` of data values to fit.
    """
    nobs = len(data)
    nexc = len(data[data > mu])
    rate = float(nexc)/float(nobs)
    gpd = genpareto.fit(data[data > mu] - mu)

    return gpd

def plotDiagnostics(data, mu, xi, sigma):
    """
    Create a 4-panel diagnostics plot of the fitted distribution.

    :param data: :class:`numpy.ndarray` of observed data values (in units
                 of metres/second).
    :param float mu: Selected threshold value.
    :param float xi: Fitted shape parameter.
    :param float sigma: Fitted scale parameter.

    """
    fig, ax = plt.subplots(2, 2)
    axes = ax.flatten()
    # Probability plots
    sortedmax = np.sort(data[data > mu])   
    gpdf = fittedPDF(data, mu, xi, sigma)
    pp_x = sm.ProbPlot(sortedmax)
    pp_x.ppplot(xlabel="Empirical", ylabel="Model", ax=axes[0], line='45')
    axes[0].set_title("Probability plot")

    prplot = sm.ProbPlot(sortedmax, genpareto, distargs=(xi,),
                         loc=mu, scale=sigma)
    prplot.qqplot(xlabel="Model", ylabel="Empirical", ax=axes[1], line='45')
    axes[1].set_title("Quantile plot")

    ax2 = axes[2]
    rp = np.array([1, 2, 5, 10, 20, 50, 100, 200,
                   500, 1000, 2000, 5000, 10000])
    rate = float(len(sortedmax)) / float(len(data))
    rval = returnLevels(rp, mu, xi, sigma, rate)

    emprp = empiricalReturnPeriod(np.sort(data))
    ax2.semilogx(rp, rval, label="Fitted RP curve", color='r')
    ax2.scatter(emprp[emprp > 1], np.sort(data)[emprp > 1],
                color='b', label="Empirical RP", s=100)
    ax2.legend(loc=2)
    ax2.set_xlabel("Return period")
    ax2.set_ylabel("Return level")
    ax2.set_title("Return level plot")
    ax2.grid(True)
    maxbin = 4 * np.ceil(np.floor(data.max() / 4) + 1)
    sns.distplot(sortedmax, bins=np.arange(mu, maxbin, 2),
                 hist=True, axlabel='Wind speed (m/s)',
                 ax=axes[3])
    axes[3].plot(sortedmax, gpdf, color='r')
    axes[3].set_title("Density plot")
    plt.tight_layout()
    
def plotFit(data, mu, xi, sigma, title):
    """
    Plot a fitted distribution, with approximate 90% confidence interval
    and empirical return period values.

    :param data: :class:`numpy.ndarray` of observed data values.
    :param float mu: Selected threshold value.
    :param float xi: Fitted shape parameter.
    :param float sigma: Fitted scale parameter.
    :param str title: Title string for the plot.
    :param str figfile: Path to store the file (includes image format)

    """

    rp = np.array([1, 2, 5, 10, 20, 50, 100, 200,
                   500, 1000, 2000, 5000, 10000])
    rate = float(len(data[data > mu])) / float(len(data))
    rval = returnLevels(rp, mu, xi, sigma, rate)

    emprp = empiricalReturnPeriod(data)
    #err = returnPeriodUncertainty(data, mu, xi, sigma, rp)

    sortedmax = np.sort(data)
    fig, ax1 = plt.subplots(1, 1)
    ax1.semilogx(rp, rval, label="Fitted RP curve")
    print rp
    print rval
    #ax1.semilogx(rp, rval + 1.96 * err, label="90% CI",
    #             linestyle='--', color='0.5')
    #ax1.semilogx(rp, rval - 1.96 * err, linestyle='--', color='0.5')

    ax1.scatter(emprp[emprp > 1], sortedmax[emprp > 1], s=100,
                color='r', label="Empirical RP")

    title_str = (title + "\n" +
                 r"$\mu$ = {0:.3f}, $\xi$ = {1:.5f}, $\sigma$ = {2:.4f}".
                 format(mu, xi, sigma))
    ax1.set_title(title_str)
    ax1.legend(loc=2)
    ax1.set_ylim((0, 100))
    ax1.set_xlim((1, 10000))
    ax1.set_ylabel('Wind speed (m/s)')
    ax1.set_xlabel('Return period (years)')
    ax1.grid(which='major')
    ax1.grid(which='minor', linestyle='--', linewidth=1)

NAMES = ['dc', 'StnNum', 'Year', 'Month', 'Day', 'Speed', 
         'QSpeed', 'Dir', 'QDir', 'Time', 'QTime']
CONVERT = {'Speed': lambda s: float(s or 0)}
stations = ["4032", "8051", "14040", "14161", "14508", "31011", "40214"]

input_path = 'N:\\climate_change\\CHARS\\B_Wind\\data\\raw\\obs\\daily\\'
basename = "DC02D_Data_{0:06d}_999999997960863.txt"
#basename = "DC02D_Data_{0:06d}_99999999720437.txt"
stnNum = "4032"
stnName = "Port Hedland"

fname = pjoin(input_path, basename.format(int(stnNum)))
if os.path.exists(fname):
    df = pd.read_csv(fname, skipinitialspace=True, skiprows=1, names=NAMES, 
                     parse_dates=[['Year', 'Month', 'Day', 'Time']], 
                     date_parser=parse, index_col=False, converters=CONVERT)
    df.describe()
else:
    print "{0} does not exist".format(fname)
    

plt.figure(figsize=(12,6))
plt.title('Daily maximum wind speeds for {0}'.format(stnName))
plt.xlabel('Year')
plt.ylabel('Wind speed (m/s)')
x = [idx for idx in df.Year_Month_Day_Time]
y = df.Speed
plt.plot(x,y)
plt.axhline(np.median(y), linestyle='--', lw=1)

quality = df['QSpeed'].fillna("X").map(lambda x: x in ['Y','N','X',' ', np.nan])
dmax = df['Speed'][df['Speed'].notnull() & quality]

@timer
def selectThreshold(data):
    """
    Select an appropriate threshold for fitting a generalised pareto
    distribution. 
    
    The only constraint placed on the selection is that the shape 
    parameter is negative (such that the distribution is bounded).
    
    :param data: :class:`numpy.ndarray` containing the observed values (with 
                 missing values removed).
    :returns: tuple of the shape, scale and threshold.
    """
    
    sh = []
    sc = []
    t = []
    q1000list = []
    q10000list = []
    
    eps = -0.01
    datamax = data.max()
    nobs = len(data)
    for mu in np.arange(np.median(data), datamax, 0.005):
        nexc = len(data[data > mu]) 
        rate = nexc / nobs
        if nexc < 5:
            break

        pp = calculateShape(mu, data)
        q1000, q10000 = returnLevels(np.array([1000, 10000]), mu, pp[0], pp[2], rate)
        if np.isnan(q1000):
            continue

        if np.isnan(q10000):
            continue

        qdiff = np.abs(q10000 - q1000)
        if pp[0] < eps and qdiff < 0.12*q10000 and qdiff > -eps: 
            t.append(mu)
            sh.append(pp[0])
            sc.append(pp[2])
            q1000list.append(q1000)
            q10000list.append(q10000)
            
    if len(t) == 0:
        print "No suitable shape parameters identified"
        return 0, 0, 0
    Av1000 = np.mean(np.array(q1000list))
    Av10000 = np.mean(np.array(q10000list))
    Av1000 = np.ceil(Av1000 + 0.05*Av1000)
    Av10000 = np.ceil(Av10000 + 0.05*Av10000)

    idx1000 = find_nearest_index(np.array(q1000list), Av1000)
    idx10000 = find_nearest_index(np.array(q10000list), Av10000)
    
    u1000 = t[idx1000]
    u10000 = t[idx10000]

    if u1000 > u10000:
        shmax = sh[idx1000]
        scmax = sc[idx1000]
    else:
        shmax = sh[idx10000]
        scmax = sc[idx10000]

    return shmax, scmax, u1000    

@timer
def selThreshold(dmax):
    """
    Select the best fitting threshold that maximises the return period values, but minimises the $R^$ value
    when fitted against the observed distribution.
    """
    eps = -0.01
    datamax = data.max()
    nobs = len(data)
    mu = np.median(data)
    while mu < datamax:
        nexc = len(data[data > mu])
        exceed = data[data > mu]
        rate = nexc / nobs
        if nexc < 10:
            break
        pp = calculateShape(mu, data)
        
        if pp[0] > eps:
            break
            
        emppdf = empiricalPDF(exceed)
        
        try:
            popt, pcov = curve_fit(lambda x, xi, sigma:                                    genpareto.pdf(x, xi, loc=mu, scale=sigma),
                                   np.sort(exceed), emppdf, (xi, sigma))
        except RuntimeError as e:
            return 0.
        sd = np.sqrt(np.diag(pcov))

xi, sigma, mu = selectThreshold(dmax)
print xi, sigma, mu

plotFit(dmax, mu, xi, sigma, stnName)

plotDiagnostics(dmax, mu, xi, sigma)

npy = 365.25
dmax = df['Speed'][df['Speed'].notnull()]
nobs = len(dmax)
sortedmax = np.sort(dmax)

# Empirical return periods:
emprp = 1./(1. - np.arange(1, nobs+1, 1)/(nobs + 1))/npy

# Start with a threshold that is half the maximum observed value
thresh = np.median(dmax) #dmax.max()/2.#sortedmax[RP2 > 1][0]
nexc = len(dmax[dmax > thresh])
rate = float(nexc)/float(nobs)

pp = genpareto.fit(dmax[dmax > thresh] - thresh)
npy = 365.25
n = len(dmax)

x = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
rpfit = returnLevels(x, thresh, pp[0], pp[2], rate)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

# Bootstrap resampling:
for i in range(100):
    data = np.random.choice(dmax, nobs-1, replace=True)
    sfit = genpareto.fit(data[data > thresh] - thresh)
    #if sfit[0] < 0.0:
    srp = returnLevels(x, thresh, sfit[0], sfit[2], rate)
    ax1.semilogx(x, srp, alpha=0.1, color='0.5')
    
ax1.semilogx(x, rpfit, label=r"$\mu$={0:.2f} m/s".format(thresh))
ax1.semilogx(emprp[emprp > 1], sortedmax[emprp > 1], marker='x',
             color='r',label="Empirical RP")
#plt.xscale('log')
ax1.set_ylim((0, 100))
ax1.set_xlim((1, 10000))
ax1.set_ylabel('Wind speed (m/s)')
ax1.set_xlabel('Return period (years)')
ax1.legend()

# Testing range of threshold values
for t in np.arange(thresh, dmax.max(), 0.01):
    trate = float(len(dmax[dmax > t]))/float(nobs)
    tfit = genpareto.fit(dmax[dmax > t] - t)
    if tfit[0] < 0.0:
        trp = returnLevels(x, t, tfit[0], tfit[2], trate)
        #if np.abs(trp[9] - trp[-1]) > 0.5:
        ax2.semilogx(x, trp, alpha=0.1, color='0.5')
        
ax2.semilogx(x, rpfit)
ax2.semilogx(emprp[emprp > 1], sortedmax[emprp > 1], marker='x', color='r')
plt.xscale('log')
ax2.set_ylim((0, 100))
ax2.set_xlim((1, 10000))
ax2.set_ylabel('Wind speed (m/s)')
ax2.set_xlabel('Return period (years)')   

def plot_gpd(data, threshold):
    ax = sns.distplot(data[data >= threshold], bins = np.arange(threshold, 250, 2.5),
                     hist=True, fit=genpareto, axlabel='Wind speed (km/h)', 
                     kde_kws={'label':'KDE fit'},
                     fit_kws={'label':'GPD fit',
                              'color':'red'})
    ax.legend()
    params = genpareto.fit(data[data >= threshold], fscale=threshold)
    print "Fit parameters: ", params
    print "Crossing rate: ", float(len(dmax[dmax >= threshold]))/float(len(dmax))
interact(plot_gpd, data=fixed(dmax), threshold=(0, 150., 1.))



