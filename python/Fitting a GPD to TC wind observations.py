get_ipython().magic('matplotlib inline')

from __future__ import division
import seaborn as sns
import pandas as pd
import numpy as np

from functools import wraps
import time
import os
from os.path import join as pjoin
from glob import glob

from datetime import datetime
import matplotlib.pyplot as plt

from scipy.stats import genpareto
import statsmodels.api as sm
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.distributions.empirical_distribution import ECDF

import Utilities.metutils as metutils

from return_period import returnLevels, empiricalReturnPeriod, returnPeriodUncertainty
from distributions import fittedPDF

from IPython.html.widgets import interact, fixed
from IPython.html import widgets

sns.set_context("poster")
sns.set_style("ticks")

def parseTime(yr, month, day, hour, minute):
    """
    Parse year, month and day as strings and return a datetime.
    
    Handles the case of a missing time string (Pandas returns nan 
    if the field is empty).
    """
    timestr = '{0}-{1:02d}-{2:02d} {3:02d}:{4:02d}'.format(yr, int(month), int(day), int(hour), int(minute))
    
    return datetime.strptime(timestr, '%Y-%m-%d %H:%M')

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
                 ax=axes[3], kde_kws={"label":"Empirical PDF"})
    axes[3].plot(sortedmax, gpdf, color='r', label="Fitted PDF")
    axes[3].set_title("Density plot")
    axes[3].legend(loc=1)
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
    err = returnPeriodUncertainty(data, mu, xi, sigma, rp)

    sortedmax = np.sort(data)
    fig, ax1 = plt.subplots(1, 1)
    ax1.semilogx(rp, rval, label="Fitted RP curve")
    ax1.semilogx(rp, rval + 1.96 * err, label="95% CI",
                 linestyle='--', color='0.5')
    ax1.semilogx(rp, rval - 1.96 * err, linestyle='--', color='0.5')

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

DTYPE = [('id', 'S8'), ('hm', 'S2'), ('StnNum', 'i'), ('Year', 'i'), ('Month', 'i'), 
         ('Day', 'i'), ('Hour', 'i'), ('Minute', 'i'), ('dtStdYear', 'i'), ('dtStdMonth', 'i'), 
         ('dtStdDay', 'i'), ('dtStdHour', 'i'), ('dtStdMinute', 'i'), ('Speed', 'f8'), 
         ('QSpeed', 'S1'), ('Dir', 'f8'), ('QDir', 'S1'), ('Gust', 'f8'), ('QGust', 'S1'), ('AWSFlag', 'S2'),
         ('end', 'S1'), ('TCName', 'S10')]
NAMES = [fields[0] for fields in DTYPE]
CONVERT = {'Speed': lambda s: metutils.convert(float(s or 0), 'kmh', 'mps'),
           'Gust': lambda s: metutils.convert(float(s or 0), 'kmh', 'mps')}

@timer
def selectThreshold(data, start=None):
    """
    Select an appropriate threshold for fitting a generalised pareto
    distribution. 
    
    The only constraint placed on the selection is that the shape 
    parameter is negative (such that the distribution is bounded).
    
    :param data: :class:`numpy.ndarray` containing the observed values (with 
                 missing values removed).
    :param start: starting point for the threshold value. If not given, 
                  defaults to the median of the ``data`` variable.
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
    if start:
        startValue = start
    else:
        startValue = np.median(data)
    for mu in np.arange(startValue, datamax, 0.01):
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
        if pp[0] < eps and qdiff < 0.2*q10000 and qdiff > -eps: 
            t.append(mu)
            sh.append(pp[0])
            sc.append(pp[2])
            q1000list.append(q1000)
            q10000list.append(q10000)
            
    if len(t) == 0:
        #print "No suitable shape parameters identified"
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

input_path = 'N:\\climate_change\\CHARS\\B_Wind\\data\\derived\\obs\\tc\\ibtracs'
#input_path = "C:\\WorkSpace\\data\\Derived\\obs\\tc\\ibtracs\\"
basename = 'bom_{0:06d}.csv'
stnNum = "4032"
stnName = "Port Hedland"

fname = pjoin(input_path, basename.format(int(stnNum)))
if os.path.exists(fname):
    df = pd.read_csv(fname, skipinitialspace=True, skiprows=1, names=NAMES, 
                     parse_dates=[['Year', 'Month', 'Day', 'Hour', 'Minute']], 
                     date_parser=parseTime, index_col=False, converters=CONVERT)
    df.describe()
else:
    print "{0} does not exist".format(fname)

plt.figure(figsize=(12,6))
plt.title('TC-related gust wind speeds for {0}'.format(stnName))
plt.xlabel('Year')
plt.ylabel('Gust wind speed (m/s)')
x = [idx for idx in df.Year_Month_Day_Hour_Minute]
y = df.Gust
plt.scatter(x,y)
plt.axhline(np.median(y), linewidth=1, linestyle='--')

quality = df['QGust'].fillna("X").map(lambda x: x in ['Y','N','X',' ', np.nan])
dmax = df['Gust'][df['Gust'].notnull() & quality]

xi, sigma, mu = selectThreshold(dmax, start=np.min(dmax))
print xi, sigma, mu

tdelta = df.Year_Month_Day_Hour_Minute.max().year - df.Year_Month_Day_Hour_Minute.min().year
dummydata = np.zeros(int(tdelta+1)*365)
ndata = len(dmax)
dummydata[-ndata:] = dmax

plotFit(dummydata, mu, xi, sigma, stnName)

plotDiagnostics(dummydata, mu, xi, sigma)



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

