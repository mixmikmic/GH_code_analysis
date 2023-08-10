# Import Python libraries
import numpy as np
import pandas as pd

# Import Visualization libraries
get_ipython().magic('matplotlib inline')
import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

# Load data with Pandas
file_url = "https://raw.githubusercontent.com/gcmatos/structural-geology/master/data/aperture-spacing.csv"
d = pd.read_csv(file_url)
d.head()

# Statistic summary
d.describe()

# Aperture distribution
with sns.axes_style("whitegrid"):
    sns.distplot(d.Aperture, kde=False, color='k');

with sns.axes_style("white"):
    sns.jointplot(x="Aperture", y="Spacing", data=d, color='k')

# Aggregation data by Aperture values cummulative sum
A = pd.DataFrame(d.Aperture                 .value_counts()                 .sort_index(ascending=False))
A.columns = ['Frequency']
A['ApertureClasses'] = A.index
A.index = np.arange(0,len(A),1)

# Calculate the sum of scanline lengths for Intensity
TotalLength = d.Spacing.sum() + d.Aperture.sum()

# Converting Frequency into Intensity (1/m)
A['Intensity'] = A.Frequency / (TotalLength/10E3)
A['CumIntensity'] = A.Frequency.cumsum() / (TotalLength/10E3)
A.head()

# Plotting script
# Variables
x = A.ApertureClasses
y = A.CumIntensity
# Plotting
ax = plt.axes()
#plt.figure(figsize=(10,5));
plt.plot(x, y, 'o', color='black');
plt.xlabel('Aperture (mm)');
plt.ylabel('Intensity (1/m)');
ax.set_title('Aperture Distribution');

from scipy import optimize
from scipy.stats import powerlaw

# Define function for calculating a power law
powerlaw = lambda x, amp, index: amp * (x**index)

# Note: all positive, non-zero data
xdata = A.ApertureClasses
ydata = A.CumIntensity
yerr = ydata

logx = np.log10(xdata)
logy = np.log10(ydata)
logyerr = yerr / ydata

# define our (line) fitting function
fitfunc = lambda p, x: p[0] + p[1] * x # coeficientes de ajuste da função
errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

pinit = [1.0, -1.0]
out = optimize.leastsq(errfunc, pinit,
                       args=(logx, logy, logyerr), 
                       full_output=1)

pfinal = out[0] # a
covar = out[1]
print(pfinal)
print(covar)

index = pfinal[1]
amp = 10.0**pfinal[0] # b tirado pelo inverso do log

indexErr = np.sqrt( covar[0][0] )
ampErr = np.sqrt( covar[1][1] ) * amp

plt.clf();
plt.figure(figsize=(10,5));
plt.plot(xdata, powerlaw(xdata, amp, index));     # Fit
plt.errorbar(xdata, ydata, yerr=yerr, fmt='k.');  # Data
plt.text(8, 80, 'Ampli = %5.2f +/- %5.2f' % (amp, ampErr));
plt.text(8, 90, 'Index = %5.2f +/- %5.2f' % (index, indexErr));
plt.title('Best Fit Power Law');
plt.xlabel('Aperture (mm)');
plt.ylabel('Intensity (1/m)');
plt.xlim(0, xdata.max() + 1);

plt.figure(figsize=(10,5));
plt.loglog(xdata, powerlaw(xdata, amp, index));
plt.errorbar(xdata, ydata, yerr=logyerr, fmt='k.');  # Data
plt.title('Best Fit Power Law');
plt.xlabel('Aperture (mm)');
plt.ylabel('Intensity (1/m)');

from scipy.optimize import curve_fit

def exponential_func(x, a, b, c):
    return a * np.exp(-b*x) + c

popt, pcov = curve_fit(exponential_func, xdata, ydata, p0=(1, 1e-6, 1))
print(popt)
print(pcov)

xx = np.linspace(xdata.min(), xdata.max(), 1000)
yy = exponential_func(xx, *popt)

plt.plot(x, y, 'ko', xx, yy);
plt.title('Exponential Fit');
plt.xlabel('Aperture (mm)');
plt.ylabel('Intensity (1/m)');

