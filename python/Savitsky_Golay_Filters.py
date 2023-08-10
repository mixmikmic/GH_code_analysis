import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
# sns.set_style('white')
np.set_printoptions(suppress=True, precision=3)
get_ipython().magic('matplotlib inline')

np.random.seed(628)
hours = np.arange(-12, 48, dtype='float')

# PRETEND YOU'RE NOT SEEING THIS!
visits = 1000 #baseline
visits = visits + 200/(np.exp(-1.25*hours+5) + 1) # A nice sigmoid...
no_noise_visits = visits.copy()
visits = visits + 30*np.random.randn(len(hours))
visits = np.around(visits)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))
axes[0].plot(hours, visits, color='k', label='Visits')
axes[0].plot(hours, no_noise_visits, color='r', label='Visits w/o noise')
axes[0].set_xlabel('Hours', fontsize=20)
axes[0].set_ylabel('Visits', fontsize=20)
axes[0].set_title('Whole dataset', fontsize=20)
axes[0].tick_params(labelsize=14)
axes[0].legend(fontsize=14, loc='lower right')
sns.despine()

axes[1].plot(hours[10:25], visits[10:25], color='k', label='Visits')
axes[1].plot(hours[10:25], no_noise_visits[10:25], color='r', label='Visits w/o noise')
axes[1].set_xlabel('Hours', fontsize=20)
axes[1].set_ylabel('Visits', fontsize=20)
axes[1].set_title('Zoomed in', fontsize=20)
axes[1].tick_params(labelsize=14)
axes[1].legend(fontsize=14, loc='lower right')

visits_3hr = (visits[:-2] + visits[1:-1] + visits[2:])/3.0
visits_5hr = (visits[:-4] + visits[1:-3] + visits[2:-2] + visits[3:-1] + visits[4:])/5.0
visits_10hr = (visits[:-9] + visits[1:-8] + visits[2:-7] + visits[3:-6] + visits[4:-5] +
               visits[5:-4] + visits[6:-3] + visits[7:-2] + visits[8:-1] + visits[9:])/10.0

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,6))
axes[0].plot(hours, visits, color='k', alpha=0.6, linewidth=2, label='Visits')
axes[0].plot(hours, no_noise_visits, color='k', linestyle=':', linewidth=2, label='No noise')
axes[0].plot(hours[1:-1], visits_3hr, color='r', linewidth=2, label='3hr avg')
axes[0].set_title('3hr avg', fontsize=20)
axes[0].tick_params(labelsize=14)
axes[0].legend(fontsize=14, loc='lower right')

axes[1].plot(hours, visits, color='k', alpha=0.6, linewidth=2, label='Visits')
axes[1].plot(hours, no_noise_visits, color='k', linestyle=':', linewidth=2, label='No noise')
axes[1].plot(hours[2:-2], visits_5hr, color='b', linewidth=2, label='5hr avg')
axes[1].set_title('5hr avg', fontsize=20)
axes[1].tick_params(labelsize=14)
axes[1].legend(fontsize=14, loc='lower right')

axes[2].plot(hours, visits, color='k', alpha=0.6, linewidth=2, label='Visits')
axes[2].plot(hours, no_noise_visits, color='k', linestyle=':', linewidth=2, label='No noise')
axes[2].plot(hours[4:-5], visits_10hr, color='g', linewidth=2, label='10hr avg')
axes[2].set_title('10hr avg', fontsize=20)
axes[2].tick_params(labelsize=14)
axes[2].legend(fontsize=14, loc='lower right')

# plt.plot(hours, visits, color='k', label='data')
# plt.plot(hours[1:-1], visits_3hr, color='r', label='3hr avg')
# plt.plot(hours[2:-2], visits_5hr, color='b', label='5hr avg')
# plt.plot(hours[4:-5], visits_10hr, color='g', label='10hr avg')
# plt.xlabel('Hours', fontsize=20)
# plt.ylabel('Visits', fontsize=20)
# plt.tick_params(labelsize=14)
# plt.legend(fontsize=14, loc='lower right')
sns.despine()

dx_gaussian = 0.03
x_gaussian = np.arange(-1, 1+dx_gaussian, dx_gaussian)
y_gaussian = sp.stats.norm.pdf(x_gaussian, loc=0.3, scale=0.1)

### 11 point moving average
x_gaussian11 = x_gaussian[5:-5]
y_gaussian11 = (y_gaussian[:-10] + y_gaussian[1:-9] + y_gaussian[2:-8] + y_gaussian[3:-7] +
                y_gaussian[4:-6] + y_gaussian[5:-5] + y_gaussian[6:-4] + y_gaussian[7:-3] +
                y_gaussian[8:-2] + y_gaussian[9:-1] + y_gaussian[10:])/11.0

plt.figure(figsize=(8,8))
plt.plot(x_gaussian, y_gaussian, color='k', label='original')
plt.plot(x_gaussian11, y_gaussian11, color='r', label='11 pt avg')
plt.legend(fontsize=14)
plt.axis([-1, 1, 0, 4])
sns.despine()

area_gaussian = np.sum(y_gaussian)*dx_gaussian
print('Area of Gaussian: %.3f' % area_gaussian)
mean_gaussian = np.sum(y_gaussian*x_gaussian*dx_gaussian)
print('Mean of Gaussian: %.3f' % mean_gaussian)
std_gaussian = np.sqrt(np.sum(y_gaussian*x_gaussian*x_gaussian)*dx_gaussian - mean_gaussian**2)
print('StDev of Gaussian: %.3f' % std_gaussian)

area_gaussian11 = np.sum(y_gaussian11)*dx_gaussian
print('Area of 11-pt-avg Gaussian: %.3f' % area_gaussian11)
mean_gaussian11 = np.sum(y_gaussian11*x_gaussian11*dx_gaussian)
print('Mean of 11-pt-avg Gaussian: %.3f' % mean_gaussian11)
std_gaussian11 = np.sqrt(np.sum(y_gaussian11*x_gaussian11*x_gaussian11)*dx_gaussian - mean_gaussian11**2)
print('StDev of 11-pt-avg Gaussian: %.3f' % std_gaussian11)

from scipy.signal import savgol_coeffs, savgol_filter

SG_11_coefs = savgol_coeffs(11, 2)
print(SG_11_coefs)
y_SG = savgol_filter(y_gaussian, 11, 2, delta=dx_gaussian)

plt.figure(figsize=(8,8))
plt.plot(x_gaussian, y_gaussian, color='k', label='original')
plt.plot(x_gaussian, y_SG, color='r', label='SG filter N = 2, L = 11')
plt.legend(fontsize=14, loc='upper left')
plt.axis([-1, 1, 0, 4])
sns.despine()

area_gaussian = np.sum(y_gaussian)*dx_gaussian
print('Area of Gaussian: %.3f' % area_gaussian)
mean_gaussian = np.sum(y_gaussian*x_gaussian*dx_gaussian)
print('Mean of Gaussian: %.3f' % mean_gaussian)
std_gaussian = np.sqrt(np.sum(y_gaussian*x_gaussian*x_gaussian)*dx_gaussian - mean_gaussian**2)
print('StDev of Gaussian: %.3f' % std_gaussian)

area_SG = np.sum(y_SG)*dx_gaussian
print('Area of 11-2-SG Gaussian: %.3f' % area_SG)
mean_SG = np.sum(y_SG*x_gaussian*dx_gaussian)
print('Mean of 11-2-SG Gaussian: %.3f' % mean_SG)
std_SG = np.sqrt(np.sum(y_SG*x_gaussian*x_gaussian)*dx_gaussian - mean_SG**2)
print('StDev of 11-2-SG Gaussian: %.3f' % std_SG)

SGvisits = savgol_filter(visits, 13, 6, delta=1, mode='mirror')

plt.figure(figsize=(8,8))
plt.plot(hours, visits, color='k', alpha=0.6, label='Visits')
plt.plot(hours, SGvisits, color='r', label='SG')
plt.plot(hours, no_noise_visits, color='k', linestyle=':', linewidth=2, label='No noise')
plt.legend(loc='lower right', fontsize=14)
plt.tick_params(labelsize=14)
plt.xlabel('Hours', fontsize=20)
plt.ylabel('Visits', fontsize=20)
sns.despine()

dumb_deriv = savgol_filter(visits, 3, 1, deriv=1, delta=1, mode='mirror')
# ^^^ this is the -1, 0, 1 centered difference derivative
deriv_on_nonoise = savgol_filter(no_noise_visits, 3, 1, deriv=1, delta=1, mode='mirror')
SGvisits_deriv = savgol_filter(visits, 15, 4, deriv=1, delta=1, mode='mirror')
deriv_on_5hr = savgol_filter(visits_5hr, 3, 1, deriv=1, delta=1, mode='mirror')
#hours[2:-2], visits_5hr


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

axes[0].plot(hours, dumb_deriv, color='k', alpha=0.6, label='Simple derivative')
axes[0].plot(hours, SGvisits_deriv, color='r', label='SG derivative')
axes[0].plot(hours[2:-2], deriv_on_5hr, color='b', alpha=0.6, label='Derivative on 5hr avg')
axes[0].legend(loc='lower right', fontsize=14)
axes[0].set_title('Data', fontsize=20)
axes[0].tick_params(labelsize=14)

axes[1].plot(hours, deriv_on_nonoise, color='k', alpha=0.6, label='True derivative')
axes[1].plot(hours, SGvisits_deriv, color='r', label='SG derivative')
axes[1].plot(hours[2:-2], deriv_on_5hr, color='b', alpha=0.6, label='Derivative on 5hr avg')
axes[1].legend(loc='upper right', fontsize=14)
axes[1].set_title('Versus True result', fontsize=20)
axes[1].tick_params(labelsize=14)

from scipy.fftpack import fft, fftfreq, fftshift
from scipy.signal import freqz
from ipywidgets import interact

def Fourier_SG_coefs(M, N):
    # Just so we don't get crazy N's
    if N > 2*M:
        N = 2*M
    SG_coefs = savgol_coeffs(2*M+1, N)
    w, h = freqz(SG_coefs)
    plt.figure(figsize=(8,8))
    plt.plot(w, 20*np.log10(abs(h)), color='k')
    plt.ylabel('Amplitude [dB]', fontsize=20)
    plt.xlabel('Freq [rad/sample]', fontsize=20)
    plt.tick_params(labelsize=14)
    plt.title('SG M = %d, N = %d' % (M, N), fontsize=20)

interact(Fourier_SG_coefs, M=(1, 25, 1), N=(1, 25, 1))

from IPython.display import Image
Image('SG_Fourier.png')
# Image from http://www-inst.eecs.berkeley.edu/~ee123/fa11/docs/SGFilter.pdf



