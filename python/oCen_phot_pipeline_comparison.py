get_ipython().magic('matplotlib inline')

#import notebook
#from notebook.nbextensions import enable_nbextension
#enable_nbextension('notebook', 'usability/codefolding/main')
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd
from scipy.optimize import curve_fit
import seaborn as sns
import markdown
#import scipy.stats as stats
import sys
sys.path.append('/Users/vs/Dropbox/Python')
import itertools
import glob
import re
import os
from astropy.stats import sigma_clip
import scipy


#import reddening_laws as red
bigfontsize=20
labelfontsize=16
tickfontsize=16
sns.set_context('talk')
mp.rcParams.update({'font.size': bigfontsize,
                     'axes.labelsize':labelfontsize,
                     'xtick.labelsize':tickfontsize,
                     'ytick.labelsize':tickfontsize,
                     'legend.fontsize':tickfontsize,
                     })

comparison_data = pd.read_csv('/Users/vs/Dropbox/Publications/omegaCen/phot_comparison.raw', skiprows=3, header=None, names = ['ID', 'xc', 'yc', 'mag_new', 'err_new', 'mag_old', 'err_old', 'chi', 'sharp'], delim_whitespace=True)

comparison_data

comparison_data['mag_diff'] = comparison_data['mag_new'] - comparison_data['mag_old']
comparison_data['err_diff'] = np.sqrt((comparison_data['err_new'])**2 + (comparison_data['err_old'])**2)

clipped['mag_new'<15]

## Calculate average offset between the two reductions

## Sigma clipping the sample to get a representative value - threshold is 3 sigma
## Also compare to average of every matched star to see if it makes much difference

## Only using mag_new<15 in this calculation


clipped = sigma_clip(comparison_data.mag_diff[comparison_data.mag_new<14], sigma=3)
av_diff_clipped = np.ma.mean(clipped)
sdev_diff_clipped = np.ma.std(clipped)

av_diff_14mag = np.ma.mean(comparison_data.mag_diff[comparison_data.mag_new<14])
sdev_diff_14mag = np.ma.std(comparison_data.mag_diff[comparison_data.mag_new<14])

av_diff_whole = np.ma.mean(comparison_data.mag_diff)
sdev_diff_whole = np.ma.std(comparison_data.mag_diff)

print av_diff_clipped, sdev_diff_clipped
print av_diff_14mag, sdev_diff_14mag
print av_diff_whole, sdev_diff_whole

sns.set(style="darkgrid")
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})
sns.set_palette("Set2", 2)
palette = itertools.cycle(sns.color_palette())


mp.plot(comparison_data.mag_new, comparison_data.mag_diff, 'o', ls='None', ms=2, color='Grey', label='All stars')
mp.plot(comparison_data.mag_new[comparison_data.mag_new<14], comparison_data.mag_diff[comparison_data.mag_new<14], 'o', ls='None', ms=2, color=next(palette), label='Mag < 14')
mp.plot(comparison_data.mag_new[comparison_data.mag_new<14], clipped, 'o', ls='None', ms=2, color=next(palette), label='Clipped sample')

mp.xlim(8, 19)
mp.ylim(-0.75, 0.75)
mp.xlabel('[3.6] (S19.2 Reduction)')
mp.ylabel('New mag - Old mag')
mp.suptitle('Comparison of new and old mean magnitudes for omega Cen Field 1')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5), markerscale=3)

## Is there a trend with mag?

bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(comparison_data.mag_new, comparison_data.mag_diff, bins=100)

bin_width = (bin_edges[1] - bin_edges[0])
bin_centers = bin_edges[1:] - bin_width/2

bin_means_c, bin_edges_c, binnumber_c = scipy.stats.binned_statistic(comparison_data.mag_new[comparison_data.mag_new<14], clipped, bins=20)

bin_width_c = (bin_edges_c[1] - bin_edges_c[0])
bin_centers_c = bin_edges_c[1:] - bin_width_c/2

def fit_a_line(x, slope, zp):
    return slope*x + zp

#popt, pcov = curve_fit(fit_a_line, bin_centers,  bin_means)

#slope = popt[0]
#zp = popt[1]
#eslope = np.sqrt(float(pcov[0][0]))
#ezp = np.sqrt(float(pcov[1][1]))
#x1 = np.arange(7,16,1)

mp.plot(comparison_data.mag_new, comparison_data.mag_diff, 'o', ls='None', ms=2, color='Grey', label='All stars')
mp.plot(comparison_data.mag_new[comparison_data.mag_new<14], comparison_data.mag_diff[comparison_data.mag_new<14], 'o', ls='None', ms=2, color=next(palette), label='Mag < 14')
mp.plot(comparison_data.mag_new[comparison_data.mag_new<14], clipped, 'o', ls='None', ms=2, color=next(palette), label='Clipped sample')
mp.plot(bin_centers, bin_means, 'o', ls='-',color='Grey', lw=2)
next(palette)
mp.plot(bin_centers_c, bin_means_c, 'o', ls='-',color=next(palette), lw=2)

#mp.plot(x1, slope*x1 + zp, ls='-', color='Grey')
#mp.plot(x1, (slope+2*eslope)*x1 + (zp-2*ezp), ls='-.', color='Grey')
#mp.plot(x1, (slope-2*eslope)*x1 + (zp+2*ezp), ls='-.', color='Grey')



mp.xlim(8, 19)
mp.ylim(-0.75, 0.75)
mp.xlabel('[3.6] (S19.2 Reduction)')
mp.ylabel('New mag - Old mag')
mp.suptitle('Is the offset magnitude dependent?')
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5), markerscale=3)

sns.distplot(comparison_data.mag_new, label='Full Sample', kde=False)
sns.distplot(comparison_data.mag_new[comparison_data.mag_new<14], label='[3.6]<14', kde=False)
mp.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))

len(comparison_data.err_diff[comparison_data.mag_new<15])

len(clipped)

len(comparison_data.mag_new[comparison_data.mag_new<15]), len(comparison_data.mag_diff[comparison_data.mag_new<15]), len(comparison_data.err_diff[comparison_data.mag_new<15])

comparison_data.err_diff[comparison_data.mag_new<15]



