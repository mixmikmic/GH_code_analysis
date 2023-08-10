#!/usr/bin/env python
"""choose_skies.ipynb: a notebook to look at detected GW skies """

__author__ = "Chiara M. F. Mingarelli"
__copyright__ = "Copyright 2017, GWASAP project"
__credits__ = ["Chiara Mingarelli"]
__license__ = "GPL"
__version__ = "0.0.1"
__maintainer__ = "Chiara Mingarelli"
__email__ = "mingarelli@gmail.com"
__status__ = "prototype"

from __future__ import division
from IPython.display import display, Math, Latex
import math
from math import sqrt, cos, sin, pi
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interp1d
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.mlab as ml
from matplotlib.ticker import FormatStrFormatter, LinearLocator, NullFormatter, NullLocator, MultipleLocator
import matplotlib.ticker
import matplotlib.colors
from matplotlib.font_manager import FontProperties
from matplotlib import rc, text
import healpy as hp
import plot
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import collections

get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')

det_sky = np.genfromtxt("../detectedSkies/detected_skies.txt")

def find_nearest(array,value):
    #returns index of matching value in array
    idx = (np.abs(array-value)).argmin()
    return idx 

def find_gal_idx(nameList, name):
    ans = [i for i, x in enumerate(nameList) if x == name]
    return ans

def find_gal_dec_idx(nameList, dec):
    ans = [i for i, x in enumerate(nameList) if x >= dec]
    return ans

def find_gal_dec_idx_neg(nameList, dec):
    ans = [i for i, x in enumerate(nameList) if x <= dec]
    return ans

det_DEC = det_sky[:,1]*180/pi

det_name = np.genfromtxt("../detectedSkies/detected_skies.txt", usecols = (6), dtype='S13')
det_file = np.genfromtxt("../detectedSkies/detected_skies.txt", usecols = (4), dtype=None)

det_name = det_name.tolist()

len(det_name)

rm_dec_pos = find_gal_dec_idx(det_DEC, 70.0)
rm_dec_neg = find_gal_dec_idx_neg(det_DEC, -70.0)

# concatenate ±70 deg in declination galaxy indices
all_bad_idx = rm_dec_neg + rm_dec_pos 

new_det_sky = np.delete(det_sky, all_bad_idx ,0)

new_det_sky_name = np.delete(det_name, all_bad_idx,0)
new_det_sky_file= np.delete(det_file, all_bad_idx,0)

new_det_sky.shape

# Parameters of detected skies once the list has had the bad galaxies removed
det_RA = new_det_sky[:,0]*180/pi
det_DEC = new_det_sky[:,1]*180/pi
det_freq = new_det_sky[:,2]
det_strain = new_det_sky[:,3]
mchirp_rec = new_det_sky[:,5]
q_rec = new_det_sky[:,6]
det_name = new_det_sky_name
det_dist = new_det_sky[:,8]
mstar_list = new_det_sky[:,9]
save_p= new_det_sky[:,10]
det_file = new_det_sky_file

# load pulsar positions on the sky

p_pos = np.genfromtxt("pulsar_positions.txt", skip_header=0, usecols = (1,2) )
p_RA = p_pos[:,0]
p_DEC = pi/2-p_pos[:,1]
p_name = np.genfromtxt("pulsar_positions.txt", skip_header=0, usecols = (0), dtype = 'S13' )

Mp_pos = np.genfromtxt("more_pulsars.txt", usecols = (1,2) )
Mp_RA = Mp_pos[:,0]
Mp_DEC = Mp_pos[:,1]
Mp_name = np.genfromtxt("more_pulsars.txt", usecols = (0), dtype = 'S13' )

range(1)

# Make maps of all the detected sources

# To plot a single sky here, set up for multiple detected skies 
det_freq = []
det_freq.append(3.78653726523e-09)
#for ii in range(len(det_RA)):
for ii in range(1):
    scat_sky=np.genfromtxt("../scatter_maps/scatterData_freq_"+str(det_freq[ii])+"Hz.dat")
    scat_ra = scat_sky[:,0]
    scat_dec = scat_sky[:,1]
    scat_strain = np.log10(scat_sky[:,2])
    #look_up_freq = det_freq[ii]
    ax = plt.subplot(111, projection='astro mollweide')
    colors = scat_strain
    sky=plt.scatter(scat_ra, scat_dec, c = colors, edgecolors='none', cmap ='viridis_r',  rasterized=True)
    #plt.suptitle(str(det_name[ii])+", GW sky at $f=$ "+str('%.2e' %det_freq[ii])+" Hz, $h=$"+str('%.2e'%det_strain[ii]),\
    #             y=0.3)
    plt.suptitle(r"$\log_{10}(h)$, GW sky at $f=$ "+str('%.2e' %det_freq[ii])+" Hz", y=0.3)
    plt.scatter(Mp_RA*pi/180, Mp_DEC*pi/180, color ='white', marker = "*", alpha=1)
    plt.scatter(p_RA, p_DEC, marker = "*", color = '#ff7f0e', s = 100)
    plt.colorbar(sky, orientation = 'horizontal')
    ax.grid(linewidth=0.5)
    #ax.set_axisbelow(True)
    #plt.savefig("../detectedSkies/hercules_skies/log_detected_sky_"+str(ii)+"_"+str(det_name[ii])+"_wPulsars.pdf", dpi=400)
    #plt.savefig("../detectedSkies/hercules_skies/log_detected_sky_"+str(ii)+"_wPulsars.pdf", dpi=400)
    #plt.clf()
    #plt.savefig("allEPTA_pulsars.pdf", dpi=400)

# indices of skies which failed 2nd inspection, you need to put these in by hand
sec_insp_idx = [0,2,4,6,8, 9,10,16,19,21, 28,33,34,41, 44,49,50,53,56,57,61,65,66,70,72,76,80,89,93,108,109,114,               120, 131,134,139,140,142,143,144,147,148,151,159,164,166,175,]

final_sky = np.delete(new_det_sky, sec_insp_idx,0)
final_sky_name = np.delete(new_det_sky_name, sec_insp_idx,0)
final_sky_file = np.delete(new_det_sky_file, sec_insp_idx,0)

final_sky_file

# final galaxy list and parameters

det_RA = final_sky[:,0]*180/pi
det_DEC = final_sky[:,1]*180/pi
det_freq = final_sky[:,2]
det_strain = final_sky[:,3]
mchirp_rec = final_sky[:,5]
q_rec = final_sky[:,6]
det_name = final_sky_name
det_dist = final_sky[:,7]
mstar_list = final_sky[:,8]
save_p= final_sky[:,9]
det_file = final_sky_file

def hist_weight(lst):
    return np.ones_like(lst)/len(lst)

def auto_bin(bin_size, min_edge, max_edge):
    #bin_size = 0.05; min_edge = -3.5; max_edge = -1
    N = (max_edge-min_edge)/bin_size; Nplus1 = N + 1
    bin_list = np.linspace(min_edge, max_edge, Nplus1)
    return bin_list

# What is the distribution of frequencies where sources were detected?
#bin_size = 0.1; min_edge = -9.0; max_edge = -8
bin_list = auto_bin(0.1,-9,-8)
weights = hist_weight(np.log10(det_freq))
plt.hist(np.log10(det_freq),bin_list, align='left',  alpha=0.8, edgecolor = 'black', weights = weights)
plt.xlabel(r"Distribution of $\log_{10}(f_\mathrm{GW})$ detected (Hz)")
plt.ylabel("Probability")
plt.legend(frameon=False)
plt.tight_layout()
#plt.savefig("det_gw_freqs_3Mar17.pdf")

det_dist = det_dist.tolist()

bin_list = auto_bin(10,10,220)
weights = hist_weight(det_dist)
ax = fig.add_subplot(111)
plt.hist(det_dist, bin_list, weights=weights, alpha = 0.8, align='left', edgecolor = 'black')
majorLocator = MultipleLocator(25)
minorLocator = MultipleLocator(5)
ax.xaxis.set_minor_locator(minorLocator)
ax.xaxis.set_major_locator(majorLocator)
plt.xlabel("Distance to Galaxy (Mpc)")
plt.ylabel("Probability")
plt.legend(frameon=False)
plt.tight_layout()
#plt.savefig("distance_to_detected_gals_3Mar17_10bins.pdf")

catalog_list = gal_name

len(catalog_list)

histo_data = dict((x, catalog_list.count(x)) for x in catalog_list)

d= histo_data

X = np.arange(len(d))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.barh(X, d.values(), alpha=0.5, linewidth = 0.5)
ax.set_yticks(X)
#ax.set_yticklabels(large_label)
plt.setp(ax.get_yticklabels(), fontsize=5)
ax.set_yticklabels(d.keys())
majorLocator = MultipleLocator(1)
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
ax.xaxis.set_major_locator(majorLocator)
xmax = max(d.values()) + 1
plt.xlim(0, xmax)
plt.xlabel("Count")

#plt.savefig("64k_galaxy_count_revised_1Feb17.pdf", bbox_inches='tight')
plt.show()

# try using a sorted list

histo_data = sorted(histo_data.items(), key=lambda x:x[1]) # sort histogram by count

value = np.zeros([len(histo_data)])
label = []

for j in range(len(histo_data)):
    value[j] = histo_data[j][1]
    label.append(histo_data[j][0])

X = np.arange(len(value))
fig = plt.figure()
ax = fig.add_subplot(111)
ax.barh(X, value, alpha=0.5, linewidth = 0.4)
ax.set_yticks(X)
plt.setp(ax.get_yticklabels(), fontsize=1.5)
ax.set_yticklabels(label)
majorLocator = MultipleLocator(1)
minorLocator = MultipleLocator(1)
ax.xaxis.set_minor_locator(minorLocator)
ax.xaxis.set_major_locator(majorLocator)
#plt.xticks(np.arange(0,4, 1))

#plt.savefig("sorted_75k_galaxies_3Mar17.pdf")

# Make LaTex tables for paper by adding  & and \\
det_nameR = []
det_distR = []
mstar_listR = []
mchirp_recR = []
q_recR = []
det_freqR = []
det_strainR = []

# final galaxy list and parameters

det_RA = final_sky[:,0]*180/pi
det_DEC = final_sky[:,1]*180/pi
det_freq = final_sky[:,2]
det_strain = final_sky[:,3]
mchirp_rec = final_sky[:,5]
q_rec = final_sky[:,6]
det_name = final_sky_name
det_dist = final_sky[:,8]
mstar_list = final_sky[:,9]
save_p= final_sky[:,10]
det_file = final_sky_file

# 34 detected galaxies; add "&" and slashes for latex table 
for ii in range(34):
    det_nameR.append(str(det_name[ii])+str(" & "))
    det_distR.append(str(det_dist[ii])+str(" & "))
    mstar_listR.append(str("%.2f" %mstar_list[ii])+str(" & "))
    mchirp_recR.append(str('%.2e'%mchirp_rec[ii])+str(" & "))
    q_recR.append(str("%.2f" %q_rec[ii])+str(" & "))
    det_freqR.append(str("%.2e" %det_freq[ii])+str(" & "))
    det_strainR.append(str("%.2e" %det_strain[ii])+str(" \\\ "))

gal_name = []

for ii in range(len(final_sky_file)):
    sky_q = np.genfromtxt("../gw_skies/"+str(final_sky_file[ii]), usecols = 5)
    #for qdet in sky_q:
    index = find_nearest(sky_q, q_rec[ii])
    print sky_q[index]
    gal_name.append(np.genfromtxt("../gw_skies/"+str(final_sky_file[ii]), usecols = 6, dtype='str')[index])

# Save final sky

#dest_file = "../detectedSkies/hercules_skies/herc_final_detected_skies.txt"    
result_file = open(dest_file, "a+") # the a+ allows you to create the file and write to it.

#for N, D, E, M, Q, F, H, FN in zip (det_nameR, det_distR, mstar_listR, mchirp_recR, q_recR, det_freqR, det_strainR, final_sky_file):
for N, D, E, M, Q, F, H, FN in zip (gal_name, det_dist, mstar_list, mchirp_rec, q_rec, det_freq, det_strain, final_sky_file):
    result_file.write('{0} {1} {2} {3} {4} {5} {6} {7}\n'.format(N, D, E, M, Q, F, H, FN))
result_file.close()

fig_width_pt = 245.27 #513.17           # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0      # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height =fig_width*golden_mean       # height in inches
fig_size = [fig_width,fig_height]

params = {'backend': 'pdf',
        'axes.labelsize': 7,
        'lines.markersize': 4,
        'font.size': 7,
        'xtick.major.size':6,
        'xtick.minor.size':3,  
        'ytick.major.size':6,
        'ytick.minor.size':3, 
        'xtick.major.width':0.5,
        'ytick.major.width':0.5,
        'xtick.minor.width':0.5,
        'ytick.minor.width':0.5,
        'lines.markeredgewidth':1,
        'axes.linewidth':1.2,
        'legend.fontsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'savefig.dpi':200,
        'path.simplify':True,
        'font.family': 'serif',
        'font.serif':'Times',
        'text.latex.preamble': [r'\usepackage{amsmath}'],
        'text.usetex':True,
        #'axes.color_cycle': ['b', 'lime', 'r', 'purple', 'g', 'c', 'm', 'orange', 'darkblue', \
        #                        'darkcyan', 'y','orangered','chartreuse','brown','deeppink','lightgreen', 'k'],
        #'font.serif':cm,
        'figure.figsize': fig_size}
plt.rcParams.update(params)



