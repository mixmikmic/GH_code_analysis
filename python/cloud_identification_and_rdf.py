# Import modules and load fields
import sys
sys.path.append('../python_scripts')   # So that we can later use the functions from helpers.py
import matplotlib
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
# from cosmo_utils.pyncdf import getfobj_ncdf   # This is a module developed specifically for COSMO output
import urllib

# Pick 6 examples
alims = (50, -51, 50, -51)   # Limits for analysis area
dates = ['2016052800', '2016052800', '2016060300', 
         '2016060300', '2016053000', '2016052800']
times = [15, 20, 13, 21, 6, 9]
tags = ['a) Scattered', 'b) Clustered', 'c) Scattered', 
        'd) Clustered', 'e) Single large cloud', 'f) Hardly any clouds']

# Pick variable [prec or w]
var = 'prec'

# Plot the fields
fig, axmat = plt.subplots(2, 3, figsize = (13, 10))
cm = plt.cm.cool
cm.set_under(color = 'white')
field_list = []
example_field_url = 'https://github.com/raspstephan/convective_variability_analysis/raw/master/jupyter_notebooks/example_fields/'
savedir = '/home/s/S.Rasp/tmp/'  # Local directory where files will be saved

for d, t, tag, ax in zip(dates, times, tags, np.ravel(axmat)):
    
    if var == 'prec':
        filename = d + '_00' + str(t).zfill(2) + '0000_prec.npy'
        url = example_field_url + filename
        f = open(savedir + filename, 'w')
        f.write(urllib.urlopen(url).read())
        f.close()
        field = np.load(savedir + filename)
        # For my local use
        # fn = ('/project/meteo/scratch-old/users/stephan.rasp/convective_variability_data/raw_data/' + 
        #       d + '/deout_ceu_pspens/det/OUTPUT/lfff00' + str(t).zfill(2) + '0000.nc_30m_surf')
        # field = getfobj_ncdf(fn, 'PREC_ACCUM').data[alims[0]:alims[1], alims[2]:alims[3]]
    elif var == 'w':
        filename = d + '_00' + str(t).zfill(2) + '0000_w.npy'
        url = example_field_url + filename
        f = open(savedir + filename, 'w')
        f.write(urllib.urlopen(url).read())
        f.close()
        field = np.load(savedir + filename)
        # For my local use
        # fn = ('/project/meteo/scratch-old/users/stephan.rasp/convective_variability_data/raw_data/' + 
        #       d + '/deout_ceu_pspens/det/OUTPUT/lfff00' + str(t).zfill(2) + '0000.nc_30m')
        # field = getfobj_ncdf(fn, 'W').data[30, alims[0]:alims[1], alims[2]:alims[3]]   # At model level 30
    else: 
        raise Exception('Wrong variable!')
        
    field_list.append(field)
    im = ax.imshow(field, cmap=cm, origin='lower', vmin=1)
    plt.sca(ax)
    cb = fig.colorbar(im, shrink = 0.6, orientation='horizontal', pad=0.07)
    if var == 'prec':
        cb.set_label('Precipitation [mm/h]')
    else:
        cb.set_label('W [m/s]')
    ax.set_title(tag + ' ' + d + ' ' + str(t).zfill(2) + ' UTC')

plt.tight_layout()

import helpers

threshold = 1.
labels_list = []
fig, axmat = plt.subplots(2, 3, figsize = (13, 10))
cm_prism = plt.cm.prism
cm_prism.set_under(color = 'white')
for field, d, t, tag, ax in zip(field_list, dates, times, tags, np.ravel(axmat)):
    cov_frac = np.sum(field > threshold) / np.float(field.size)
    labels = helpers.identify_clouds(field, threshold, water=False)[0]
    labels_list.append(labels)
    ax.imshow(labels, cmap=cm_prism, origin='lower', vmin = 1)
    ax.set_title(tag + ' ' + d + ' ' + str(t).zfill(2) + ' UTC' + 
                '\n Coverage fraction: {:.2f}%'.format(cov_frac * 100.))
plt.tight_layout()

ex_ind = 3   # Pick one example case

plims = (50, 150, 0, 100)   # Chose which section to zoom into for plot

fig, axmat = plt.subplots(2, 3, figsize=(13, 10))
axmat[0, 0].imshow(field_list[ex_ind][plims[0]:plims[1], plims[2]:plims[3]], cmap=cm, origin='lower', vmin = 1)
axmat[0, 0].set_title('Precipitation')
axmat[0, 1].imshow(labels_list[ex_ind][plims[0]:plims[1], plims[2]:plims[3]], cmap=cm_prism, origin='lower', vmin = 1)
axmat[0, 1].set_title('Unseparated clouds')

for n, ax in zip([[[0,1,0],[1,1,1],[0,1,0]], 3, 5, 11], np.ravel(axmat)[2:]):
    labels_sep = helpers.identify_clouds(field_list[ex_ind], threshold, water=True, neighborhood = n)[0]
    ax.imshow(labels_sep[plims[0]:plims[1], plims[2]:plims[3]], cmap=cm_prism, origin='lower', vmin = 1)
    if not type(n) is int:
        ax.set_title('Separated with cross footprint')
    else:  
        ax.set_title('Separated with footprint {:d}x{:d} grid points'.format(n, n))
plt.tight_layout()

# Do the calculation with a 3x3 footprint for all examples
labels_sep_list = []
for field in field_list:
    labels_sep_list.append(helpers.identify_clouds(field, threshold, water=True, neighborhood=3)[0])

import sys
sys.path.append('../python_scripts')
from helpers import calc_rdf, pair_correlation_2d

r_max = 30   # In grid points
dr = 2       # Also in grid points

fig, axmat = plt.subplots(2, 2, figsize=(14, 10))
colors = ['salmon', 'lightblue', 'darkred', 'darkblue', 'green']   # Only 5 colors because we omit the last example
for labels, labels_sep, field, tag, c in zip(labels_list, labels_sep_list, field_list, tags, colors):
    rdf, radii = calc_rdf(labels, field, normalize=True, dx=2.8e3, r_max=r_max, dr=dr)
    axmat[0, 0].plot(radii / 1000., rdf, label=tag, c=c)
    
    rdf, radii = calc_rdf(labels_sep, field, normalize=True, dx=2.8e3, r_max=r_max, dr=dr)
    axmat[0, 1].plot(radii / 1000., rdf, c=c)
    
    rdf, radii = calc_rdf(labels, field, normalize=False, dx=2.8e3, r_max=r_max, dr=dr)
    axmat[1, 0].plot(radii / 1000., rdf, label=tag, c=c)
    
    rdf, radii = calc_rdf(labels_sep, field, normalize=False, dx=2.8e3, r_max=r_max, dr=dr)
    axmat[1, 1].plot(radii / 1000., rdf, c=c)


axmat[0, 0].set_title('Normalized RDF without separated clouds')
axmat[0, 0].set_xlabel('Radius [km]')
axmat[0, 0].axhline(1, c='r')
axmat[0, 0].legend()

axmat[0, 1].set_title('Normalized RDF with separated clouds')
axmat[0, 1].set_xlabel('Radius [km]')
axmat[0, 1].axhline(1, c='r')

axmat[1, 0].set_title('Non-Normalized RDF without separated clouds')
axmat[1, 0].set_xlabel('Radius [km]')
axmat[1, 0].legend()

axmat[1, 1].set_title('Non-Normalized RDF with separated clouds')
axmat[1, 1].set_xlabel('Radius [km]')

plt.tight_layout()

fig, axmat = plt.subplots(2, 2, figsize=(14, 10))
for n, ax in zip([[[0,1,0],[1,1,1],[0,1,0]], 3, 5, 11], np.ravel(axmat)):
    for field, tag, c in zip(field_list, tags, colors):
        labels_sep = helpers.identify_clouds(field, threshold, water=True, neighborhood = n)[0]
        rdf, radii = calc_rdf(labels_sep, field, normalize=True, dx=2.8e3, r_max=r_max, dr=dr)
        ax.plot(radii / 1000., rdf, label=tag, c=c)
    if not type(n) is int:
        ax.set_title('Cross footprint')
    else:  
        ax.set_title('Footprint: {:d}x{:d} grid points'.format(n, n))
axmat[0, 0].legend()
plt.tight_layout()





