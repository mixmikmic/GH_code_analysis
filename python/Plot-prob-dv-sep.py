from os import path

# Third-party
from astropy.table import Table
from astropy.table import vstack
import astropy.coordinates as coord
import astropy.units as u
from astropy.constants import G, c
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('apw-notebook')
get_ipython().magic('matplotlib inline')
from scipy.stats import scoreatpercentile

from comoving_rv.log import logger
from comoving_rv.db import Session, Base, db_connect
from comoving_rv.db.model import (Run, Observation, TGASSource, SimbadInfo, PriorRV,
                                  SpectralLineInfo, SpectralLineMeasurement, RVMeasurement,
                                  GroupToObservations)

import matplotlib as mpl
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

all_color = '#888888'
comoving_color = '#000000'
alpha = 0.5
sky_sep_label = r'tangential sep., $s_{\rm tan}$ [pc]'

apw_color = '#045a8d'
rave_color = '#ef8a62'

# base_path = '/Volumes/ProjectData/gaia-comoving-followup/'
base_path = '../../data/'
db_path = path.join(base_path, 'db.sqlite')
engine = db_connect(db_path)
session = Session()

tbl = Table.read('group_prob_dv.ecsv', format='ascii.ecsv')
rave_tbl = Table.read('group_prob_dv_rave.ecsv', format='ascii.ecsv')
all_tbl = vstack((tbl, rave_tbl))

# HACK:
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

for row in all_tbl:
    axes[0].plot([row['chord_length'], row['chord_length']],
                 [row['dv_15'], row['dv_85']], marker='', linestyle='-',
                 alpha=0.15, linewidth=1.5, color='#333333')

    axes[0].scatter(all_tbl['chord_length'], all_tbl['dv_50'], marker='o', 
                    s=3, color='#333333', alpha=0.4)
    
    axes[1].plot([row['sep_3d'], row['sep_3d']],
                 [row['dv_15'], row['dv_85']], marker='', linestyle='-',
                 alpha=0.15, linewidth=1.5, color='#333333')

    axes[1].scatter(all_tbl['sep_3d'], all_tbl['dv_50'], marker='o', 
                    s=3, color='#333333', alpha=0.4)

axes[0].set_xscale('log')
axes[0].set_xlim(1E-2, 10)
axes[0].set_ylim(-5, 75)

axes[0].set_ylabel(r'$|\boldsymbol{v}_1 - \boldsymbol{v}_2|$ ' + 
                   '[{0}]'.format((u.km/u.s).to_string('latex_inline')))

axes[0].set_xlabel('projected sep. [pc]')
axes[1].set_xlabel('3D sep. [pc]')

# ---

fig, ax = plt.subplots(1, 1, figsize=(6, 5))

for row in all_tbl:
    ax.plot([row['sep_2d'], row['sep_2d']],
            [row['dv_15'], row['dv_85']], marker='', linestyle='-',
            alpha=0.15, linewidth=1.5, color='#333333')

    ax.scatter(all_tbl['sep_2d'], all_tbl['dv_50'], marker='o', 
               s=3, color='#333333', alpha=0.4)


ax.set_xscale('log')
ax.set_xlim(1E-2, 10)
ax.set_ylim(-5, 75)

ax.set_ylabel(r'$|\boldsymbol{v}_1 - \boldsymbol{v}_2|$ ' + 
                   '[{0}]'.format((u.km/u.s).to_string('latex_inline')))

ax.set_xlabel('ang. sep. [deg]')

shaya = Table.read('../../data/shaya_olling2011.fit')

chord_length = []
dist = []
for shaya_id in np.unique(shaya['Seq']):
    rows = shaya[shaya['Seq'] == shaya_id]
    if len(rows) != 2:
        continue
        
    if rows['Prob'][-1] < 0.5:
        continue
    
    icrs1 = coord.ICRS(ra=rows[0]['_RAJ2000']*u.deg,
                       dec=rows[0]['_DEJ2000']*u.deg)
    icrs2 = coord.ICRS(ra=rows[1]['_RAJ2000']*u.deg,
                       dec=rows[1]['_DEJ2000']*u.deg)
    sep_2d = icrs1.separation(icrs2)
    R = np.mean(rows['Dist'])
    
    dist.append(R)
    chord_length.append((2*R*np.sin(sep_2d/2.)).value)
    
chord_length = u.Quantity(chord_length*u.pc)
dist = u.Quantity(dist*u.pc)

shaya_tbl = Table({'chord_length': chord_length, 'd_min': dist})
len(shaya_tbl)

comoving = tbl['prob'] > 0.5
rave_comoving = rave_tbl['prob'] > 0.5
comoving_all = all_tbl['prob'] > 0.5

print('apw: {0} are comoving of {1} ({2:.0%})'.format(comoving.sum(), len(tbl), 
                                                      comoving.sum()/len(tbl)))
print('RAVE: {0} are comoving of {1} ({2:.0%})'.format(rave_comoving.sum(), len(rave_tbl), 
                                                       rave_comoving.sum()/len(rave_tbl)))
print('all: {0} are comoving of {1} ({2:.0%})'.format(comoving_all.sum(), len(all_tbl),
                                                      comoving_all.sum()/len(all_tbl)))

apw_f_samples = np.ravel(np.load('../../data/sampler_chain_apw.npy')[:,100::4,0])
rave_f_samples = np.ravel(np.load('../../data/sampler_chain_rave.npy')[:,100::4,0])

fig, ax = plt.subplots(1, 1, figsize=(5,5))

bins = np.linspace(0, 1, 55)
ax.hist(apw_f_samples, bins=bins, alpha=0.75,
        normed=True, color=apw_color, label='this work')
ax.hist(rave_f_samples, bins=bins, alpha=0.75,
        normed=True, color=rave_color, label='RAVE');

ax.legend(title='RV source:')

ax.set_xlabel('$f$')
ax.set_ylabel(r'$p(f \,|\, {\rm data})$')

ax.set_xlim(0, 1)
fig.savefig('f_samples.pdf')

cmap = plt.get_cmap('coolwarm_r')

for title, name, _tbl in zip(['RV source: this work', 'RV source: RAVE'], 
                             ['apw', 'rave'],
                             [tbl, rave_tbl]):
    
    fig,axes = plt.subplots(1, 2, figsize=(12,5), sharey=True)

    for ax in axes:
        for row in _tbl:
            color = cmap(row['prob'])
            ax.plot([row['chord_length'], row['chord_length']],
                    [row['dv_15'], row['dv_85']], marker='', linestyle='-',
                    color=color, alpha=0.5, linewidth=1.5)

        ax.scatter(_tbl['chord_length'], _tbl['dv_50'], marker='o', 
                   s=3, color='#333333', alpha=0.9)

        ax.set_ylim(-5, 75)

        ax.set_xlabel(sky_sep_label)

    axes[0].axhline(0, color='#000000', zorder=-100, alpha=0.15)
    axes[1].axhline(0, color='#000000', zorder=-100, alpha=0.15)
        
    axes[0].set_xscale('log')

    axes[0].text(2E-1, 67, r'$<1\,{\rm pc}$', fontsize=20)
    axes[1].text(7.4, 67, r'$1$--$10\,{\rm pc}$', fontsize=20)

    axes[0].set_xlim(1E-3, 1)
    axes[1].set_xlim(0, 10)

    axes[0].set_ylabel(r'$|\boldsymbol{v}_1 - \boldsymbol{v}_2|$ ' + 
                       '[{0}]'.format((u.km/u.s).to_string('latex_inline')))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize())
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes.ravel().tolist())
    cb.set_label('comoving prob.')
    
    fig.suptitle(title, fontsize=24)

    fig.savefig('dx-dv-{0}.pdf'.format(name))

for row in tbl[(tbl['prob'] < 0.1) & (tbl['chord_length'] < 1) & (tbl['dv_50'] > 20)]:
    gid = row['group_id']
    for obs in session.query(Observation).filter(Observation.group_id == gid).all():
        print(obs.simbad_info)
        print(obs.rv_measurement.rv, obs.rv_measurement.err)
    print('---')

fig, axes = plt.subplots(1, 2, figsize=(10,4.5))

sep2d_bins = np.logspace(-3., 1, 13)
sep2d_bins_lin = np.linspace(0., 10, 10)

_ = axes[0].hist(tbl['chord_length'], bins=sep2d_bins, 
                 color=all_color, alpha=alpha)
axes[0].hist(tbl['chord_length'][comoving], bins=sep2d_bins, 
             color=comoving_color, alpha=alpha)
axes[0].set_xlabel(sky_sep_label)

axes[0].set_xlim(1e-3, 1e1)
axes[0].set_xscale('log')
axes[0].set_ylim(1, 220)
axes[0].set_yscale('log')

axes[1].hist(tbl['chord_length'], bins=sep2d_bins_lin, 
             color=all_color, alpha=alpha)
axes[1].hist(tbl['chord_length'][comoving], bins=sep2d_bins_lin, 
            color=comoving_color, alpha=alpha)
axes[1].set_xlabel(sky_sep_label)

axes[1].set_xlim(axes[0].get_xlim())
# axes[1].set_xscale('log')
axes[1].set_ylim(0, 60)
# axes[1].set_yscale('log')

fig.tight_layout()

fig.suptitle('Observed comoving pairs', fontsize=20)
fig.subplots_adjust(top=0.9)

# fig.savefig('separation-hist.pdf')

# Weighted histograms instead
fig, axes = plt.subplots(1, 2, figsize=(10,4.5))

sep2d_bins = np.logspace(-3., 1, 13)
sep2d_bins_lin = np.linspace(0., 10, 13)

_ = axes[0].hist(all_tbl['chord_length'], bins=sep2d_bins, 
                 weights=all_tbl['prob'], color=comoving_color, alpha=alpha)
axes[0].set_xlabel(sky_sep_label)

axes[0].set_xlim(1e-3, 1e1)
axes[0].set_xscale('log')
axes[0].set_ylim(1, 220)
axes[0].set_yscale('log')

axes[1].hist(all_tbl['chord_length'], bins=sep2d_bins_lin, 
             weights=all_tbl['prob'], color=comoving_color, alpha=alpha)
axes[1].set_xlabel(sky_sep_label)

axes[1].set_xlim(axes[0].get_xlim())
# axes[1].set_xscale('log')
axes[1].set_ylim(0, 60)
# axes[1].set_yscale('log')

fig.tight_layout()

fig.suptitle('Weighted by probability', fontsize=20)
fig.subplots_adjust(top=0.9)

fig.savefig('separation-hist.pdf')

mask = ((all_tbl['prob'] > 0.5) & 
        (all_tbl['sep_3d'].to(u.pc) < 10*u.pc) &
        (all_tbl['d_min'].to(u.pc) < (200.*u.pc)))
print('Total number of confirmed pairs within 200 pc:', mask.sum())

plt.hist(all_tbl['sep_2d'][mask], bins=np.logspace(-3, 1, 13),
         color=comoving_color, alpha=alpha)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta \theta$ [deg]')

# chord_length = []
# dist = []
# for shaya_id in np.unique(shaya['Seq']):
#     rows = shaya[shaya['Seq'] == shaya_id]
#     if len(rows) != 2:
#         continue
        
#     if rows['Prob'][-1] < 0.5:
#         continue
    
#     icrs1 = coord.ICRS(ra=rows[0]['_RAJ2000']*u.deg,
#                        dec=rows[0]['_DEJ2000']*u.deg)
#     icrs2 = coord.ICRS(ra=rows[1]['_RAJ2000']*u.deg,
#                        dec=rows[1]['_DEJ2000']*u.deg)
#     sep_2d = icrs1.separation(icrs2)
#     R = np.mean(rows['Dist'])
    
#     dist.append(R)
#     chord_length.append((2*R*np.sin(sep_2d/2.)).value)
    
# chord_length = u.Quantity(chord_length*u.pc)
# dist = u.Quantity(dist*u.pc)

# shaya_tbl = Table({'chord_length': chord_length, 'd_min': dist})

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import LogLocator, AutoMinorLocator

len(tbl[comoving])+len(rave_tbl[rave_comoving])

fig, ax = plt.subplots(figsize=(8, 8))

divider = make_axes_locatable(ax)
ax_hist_x = divider.append_axes("top", 1.4, pad=0.4, sharex=ax)
ax_hist_y = divider.append_axes("right", 1.4, pad=0.4, sharey=ax)

ax.scatter(tbl['chord_length'][comoving], tbl['d_min'][comoving],
           marker='o', s=10, color=apw_color, label='this work')
ax.scatter(rave_tbl['chord_length'][rave_comoving], rave_tbl['d_min'][rave_comoving],
           marker='s', s=10, color=rave_color, label='RAVE')
ax.set_xscale('log')

xbins = np.logspace(-3, 1, 10)
ybins = np.linspace(0, 200, 10)
ax_hist_x.hist(all_tbl['chord_length'][mask], color=comoving_color, 
               alpha=alpha, bins=xbins)
ax_hist_y.hist(all_tbl['d_min'][mask], bins=ybins, 
               color=comoving_color, alpha=alpha, orientation='horizontal')

ax.legend(loc='lower left', fontsize=16)

ax.set_xlim(2e-3, 1E1)
ax_hist_x.set_xlim(ax.get_xlim())
ax_hist_x.set_yscale('log')
ax_hist_x.set_ylim(3, 2E2)
ax_hist_x.set_yticks([10, 100])
ax_hist_x.yaxis.set_ticks(list(np.arange(2, 10)) + list(np.arange(2, 10)*10), minor=True)
ax_hist_x.set_ylabel('$N$')

ax.set_ylim(0, 200)
ax_hist_y.set_ylim(ax.get_ylim())
ax_hist_y.set_xlim(0, 40)
ax_hist_y.set_xticks([0, 10, 20, 30, 40])
ax_hist_y.set_xlabel('$N$')

# make some labels invisible
plt.setp(ax_hist_x.get_xticklabels() + ax_hist_y.get_yticklabels(),
         visible=False)

ax.set_xlabel(sky_sep_label)
ax.set_ylabel(r'mean distance, $\bar{d}$ [pc]')

fig.savefig('separation-with-rave.pdf')

# fig, ax = plt.subplots(figsize=(8, 8))

# divider = make_axes_locatable(ax)
# ax_hist_x = divider.append_axes("top", 1.4, pad=0.4, sharex=ax)
# ax_hist_y = divider.append_axes("right", 1.4, pad=0.4, sharey=ax)

# ax.scatter(tbl['chord_length'], tbl['d_min'],
#            marker='o', s=10, color='#67a9cf', label='this work', alpha=0.7)
# ax.scatter(rave_tbl['chord_length'], rave_tbl['d_min'],
#            marker='s', s=10, color='#ef8a62', label='RAVE', alpha=0.7)
# ax.scatter(shaya_tbl['chord_length'], shaya_tbl['d_min'],
#            marker='^', s=6, color='#31a354', label='SO11', alpha=0.7)
# ax.set_xscale('log')

# xbins = np.logspace(-3, 1, 21)
# ybins = np.linspace(0, 100, 21)
# ax_hist_x.hist(np.concatenate((all_tbl['chord_length'][mask], shaya_tbl['chord_length'])), 
#                color=comoving_color, alpha=alpha, bins=xbins)
# ax_hist_y.hist(np.concatenate((all_tbl['d_min'][mask], shaya_tbl['d_min'])), bins=ybins, 
#                color=comoving_color, alpha=alpha, orientation='horizontal')

# ax.legend(loc='lower left', fontsize=12, ncol=3)

# ax.set_xlim(2e-3, 1E1)
# ax_hist_x.set_xlim(ax.get_xlim())
# ax_hist_x.set_yscale('log')
# ax_hist_x.set_ylim(8E-1, 1.5E2)
# ax_hist_x.set_yticks([1, 10, 100])
# ax_hist_x.yaxis.set_ticks(list(np.arange(2, 10)) + list(np.arange(2, 10)*10), minor=True)
# ax_hist_x.set_ylabel('$N$')

# ax.set_ylim(0, 100)
# ax_hist_y.set_ylim(ax.get_ylim())
# ax_hist_y.set_xlim(0, 50)
# ax_hist_y.set_xticks([0, 50, 100, 150])
# ax_hist_y.set_xlabel('$N$')

# # make some labels invisible
# plt.setp(ax_hist_x.get_xticklabels() + ax_hist_y.get_yticklabels(),
#          visible=False)

# ax.set_xlabel(r'chord length, $\hat{s}$ [pc]')
# ax.set_ylabel(r'mean distance, $\bar{d}$ [pc]')

# # fig.savefig('separation-with-shaya.pdf')



