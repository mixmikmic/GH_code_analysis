from __future__ import division, print_function, unicode_literals
get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity
from sklearn.metrics import precision_recall_curve

from utils.metrics import threshold_at_completeness_of, threshold_at_purity_of
from utils.bootstrap import (
    kde_purity,
    kde_completeness,
    get_stellar_fraction,
    confidence_band,
    get_log_density
)

plt.rc('legend', fontsize=10)

df = pd.read_csv("clens_test.csv")
g_r = df["MAG_g"].values - df["MAG_r"].values

cnn_pred = np.load("clens_convnet_pred.npy")
y_true = np.load("clens_test_labels.npy")

phot_pred = np.loadtxt('clens_tpc.0.mlz', unpack=True, usecols=(2,))
morph_pred= np.loadtxt('clens_tpc.1.mlz', unpack=True, usecols=(2,))

mask = ((g_r > -10) & (g_r < 10))
g_r, y_true, cnn_pred, phot_pred, morph_pred = map(
    lambda x: x[mask],
    [g_r, y_true, cnn_pred, phot_pred, morph_pred]
)

bins = np.arange(-0.75, 2.75, 0.05)

stars_frac_med, stars_frac_lower, stars_frac_upper = confidence_band(
    get_stellar_fraction, y_true, y_true, g_r, bins=bins, n_boots=1000)

cnn_cut, _ = threshold_at_completeness_of(1 - y_true, 1 - cnn_pred, 0.96)

cnn_g_med, cnn_g_lower, cnn_g_upper = confidence_band(
    kde_purity, 1 - y_true, 1- cnn_pred, g_r, bins=bins, p_cut=cnn_cut, n_boots=1000)

phot_p_cut, _ = threshold_at_completeness_of(1 - y_true, 1 - phot_pred, 0.96)

phot_g_med, phot_g_lower, phot_g_upper = confidence_band(
    kde_purity, 1 - y_true, 1 - phot_pred, g_r, bins=bins, p_cut=phot_p_cut, n_boots=1000)

morph_p_cut, _ = threshold_at_completeness_of(1 - y_true, 1 - morph_pred, 0.96)

morph_g_med, morph_g_lower, morph_g_upper = confidence_band(
    kde_purity, 1 - y_true, 1 - morph_pred, g_r, bins=bins, p_cut=morph_p_cut, n_boots=1000)

cnn_p_cut, _ = threshold_at_purity_of(y_true, cnn_pred, 0.97)

cnn_s_med, cnn_s_lower, cnn_s_upper = confidence_band(
    kde_completeness, y_true, cnn_pred, g_r, bins=bins, p_cut=cnn_p_cut, n_boots=10000)

phot_p_cut, _ = threshold_at_purity_of(y_true, phot_pred, 0.97)

phot_s_med, phot_s_lower, phot_s_upper = confidence_band(
    kde_completeness, y_true, phot_pred, g_r, bins=bins, p_cut=phot_p_cut, n_boots=10000)

morph_p_cut, _ = threshold_at_purity_of(y_true, morph_pred, 0.97)

morph_s_med, morph_s_lower, morph_s_upper = confidence_band(
    kde_completeness, y_true, morph_pred, g_r, bins=bins, p_cut=morph_p_cut, n_boots=10000)

alpha = 0.2

p = sns.color_palette()
sns.set(style="ticks")

fig = plt.figure(figsize=(6, 8))

ax0 = plt.subplot2grid((6, 8), (0, 0), colspan=8)
ax1 = plt.subplot2grid((6, 8), (1, 0), colspan=8)
ax2 = plt.subplot2grid((6, 8), (2, 0), colspan=8, rowspan=2)
ax3 = plt.subplot2grid((6, 8), (4, 0), colspan=8, rowspan=2)

plt.setp(ax0.get_xticklabels(), visible=False)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)

ax0.hist(g_r, bins=bins, histtype='bar', color=p[1], alpha=alpha)
log_dens_g_r = get_log_density(g_r, bins=bins)
ax0.plot(bins, len(g_r) * np.exp(log_dens_g_r) / np.exp(log_dens_g_r).sum(), color=p[1])
ax0.set_xlim(-0.4, 2.25)
ax0.set_yticks([200, 400, 600])
ax0.set_ylabel('$N$')

ax1.plot(bins, stars_frac_med, label='stars', color=p[3])
# error bars are too small
ax1.fill_between(bins, stars_frac_lower, stars_frac_upper, color=p[3], alpha=alpha)
ax1.set_xlim(-0.25, 2.25)
ax1.set_ylim(0, 0.45)
ax1.set_yticks([0, 0.2, 0.4])
ax1.set_ylabel('stellar fraction')
#ax1.legend(loc='upper right')

ax2.plot(bins, cnn_g_med, label='ConvNet', color=p[2])
ax2.fill_between(bins, cnn_g_lower, cnn_g_upper, color=p[2], alpha=alpha)

ax2.plot(bins, morph_g_med, label='$\mathregular{TPC_{morph}}$', ls='--', color=p[0])
ax2.fill_between(bins, morph_g_lower, morph_g_upper, color=p[0], alpha=alpha)

ax2.plot(bins, phot_g_med, label='$\mathregular{TPC_{phot}}$', ls='--', color=p[4])
ax2.fill_between(bins, phot_g_lower, phot_g_upper, color=p[4], alpha=alpha)

ax2.legend(loc='lower center')

ax2.set_xlim(-0.25, 2.25)
ax2.set_ylim(0.83, 1.03)
ax2.set_yticks([0.84, 0.88, 0.92, 0.96, 1.0])
ax2.set_ylabel(r'$p_g(c_g=0.96)$', fontsize=14)

ax3.plot(bins, cnn_s_med, label='ConvNet', ls='-', color=p[2])
ax3.fill_between(bins, cnn_s_lower, cnn_s_upper, color=p[2], alpha=alpha)

ax3.plot(bins, phot_s_med, label='$\mathregular{TPC_{phot}}$', ls='--', color=p[4])
ax3.fill_between(bins, phot_s_lower, phot_s_upper, color=p[4], alpha=alpha)

ax3.plot(bins, morph_s_med, label='$\mathregular{TPC_{morph}}$', ls='--', color=p[0])
ax3.fill_between(bins, morph_s_lower, morph_s_upper, color=p[0], alpha=alpha)

ax3.set_ylabel(r'$c_s(p_s=0.97)$', fontsize=14)
ax3.set_xlim(-0.25, 2.25)
ax3.set_ylim(0.25, 1.05)
ax3.set_yticks([0.25, 0.50, 0.75, 1.0])

ax3.set_xlabel(r'$g-r$')

plt.subplots_adjust(hspace=0)
plt.savefig('figures/clens_g_r.pdf', bbox_inches='tight')

plt.show()



