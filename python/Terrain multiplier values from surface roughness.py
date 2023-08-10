get_ipython().magic('matplotlib inline')

import numpy as np
from scipy import optimize, interpolate
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_context("poster")
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display

z0 = np.array([2., 1., 0.8, 0.4, 0.2, 0.08, 0.06, 0.04, 0.02, 0.008, 0.006, 0.004, 0.002])
cat = np.array([4., 3.7, 3.6, 3.3, 3.0, 2.6, 2.48, 2.30, 2.0, 1.6, 1.48, 1.30, 1.0])


def func(z0, a, b):
    return a + b * np.log(z0)

ax = sns.regplot(z0, cat, fit_reg=True, logx=True)
zopt, zcov = optimize.curve_fit(func, z0, cat)
label = "T = {0:.5f} + {1:.5f} * log($z_0$)".format(*zopt)
plt.plot(z0, func(z0, *zopt), 'r-', label=label)

plt.plot(z0, np.log10(z0/2.)+4., 'k--', lw=0.5, label="AS/NZS 1170.2 Supp 1")

ax.set_xlim(0.001, 2.5)
ax.set_ylim(0.5, 5)
ax.set_xscale("log")
ax.set_xlabel("Roughness length ($z_0$)")
ax.set_ylabel("Terrain category")
ax.legend()

mz_val = {3: np.array([0.99, 0.91, 0.83, 0.75]),
          5: np.array([1.05, 0.91, 0.83, 0.75]), 
          10: np.array([1.12, 1.00, 0.83, 0.75]), 
          15: np.array([1.16, 1.05, 0.89, 0.75]), 
          20: np.array([1.19, 1.08, 0.94, 0.75]), 
          30: np.array([1.22, 1.12, 1.00, 0.80]), 
          40: np.array([1.24, 1.16, 1.04, 0.85]), 
          50: np.array([1.25, 1.18, 1.07, 0.90]), 
          75: np.array([1.27, 1.22, 1.12, 0.98]), 
          100: np.array([1.29, 1.24, 1.16, 1.03]), 
          150: np.array([1.31, 1.27, 1.21, 1.11]), 
          200: np.array([1.32, 1.29, 1.24, 1.16])}

mz_cat = np.array([1., 2., 3., 4.])

cmap = sns.husl_palette(12)

fig, ax = plt.subplots(1, 1)
n = 0
for key, val in mz_val.iteritems():
    sns.regplot(mz_cat, val, label="{0}".format(key), ax=ax, color=cmap[n], ci=None)
    n += 1
    
ax.set_xlabel("Terrain category")
ax.set_ylabel(r"$M_{z,cat}$")
ax.legend(title="Height", ncol=3)

def fit_mzcat(category, a, b):
    return a + b *np.log(category)


def plotmzcat(height):
    
    plt.clf()
    mzvals = mz_val[height]
    ax = sns.regplot(mz_cat, mzvals, fit_reg=True, logx=True)
    
    popt, pcov = optimize.curve_fit(fit_mzcat, mz_cat, mzvals)
    label = r"$M_z$ = {0:.5f} + {1:.5f} * log(T)".format(*popt)
    plt.plot(mz_cat, fit_mzcat(mz_cat, *popt), 'r-', label=label)
    ax.set_ylabel("$M_{z,cat}$")
    ax.set_xlabel("Terrain category")
    ax.legend()
    plt.show()
    
    
heights = sorted(mz_val.keys())
heightSelect = widgets.Dropdown(options=heights, value=heights[0], description="Height (m)")

w = interact(plotmzcat, height=heightSelect) 
display(w)

extended_z0 = np.array([10., 8., 4., 2., 1., 0.8, 0.4, 0.2, 0.08, 0.06, 0.04, 0.02, 0.008, 0.006, 0.004, 0.002])
new_mz_cat = func(extended_z0, *zopt)

def interp_mzcategory(mzvals, mz_cat, height, new_mz_cat):
    f = interpolate.interp1d(mzvals, mz_cat, )
    return f(new_mz_cat)

def plotmzcat_zo(height):
    
    plt.clf()
    mzvals = mz_val[height]
    popt, pcov = optimize.curve_fit(fit_mzcat, mz_cat, mzvals)
    fix, ax = plt.subplots(1, 1)
    label1 = r"{0:.5f} + {1:.5f} * log($z_0$)".format(*zopt)
    label = r"$M_z$ = {0:.5f} + {1:.5f} * log({2})".format(popt[0], popt[1], label1)
    plt.semilogx(extended_z0, fit_mzcat(new_mz_cat, *popt), 'r-', label=label)
    ax.set_ylabel("$M_{z,cat}$")
    ax.set_xlabel("Roughness length ($z_0$)")
    ax.legend()
    plt.show()
    
    
heights = sorted(mz_val.keys())
heightSelect = widgets.Dropdown(options=heights, value=heights[0], description="Height (m)")

w = interact(plotmzcat_zo, height=heightSelect) 
display(w)

def tableprint(height):
    titlestr = "Roughness length (m) | Terrain category | Mz,cat"
    rowfmt = "{0: ^21}|{1:18.5f}|{2:7.4f}"
    mzvals = mz_val[height]
    popt, pcov = optimize.curve_fit(fit_mzcat, mz_cat, mzvals)
    mzs = fit_mzcat(new_mz_cat, *popt)
    
    print(titlestr)
    for z, cat, mz in zip (extended_z0, new_mz_cat, mzs):
        print(rowfmt.format(z, cat, mz))

heights = sorted(mz_val.keys())
heightSelector = widgets.Dropdown(options=heights, value=heights[0], description="Height (m)")
ww = interact(tableprint, height=heightSelector) 
display(ww)



