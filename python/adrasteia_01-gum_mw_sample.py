import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
get_ipython().magic('config InlineBackend.figure_format = "retina"')
import pandas as pd
sns.set_context("talk")

names = ['byte_range', 'data_type', 'col_ID', 'desc']
fwf_cols = pd.read_fwf('../data/synthetic/gum_mw_columns.tsv',names=names)

fwf_cols.head()

col_names = fwf_cols.col_ID.values

gum_mw_alt = pd.read_fwf('../data/synthetic/gum_mw.sam', names=col_names)

gum_mw_alt.drop(["Vamp", "Vper", "Vphase", "Vtype"], inplace=True, axis=1)

gum_mw_alt.head()

col_names

gum = gum_mw_alt

plt.figure(figsize=[5, 8])
#plt.plot(gum['V-I'], gum.Mbol, '.')
plt.plot(gum.Teff, gum.Mbol, '.')
plt.xlim(10000, 2000)
plt.ylim(20, -5)
plt.xlabel("$T_{\mathrm{eff}}$")
plt.ylabel("$M_{\mathrm{bol}}$");

plt.figure(figsize=[8, 8])
#plt.plot(gum['V-I'], gum.Mbol, '.')
sc = plt.scatter(gum.r/1000.0, gum.Gmag, c=gum.Teff, s=20, marker='o', vmin=2000, vmax=10000, cmap="Spectral")
plt.xlabel("$d$ (kpc)")
plt.ylabel("$G$")
plt.hlines(12, 0, 10, colors = 'b', linestyles='--')
plt.colorbar(sc)
plt.ylim(20, 5)
plt.xlim(0, 10)

plt.figure(figsize=[8, 8])
plt.plot(gum.pmRA, gum.pmDE, '.', alpha=0.2)
plt.xlabel("$\delta_{\mathrm{RA}}$ (mas/yr)")
plt.ylabel("$\delta_{\mathrm{DEC}}$ (mas/yr)")

plt.figure(figsize=[8, 8])
pm = np.sqrt(gum.pmDE**2 + gum.pmDE**2)
plt.plot(gum.r/1000.0, pm, '.', alpha=0.5)
plt.xlabel("$d$ (kpc)")
plt.ylabel("$\delta$ (mas/yr)")

