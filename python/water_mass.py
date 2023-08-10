import gsw
import numpy as np

Te = np.linspace(0, 13, 25)
Se = np.linspace(34.4, 35.4, 25)

Tg, Sg = np.meshgrid(Te, Se)
sigma_theta = gsw.sigma0(Sg, Tg)
cnt = np.linspace(sigma_theta.min(), sigma_theta.max(), 10)

import pandas as pd

df = pd.read_csv('data/water_mass.csv', index_col='p')
df

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


fig, ax = plt.subplots(figsize=(7, 7))

cs = ax.contour(Sg, Tg, sigma_theta, colors='grey', levels=cnt, zorder=1)
kw = dict(color='blue', fontsize=14, fontweight='black')
ax.text(35.00, 11.9, 'SACW', **kw)
ax.text(34.50, 3.12, 'AAIW', **kw)
ax.text(34.82, 1.72, 'NADW', **kw)

kw = dict(color='darkorange', linestyle="none", marker='*')
ax.plot(df['SP'], df['CT'], **kw)
ax.set_xlabel('Salinity [g kg$^{-1}$]')
ax.set_ylabel('Temperature [$^\circ$C]')
ax.set_title('T-S Diagram')
ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

