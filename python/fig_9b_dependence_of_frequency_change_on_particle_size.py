import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from style_helpers import style_cycle

get_ipython().magic('matplotlib inline')
plt.style.use('style_sheets/custom_style.mplstyle')

df = pd.read_csv('../data/eigenmode_info_data_frame.csv')
df = df.query('(has_particle == True) and (x == 0) and (y == 0) and '
              '(d == 30) and (Hz == 8e4) and (Ms_particle == 1e6)')
df = df.sort_values('d_particle')

def plot_curve_for_eigenmode(ax, N, df, style_kwargs):
    df_filtered = df.query('N == {}'.format(N)).sort_values('d_particle')
    d_vals = df_filtered['d_particle']
    freq_vals = df_filtered['freq_diff'] * 1e3  # freq in MHz
    ax.plot(d_vals, freq_vals, label='N = {}'.format(N), **style_kwargs)

fig, ax = plt.subplots(figsize=(6, 6))

for N, style_kwargs in zip([1, 2, 3, 4, 5], style_cycle):
    plot_curve_for_eigenmode(ax, N, df, style_kwargs)

xmin, xmax = 7.5, 45
ymin, ymax = -50, 800

ax.plot([xmin, xmax], [0, 0], color='black', linestyle='--', linewidth=1)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel('Particle diameter (nm)')
ax.set_ylabel(r'Frequency change $\Delta\,$f (MHz)')
ax.legend(loc='upper left')

