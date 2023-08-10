get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from math import pi

from windrose import WindroseAxes, WindAxes, plot_windrose

df = pd.read_csv("samples/sample_wind_poitiers.csv", parse_dates=['Timestamp'])
#df['Timestamp'] = pd.to_timestamp()
df = df.set_index('Timestamp')
df

df['speed_x'] = df['speed'] * np.sin(df['direction'] * pi / 180.0)
df['speed_y'] = df['speed'] * np.cos(df['direction'] * pi / 180.0)

fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
#ax.set_aspect(abs(x1-x0)/abs(y1-y0))
ax.set_aspect('equal')
#_ = ax.scatter(df['speed_x'], df['speed_y'], alpha=0.25)
_ = df.plot(kind='scatter', x='speed_x', y='speed_y', alpha=0.05, ax=ax)
Vw = 80
_ = ax.set_xlim([-Vw, Vw])
_ = ax.set_ylim([-Vw, Vw])

ax = WindroseAxes.from_ax()
ax.bar(df.direction.values, df.speed.values, bins=np.arange(0.01,8,1), cmap=cm.hot, lw=3)
ax.set_legend()

#ax = new_axes()
#plot_windrose(df, bins=np.arange(0.01,8,1), cmap=cm.hot, lw=3, ax=ax)
_ = plot_windrose(df, kind='contour', bins=np.arange(0.01,8,1), cmap=cm.hot, lw=3)

bins = np.arange(0,30+1,1)
bins = bins[1:]
bins

_ = plot_windrose(df, kind='pdf', bins=np.arange(0.01,30,1))

data = np.histogram(df['speed'], bins=bins)[0]
data

def plot_month(df, t_year_month, *args, **kwargs):
    by = 'year_month'
    df[by] = df.index.map(lambda dt: (dt.year, dt.month))
    df_month = df[df[by] == t_year_month]
    ax = plot_windrose(df_month, *args, **kwargs)
    return ax


plot_month(df, (2014, 7), kind='contour', bins=np.arange(0, 10, 1), cmap=cm.hot)

plot_month(df, (2014, 8), kind='contour', bins=np.arange(0, 10, 1), cmap=cm.hot)

plot_month(df, (2014, 9), kind='contour', bins=np.arange(0, 10, 1), cmap=cm.hot)

