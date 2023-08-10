get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import numpy as np
from math import pi

from windrose import WindroseAxes

N = 500
ws = np.random.random(N)*6
wd = np.random.random(N)*360

df = pd.DataFrame({"speed": ws, "direction": wd})
df

df['speed_x'] = df['speed'] * np.sin(df['direction'] * pi / 180.0)
df['speed_y'] = df['speed'] * np.cos(df['direction'] * pi / 180.0)
fig, ax = plt.subplots(figsize=(8, 8), dpi=80)
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()
ax.set_aspect('equal')
_ = df.plot(kind='scatter', x='speed_x', y='speed_y', alpha=0.35, ax=ax)

ax = WindroseAxes.from_ax()
ax.bar(df.direction, df.speed, normed=True, opening=0.8, edgecolor='white')
ax.set_legend()

ax = WindroseAxes.from_ax()
ax.box(df.direction, df.speed, bins=np.arange(0, 8, 1))
ax.set_legend()

ax = WindroseAxes.from_ax()
ax.contourf(df.direction, df.speed, bins=np.arange(0, 8, 1), cmap=cm.hot)
ax.set_legend()

ax = WindroseAxes.from_ax()
ax.contourf(df.direction, df.speed, bins=np.arange(0, 8, 1), cmap=cm.hot)
ax.contour(df.direction, df.speed, bins=np.arange(0, 8, 1), colors='black')
ax.set_legend()

ax = WindroseAxes.from_ax()
ax.contour(df.direction, df.speed, bins=np.arange(0, 8, 1), cmap=cm.hot, lw=3)
ax.set_legend()

