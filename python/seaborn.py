import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from bokeh import mpl
from bokeh.plotting import output_notebook, show

output_notebook()

from scipy import optimize

# Set the palette colors.
sns.set(palette="Set2")


# Build the sin wave
def sine_wave(n_x, obs_err_sd=1.5, tp_err_sd=.3):
    x = np.linspace(0, (n_x - 1) / 2, n_x)
    y = np.sin(x) + np.random.normal(0, obs_err_sd) + np.random.normal(0, tp_err_sd, n_x)
    return y

sines = np.array([sine_wave(31) for _ in range(20)])

# Generate the Seaborn plot with "ci" bars.
ax = sns.tsplot(sines, err_style="ci_bars", interpolate=False)
xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, sines.shape[1])
out, _ = optimize.leastsq(lambda p: sines.mean(0) - (np.sin(x / p[1]) + p[0]), (0, 2))
a, b = out
xx = np.linspace(xmin, xmax, 100)
plt.plot(xx, np.sin(xx / b) + a, c="#444444")

plt.title("Seaborn tsplot with CI in bokeh.")

show(mpl.to_bokeh())

import pandas as pd
import matplotlib as mplc

# Generate the pandas dataframe
data = np.random.multivariate_normal([0, 0], [[1, 2], [2, 20]], size=100)
data = pd.DataFrame(data, columns=["X", "Y"])
mplc.rc("figure", figsize=(6, 6))

# Just plot seaborn kde
sns.kdeplot(data, cmap="BuGn_d")

plt.title("Seaborn kdeplot in bokeh.")

show(mpl.to_bokeh())

planets = sns.load_dataset("planets")

tips = sns.load_dataset("tips")

sns.set_style("whitegrid")

ax = sns.violinplot(x="day", y="total_bill", hue="sex",
                    data=tips, palette="Set2", split=True,
                    scale="count", inner="stick")

plt.title("Seaborn violin plot in bokeh.")

show(mpl.to_bokeh())



