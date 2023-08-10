import matplotlib
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
seaborn_style = [style for style in matplotlib.style.available if "seaborn" in style]
seaborn_style

import numpy as np

colors = {
    "seaborn-bright": ['003FFF', '03ED3A', 'E8000B', '8A2BE2', 'FFC400', '00D7FF'],
    "seaborn-colorblind": ['0072B2', '009E73', 'D55E00', 'CC79A7', 'F0E442', '56B4E9'],
    "seaborn-dark-palette": ['001C7F', '017517', '8C0900', '7600A1', 'B8860B', '006374'],
    "seaborn-deep": ['4C72B0', '55A868', 'C44E52', '8172B2', 'CCB974', '64B5CD'],
    "seaborn-muted": ['4878CF', '6ACC65', 'D65F5F', 'B47CC7', 'C4AD66', '77BEDB'],
    "seaborn-pastel": ['92C6FF', '97F0AA', 'FF9F9A', 'D0BBFF', 'FFFEA3', 'B0E0E6']    
}
f, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey=True)
x = np.linspace(0, 1, 100)
for ax, cname in zip(axes.flat, colors):
    for i, color in enumerate(colors[cname]):
        ax.plot(x, (i+1)*x**2 + i, color="#" + color, linewidth=5)
        ax.set_title(cname)

base = "/Library/Frameworks/Python.framework/Versions/3.5/lib/python3.5/site-packages/matplotlib/mpl-data/stylelib/"
styles = ["seaborn-notebook", "seaborn-paper", "seaborn-poster", "seaborn-talk"]
f, axes = plt.subplots(2, 2, figsize=(6, 6))
x = np.linspace(0, 1, 100)
for ax, style in zip(axes.flat, styles):
    rc = matplotlib.style.core.rc_params_from_file(base + style + ".mplstyle")
    matplotlib.rcParams.update(rc)
    for i in range(6):
        ax.plot(x, (i+1)*x**2 + i, linewidth=5)
        ax.set_title(style)
plt.tight_layout()

matplotlib.style.use(["seaborn-darkgrid", "seaborn-colorblind", "seaborn-notebook"])
x = np.linspace(0, 1, 100)
for i in range(6):
    plt.plot(x, (i+1)*x**2 + i)
plt.title("With the darkgrid, colorblind and notebook styles")

matplotlib.style.use(["seaborn-whitegrid", "seaborn-deep", "seaborn-notebook"])
x = np.linspace(0, 1, 100)
for i in range(6):
    plt.plot(x, (i+1)*x**2 + i)
plt.title("With the whitegrid, deep and notebook styles")

matplotlib.style.use(["seaborn-ticks", "seaborn-deep", "seaborn-paper"])
x = np.linspace(0, 1, 100)
for i in range(6):
    plt.plot(x, (i+1)*x**2 + i)
plt.title("With the white, ticks, deep and notebook styles")

