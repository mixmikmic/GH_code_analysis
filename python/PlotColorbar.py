get_ipython().magic('matplotlib inline')
from matplotlib import pyplot as plt
import matplotlib as mpl

import seaborn as sns

sns.set_style("ticks")
sns.set_context("talk")

output_dir = "out/"
output_suffix = ""
output_formats = [".png", ".pdf"]

def save_figure(fig, name):
    for output_format in output_formats:
        fig.savefig(output_dir + "/" + name + output_suffix + output_format)
    return None

mpl.rc('savefig', dpi=300)

x = [0, 16]
y = [0, 16]
c = [0, 16.0]
fig, ax = plt.subplots(1, 1, figsize=(6,4))
sc = ax.scatter(x, y, c=c, cmap="Reds")
plt.gca().set_visible(False)
fig.colorbar(sc, label="Log2(CPM+1)")

max_val = 12.5
x = [0, 16]
y = [0, 16]
c = [0, max_val]
fig, ax = plt.subplots(1, 1, figsize=(6,4))
sc = ax.scatter(x, y, c=c, cmap="plasma")
plt.gca().set_visible(False)
fig.colorbar(sc, label="Log2(CPM+1)")



