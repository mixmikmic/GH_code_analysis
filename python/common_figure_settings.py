# common settings for making the figure text actual text instead of lots of tiny dots
import matplotlib
from matplotlib import rc

rc('text', usetex=False) # important!
matplotlib.rcParams['svg.fonttype'] = 'none'
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})

# Plots a heatmap as a rasterized layer instead of individual dots 
# (Gabe might have a better way of doing this)
get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_table(
    '/projects/ps-yeolab3/bay001/maps/current/se/204_01_RBFOX2.merged.r2.1.HepG2_native_cassette_exons.normed_matrix.txt',
    sep=',',
    index_col=0
)
df = df.head()
sns.heatmap(df, xticklabels=False, yticklabels=False)
plt.savefig(
    '/home/bay001/projects/codebase/temp/example_heatmap.svg',
    rasterized=True # important
)





