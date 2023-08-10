get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt

## Make some data for plotting purposes
N = 100
x = np.random.randn(N)
y = np.random.randn(N)
z = np.arange(N)
def make_plot(**kwargs):
    plt.plot(x,y,'o',**kwargs)

make_plot()

import seaborn as sns
make_plot()

for name in ['whitegrid','darkgrid','dark','white','ticks']:
    with sns.axes_style(name):
        fig = plt.figure()
        make_plot()

for name in ['paper','notebook','talk','poster']:
    with sns.plotting_context(name):
        fig = plt.figure()
        make_plot()

import seaborn as sns
sns.set(context='talk',style='ticks',font='serif',palette='muted',rc={"xtick.direction":"in","ytick.direction":"in"})
make_plot()

sns.reset_defaults()
make_plot()

#import seaborn.apionly as sns

sns.reset_defaults()
sns.palplot(sns.color_palette()) #current palette

for name in ['deep','muted','pastel','bright','dark','colorblind']:
    sns.palplot(sns.color_palette(name))

import pandas as pd

