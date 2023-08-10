# You can ignore this cell, it just imports the different modules you need :smile:
from glob import glob
from IPython.display import display # Use this to display multiple outputs within a cell
import itertools as it
import json
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import os
import sys
import pandas as pd
import requests
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook',font_scale=1.5)
import sys

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import springerFullTextApi as sft

terms = ["github"]
years = range(2017, 2010, -1)
df = sft.records_by_term_and_year(terms, years)

def plot_histogram_hits(df):
    
    fig, ax = plt.subplots()

    # Helpful command from
    # https://stackoverflow.com/questions/27365467/can-pandas-plot-a-histogram-of-dates
    ax = sns.barplot(x='year', y='hits', data=df)

    ax.set_title('Source: Springer full-text API')
    ax.set_xlabel('Year')

    sns.despine()

plot_histogram_hits(df)





