get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format='retina'")

from __future__ import absolute_import, division, print_function
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import GridSpec
import seaborn as sns
import mpld3
import numpy as np
import pandas as pd
import os, sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
sns.set_context("poster", font_scale=1.3)

import missingno as msno
import pandas_profiling

import hdbscan
from sklearn.datasets import make_blobs
import time

# Uncomment one of the following lines and run the cell:

# df = pd.read_csv("../data/redcard/redcard.csv.gz", compression='gzip')
# df = pd.read_csv("https://github.com/cmawer/pycon-2017-eda-tutorial/raw/master/data/redcard/redcard.csv.gz", compression='gzip')

def save_subgroup(dataframe, g_index, subgroup_name, prefix='../data/redcard/raw_'):
    save_subgroup_filename = "".join([prefix, subgroup_name, ".csv.gz"])
    dataframe.to_csv(save_subgroup_filename, compression='gzip')
    test_df = pd.read_csv(save_subgroup_filename, compression='gzip', index_col=g_index)
    # Test that we recover what we send in
    if dataframe.equals(test_df):
        print("Test-passed: we recover the equivalent subgroup dataframe.")
    else:
        print("Warning -- equivalence test!!! Double-check.")

def load_subgroup(filename, index_col=[0]):
    return pd.read_csv(filename, compression='gzip', index_col=index_col)

from pivottablejs import pivot_ui

temp = tidy_dyads.reset_index().set_index('playerShort').merge(clean_players, left_index=True, right_index=True)

temp.shape

pivot_ui(temp[['skintoneclass', 'position_agg', 'redcard']], )

# How many games has each player played in?
games = tidy_dyads.groupby(level=1).count()
sns.distplot(games);

(tidy_dyads.groupby(level=0)
           .count()
           .sort_values('redcard', ascending=False)
           .rename(columns={'redcard':'total games refereed'})).head()

(tidy_dyads.groupby(level=0)
           .sum()
           .sort_values('redcard', ascending=False)
           .rename(columns={'redcard':'total redcards given'})).head()

(tidy_dyads.groupby(level=1)
           .sum()
           .sort_values('redcard', ascending=False)
           .rename(columns={'redcard':'total redcards received'})).head()

total_ref_games = clean_dyads.groupby(level=0).games.sum().sort_values(ascending=False)
total_player_games = clean_dyads.groupby(level=1).games.sum().sort_values(ascending=False)

total_ref_given = clean_dyads.groupby(level=0).totalRedCards.sum().sort_values(ascending=False)
total_player_received = clean_dyads.groupby(level=1).totalRedCards.sum().sort_values(ascending=False)

sns.distplot(total_player_received, kde=False)

sns.distplot(total_ref_given, kde=False)

clean_dyads.groupby(level=1).totalRedCards.sum().sort_values(ascending=False).head()

clean_dyads.totalRedCards.sum(), clean_dyads.games.sum(), clean_dyads.totalRedCards.sum()/clean_dyads.games.sum()

temp = dyads.reset_index().set_index('playerShort')

temp2 = temp.merge(players, left_index=True, right_index=True, )

temp2.groupby('position_agg').totalRedCards.sum() / temp2.groupby('position_agg').games.sum()

players.head()

dyads.shape

sns.regplot('skintone', 'allredsStrict', data=temp2);



