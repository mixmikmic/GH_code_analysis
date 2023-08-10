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

agg_dyads = pd.read_csv("../data/redcard/raw_dyads.csv.gz", compression='gzip', index_col=[0, 1])

agg_dyads.head()

# Test if the number of games is equal to the victories + ties + defeats in the dataset

all(agg_dyads['games'] == agg_dyads.victories + agg_dyads.ties + agg_dyads.defeats)

# Sanity check passes

len(agg_dyads.reset_index().set_index('playerShort'))

agg_dyads['totalRedCards'] = agg_dyads['yellowReds'] + agg_dyads['redCards']
agg_dyads.rename(columns={'redCards': 'strictRedCards'}, inplace=True)

agg_dyads.head()

player_dyad = (clean_players.merge(agg_dyads.reset_index().set_index('playerShort'),
                                   left_index=True,
                                   right_index=True))

player_dyad.head()

clean_dyads = (agg_dyads.reset_index()[agg_dyads.reset_index()
                                   .playerShort
                                   .isin(set(players.index))
                                  ]).set_index(['refNum', 'playerShort'])

clean_dyads.shape, agg_dyads.shape, player_dyad.shape

# inspired by https://github.com/mathewzilla/redcard/blob/master/Crowdstorming_visualisation.ipynb
colnames = ['games', 'totalRedCards']
j = 0
out = [0 for _ in range(sum(clean_dyads['games']))]

for index, row in clean_dyads.reset_index().iterrows():
    n = row['games']
    d = row['totalRedCards']
    ref = row['refNum']
    player = row['playerShort']
    for _ in range(n):
        row['totalRedCards'] = 1 if (d-_) > 0 else 0
        rowlist=list([ref, player, row['totalRedCards']])
        out[j] = rowlist
        j += 1

tidy_dyads = pd.DataFrame(out, columns=['refNum', 'playerShort', 'redcard'],).set_index(['refNum', 'playerShort'])

# 3092
tidy_dyads.redcard.sum()

# Notice this is longer than before
clean_dyads.games.sum()

tidy_dyads.shape

# Ok, this is a bit crazy... tear it apart and figure out what each piece is doing if it's not clear
clean_referees = (referees.reset_index()[referees.reset_index()
                                                 .refNum.isin(tidy_dyads.reset_index().refNum
                                                                                       .unique())
                                        ]).set_index('refNum')

clean_referees.shape, referees.shape

clean_countries = (countries.reset_index()[countries.reset_index()
                                           .refCountry
                                           .isin(clean_referees.refCountry
                                                 .unique())
                                          ].set_index('refCountry'))

clean_countries.shape, countries.shape

tidy_dyads.head()

