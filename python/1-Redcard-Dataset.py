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

# import hdbscan
from sklearn.datasets import make_blobs
import time

# Uncomment one of the following lines and run the cell:

df = pd.read_csv("../data/redcard/redcard.csv.gz", compression='gzip')
# df = pd.read_csv("https://github.com/cmawer/pycon-2017-eda-tutorial/raw/master/data/redcard/redcard.csv.gz", compression='gzip')

df.shape

df.head()

df.describe().T

df.dtypes

all_columns = df.columns.tolist()
all_columns







df.height.mean()

np.mean(df.groupby('playerShort').height.mean())

player_index = 'playerShort'
player_cols = [#'player', # drop player name, we have unique identifier
               'birthday',
               'height',
               'weight',
               'position',
               'photoID',
               'rater1',
               'rater2',
              ]

# Count the unique variables (if we got different weight values, 
# for example, then we should get more than one unique value in this groupby)
all_cols_unique_players = df.groupby('playerShort').agg({col:'nunique' for col in player_cols})

all_cols_unique_players.head()

# If all values are the same per player then this should be empty (and it is!)
all_cols_unique_players[all_cols_unique_players > 1].dropna().head()

# A slightly more elegant way to test the uniqueness
all_cols_unique_players[all_cols_unique_players > 1].dropna().shape[0] == 0

def get_subgroup(dataframe, g_index, g_columns):
    """Helper function that creates a sub-table from the columns and runs a quick uniqueness test."""
    g = dataframe.groupby(g_index).agg({col:'nunique' for col in g_columns})
    if g[g > 1].dropna().shape[0] != 0:
        print("Warning: you probably assumed this had all unique values but it doesn't.")
    return dataframe.groupby(g_index).agg({col:'max' for col in g_columns})

players = get_subgroup(df, player_index, player_cols)
players.head()

def save_subgroup(dataframe, g_index, subgroup_name, prefix='../data/redcard/raw_'):
    save_subgroup_filename = "".join([prefix, subgroup_name, ".csv.gz"])
    dataframe.to_csv(save_subgroup_filename, compression='gzip')
    test_df = pd.read_csv(save_subgroup_filename, compression='gzip', index_col=g_index)
    # Test that we recover what we send in
    if dataframe.equals(test_df):
        print("Test-passed: we recover the equivalent subgroup dataframe.")
    else:
        print("Warning -- equivalence test!!! Double-check.")

save_subgroup(players, player_index, "players")

club_index = 'club'
club_cols = ['leagueCountry']
clubs = get_subgroup(df, club_index, club_cols)
clubs.head()

clubs['leagueCountry'].value_counts()

save_subgroup(clubs, club_index, "clubs")

referee_index = 'refNum'
referee_cols = ['refCountry']
referees = get_subgroup(df, referee_index, referee_cols)
referees.head()

referees.refCountry.nunique()

save_subgroup(referees, referee_index, "referees")

country_index = 'refCountry'
country_cols = ['Alpha_3', # rename this name of country
                'meanIAT',
                'nIAT',
                'seIAT',
                'meanExp',
                'nExp',
                'seExp',
               ]
countries = get_subgroup(df, country_index, country_cols)
countries.head()

rename_columns = {'Alpha_3':'countryName'}
countries = countries.rename(columns=rename_columns)
countries.head()

countries.shape

save_subgroup(countries, country_index, "countries")

# Ok testing this out: 
test_df = pd.read_csv("../data/redcard/raw_countries.csv.gz", compression='gzip', index_col=country_index)

for (_, row1), (_, row2) in zip(test_df.iterrows(), countries.iterrows()):
    if not row1.equals(row2):
        print(row1)
        print()
        print(row2)
        print()
        break

row1.eq(row2)

row1.seIAT - row2.seIAT

countries.dtypes

test_df.dtypes

countries.head()

test_df.head()

countries.tail()

test_df.tail()

dyad_index = ['refNum', 'playerShort']
dyad_cols = ['games',
             'victories',
             'ties',
             'defeats',
             'goals',
             'yellowCards',
             'yellowReds',
             'redCards',
            ]

dyads = get_subgroup(df, g_index=dyad_index, g_columns=dyad_cols)

dyads.head()

dyads.shape

dyads[dyads.redCards > 1].head(10)

save_subgroup(dyads, dyad_index, "dyads")



