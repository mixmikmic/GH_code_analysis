import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.colors import LogNorm

from textblob.sentiments import NaiveBayesAnalyzer

import pandas as pd
import sqlite3
from textblob import TextBlob

import seaborn as sns
sns.set(color_codes=True)

# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import spacy
nlp = spacy.load('en')

import re

df = pd.read_csv('../pitchfork3.csv')

years = np.arange(1999, 2018)
df_new = df[df['new_album'] == 1]
df_reissue = df[df['new_album'] == 0]

df_spotify = pd.read_csv('spotify_with_rank.csv')

df_merged = df_new.merge(df_spotify, how='left', on=['artist', 'title'])

df = df_merged[df_merged['best_new_music'] == 1]

df.drop_duplicates(inplace=True)

df[df['pub_year'] > 1999].sort_values('album_popularity mean', ascending=False)[['artist','title','score','album_popularity mean']].head(10)

df[df['pub_year'] > 1999].sort_values('album_popularity mean', ascending=True)[['artist','title','score','album_popularity mean']].head(10)

df[df['pub_year'] > 1999].sort_values('danceability mean', ascending=False)         [['artist','title','score','danceability mean']].head(10)

df[df['pub_year'] > 1999].sort_values('danceability mean', ascending=True)         [['artist','title','score','danceability mean']].head(10)

df[df['pub_year'] > 1999].sort_values('valence mean', ascending=False)         [['artist','title','score','valence mean']].head(10)

df[df['pub_year'] > 1999].sort_values('valence mean', ascending=True)         [['artist','title','score','valence mean']].head(10)

df[df['pub_year'] > 1999].sort_values('energy mean', ascending=False)         [['artist','title','score','energy mean']].head(10)

df[df['pub_year'] > 1999].sort_values('energy mean', ascending=True)         [['artist','title','score','energy mean']].head(10)

df[df['pub_year'] > 1999].sort_values('acousticness mean', ascending=False)         [['artist','title','score','acousticness mean']].head(10)

df[df['pub_year'] > 1999].sort_values('acousticness mean', ascending=True)         [['artist','title','score','acousticness mean']].head(10)

df[df['pub_year'] > 1999].sort_values('instrumentalness mean', ascending=False)         [['artist','title','score','instrumentalness mean']].head(10)

df[df['pub_year'] > 1999].sort_values('instrumentalness mean', ascending=True)         [['artist','title','score','instrumentalness mean']].head(10)

df[df['pub_year'] > 1999].sort_values('tempo mean', ascending=False)         [['artist','title','score','tempo mean']].head(10)

df[df['pub_year'] > 1999].sort_values('tempo mean', ascending=True)         [['artist','title','score','tempo mean']].head(10)

df[df['pub_year'] > 1999].sort_values('loudness mean', ascending=False)         [['artist','title','score','loudness mean']].head(10)

df[df['pub_year'] > 1999].sort_values('loudness mean', ascending=True)         [['artist','title','score','loudness mean']].head(10)

