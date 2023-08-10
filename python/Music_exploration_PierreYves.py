import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import h5py as h5
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = (8,6)
sns.set_palette('Dark2')
sns.set_style('whitegrid')

con_simi = sqlite3.connect('MillionSongSubset/AdditionalFiles/subset_artist_similarity.db')
con_term = sqlite3.connect('MillionSongSubset/AdditionalFiles/subset_artist_term.db')
con_meta = sqlite3.connect('MillionSongSubset/AdditionalFiles/subset_track_metadata.db')

cur_simi = con_simi.cursor()
cur_term = con_term.cursor()
cur_meta = con_meta.cursor()

res = con_simi.execute("SELECT name FROM sqlite_master WHERE type='table';")
for name in res:
    print(name[0])

res = con_term.execute("SELECT name FROM sqlite_master WHERE type='table';")
for name in res:
    print(name[0])

res = con_meta.execute("SELECT name FROM sqlite_master WHERE type='table';")
for name in res:
    print(name[0])

#cur_meta.execute("SELECT *\
#                  FROM songs")
#con_meta.commit()

songs = pd.read_sql_query('SELECT * FROM songs WHERE year!=0',con_meta)

songs.head(5)

songs.artist_hotttnesss.hist(bins=np.linspace(0.0,1.0,41));

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                       figsize=(15,8))

ax[0].scatter(songs.year, songs.artist_hotttnesss, marker='.')
ax[1].hexbin(songs.year, songs.artist_hotttnesss, cmap='viridis', gridsize=41, mincnt=1.0)

plt.subplots_adjust(wspace=0.02);

fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True,
                       figsize=(15,12))

ax[0,0].scatter(songs.year, songs.artist_familiarity, marker='.')
ax[0,1].hexbin(songs.year, songs.artist_familiarity, cmap='viridis', gridsize=41, mincnt=1.0)

ax[1,0].scatter(songs.year, songs.artist_hotttnesss, marker='.')
ax[1,1].hexbin(songs.year, songs.artist_hotttnesss, cmap='viridis', gridsize=41, mincnt=1.0)

plt.subplots_adjust(wspace=0.02, hspace=0.05)
ax[-1,-1].set_xlim(1920,songs.year.max());

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                       figsize=(15,8))

ax[0].scatter(songs.artist_familiarity, songs.artist_hotttnesss, marker='.')
ax[1].hexbin(songs.artist_familiarity, songs.artist_hotttnesss, cmap='viridis', gridsize=51, mincnt=1.0)

plt.subplots_adjust(wspace=0.02);

sns.lmplot(data=songs, x='artist_familiarity', y='artist_hotttnesss',
           markers='.', size=10);

tmp = songs.groupby('year').mean()
tmp[['artist_familiarity','artist_hotttnesss']].plot();



