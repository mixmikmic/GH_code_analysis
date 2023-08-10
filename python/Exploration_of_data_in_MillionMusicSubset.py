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

path_to_data = '../MillionSongSubset/'

con_simi = sqlite3.connect(path_to_data+'AdditionalFiles/subset_artist_similarity.db')
con_term = sqlite3.connect(path_to_data+'AdditionalFiles/subset_artist_term.db')
con_meta = sqlite3.connect(path_to_data+'AdditionalFiles/subset_track_metadata.db')

cur_simi = con_simi.cursor()
cur_term = con_term.cursor()
cur_meta = con_meta.cursor()

# subset_artist_similarity.db
res = con_simi.execute("SELECT name FROM sqlite_master WHERE type='table';")
for name in res:
    print(name[0])

# subset_artist_term
res = con_term.execute("SELECT name FROM sqlite_master WHERE type='table';")
for name in res:
    print(name[0])

# subset_track_metadata
res = con_meta.execute("SELECT name FROM sqlite_master WHERE type='table';")
for name in res:
    print(name[0])

songs = pd.read_sql_query('SELECT * FROM songs WHERE year!=0',con_meta)

songs.head(5)

songs.artist_hotttnesss.hist(bins=np.linspace(0.0,1.0,41));
plt.xlabel('Artist Hotness')

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
ax[-1,-1].set_xlim(1920,songs.year.max());
plt.subplots_adjust(wspace=0.02, hspace=0.05)

fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                       figsize=(15,8))

ax[0].scatter(songs.artist_familiarity, songs.artist_hotttnesss, marker='.')
ax[1].hexbin(songs.artist_familiarity, songs.artist_hotttnesss, cmap='viridis', gridsize=51, mincnt=1.0)

plt.subplots_adjust(wspace=0.02);
### Artist_hotttnesss vs artist familiarity
sns.lmplot(data=songs, x='artist_familiarity', y='artist_hotttnesss',
           markers='.', size=10);

tmp = songs.groupby('year').mean()
tmp[['artist_familiarity','artist_hotttnesss']].plot();

with pd.HDFStore(path_to_data+'AdditionalFiles/subset_msd_summary_file.h5') as store:
    print(store)
    analysis_summary = store.select('analysis/songs')
    metadata_summary = store.select('metadata/songs')
    musicbrainz_summary = store.select('musicbrainz/songs')                  

analysis_summary.head()

metadata_summary.head()

musicbrainz_summary.head()



