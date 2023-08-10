import pandas as pd
import reverse_geocode
import matplotlib.pyplot as plt
import numpy as np

colNames = ['drop','year','artist_name','artist_id','artist_hotness','artist_familarity','artist_location','song_title','song_hotness','album_name','lat','long']

len(colNames)

songs_df = pd.read_csv('files/moreFeaturesLatLong.csv', index_col=False, encoding='iso-8859-1', header = None)

songs_df.head()

songs_df.tail(20)

songs_df.shape

songs_df.columns = colNames

songs_df.drop('drop', axis = 1, inplace=True)

trackid_df = pd.read_csv('files/moreFeaturesTrackID.csv', index_col=False, encoding='iso-8859-1', header = None)

trackid_df.shape

colNames = ['drop','track_id']
trackid_df.columns = colNames
trackid_df.drop('drop', axis = 1, inplace=True)
trackid_df.shape

trackid_df.head()

songs_df.insert(0, 'track_id', trackid_df['track_id'])
songs_df.head()

songs_dropped = songs_df.dropna(subset=['lat','long'])

songs_dropped.reset_index(drop=True, inplace=True)

songs_dropped.ix[:,10:12].head(20)

songs_dropped.tail()

size = songs_dropped.shape[0]
size

get_ipython().system(' pip install --user reverse_geocode')

countries = []
for i in range(0, size):
    lat = songs_dropped.iloc[i]['lat']
    lon = songs_dropped.iloc[i]['long']
    l = list()
    l.append((lat,lon))
    res = reverse_geocode.search(l)
    countries.append(res[0]['country'])

country = pd.DataFrame(countries)
country.columns = ['country']
country.head()

country.shape

data_withCountry_df = pd.concat([songs_dropped, country], axis=1)
data_withCountry_df.tail(20)

res = pd.DataFrame(data_withCountry_df.groupby(by='country').count())
res = pd.DataFrame(res['track_id'])
res.reset_index(inplace=True)
res.columns = ['country','count']
res.sort_values(by='count', axis=0, ascending=False, inplace=True)
top_countries = res.head(10)
top_countries.head()

np.ravel(top_countries['country'])

plt.figure(figsize = (12,8))
plt.bar(range(10), np.ravel(top_countries['count']), align='center')
plt.xticks(range(10), np.ravel(top_countries['country']), size='small')
plt.title('Top Ten Countries By Song Frequency',fontweight="bold", fontsize=18)
plt.xlabel('Countries', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.show()

res.head()

songs_unitesStates_df = data_withCountry_df.loc[data_withCountry_df['country'] == 'United States']
songs_unitesStates_df.tail()

list_trackids = songs_unitesStates_df['track_id']
#list_trackids

songs_unitesStates_df.shape

colNames = ['year', 'artist_name', 'artist_id', 'song_title', 'song_beats_persecond', 'song_duration', 'song_end_fade_in', 'song_start_fade_out', 'song_key', 'song_loudness', 'song_loudness_max', 'song_loudness_min', 'song_loudness_med', 'song_loudness_time_max', 'song_loudness_time_min', 'song_loudness_time_med', 'song_mode', 'song_tempo', 'song_time_signature', 'pitch0', 'pitch1', 'pitch2', 'pitch3', 'pitch4', 'pitch5', 'pitch6', 'pitch7', 'pitch8', 'pitch9', 'pitch10', 'pitch11', 'pitch12', 'pitch13', 'pitch14', 'pitch15', 'pitch16', 'pitch17', 'pitch18', 'pitch19', 'pitch20', 'pitch21', 'pitch22', 'pitch23', 'pitch24', 'pitch25', 'pitch26', 'pitch27', 'pitch28', 'pitch29', 'pitch30', 'pitch31', 'pitch32', 'pitch33', 'pitch34', 'pitch35', 'pitch36', 'pitch37', 'pitch38', 'pitch39', 'pitch40', 'pitch41', 'pitch42', 'pitch43', 'pitch44', 'pitch45', 'pitch46', 'pitch47', 'pitch48', 'pitch49', 'pitch50', 'pitch51', 'pitch52', 'pitch53', 'pitch54', 'pitch55', 'pitch56', 'pitch57', 'pitch58', 'pitch59', 'pitch60', 'pitch61', 'pitch62', 'pitch63', 'pitch64', 'pitch65', 'pitch66', 'pitch67', 'pitch68', 'pitch69', 'pitch70', 'pitch71', 'pitch72', 'pitch73', 'pitch74', 'pitch75', 'pitch76', 'pitch77', 'pitch78', 'pitch79', 'pitch80', 'pitch81', 'pitch82', 'pitch83', 'pitch84', 'pitch85', 'pitch86', 'pitch87', 'pitch88', 'pitch89', 'timbre0', 'timbre1', 'timbre2', 'timbre3', 'timbre4', 'timbre5', 'timbre6', 'timbre7', 'timbre8', 'timbre9', 'timbre10', 'timbre11', 'timbre12', 'timbre13', 'timbre14', 'timbre15', 'timbre16', 'timbre17', 'timbre18', 'timbre19', 'timbre20', 'timbre21', 'timbre22', 'timbre23', 'timbre24', 'timbre25', 'timbre26', 'timbre27', 'timbre28', 'timbre29', 'timbre30', 'timbre31', 'timbre32', 'timbre33', 'timbre34', 'timbre35', 'timbre36', 'timbre37', 'timbre38', 'timbre39', 'timbre40', 'timbre41', 'timbre42', 'timbre43', 'timbre44', 'timbre45', 'timbre46', 'timbre47', 'timbre48', 'timbre49', 'timbre50', 'timbre51', 'timbre52', 'timbre53', 'timbre54', 'timbre55', 'timbre56', 'timbre57', 'timbre58', 'timbre59', 'timbre60', 'timbre61', 'timbre62', 'timbre63', 'timbre64', 'timbre65', 'timbre66', 'timbre67', 'timbre68', 'timbre69', 'timbre70', 'timbre71', 'timbre72', 'timbre73', 'timbre74', 'timbre75', 'timbre76', 'timbre77', 'timbre78', 'timbre79', 'timbre80', 'timbre81', 'timbre82', 'timbre83', 'timbre84', 'timbre85', 'timbre86', 'timbre87', 'timbre88', 'timbre89']

all_songs_df = pd.read_csv('write_clean.csv', index_col=False, encoding='iso-8859-1', header = None)
all_songs_df.columns = colNames
all_songs_df.insert(0, 'track_id', trackid_df['track_id'])
all_songs_df.shape

all_songs_df.head()

final_df = all_songs_df.loc[all_songs_df['track_id'].isin(list_trackids)]
final_df.head()

final_df.shape

final_df.to_csv('us_trackid.csv', header=True,index=True, encoding = 'iso-8859-1')

import sqlite3
engine = sqlite3.connect('USDB')
final_df.to_sql('songs', engine, if_exists='replace', index = False)

songs_unitedKingdom_df = data_withCountry_df.loc[data_withCountry_df['country'] == 'United Kingdom']
songs_unitedKingdom_df.tail()

colNames = ['year', 'artist_name', 'artist_id', 'song_title', 'song_beats_persecond', 'song_duration', 'song_end_fade_in', 'song_start_fade_out', 'song_key', 'song_loudness', 'song_loudness_max', 'song_loudness_min', 'song_loudness_med', 'song_loudness_time_max', 'song_loudness_time_min', 'song_loudness_time_med', 'song_mode', 'song_tempo', 'song_time_signature', 'pitch0', 'pitch1', 'pitch2', 'pitch3', 'pitch4', 'pitch5', 'pitch6', 'pitch7', 'pitch8', 'pitch9', 'pitch10', 'pitch11', 'pitch12', 'pitch13', 'pitch14', 'pitch15', 'pitch16', 'pitch17', 'pitch18', 'pitch19', 'pitch20', 'pitch21', 'pitch22', 'pitch23', 'pitch24', 'pitch25', 'pitch26', 'pitch27', 'pitch28', 'pitch29', 'pitch30', 'pitch31', 'pitch32', 'pitch33', 'pitch34', 'pitch35', 'pitch36', 'pitch37', 'pitch38', 'pitch39', 'pitch40', 'pitch41', 'pitch42', 'pitch43', 'pitch44', 'pitch45', 'pitch46', 'pitch47', 'pitch48', 'pitch49', 'pitch50', 'pitch51', 'pitch52', 'pitch53', 'pitch54', 'pitch55', 'pitch56', 'pitch57', 'pitch58', 'pitch59', 'pitch60', 'pitch61', 'pitch62', 'pitch63', 'pitch64', 'pitch65', 'pitch66', 'pitch67', 'pitch68', 'pitch69', 'pitch70', 'pitch71', 'pitch72', 'pitch73', 'pitch74', 'pitch75', 'pitch76', 'pitch77', 'pitch78', 'pitch79', 'pitch80', 'pitch81', 'pitch82', 'pitch83', 'pitch84', 'pitch85', 'pitch86', 'pitch87', 'pitch88', 'pitch89', 'timbre0', 'timbre1', 'timbre2', 'timbre3', 'timbre4', 'timbre5', 'timbre6', 'timbre7', 'timbre8', 'timbre9', 'timbre10', 'timbre11', 'timbre12', 'timbre13', 'timbre14', 'timbre15', 'timbre16', 'timbre17', 'timbre18', 'timbre19', 'timbre20', 'timbre21', 'timbre22', 'timbre23', 'timbre24', 'timbre25', 'timbre26', 'timbre27', 'timbre28', 'timbre29', 'timbre30', 'timbre31', 'timbre32', 'timbre33', 'timbre34', 'timbre35', 'timbre36', 'timbre37', 'timbre38', 'timbre39', 'timbre40', 'timbre41', 'timbre42', 'timbre43', 'timbre44', 'timbre45', 'timbre46', 'timbre47', 'timbre48', 'timbre49', 'timbre50', 'timbre51', 'timbre52', 'timbre53', 'timbre54', 'timbre55', 'timbre56', 'timbre57', 'timbre58', 'timbre59', 'timbre60', 'timbre61', 'timbre62', 'timbre63', 'timbre64', 'timbre65', 'timbre66', 'timbre67', 'timbre68', 'timbre69', 'timbre70', 'timbre71', 'timbre72', 'timbre73', 'timbre74', 'timbre75', 'timbre76', 'timbre77', 'timbre78', 'timbre79', 'timbre80', 'timbre81', 'timbre82', 'timbre83', 'timbre84', 'timbre85', 'timbre86', 'timbre87', 'timbre88', 'timbre89']

all_songs_df = pd.read_csv('write_clean.csv', index_col=False, encoding='iso-8859-1', header = None)
all_songs_df.columns = colNames
all_songs_df.reset_index(inplace=True)
all_songs_df = all_songs_df.rename(columns={'index': 'track_id'})
all_songs_df.shape

list_trackids = songs_unitedKingdom_df['track_id']

final_df = all_songs_df.loc[all_songs_df['track_id'].isin(list_trackids)]
final_df.head()

final_df.shape

import sqlite3
engine = sqlite3.connect('UKDB')
final_df.to_sql('songs', engine, if_exists='replace', index = False)

