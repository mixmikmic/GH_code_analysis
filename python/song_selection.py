import pandas
import numpy as np
import tensorflow as tf
from sklearn.cross_validation import train_test_split

# song_df = pandas.read_table('./dataset/10000.txt', header=None)
# song_df.columns=['user_id', 'song_id', 'listen_count']

song_df2 = pandas.read_csv('./dataset/MSD_songs.csv', sep='\t')
song_df2['title'] = song_df2['title'].map(str) + ' - ' + song_df2['artist_name']
song_df2 = song_df2.drop(['dig7_id', 'release', 'artist_name', 'year'], axis=1)
tag_df2 = pandas.read_csv('./dataset/LAST_FM_tags.csv', sep='\t')
tag_df2 = pandas.merge(tag_df2,song_df2, on='track_id', how='inner')
print(song_df2.head())
print(tag_df2.head())

print(len(tag_df2[tag_df2['tags']=='favorite']))
print(len(tag_df2[tag_df2['tags']=='favourites']))
print(tag_df2[tag_df2['tags']=='favourites'].head(1))

popular_tag_df = tag_df2.groupby('tags').agg({'tags': 'count'}).sort_values('tags',ascending=False)
popular_tag_df = popular_tag_df[:50]
popular_tag_df.head(50)

popular_tag_df.index

popular_processed_tag_df = pandas.merge(tag_df2, pandas.DataFrame({'tags':popular_tag_df.index}), on='tags', how='inner')
# print len(popular_processed_tag_df['tags'].unique())
len(popular_processed_tag_df['track_id'].unique())

len(tag_df2['track_id'].unique())

# tag_df2[tag_df2['track_id'] == 'TRAYJOF128F92F9EEA']
selected_songs_df = popular_processed_tag_df.groupby('track_id', as_index=False).last()
print(len(selected_songs_df))
selected_songs_df.head()

df = pandas.read_csv('./dataset/selected_song_based_on_tags.csv', sep='\t')

print(len(df[df['preview_url']=='not found']))
print(len(df[df['genres']!='not found']))
print(df['genres'].isnull().sum())

df[(df['genres']!='not found') & (df['genres'].notnull()==True)].head()

from time import sleep
import sys

for i in range(21):
    sys.stdout.write('\r')
    # the exact output you're looking for:
    sys.stdout.write("[%-20s] %d%%" % ('='*i, 5*i))
    sys.stdout.flush()
    print('\naaa')
    sleep(0.25)

