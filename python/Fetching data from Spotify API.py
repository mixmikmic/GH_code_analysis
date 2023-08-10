import pandas as pd
from pandas.io.json import json_normalize
import requests
import json
from requests.auth import HTTPBasicAuth

get_ipython().magic('store -r CLIENT_ID')
get_ipython().magic('store -r CLIENT_SECRET')
get_ipython().magic('store -r REDIRECT_URI')

def api_request():
    auth = requests.post('https://accounts.spotify.com/api/token',data={'grant_type':'client_credentials'}, auth=HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET))
    if auth.status_code == 200:
        auth_token = auth.json()['access_token']
    else:
        output = 'failed'
    return auth_token 

auth_token = api_request()

# Enter the playlist name 
playlist_name = 'Classics 1000'

req = requests.get(url='https://api.spotify.com/v1/search', params= {'q':playlist_name, 'type':'playlist'}, 
                   headers={"Authorization": "Bearer %s" %auth_token})
resp = req.json() 
playlist_df = json_normalize(resp['playlists']['items'])
#playlist_df.columns
playlist_df = playlist_df[['name', 'id','owner.display_name', 'tracks.total', 'owner.id']]
playlist_df.columns = [c.replace('.', '_') for c in playlist_df.columns]

playlist_df

# enter index playlist
n = 1
name = playlist_df['name'][n]
total = playlist_df['tracks_total'][n]
print 'Playlist name: %s' % name
print 'Total numeber of tracks: %d' % total

## we need to support pagination
def get_playlist_tracks():
    #columns=['track.id','track.artists', 'track.name' ]
    playlist_tracks=pd.DataFrame()
    limit=50 #set the number of albums to retrieve from API, 50 is the max
    offset=0 #The index of the first album to return. Default: 0 (i.e., the first album)
    total=limit
    while total> offset:
        req = requests.get(url='http://api.spotify.com/v1/users/{user_id}/playlists/{id}/tracks'.format(user_id=playlist_df['owner_id'][n], id=playlist_df['id'][n]),
                 params= {'limit':limit, 'offset':offset}, headers={"Authorization": "Bearer %s" %auth_token})
        resp_albums = req.json()
        playlist_tracks = playlist_tracks.append(json_normalize(resp_albums['items']), ignore_index=True)
        ##read the total from response and update offset
        total = playlist_df['tracks_total'][n]  #resp_albums['total']
        offset = offset + limit 
        print ('TOTAL: %d, OFFSET: %d' % (total, offset))
    return playlist_tracks 

tracks_df = get_playlist_tracks()
#df.columns

tracks_df.columns = [c.replace('.', '_') for c in tracks_df.columns]
tracks_df.head()

tracks_df["artist_name"]=tracks_df.track_artists.apply(lambda x: x[0]["name"])
tracks_df=tracks_df.drop(["track_artists"], axis=1)

tracks_df = tracks_df[['track_name','artist_name', 'track_id', 'track_popularity', 'track_duration_ms']]
tracks_df.head()

tracks_df=tracks_df.drop_duplicates(['track_id'])
tracks_df['id'] = tracks_df['track_id']
tracks_df = tracks_df

def get_audio_features():
    auth_token = api_request()    
    ##using access token you can now autorize and get audio features for a track 
    track_audio_features= list(map(lambda x: requests.get('https://api.spotify.com/v1/audio-features/{track_id}'.format(track_id=x), 
                                                          headers={"Authorization": "Bearer %s" %auth_token}).json(), tracks_df.track_id))

    return json_normalize(track_audio_features)

features_df = get_audio_features()
features_df.head()

features_df['track_id'] = features_df['id']
features_df.tail()

result = pd.merge(features_df, tracks_df, how='left', on=['track_id', 'track_id'])
result.head()

results = pd.DataFrame(result)
results.columns

df = results[['acousticness', 'instrumentalness', 'loudness', 'tempo','danceability', 'liveness' ,'valence', 'energy', 'speechiness']]
df.describe()

df.to_csv('playlist_data.csv', encoding='utf-8', index=False)







