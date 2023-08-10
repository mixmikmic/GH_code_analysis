import pandas as pd 
import spotipy 
import json
import time
import re

sp = spotipy.Spotify() 
from spotipy.oauth2 import SpotifyClientCredentials 
cid ="73ae7d6ba1134544beb303a0bfade356" 
secret = "4e7f26c2bc44499ea80ab14258a37fb3" 
client_credentials_manager = SpotifyClientCredentials(client_id=cid, client_secret=secret) 
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager) 
sp.trace=False 

# this works ;)
def get_artist(name):
    results = sp.search(q='artist:' + name, type='artist')
    items = results['artists']['items']
    if len(items) > 0:
        return items[0]
    else:
        return None

def show_album_tracks(album):
    tracks = []
    results = sp.album_tracks(album['id'])
    tracks.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        tracks.extend(results['items'])
    for track in tracks:
        #print('Track Name: ', track['name'])
        tn = track['name']
        track_name.append(tn)
        #print(track['uri'])
        tu = track['uri']
        track_uri.append(tu)
        artist_rep.append(artist['name']) #add artist name to df
        #print()

def show_artist_albums(id):
    albums = []
    results = sp.artist_albums(artist['id'], album_type='album')
    albums.extend(results['items'])
    while results['next']:
        results = sp.next(results)
        albums.extend(results['items'])
    unique = set()  # skip duplicate albums
    for album in albums:
        name = album['name'].lower()
        
        if not name in unique:  
            #print('Album Name:',name)
            unique.add(name)
            album_name.append(name)  # make list of albums
            show_album_tracks(album)

def show_artist(artist):
    an = artist['name']
    print(artist['name'])
    artist_sp_name.append(an)    # make list of artist names


artists80 = ['Bon Jovi','Fleetwood Mac','Rod Stewart','Michael Jackson','Rush','Elton John','Bruce Springsteen',
             'Aerosmith','Paul McCartney','Depeche Mode','Cyndi Lauper', 'David Bowie','Queen' 
              ]

artist_sp_name = []
album_name = []
track_name = []
track_uri = []
artist_rep = []
for star in artists80:
    try:
        artist = get_artist(star)
    except:
        print("Connection refused by the server..")
        time.sleep(5)
        print("Was a nice sleep, now let me continue...")
        continue        
    try:
        show_artist(artist)
    except:
        print("Connection refused by the server..")
        time.sleep(5)
        print("Was a nice sleep, now let me continue...")
        continue
    try:
        show_artist_albums(artist)
    except:
        print("Connection refused by the server..")
        time.sleep(5)
        print("Was a nice sleep, now let me continue...")
        continue
    

tracks = pd.DataFrame()
tracks['uri']= track_uri
tracks['name'] = track_name
tracks['artist_rep'] = artist_rep

tracks.head()

features_df = pd.DataFrame(columns = ['acousticness','analysis_url','danceability','duration_ms','energy','id','instrumentalness','key',
                  'liveness','loudness','mode','speechiness','tempo','time_signature','track_href','type','uri','valence'])
for tid in tracks['uri']:
    try:
        features = sp.audio_features(tid) 
        features_df = features_df.append(features)
    except:
        print("Connection refused by the server..")
        time.sleep(5)
        print("Was a nice sleep, now let me continue...")
        continue

features_df.head()

track_features =  pd.merge(tracks,features_df, on='uri', how='inner')

track_features.head()

track_features.shape

# this function cleans the end of a track (space hyphen space)
def clean_track(track_features):
    new_track = re.split(r"\b - \b",track_features['name'], 1)[0]
    return new_track

# apply function and create new column
track_features['name_clean']  = track_features.apply(clean_track, axis=1)

track_features.head()

# remove dupes 
track_features = track_features.drop_duplicates(subset=['name_clean', 'artist_rep'])

track_features.shape

# download list:
filename = 'spotify.csv'
track_features.to_csv(filename, index=False, encoding='utf-8')



