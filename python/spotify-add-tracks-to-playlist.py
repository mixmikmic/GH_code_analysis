import spotipy, spotipy.util as util, pandas as pd, math, time
from keys import spotify_username, spotify_client_id, spotify_client_secret, spotify_redirect_uri

pause = 0.5 #how long to pause between requests
limit = 100 #number of tracks to add per request (max=100)
max_playlist_size = 11000 #max number of tracks spotify allows in a single playlist

df = pd.read_csv('data/playlist_tracks.csv', encoding='utf-8')
print(len(df))
df['playlist_id'].value_counts()

df = df.sort_values(by='duration_ms', ascending=True)

df_unique = df.drop_duplicates(subset='artist_id', keep='first')
df_unique = df_unique.sort_values(by='duration_ms', ascending=False)
print(len(df_unique))
df_unique['playlist_id'].value_counts()

# define scope to work with playlists then get an auth token
scope = 'playlist-read-private playlist-modify-private playlist-modify-public'
token = util.prompt_for_user_token(spotify_username, scope, spotify_client_id, spotify_client_secret, spotify_redirect_uri)

# open a connection to spotify
sp = spotipy.Spotify(auth=token)

# get the user id associated with this username
user = sp.user(spotify_username)
user_id = user['id']

all_tracks = list(df_unique['track_id'].values)
len(all_tracks)

# if length > max_playlist_size, you need multiple playlists
playlists_tracks = []
for n in range(math.ceil(len(all_tracks) / max_playlist_size)):
    offset = n * max_playlist_size
    playlists_tracks.append(all_tracks[offset:offset+max_playlist_size])

current_time = int(time.time())
for playlist_tracks, i in zip(playlists_tracks, range(len(playlists_tracks))):
    
    new_playlist_name = 'Big List {}_{}'.format(current_time, i+1)
    result = sp.user_playlist_create(user=user_id, name=new_playlist_name, public=False)
    playlist_id = result['id']
    print('Created playlist "{}", ID={}'.format(new_playlist_name, playlist_id))
          
    call_count = math.ceil(len(playlist_tracks) / limit)
    tracks_added = []
    for n in range(call_count):
        time.sleep(pause)
        offset = n * limit
        tracks_batch = playlist_tracks[offset:offset+limit]
        snapshot_id = sp.user_playlist_add_tracks(user=user_id, playlist_id=playlist_id, tracks=tracks_batch)
        tracks_added.extend(tracks_batch)
        
    print('  Added {:,} tracks to "{}"'.format(len(tracks_added), new_playlist_name))



