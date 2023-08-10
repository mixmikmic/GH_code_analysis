import spotipy, spotipy.util as util, pandas as pd, math, time
from keys import spotify_username, spotify_client_id, spotify_client_secret, spotify_redirect_uri

# define the playlist ids here
playlist_ids = ['2UfBZ8aRbMniK0052b5uYb', '25suCiMhFOWCWLG2JLfhot']
limit = 100 #number of tracks to download per request (max=100)
pause = 0.5 #how long to pause between requests

# define scope to work with playlists then get an auth token
scope = 'playlist-read-private playlist-modify-private playlist-modify-public'
token = util.prompt_for_user_token(spotify_username, scope, spotify_client_id, spotify_client_secret, spotify_redirect_uri)

# function to determine how many tracks total are in a playlist
def get_playlist_track_count(playlist_id):
    playlist = sp.user_playlist(user=user_id, playlist_id=playlist_id)
    track_count = playlist['tracks']['total']
    return track_count

# function to iteratively get all the tracks from a playlist
def get_playlist_tracks_details(playlist_id, user_id, track_count, limit):
    playlist_tracks = []
    call_count = math.ceil(track_count / limit)
    for n in range(call_count):
        time.sleep(pause)
        offset = n * limit
        tracks_batch = sp.user_playlist_tracks(user=user_id, playlist_id=playlist_id, fields=None, limit=limit, offset=offset)
        for item in tracks_batch['items']:
            item['playlist_id'] = playlist_id
        playlist_tracks.extend(tracks_batch['items'])
    return playlist_tracks

# open a connection to spotify
sp = spotipy.Spotify(auth=token)

# get the user id associated with this username
user = sp.user(spotify_username)
user_id = user['id']

# for each playlist id in the list, get the details of all the tracks in it
tracks = []
for playlist_id, n in zip(playlist_ids, range(len(playlist_ids))):
    
    print('playlist {} of {}: {}'.format(n+1, len(playlist_ids), playlist_id))
    
    track_count = get_playlist_track_count(playlist_id)
    print('  {:,} tracks in playlist {}'.format(track_count, playlist_id))
    
    new_tracks = get_playlist_tracks_details(playlist_id, user_id, track_count, limit)
    tracks.extend(new_tracks)
    print('  {:,} tracks retrieved from API'.format(len(new_tracks)))

# for each downloaded track, extract the details to an object and append to list
track_objects = []
for track in tracks:
    track_obj = {}
    track_obj['album_name'] = track['track']['album']['name']
    track_obj['album_id'] = track['track']['album']['id']
    track_obj['artist_name'] = track['track']['artists'][0]['name']
    track_obj['artist_id'] = track['track']['artists'][0]['id']
    track_obj['track_name'] = track['track']['name']
    track_obj['track_id'] = track['track']['id']
    track_obj['duration_ms'] = track['track']['duration_ms']
    track_obj['playlist_id'] = track['playlist_id']
    track_objects.append(track_obj)

# create a dataframe from the list of objects
df = pd.DataFrame(track_objects)
print(len(df))
df.head()

# save the dataframe to csv
filename = 'data/playlist_tracks.csv'
df.to_csv(filename, index=False, encoding='utf-8')



