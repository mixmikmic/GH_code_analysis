import spotipy, spotipy.util as util, pandas as pd, math, time
from keys import spotify_username, spotify_client_id, spotify_client_secret, spotify_redirect_uri

limit = 50 #max playlists you can retrieve in a single request
pause = 0.5 #how long to pause between requests

# define scope to work with playlists then get an auth token
scope = 'playlist-read-private playlist-modify-private playlist-modify-public'
token = util.prompt_for_user_token(spotify_username, scope, spotify_client_id, spotify_client_secret, spotify_redirect_uri)

# open a connection to spotify
sp = spotipy.Spotify(auth=token)

# get the user id associated with this username
user = sp.user(spotify_username)
user_id = user['id']

def get_playlists(user_id):
    
    response = sp.user_playlists(user=user_id, limit=limit, offset=0)
    playlists = response['items']
    playlists_count = response['total']
    additional_call_count = math.ceil((playlists_count - limit) / limit)
    
    for n in range(additional_call_count):
        time.sleep(pause)
        offset = (n + 1) * limit
        playlists_batch = sp.user_playlists(user=user_id, limit=limit, offset=offset)
        playlists.extend(playlists_batch['items'])

    print('Retrieved {} playlists via {} API calls'.format(len(playlists), additional_call_count + 1))
    return playlists

def make_df(playlists):
    df = pd.DataFrame(playlists)
    drop_cols = ['external_urls', 'href', 'images', 'snapshot_id', 'uri']
    df = df[[col for col in df.columns if col not in drop_cols]]
    df['owner'] = df['owner'].map(lambda x: x['id'])
    df['tracks'] = df['tracks'].map(lambda x: x['total'])
    return df

playlists = get_playlists(user_id)

df = make_df(playlists)
df.head()

