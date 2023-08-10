import os

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util

def get_creds():
    """ Client credentials flow is appropriate for requests that do not require access to a user’s private data. 
    To support the Client Credentials Flow Spotipy provides a class "SpotifyClientCredentials".
    """
    creds = SpotifyClientCredentials(client_id = os.environ['spotipy_id'],
                                     client_secret = os.environ['spotipy_secret'])
    return creds

# connect
creds = get_creds()
sp = spotipy.Spotify(client_credentials_manager=creds)

# get playlists from some user (me in this case)
playlists = sp.user_playlists('bramvcamp')

# count nr playlists (public)
print "n playlists=", len(playlists['items'])

# show first 3 playlists, show count n tracks
playlist_ids = []
for elem in playlists['items'][0:3]:  # explore the first 3 playlists
    print elem['name'], "n tracks=", elem['tracks']['total'] 

def get_token(scope=None):
    '''
    To support the Authorization Code Flow Spotipy provides a utility method "util".
    This creates a token while allowing the user to set the scope of using the api.
    Info on scope: https://developer.spotify.com/web-api/using-scopes/
    
    If credentials are stored as local variables the function will fetch these. For windows: https://tinyurl.com/y78qpmn8
    If not stored locally, the user has to enter the credentials manually.
    
    '''        
    redirect_uri = 'http://example.com/callback/'
    
    try:
        try:
            username = os.environ['spotipy_user']
            client_id = os.environ['spotipy_id']
            client_secret = os.environ['spotipy_secret']
        
        except:  # credentials not stored as system variables
            username = raw_input('user=')
            client_id = raw_input('id=')
            client_secret = raw_input('secret=')
        token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)
    
    except:  # cache file for user already exists
        os.remove('.cache-'+username)
        token = util.prompt_for_user_token(username, scope, client_id, client_secret, redirect_uri)

    return token

# connect
token = get_token(scope='playlist-read-private')
sp = spotipy.Spotify(auth=token)

# get playlists from current user (me in this case)
playlists = sp.current_user_playlists()

# count nr playlists (private)
print "n playlists=", len(playlists['items'])

