import os

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

def get_creds():
    """ Client credentials flow is appropriate for requests that do not require access to a user’s private data. 
    To support the Client Credentials Flow Spotipy provides a class "SpotifyClientCredentials".
    """
    creds = SpotifyClientCredentials(client_id = os.environ['spotipy_id'],
                                     client_secret = os.environ['spotipy_secret'])
    return creds

# get playlists from user
creds = get_creds()
sp = spotipy.Spotify(client_credentials_manager=creds)
playlists = sp.user_playlists('bramvcamp')
print "n playlists=", len(playlists)

# explore structure of playlists 
print type(playlists)
print playlists.keys()
print playlists['items'][0].keys()

playlist_ids = []
for elem in playlists['items'][0:3]:  # explore the first 3 playlists
    print elem['name'], "n tracks=", elem['tracks']['total']

