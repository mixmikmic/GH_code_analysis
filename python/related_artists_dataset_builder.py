import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import numpy as np
import pandas as pd
from IPython.display import clear_output
from IPython.display import display

# Establish a session and instantiate it
client_credentials_manager = SpotifyClientCredentials(client_id='ccae2e7a3a0f48088d24563289eb71d0',
                                                      client_secret='0204fbff5f2c438a816b7c01ede9d9ce')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

artist_name = 'Jorge Drexler'
related_artists = []
result = sp.search(q='artist:' + artist_name, type='artist')
try:
    name = result['artists']['items'][0]['name']
    uri = result['artists']['items'][0]['uri']

    related = sp.artist_related_artists(uri)
    print('Related artists for', name)
    for artist in related['artists']:
        related_artists.append(artist['name'])
    print(related_artists)
except:
    print("Error!")

related_artists[2] = 'Fito Páez'
related_artists[-3] = 'Seru Giran'
related_artists[12] = 'Sui Generis (AR)'

import requests
import json
import requests
from bs4 import BeautifulSoup
import re

CLIENT_ACCESS_TOKEN = "lTv0ZCyw1mkXQ7_9bZ9GmEdc9vPh2_VQzhFCpERSOYiFA1fXjEVrI8BY-D9_k6E2"
BASE_URI = "https://api.genius.com"

#from: https://gist.github.com/imdkm/a60247b59ff1881fa4bb8846a9b44c96
# send request and get response in json format.
def _get(path, params=None, headers=None):

    # generate request URL
    requrl = '/'.join([BASE_URI, path])
    token = "Bearer {}".format(CLIENT_ACCESS_TOKEN)
    if headers:
        headers['Authorization'] = token
    else:
        headers = {"Authorization": token}

    response = requests.get(url=requrl, params=params, headers=headers)
    response.raise_for_status()

    return response.json()

artists_by_ids = {}
for artist_name in related_artists:
    print("searching " + artist_name + "'s artist id. \n")

    # find artist id from given data.
    find_id = _get("search", {'q': artist_name})
    for hit in find_id["response"]["hits"]:
       if hit["result"]["primary_artist"]["name"] == artist_name:
           artist_id = hit["result"]["primary_artist"]["id"]
           break
    
    artists_by_ids[artist_id] = artist_name
    print("-> " + artist_name + "'s id is " + str(artist_id) + "\n")

print(len(artists_by_ids))
print(len(set(list(artists_by_ids.keys()))))

def get_artist_songs(artist_id):

    current_page = 1
    next_page = True
    songs = []

    while next_page:
        path = "artists/{}/songs/".format(artist_id)
        params = {'page': current_page}
        data = _get(path=path, params=params)
        page_songs = data['response']['songs']
        if page_songs:
            songs += page_songs
            current_page += 1
        else:
            next_page = False

    return songs

def scrape_lyrics(url):

    response = requests.get(url)
    html = response.text

    soup = BeautifulSoup(html, 'html.parser')
    lyrics = soup.find("div", {"class": "lyrics"})
    lyrics = cleanhtml(str(lyrics))
    return lyrics

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


##### COMPLETE THIS LIST!
#artists_downloaded_already = ['Kevin Johansen', 'Pedro Guerra', 'Fito Páez']
### OR
df = pd.read_csv('data/related_raw_v1.csv', sep=';')
artists_downloaded_already = df.artist.unique().tolist()

with open('data/related_raw_v1.csv', 'a') as f:
    for an_id in artists_by_ids.keys():
        if artists_by_ids[an_id] in artists_downloaded_already:
            continue
        songs = get_artist_songs(an_id)
        songs = [song for song in songs if song['primary_artist']['id'] == an_id]

        for idx, song in enumerate(songs):
            clear_output(wait=True)
            print('Looking for %s with id %s' % (str(artists_by_ids[an_id]),str(an_id)))
            print('Song %s/%s' % (str(idx),str(len(songs))))
            url = song['url']
            lyrics = scrape_lyrics(url)
            song['lyrics'] = lyrics
            song['artist'] = artists_by_ids[an_id]
        df = pd.read_json(json.dumps(songs))
        df.to_csv(f, header=False, sep=';', index=False)
        artists_downloaded_already.append(artists_by_ids[an_id])
        display(df.tail(10))

df = pd.read_csv('data/related_raw_v1.csv', sep=';')
df.sample(10)

df = df[['artist', 'title', 'lyrics']]
df['lyrics'] = df['lyrics'].str.replace('\n', ' ')
df.sample(10)

df.to_csv('data/related_lyrics.csv', sep=';',index=False)

df



