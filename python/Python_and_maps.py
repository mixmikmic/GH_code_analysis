import json

from tweepy import API
from tweepy import OAuthHandler
from tweepy import Cursor

import folium

consumer_key    = 'XXXXXXXXXXXXXXXXXXXXXXXX'
consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXX' 
access_token    = 'XXXXXXXXXXXXXXXXXXXXXXXX'
access_secret   = 'XXXXXXXXXXXXXXXXXXXXXXXX'

def get_twitter_auth():
    """Setup Twitter Authentication.
    
    Return: tweepy.OAuthHandler object
    """
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    return auth
    
def get_twitter_client():
    """Setup Twitter API Client.
    
    Return: tweepy.API object
    """
    auth = get_twitter_auth()
    client = API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True, compression=True)
    return client

client = get_twitter_client()

with open('home_timeline.jsonl','w') as f:
    for page in Cursor(client.home_timeline, count=200).pages(4): # limit of 800 for you
        for status in page:
            f.write(json.dumps(status._json)+'\n')
            

tweets = 'home_timeline.jsonl'    # Contains tweets
geo_tweets = 'home.geo.json'      # Output file

with open(tweets,'r') as f:
    geo_data = {
        "type": "FeatureCollection",
        "features": [],
    }
    for line in f:
        tweet = json.loads(line)
        try:
            if tweet['coordinates']:
                geo_json_feature  = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": tweet['coordinates']['coordinates'],
                    },
                    "properties": {
                        "text": tweet['text'],
                        "created_at": tweet['created_at']
                    },
                }
                geo_data['features'].append(geo_json_feature)
        except KeyError:
            # json doc is not a tweet
            continue
                
with open(geo_tweets, 'w') as f:
    f.write(json.dumps(geo_data, indent=4))

def make_map(geojson_file, map_file):
    # Create folium map centered at (latitude, longitude)
    tweet_map = folium.Map(location=[50,-50], zoom_start=2)
    # In case Tweets get too clustered
    marker_cluster = folium.MarkerCluster().add_to(tweet_map)
    
    geodata = json.load(open(geojson_file))
    for tweet in geodata['features']:
        tweet['geometry']['coordinates'].reverse()
        marker = folium.Marker(tweet['geometry']['coordinates'], popup=tweet['properties']['text'])
        marker.add_to(marker_cluster)
        
    tweet_map.save(map_file)

make_map(geo_tweets, 'example.html')

from IPython.display import IFrame
IFrame('example.html', width=700, height=350)



