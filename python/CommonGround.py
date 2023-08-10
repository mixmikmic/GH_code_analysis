# Import dependencies
import json
from dotenv import load_dotenv
from os.path import join, dirname
import os
import requests
import pandas as pd
import numpy as np
from collections import Counter
import watson_developer_cloud
import watson_developer_cloud.natural_language_understanding.features.v1 as features

# Load credentials for Watson and Foursquare
load_dotenv('.env')

from User import Users

A = Users('a', 'Food')
A.eval_user()
A.concepts_for_all_relevant_visits()
A.all_concepts()
ra = A.pool_concepts()
ma = A.mean_concepts(ra)

B = Users('y', 'Food')
B.eval_user()
B.concepts_for_all_relevant_visits()
B.all_concepts()
rb = B.pool_concepts()
mb = B.mean_concepts(rb)

for i in ma:
    print("{}\t{}".format(i, ma[i]))
print()

for i in mb:
    print("{}\t{}".format(i, mb[i]))

from __future__ import print_function
from geopy.geocoders import Nominatim # module to convert an address into latitude and longitude values
import requests # library to handle requests
import pandas as pd # library for data analsysis
import numpy as np # library to handle data in a vectorized manner
import random # library for random number generation
import folium
import matplotlib.cm as cm
#from shapely.geometry import MultiPoint

# tranforming json file into a pandas dataframe library
from pandas.io.json import json_normalize

address_U1 = "176 34th Street Brooklyn, NY"
address_U2 = "1660 Madison Avenue New York, NY"# Geolocator translate addrss into lat, lon
geolocator = Nominatim()
location_U1 = geolocator.geocode(address_U1)
latitude_U1 = location_U1.latitude
longitude_U1 = location_U1.longitude
location_U2 = geolocator.geocode(address_U2)
latitude_U2 = location_U2.latitude
longitude_U2 = location_U2.longitude
print (latitude_U1, longitude_U1, latitude_U2, longitude_U2)

# Multipoint has a function called centroid, that find the midpoint among any number of points listed
points = MultiPoint([(latitude_U1, longitude_U1), (latitude_U2, longitude_U2)])
latitude = points.centroid.x
longitude = points.centroid.y
print (points.centroid) #True centroid, not necessarily an existing point on map

latitude = 40.72604815

longitude = -73.97559250420741

map_nyc = folium.Map(location=[40.765937,-73.977304], zoom_start=11)
folium.CircleMarker([latitude_U1, longitude_U1], color = 'blue',
                   radius = 7).add_to(map_nyc)
folium.CircleMarker([latitude_U2, longitude_U2], color = 'blue',
                   radius = 7).add_to(map_nyc)
folium.CircleMarker([latitude, longitude], color = 'green',
                   radius = 7).add_to(map_nyc)
map_nyc

foursquare_client_id=os.environ.get('FOURSQUARE_CLIENT_ID')

foursquare_client_secret=os.environ.get('FOURSQUARE_CLIENT_SECRET')

# Foursquare credentials
CLIENT_ID = foursquare_client_id;
CLIENT_SECRET = foursquare_client_secret;
VERSION = "20170511"
LIMIT = 30

# these ids help us to find the class of places we want to
food_id = "4d4b7105d754a06374d81259"
bars_id = "4d4b7105d754a06376d81259"

radius = 500 # range in meters
category_id = food_id
price = "2,3" # price range. 1 very cheap, 4 very expensivecategory_id = food_id
url="https://api.foursquare.com/v2/venues/explore?client_id={}&client_secret={}&ll={},{}&v={}&radius={}&categoryId={}&limit={}&price={}".format(CLIENT_ID, CLIENT_SECRET, latitude, longitude, VERSION, radius, category_id, LIMIT, price)
results = requests.get(url).json()

items = results["response"]["groups"][0]["items"]
items[0]["venue"].keys()

# function that extracts the category of the venue
def get_category_type(row):
   try:
       categories_list = row["categories"]
   except:
       categories_list = row["venue.categories"]
       
   if len(categories_list) == 0:
       return None
   else:
       return categories_list[0]["name"].encode('ascii',errors='ignore')

dataframe = json_normalize(items) # flatten JSON# filter columns
filtered_columns = ['venue.name', 'venue.url', 'venue.categories'] + ["venue.rating"] +                   ["venue.id"] + ['venue.hours.isOpen'] + ['venue.price.tier'] +                   [col for col in dataframe.columns if col.startswith('venue.location.')]
                 
dataframe_filtered = dataframe.ix[:, filtered_columns]# filter the category for each row
dataframe_filtered['venue.categories'] = dataframe_filtered.apply(get_category_type, axis=1)# clean columns
dataframe_filtered.columns = [col.split(".")[-1] for col in dataframe_filtered.columns]# filter just open places
open_places = dataframe_filtered[dataframe_filtered['isOpen'] == True]
dataframe_filtered.head(10)

list_id = list(dataframe_filtered['id'].astype(str))
print ('number of places found:', len(list_id))
print ('radius: %d m' %radius)
print ('price class:', price)
list_id[:10]

venues_id = list_id

def pull_foursquare_json(venue_id):
    VERSION = "20170511"
    foursquare_client_id=os.environ.get('FOURSQUARE_CLIENT_ID')
    foursquare_client_secret=os.environ.get('FOURSQUARE_CLIENT_SECRET')
    url="https://api.foursquare.com/v2/venues/{}/tips?client_id={}&client_secret={}&v={}&limit=150".format(venue_id, foursquare_client_id, foursquare_client_secret, VERSION)
    json_f = requests.get(url).json()
    return json_f

def tips_list(json_file):
    try:
        num_tips = json_file['response']['tips']['count']
        return [json_file['response']['tips']['items'][k]['text'] for k in range(num_tips)]
    except:
        return [""]

def combine_u_prefs(u1_dict, u2_dict, k):
    u1_counter = Counter(u1_dict)
    u2_counter = Counter(u2_dict)
    u_prefs = u1_counter + u2_counter
    # Pull k highest values
    u_prefs_top = dict(u_prefs.most_common(k))
    vallist = [val for val in u_prefs_top.values()]
    factor = np.median(vallist)
    normed_val = [val/factor for val in vallist]
    topic_idx = [key for key in u_prefs_top.keys()]
    pref_vec = pd.Series(data=normed_val, index=topic_idx)
    return pref_vec, topic_idx

pref_vec, topic_idx = combine_u_prefs(ma, mb, 20)

def construct_matrix(venue_ids, topic_idx):
    empty_matrix = pd.DataFrame(index=venue_ids, columns=topic_idx)
    return empty_matrix

empty_mat = construct_matrix(venues_id, topic_idx)

def sentiment(tips):
    # Helper function to return text sentiment analysis
    # Load Watson credentials
    username=os.environ.get('NLU_USERNAME')
    password = os.environ.get('NLU_PASSWORD')
    nlu = watson_developer_cloud.NaturalLanguageUnderstandingV1(version='2017-02-27',
        username=username, password=password)
    output = nlu.analyze(text=tips, features=[features.Sentiment()])
    return output['sentiment']['document']['score']

def fill_sentiment_matrix(mat):
    for j in range(mat.shape[0]):
        venue_id = mat.index[j]
        json_f = pull_foursquare_json(venue_id)
        tips = tips_list(json_f)
        for k in range(mat.shape[1]):
            topic = mat.columns[k]
            score = np.median([sentiment(tip) for tip in tips if topic in tip])
            mat.loc[venue_id, topic] = score
    return mat.fillna(0)

sent_mat = fill_sentiment_matrix(empty_mat)

def recommend(score_mat, user_vec, venues_ids, top_n):
    score_vec = pd.Series(np.dot(score_mat, user_vec), index=venues_ids)
    return score_vec.sort_values(ascending=False)[:top_n].index.values

recommend(sent_mat, pref_vec, venues_id, 5)

recom = ['3fd66200f964a5203ce51ee3', '3fd66200f964a5203fe51ee3',
       '515392c4e4b0f03a43c69fce', '4ae1b55df964a520df8621e3',
       '49e8896cf964a5204e651fe3']

# Recommendation list
recommended = dataframe_filtered[dataframe_filtered['id'].isin(recom)]

recommended

