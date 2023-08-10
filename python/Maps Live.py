import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import Normalizer, StandardScaler


from folium import plugins

import matplotlib.pyplot as plt

#sql stuff:
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import psycopg2
get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

#FUNCTIONS:

def get_nbds(new_descp):
    """
    builds a score for each neighborhood given a description as follows:
    ass up the distances
    """
    neighbors = knn.kneighbors(model.transform([new_descp]))
    closest_listings = neighbors[1][0]
    results = train.iloc[closest_listings][["neighbourhood_cleansed"]]
    results["distance"] = neighbors[0][0]

    #invert the distance:
    results["distance"] = results["distance"].max() + 1 - results["distance"]
    nbd_score = results.groupby("neighbourhood_cleansed")["distance"].sum().sort_values(ascending = False)


    nbd_score = pd.concat((nbd_score, nbd_counts), 1)
    nbd_score["weighted_score"] = nbd_score["distance"]/np.log(nbd_score["neighbourhood_cleansed"])

    return nbd_score

def locations_of_best_match(new_descp):
    neighbors = knn.kneighbors(model.transform([new_descp]))
    closest_listings = neighbors[1][0]
    results = train.iloc[closest_listings]
    return results


def draw_point_map(results, nr_pts = 300):
    map_osm = folium.Map(tiles='Cartodb Positron', location = [40.661408, -73.961750])
    #this is stupidly slow:
    for index, row in results[:nr_pts].iterrows():
        folium.CircleMarker(location=[row["latitude"], row["longitude"]], radius=row["latitude"], color = "pink").add_to(map_osm)

    return(map_osm)



def get_heat_map(descp):
    map_osm = folium.Map(tiles='Cartodb Positron', location = [40.7831, -73.970], zoom_start=13)
    results = locations_of_best_match(descp)
    temp = results[["latitude", "longitude"]].values.tolist()
    

    map_osm.add_children(plugins.HeatMap(temp, min_opacity = 0.4, radius = 20, blur = 30,
                                         gradient = return_color_scale(scale_1),
                                         name = descp))
    
    
    folium.LayerControl().add_to(mapa)
    return map_osm

dbname = 'airbnb_db'
username = 'alexpapiu'
con = psycopg2.connect(database = dbname, user = username)
train = pd.read_sql_query("SELECT * FROM location_descriptions", con)

nbd_counts = train["neighbourhood_cleansed"].value_counts()

descp = train[["id", "neighborhood_overview"]]

descp = descp.drop_duplicates()

#MODEL:
model = make_pipeline(TfidfVectorizer(stop_words = "english", min_df = 5, ngram_range = (1,2)),
                      TruncatedSVD(100),
                      Normalizer())

knn = NearestNeighbors(500, metric = "cosine", algorithm = "brute")


X = descp["neighborhood_overview"]
X_proj = model.fit_transform(X)
knn.fit(X_proj)

model

from sklearn.externals import joblib
joblib.dump(model, 'tf_idf_model.pkl') 
knn.dump(knn, 'knn.pkl') 

model = joblib.load('tf_idf_model.pkl') 

descp = "jamaican cusine culture gritty"

nbd_score = get_nbds(descp)

nbd_score["weighted_score"].dropna().sort_values().tail(12).plot(kind = "barh")

results = locations_of_best_match(descp)

get_heat_map("hip urban gritty")

def add_heat_layer(mapa, descp, scale  = scale_2):

    results = locations_of_best_match(descp)
    temp = results[["latitude", "longitude"]].values.tolist()

    mapa.add_children(plugins.HeatMap(temp, min_opacity = 0.4, radius = 30, blur = 30,
                                      gradient = return_color_scale(scale),
                                      name = descp))
    return mapa

scale_2 = ["#f2eff1", "#f2eff1", "#3E4A89", "#31688E", "#26828E", "#1F9E89", "#35B779",
               "#6DCD59", "#B4DE2C", "#FDE725"]

scale_1 = ["#f2eff1", "#f2eff1", "#451077", "#721F81", "#9F2F7F", "#CD4071",
           "#F1605D",  "#FD9567",  "#FEC98D", "#FCFDBF"]

scale_3 = ["#f2eff1", "#f2eff1", "#4B0C6B", "#781C6D", "#A52C60", "#CF4446",
           "#ED6925", "#FB9A06", "#F7D03C", "#FCFFA4"]

def return_color_scale(scale):
    df = pd.Series(scale)
    df.index = np.power(df.index/10, 1/2)
    return df.to_dict()

mapa = get_heat_map("hip cute thrift stores bars")
add_heat_layer(mapa, "gritty authentic")
add_heat_layer(mapa, "chinese", scale = scale_3)

folium.LayerControl().add_to(mapa)

mapa

mapa.save("map_test")

ls

def get_heat_map(descp):
    map_osm = folium.Map(tiles='Cartodb Positron', location = [40.7831, -73.970], zoom_start=13)
    results = locations_of_best_match(descp)
    temp = results[["latitude", "longitude"]].values.tolist()
    

    map_osm.add_children(plugins.HeatMap(temp, min_opacity = 0.4, radius = 30, blur = 30,
                                         gradient = return_color_scale(scale_1),
                                         name = descp))
    
    
    folium.LayerControl().add_to(mapa)
    return map_osm

get_heat_map("cocktail")

danger = train[train.neighborhood_overview.str.contains("dangerous")]

danger.neighborhood_overview.iloc[1]

get_heat_map("fashion")

import folium
map_osm = folium.Map(location=[45.5236, -122.6750])

m.create_map(path='map.html')

map_osm = folium.Map(location=[45.5236, -122.6750])

