import pandas as pd
from pandas.io.json import json_normalize

df_geojson = pd.read_json("data/in/countries.geo.json")

df_geojson["features"][:5]

# Countries of interest:
countries = [u"C\xf4te d'Ivoire (Ivory Coast)", u'Italy', u'USA', 
                         u'South Africa', u'Philippines', u'Democratic Republic of the Congo', 
                         u'Gabon', u'Sudan (South Sudan)', u'Uganda', u'England', u'Russia']

ds_features = df_geojson["features"]

ds_features[1]["properties"]["name"]

dict_geojson = {}

for i in ds_features:
    dict_geojson[i["properties"]["name"]] = i["geometry"]

dict_geojson.keys()[:10]

for c in countries:
    if c in dict_geojson.keys():
        print c, "is in dict_geojson"
    else:
        print c, "has a different spelling in dict_geojson"

geojson_list = []

u"Ivory Coast" in dict_geojson.keys()

u"South Sudan" in dict_geojson.keys()

u"United Kingdom" in dict_geojson.keys()

# Countries with the same spelling as in the geo.json file
countries2 = [u'Ivory Coast', u'Italy', u'United States of America', 
              u'South Africa', u'Philippines', u'Democratic Republic of the Congo', 
              u'Gabon', u'South Sudan', u'Uganda', u'United Kingdom', u'Russia']

for c in countries2:
    geojson_list.append(dict_geojson[c])

geojson_list[0]

df_eb = pd.read_csv("data/out/ebola-outbreaks-before-2014-coordinates.csv", encoding="utf-8", index_col=False)

df_eb = df_eb.drop(df_eb.columns[0], axis=1)

df_eb.head()

all_countries = list(df_eb["Country name"])

print all_countries[:5]

all_geom = [geojson_list[countries.index(country)] for country in all_countries]

all_geom[0]

df_eb.insert(10, "Geometry (geojson)", all_geom)

df_eb.head()

df_eb.to_csv("data/out/ebola-outbreaks-before-2014-geometry.csv", encoding="utf-8", index_col=False)



