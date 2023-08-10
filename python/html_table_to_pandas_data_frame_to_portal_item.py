import pandas as pd

df = pd.read_html("https://en.wikipedia.org/wiki/Number_of_guns_per_capita_by_country")[0]

df.columns = df.iloc[0]
df = df.reindex(df.index.drop(0))

df.head()

df.iloc[0,2] = 112.6

df.dtypes

converted_column = pd.to_numeric(df["Guns per 100 Residents"], errors = 'coerce')
df['Guns per 100 Residents'] = converted_column
df.head()

from arcgis.gis import GIS
import json

gis = GIS("https://www.arcgis.com", "arcgis_python", "P@ssword123")

fc = gis.content.import_data(df, {"CountryCode":"Country"})

map1 = gis.map('UK')

map1

map1.add_layer(fc, {"renderer":"ClassedSizeRenderer",
               "field_name": "Guns_per_100_Residents"})

item_properties = {
    "title": "Worldwide gun ownership",
    "tags" : "guns,violence",
    "snippet": " GSR Worldwide gun ownership",
    "description": "test description",
    "text": json.dumps({"featureCollection": {"layers": [dict(fc.layer)]}}),
    "type": "Feature Collection",
    "typeKeywords": "Data, Feature Collection, Singlelayer",
    "extent" : "-102.5272,-41.7886,172.5967,64.984"
}

item = gis.content.add(item_properties)

search_result = gis.content.search("Worldwide gun ownership")
search_result[0]

