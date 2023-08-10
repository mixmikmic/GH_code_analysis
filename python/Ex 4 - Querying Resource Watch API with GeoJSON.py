# Data fetching library
import requests as req
# used below: 'res' stands for 'response'

# File management libraries
import os
import json

# Data manipulation libraries
import pandas as pd
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

# Base URL for getting dataset metadata from RW API
# Metadata = Data that describes Data 
url = "https://api.resourcewatch.org/v1/dataset?sort=slug,-provider,userId&status=saved&includes=metadata,vocabulary,widget,layer"

# page[size] tells the API the maximum number of results to send back
# There are currently between 200 and 300 datasets on the RW API
payload = { "application":"rw", "page[size]": 1000}

# Request all datasets, and extract the data from the response
res = req.get(url, params=payload)
data = res.json()["data"]

#############################################################

### Convert the json object returned by the API into a pandas DataFrame
# Another option: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.io.json.json_normalize.html
datasets_on_api = {}
for ix, dset in enumerate(data):
    atts = dset["attributes"]
    metadata = atts["metadata"]
    layers = atts["layer"]
    widgets = atts["widget"]
    tags = atts["vocabulary"]
    datasets_on_api[atts["name"]] = {
        "rw_id":dset["id"],
        "table_name":atts["tableName"],
        "provider":atts["provider"],
        "date_updated":atts["updatedAt"],
        "num_metadata":len(metadata),
        "metadata": metadata,
        "num_layers":len(layers),
        "layers": layers,
        "num_widgets":len(widgets),
        "widgets": widgets,
        "num_tags":len(tags),
        "tags":tags
    }

# Create the DataFrame, name the index, and sort by date_updated
# More recently updated datasets at the top
current_datasets_on_api = pd.DataFrame.from_dict(datasets_on_api, orient='index')
current_datasets_on_api.index.rename("Dataset", inplace=True)
current_datasets_on_api.sort_values(by=["date_updated"], inplace=True, ascending = False)

# View datasets on the Resource Watch API
current_datasets_on_api.head()

# View all providers of RW data
current_datasets_on_api["provider"].unique()

# Choose only datasets stored on:
## cartodb, csv, gee, featureservice, bigquery, wms, json, rasdaman
provider = "cartodb"
carto_ids = (current_datasets_on_api["provider"]==provider)
carto_data = current_datasets_on_api.loc[carto_ids]

print("Number of Carto datasets: ", carto_data.shape[0])

carto_data.head()

# Store your data in a "data" folder in the same location
# As this notebook
DATA_FOLDER = os.getcwd() + "/data/"

# src: geojson.io
geojson_obj = json.load(open(DATA_FOLDER + "points_and_poly.json"))

geojson_obj

# Template query string used to query RW datasets
query_base = "https://api.resourcewatch.org/v1/query/{}?sql={}"

# Template SQL string used in RW query
sql = "".join(["SELECT * FROM {} WHERE ",
"ST_Intersects({}, ",
"{}.the_geom)"])

# Create the queries for points and polygons in your GeoJSON
def make_point_query(point):
    point_template = "ST_GeomFromText('POINT({})', 4326)"
    
    point_coords = str(point[0]) + " " + str(point[1])
    
    return(point_template.format(point_coords))

def make_poly_query(poly):
    poly_template = "ST_GeomFromText('POLYGON(({}))', 4326)"

    poly_coords = ""
    for ix, point in enumerate(poly):
        if(ix < len(poly)-1):
            poly_coords += str(point[0]) + " " + str(point[1]) + ", "
        else:
            poly_coords += str(point[0]) + " " + str(point[1])

    return(poly_template.format(poly_coords))

for feature in geojson_obj["features"]:
    if feature["geometry"]["type"] == "Point":
        point = feature["geometry"]["coordinates"]
        feature["properties"].update(
            query=make_point_query(point)
        )
    elif feature["geometry"]["type"] == "Polygon":
        poly = feature["geometry"]["coordinates"][0]
        feature["properties"].update(
            query=make_poly_query(poly)
        )

# Pick a dataset from carto_data
dataset = 'Percentage of Urban Population with Access to Electricity'

# Select the Carto table name, and Resource Watch ID (rw_id)
# The rw_id is needed to query the RW API
table_name = carto_data.loc[dataset, "table_name"]
rw_id = carto_data.loc[dataset, "rw_id"]

for feature in geojson_obj["features"]:
    geom = feature["properties"]["query"]
    
    # Use the templates defined above to create/send a query to RW API
    query_sql = sql.format(table_name, geom, table_name)
    query = query_base.format(rw_id, query_sql)    
    res = req.get(query)
    
    # Try, except: useful in python to catch errors,
    # and provide an alternative action if an error occurs
    try:
        data = res.json()["data"]
        total_data = [[elem["country_name"], elem["yr_2014"]] for elem in data]
        feature["properties"].update(
            per_urban_access_to_electricity=total_data
        )
    except:
        feature["properties"].update(
            per_urban_access_to_electricity="No matching data found"
        )

geojson_obj

