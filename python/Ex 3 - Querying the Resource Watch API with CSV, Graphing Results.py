# Data fetching library
import requests as req
# used below: 'res' stands for 'response'

# File management library
import os

# Data manipulation libraries
import pandas as pd
pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

# Data visualization library
## Uses Vega-Lite, which can be easily put in websites
from altair import *

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

# src: https://developers.google.com/public-data/docs/canonical/countries_csv
country_points = pd.read_csv(DATA_FOLDER + "country_points.tsv", sep="\t")

country_points.head()

# Template query string used to query RW datasets
query_base = "https://api.resourcewatch.org/v1/query/{}?sql={}"

# Template SQL string used in RW query
sql = "".join(["SELECT * FROM {} WHERE ",
"ST_Intersects({}, ",
"{}.the_geom)"])

# Pick a dataset from carto_data
dataset = 'Percentage of Urban Population with Access to Electricity'

# Select the Carto table name, and Resource Watch ID (rw_id)
# The rw_id is needed to query the RW API
table_name = carto_data.loc[dataset, "table_name"]
rw_id = carto_data.loc[dataset, "rw_id"]

def query_api(row):
    # Construct a Well-Known-Text (WKT) Point string
    # WKT formats points: 'POINT(Latitude Longitude)'
    # https://www.drupal.org/node/511370
    point = "ST_GeomFromText('POINT({} {})', 4326)".format(row.longitude, row.latitude)
    
    # Use the templates defined above to create/send a query to RW API
    query_sql = sql.format(table_name, point, table_name)
    query = query_base.format(rw_id, query_sql)    
    res = req.get(query)

    # Try, except: useful in python to catch errors,
    # and provide an alternative action if an error occurs
    try:
        data = res.json()["data"]
        return(data[0]["yr_2014"])
    except:
        return("No matching data found")

country_points["% Urban Population with Electricity Access"] = pd.Series(query_api(row) for row in country_points.itertuples())

country_points.head()

# Add a plot to show something about it

Chart(country_points).mark_bar().encode(
x=X("% Urban Population with Electricity Access", bin=Bin()),
y="count(*):Q")

