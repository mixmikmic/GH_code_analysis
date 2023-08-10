# Data fetching library
import requests as req
# used below: 'res' stands for 'response'

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

