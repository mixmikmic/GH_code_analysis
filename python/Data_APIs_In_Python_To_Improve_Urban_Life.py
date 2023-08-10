get_ipython().magic('matplotlib inline')
import json
import pandas as pd

# Read and return the "api_key" key from a JSON file
def read_key(fn):
    with open(fn) as json_file:
        data = json.load(json_file)
        return data["api_key"]

# Before running you will need to install the 'census' package with
# pip install census
from census import Census

census_api_key = read_key("census.key")
print(census_api_key[0:10])

florida_fips = '12'
alachua_fips = '001'

census_api = Census(census_api_key)
census_data = census_api.acs5.state_county_tract(
    ('NAME', 'B01003_001E', 'B14006_002E', 'B17009_002E'), 
    florida_fips, alachua_fips, Census.ALL)

print(census_data[0:2]) # just the first two items

poverty_by_tract = pd.DataFrame.from_dict(census_data)
poverty_by_tract.head()

poverty_by_tract['percent'] = poverty_by_tract.apply(
    lambda row: float(row['B17009_002E']) / float(row['B01003_001E']), axis=1)

poverty_by_tract.hist(column="percent")

api_access_point = "https://data.cityofgainesville.org"

method = "resource"

dataset = "9qim-t8hy"
year = "2016"
month = "January"

api_url = ("{0}/{1}/{2}.json?year={3}&month={4}&$limit=3"
    .format(api_access_point, method, dataset, year, month))
print(api_url)

import requests
r = requests.get(api_url)
gru_data = r.json()

print(type(gru_data))
print(gru_data)

print(gru_data[0]["location_1"])

gru_dataframe = pd.read_json(api_url)
gru_dataframe.head(2)

data1 = pd.read_csv("data/32605_00100_0k_50k.csv")
data2 = pd.read_csv("data/32605_00100_50k_100k.csv")
data3 = pd.read_csv("data/32605_00100_100k_200k.csv")
data4 = pd.read_csv("data/32605_00100_200k_300k.csv")
data5 = pd.read_csv("data/32605_00100_300k_999k.csv")


appraiser_data = pd.concat([data1, data2, data3, data4, data5])
appraiser_data.head()

appraiser_data.hist("TotSqFt")

google_api_key = read_key("google.key")

# Before running you will need to install the 'census' package with
# pip install geocoder
import geocoder

g = geocoder.google("1604 NW 21ST AVE GAINESVILLE FL 32605-4062", 
                    key=google_api_key)
g.json

small_data = appraiser_data[appraiser_data["City_Desc"] == "Gainesville"].head()

def run_geocode(address):
    g = geocoder.google(address, key=google_api_key)
    return g.json

small_data["geocode_response"] = small_data.apply(lambda row: run_geocode(" ".join(
            [row["Loc_Address"], row["City_Desc"], "FL", "USA"])), axis=1)

small_data["lat"] = small_data.apply(lambda row: row["geocode_response"]["lat"], axis=1)
small_data["lng"] = small_data.apply(lambda row: row["geocode_response"]["lng"], axis=1)

small_data.head()

# Before running you will need to install the 'basemap' package with
# conda install basemap
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

my_map = Basemap(llcrnrlon=-82.5,llcrnrlat=29.4,urcrnrlon=-82.1,urcrnrlat=29.8,
             resolution='i', projection='tmerc', lat_0 = 29.65, lon_0 = -82.33)
my_map.readshapefile("data/gz_2010_12_140_00_500k", "census_tracts")


x,y = my_map(small_data["lng"].tolist(), small_data["lat"].tolist())

my_map.plot(x, y, 'bo', markersize=7)

plt.show()



