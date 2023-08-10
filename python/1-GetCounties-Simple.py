import requests
import pandas as pd
import geopandas as gpd

from shapely.geometry import LinearRing
from shapely.geometry import Polygon

get_ipython().run_line_magic('matplotlib', 'inline')

#Specify the state FIPS to extact
stateFIPS = '37' #This is NC

#Build the request and parameters to fetch county features
#  from the NOAA ArcGIS map server end point
url = 'https://nowcoast.noaa.gov/arcgis/rest/services/nowcoast/mapoverlays_political/MapServer/find'
params = {'searchText':stateFIPS,
          'contains':'true',
          'searchFields':'STATEFP',
          'sr':'',
          'layers':'2',
          'layerDefs':'',
          'returnGeometry':'true',
          'maxAllowableOffset':'',
          'geometryPrecision':'',
          'dynamicLayers':'',
          'returnZ':'false',
          'returnM':'false',
          'gdbVersion':'',
          'returnUnformattedValues':'false',
          'returnFieldName':'false',
          'datumTransformations':'',
          'layerParameterValues':'',
          'mapRangeValues':'',
          'layerRangeValues':'',
          'f':'json'}

#Fetch the data
response = requests.get(url,params)

#Convert to a JSON object (i.e. a dictionary)
results = response.json()['results']

#Create a dataFrame from the results, 
#  keeping just the attributes and geometry objects
df = pd.DataFrame(results,columns=('attributes','geometry'))

#Explode the dictionary values into fields
dfCounties = df['attributes'].apply(pd.Series)
dfGeom = df['geometry'].apply(pd.Series)

#Combine the two
dfAll = pd.concat((dfCounties,dfGeom),axis='rows')

#Define a function to convert a JSON 'ring' to a shapely polygon 
def polyFromRing(ring):
    r = LinearRing(ring)
    s = Polygon(r)
    return r

#Iterate through all records and create a 'geometry object'
dfAll['geometry']=dfAll.apply(lambda x: Polygon(x.rings[0]),axis=1)

#Convert the pandas dataFrame to a geopandas data frame
gdf=gpd.GeoDataFrame(dfAll)

#Write to a csv file
gdf.to_csv("../Data/counties_{}.csv".format(stateFIPS))

