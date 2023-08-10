import requests
import pandas as pd
import geopandas as gpd

get_ipython().run_line_magic('matplotlib', 'inline')

#Build the request and parameters to fetch county features
#  from the NOAA ArcGIS map server end point
stateFIPS = '37' #This is NC

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
respons_js = response.json()

#The 'results' object contains a record for each county returned
results = respons_js['results']
len(results)

#Within each item in the results object are the following items
results[0].keys()

#The 'attributes' item contains the feature attributes
results[0]['attributes']

#And the geometry object contains the shape
results[0]['geometry']

#Create a dataFrame from the results, 
#  keeping just the attributes and geometry objects
df = pd.DataFrame(results,columns=('attributes','geometry'))
df.head()

#Explode the dictionary values into fields
dfCounties = df['attributes'].apply(pd.Series)
dfGeom = df['geometry'].apply(pd.Series)

#Combine the two
dfAll = pd.concat((dfCounties,dfGeom),axis='rows')
dfAll.columns

#Demo creating a shapely polygnon from the JSON ring
rings = dfAll['rings'][0]
print "There is/are {} ring(s)".format(len(rings))
print "There are {} vertices in the first ring".format(len(rings[0]))
print "The first vertex is at {}".format(rings[0][0])

from shapely.geometry import LinearRing
from shapely.geometry import Polygon
ring = rings[0]
r = LinearRing(ring)
s = Polygon(r)

#https://shapely.readthedocs.io/en/latest/manual.html#polygons
from shapely.geometry import Polygon
from shapely.geometry import LinearRing
def polyFromRing(ring):
    r = LinearRing(ring)
    s = Polygon(r)
    return r
dfAll['geometry']=dfAll.apply(lambda x: Polygon(x.rings[0]),axis=1)

gdf=gpd.GeoDataFrame(dfAll)

gdf[gdf['NAME'] == 'Durham'].plot();

gdf.to_csv("counties_{}.csv".format(stateFIPS))

