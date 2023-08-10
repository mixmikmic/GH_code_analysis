import hkvwaporpy as hkv

# request the catalogus
df = hkv.read_wapor.get_catalogus()
df.head()

# show all codes
df[['caption','code']]

ds_code='L2_AET_D'

# get additional info of the dataset given a code and catalogus
df_add = hkv.read_wapor.get_additional_info(df, cube_code=ds_code)
df_add

# select single row of catalogus given a code
df_row = df.loc[df['code'] == ds_code]
df_row.T

# get data availablitiy and corresponding raster id's given date range
df_avail = hkv.read_wapor.get_data_availability(df_row, dimensions_range='[2001-11-01,2017-11-01]')
df_avail.head()

df_locations = hkv.read_wapor.get_locations(filter_value=None)
df_locations.head()

location = 'Awash'

year = df_avail.iloc[0].name
raster_id = df_avail.iloc[0]['raster_id']
location_code = df_locations.loc[df_locations['name'] == location]['code'][0]
print(raster_id)

url = hkv.read_wapor.get_coverage_url(ds_code, year, raster_id, location_code)
print(url)

import requests
from io import BytesIO
import uuid
from osgeo import gdal

resp = requests.get(url)

image_data = BytesIO(resp.content)

filename = uuid.uuid4().hex
mmap_name = "/vsimem/{}".format(filename)
gdal.FileFromMemBuffer(mmap_name, image_data.read())
dataset = gdal.Open(mmap_name)

print(gdal.Info(mmap_name))

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

im = plt.imshow(dataset.ReadAsArray())
plt.colorbar(im)

# query_json = {  
#    "type":"PixelTimeSeries",
#    "params":{  
#       "cube":{  
#          "code":"L2_AGBP_S",
#          "workspaceCode":"WAPOR",
#          "language":"en"
#       },
#       "dimensions":[  
#          {  
#             "code":"SEASON",
#             "values":[  
#                "S1"
#             ]
#          },
#          {  
#             "code":"YEAR",
#             "range":"[2014-09-01,2016-01-01]"
#          }
#       ],
#       "measures":[  
#          "AGBP_S"
#       ],
#       "point":{  
#          "crs":"EPSG:4326",
#          "x":39.40862388968785,
#          "y":10.60703111476445
#       }
#    }
# }

