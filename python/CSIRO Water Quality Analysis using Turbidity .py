from pprint import pprint
from datetime import datetime
import xarray as xr

import matplotlib
import matplotlib.image
get_ipython().magic('matplotlib inline')

import datacube
from datacube.api import API, geo_xarray
from datacube.analytics.analytics_engine import AnalyticsEngine
from datacube.execution.execution_engine import ExecutionEngine
from datacube.analytics.utils.analytics_utils import plot

print('This example runs on Data Cube v2/{}.'.format(datacube.__version__))

dc_a = AnalyticsEngine()
dc_e = ExecutionEngine()
dc_api = API()

print(dc_api.list_field_values('product')) # 'LEDAPS' should be in the list
print(dc_api.list_field_values('platform')) # 'LANDSAT_5' should be in the list

query = {
    'product': 'LEDAPS',
    'platform': 'LANDSAT_5',
}
descriptor = dc_api.get_descriptor(query, include_storage_units=False)
pprint(descriptor)

dimensions = {
    'x': {
        'range': (140, 141)
    },
    'y': {
        'range': (-35.5, -36.5)
    },
    'time': {
        'range': (datetime(2011, 10, 17), datetime(2011, 10, 18))
    }
}

red = dc_a.create_array(('LANDSAT_5', 'LEDAPS'), ['band3'], dimensions, 'red')
green = dc_a.create_array(('LANDSAT_5', 'LEDAPS'), ['band2'], dimensions, 'green')
blue = dc_a.create_array(('LANDSAT_5', 'LEDAPS'), ['band1'], dimensions, 'blue')

blue_result = dc_a.apply_expression([blue], 'array1', 'blue')
dc_e.execute_plan(dc_a.plan)
plot(dc_e.cache['blue'])

turbidity = dc_a.apply_expression([blue, green, red], '(array1 + array2 - array3) / 2', 'turbidity')

dc_e.execute_plan(dc_a.plan)
plot(dc_e.cache['turbidity'])

result = dc_e.cache['turbidity']['array_result']['turbidity']
reprojected = datacube.api.geo_xarray.reproject(result.isel(time=0), 'EPSG:3577', 'WGS84')

pprint(reprojected)

reprojected.plot.imshow()

matplotlib.image.imsave('turbidity.png', reprojected)

map(float, (reprojected.x[0], reprojected.x[-1], reprojected.y[0], reprojected.y[-1]))



