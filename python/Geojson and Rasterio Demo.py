get_ipython().magic('pylab inline')

import os
import json
import rasterio
import rasterio.features
import shapely.geometry
import pandas as pd
from affine import Affine

RASTER_FILE = os.path.join(
    os.path.expanduser('~'), 'bh', 'data', 'satellite',
    'SVDNB_npp_20140201-20140228_75N180W_vcmcfg_'
    'v10_c201507201052.avg_rade9.tif'
)

COUNTIES_GEOJSON_FILE = os.path.join(
    os.path.expanduser('~'), 'bh', 'data',
    'us_counties_5m.json'
)
STATES_TEXT_FILE = os.path.join(
    os.path.expanduser('~'), 'bh', 'data',
    'state.txt'
)

with open(COUNTIES_GEOJSON_FILE, 'r') as f:
    counties_raw_geojson = json.load(f, 'latin-1')

states_df = pd.read_csv(STATES_TEXT_FILE, sep='|').set_index('STATE')
states = states_df['STATE_NAME']

def get_county_name_from_geo_obj(geo_obj):
    """
    Use the NAME and STATE properties of a county's geojson
    object to get a name "state: county" for that county.
    """
    return u'{state}: {county}'.format(
        state=states[int(geo_obj['properties']['STATE'])],
        county=geo_obj['properties']['NAME']
    )

counties_geojson = {
    get_county_name_from_geo_obj(county_geojson): county_geojson
    for county_geojson in counties_raw_geojson['features']
}

print sorted(counties_geojson.keys())[:10]

ny_shape = shapely.geometry.shape(counties_geojson['New York: New York']['geometry'])
print '%r' % ny_shape
ny_shape

lon_min, lat_min, lon_max, lat_max = ny_shape.bounds
print lon_min, lat_min, lon_max, lat_max

raster_file = rasterio.open(RASTER_FILE, 'r')

bottom, left = raster_file.index(lon_min, lat_min)
top, right = raster_file.index(lon_max, lat_max)

raster_window = ((top, bottom+1), (left, right+1))
raster_window

ny_raster_array = raster_file.read(indexes=1, window=raster_window)
ny_raster_array.shape

from matplotlib import pyplot as plt
plt.imshow(ny_raster_array)
plt.show()

raster_file.affine

rfa = raster_file.affine
ny_affine = Affine(
  rfa.a, rfa.b, lon_min,
  rfa.d, rfa.e, lat_max
)

import rasterio.features
ny_mask = rasterio.features.rasterize(
    shapes=[(ny_shape, 0)],
    out_shape=ny_raster_array.shape,
    transform=ny_affine,
    fill=1,
    dtype=np.uint8,
)
plt.imshow(ny_mask)

plt.imshow(ny_raster_array * (1 - ny_mask))

ny_masked = np.ma.array(
    data=ny_raster_array,
    mask=ny_mask.astype(bool)
)
print 'min: {}'.format(ny_masked.min())
print 'max: {}'.format(ny_masked.max())
print 'mean: {}'.format(ny_masked.mean())
print 'standard deviation: {}'.format(ny_masked.std())

