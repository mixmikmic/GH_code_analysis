import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
import matplotlib; matplotlib.rcParams['savefig.dpi'] = 144

data = pd.read_csv('../chapter2/data/nyc_data.csv',
                   parse_dates=['pickup_datetime', 'dropoff_datetime'])

pickup = data[['pickup_longitude', 'pickup_latitude']].values
dropoff = data[['dropoff_longitude', 'dropoff_latitude']].values
pickup

print(pickup[3, 1])

pickup[1:7:2, 1:]

lon = pickup[:, 0]
lon

lat = pickup[:, 1]
lat

lon_min, lon_max = (-73.98330, -73.98025)
lat_min, lat_max = ( 40.76724,  40.76871)

in_lon = (lon_min <= lon) & (lon <= lon_max)
in_lon

in_lon.sum()

in_lat = (lat_min <= lat) & (lat <= lat_max)

in_lonlat = in_lon & in_lat
in_lonlat.sum()

np.nonzero(in_lonlat)[0]

lon1, lat1 = dropoff.T

EARTH_R = 6372.8
def geo_distance(lon0, lat0, lon1, lat1):
    """Return the distance (in km) between two points in
    geographical coordinates."""
    # from: http://en.wikipedia.org/wiki/Great-circle_distance
    # and: http://stackoverflow.com/a/8859667/1595060
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    dlon = lon0 - lon1
    y = np.sqrt(
        (np.cos(lat1) * np.sin(dlon)) ** 2
         + (np.cos(lat0) * np.sin(lat1)
         - np.sin(lat0) * np.cos(lat1) * np.cos(dlon)) ** 2)
    x = np.sin(lat0) * np.sin(lat1) +         np.cos(lat0) * np.cos(lat1) * np.cos(dlon)
    c = np.arctan2(y, x)
    return EARTH_R * c

distances = geo_distance(lon, lat, lon1, lat1)

plt.hist(distances[in_lonlat], np.linspace(0., 10., 50))
plt.xlabel('Trip distance (km)')
plt.ylabel('Number of trips')

evening = (data.pickup_datetime.dt.hour >= 19).values

n = np.sum(evening)

n

weights = np.zeros(2 * n)

weights[:n] = -1
weights[n:] = +1

points = np.r_[pickup[evening],
               dropoff[evening]]

points.shape

def lat_lon_to_pixels(lat, lon):
    lat_rad = lat * np.pi / 180.0
    lat_rad = np.log(np.tan((lat_rad + np.pi / 2.0) / 2.0))
    x = 100 * (lon + 180.0) / 360.0
    y = 100 * (lat_rad - np.pi) / (2.0 * np.pi)
    return (x, y)

lon, lat = points.T
x, y = lat_lon_to_pixels(lat, lon)

lon_min, lat_min = -74.0214, 40.6978
lon_max, lat_max = -73.9524, 40.7982

x_min, y_min = lat_lon_to_pixels(lat_min, lon_min)
x_max, y_max = lat_lon_to_pixels(lat_max, lon_max)

bin = .00003
bins_x = np.arange(x_min, x_max, bin)
bins_y = np.arange(y_min, y_max, bin)

grid, _, _ = np.histogram2d(y, x, weights=weights,
                            bins=(bins_y, bins_x))

density = 1. / (1. + np.exp(-.5 * grid))

plt.figure(figsize=(8, 8))
plt.imshow(density,
           origin='lower',
           interpolation='bicubic'
           )
plt.axis('off')
plt.tight_layout()

