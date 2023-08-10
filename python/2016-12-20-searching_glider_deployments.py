import requests

url = 'http://data.ioos.us/gliders/providers/api/deployment'

response = requests.get(url)

res = response.json()

print('Found {0} deployments!'.format(res['num_results']))

deployments = res['results']

deployment = deployments[-1]

deployment

import iris


iris.FUTURE.netcdf_promote = True


# Get this specific glider because it looks cool ;-)
for deployment in deployments:
    if deployment['name'] == 'sp064-20161214T1913':
        url = deployment['dap']

cubes = iris.load_raw(url)

print(cubes)

import numpy as np
import numpy.ma as ma
import seawater as sw
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def distance(x, y, units='km'):
    if ma.isMaskedArray(x):
        x = x.filled(fill_value=np.NaN)
    if ma.isMaskedArray(y):
        y = y.filled(fill_value=np.NaN)
    dist, pha = sw.dist(x, y, units=units)
    return np.r_[0, np.cumsum(dist)]


def apply_range(cube_coord):
    if isinstance(cube_coord, iris.cube.Cube):
        data = cube_coord.data.squeeze()
    elif isinstance(cube_coord, (iris.coords.AuxCoord, iris.coords.Coord)):
        data = cube_coord.points.squeeze()

    actual_range = cube_coord.attributes.get('actual_range')
    if actual_range is not None:
        vmin, vmax = actual_range
        data = ma.masked_outside(data, vmin, vmax)
    return data


def plot_glider(cube, cmap=plt.cm.viridis,
                figsize=(9, 3.75), track_inset=False):

    data = apply_range(cube)
    x = apply_range(cube.coord(axis='X'))
    y = apply_range(cube.coord(axis='Y'))
    z = apply_range(cube.coord(axis='Z'))
    t = cube.coord(axis='T')
    t = t.units.num2date(t.points.squeeze())

    fig, ax = plt.subplots(figsize=figsize)
    dist = distance(x, y)
    z = ma.abs(z)
    dist, _ = np.broadcast_arrays(dist[..., np.newaxis],
                                  z.filled(fill_value=np.NaN))
    dist, z = map(ma.masked_invalid, (dist, z))
    cs = ax.pcolor(dist, z, data, cmap=cmap, snap=True)
    kw = dict(orientation='horizontal', extend='both', shrink=0.65)
    cbar = fig.colorbar(cs, **kw)

    if track_inset:
        axin = inset_axes(
            ax, width=2, height=2, loc=4,
            bbox_to_anchor=(1.15, 0.35),
            bbox_transform=ax.figure.transFigure
        )
        axin.plot(x, y, 'k.')
        start, end = (x[0], y[0]), (x[-1], y[-1])
        kw = dict(marker='o', linestyle='none')
        axin.plot(*start, color='g', **kw)
        axin.plot(*end, color='r', **kw)
        axin.axis('off')

    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Depth (m)')
    return fig, ax, cbar

get_ipython().magic('matplotlib inline')

temp = cubes.extract_strict('sea_water_temperature')

fig, ax, cbar = plot_glider(temp, cmap=plt.cm.viridis,
                            figsize=(9, 4.25), track_inset=True)

bbox = [
    [-125.72, 32.60],
    [-117.57, 36.93]
]

from shapely.geometry import LineString


def parse_geometry(geometry):
    """
    Filters out potentially bad coordinate pairs as returned from
    GliderDAC. Returns a safe geometry object.

    :param dict geometry: A GeoJSON Geometry object

    """
    coords = []
    for lon, lat in geometry['coordinates']:
        if lon is None or lat is None:
            continue
        coords.append([lon, lat])
    return {'coordinates': coords}


def fetch_trajectory(deployment):
    """
    Downloads the track as GeoJSON from GliderDAC

    :param dict deployment: The deployment object as returned from GliderDAC

    """
    track_url = 'http://data.ioos.us/gliders/status/api/track/{}'.format
    response = requests.get(track_url(deployment['deployment_dir']))
    if response.status_code != 200:
        raise IOError("Failed to get Glider Track for %s" % deployment['deployment_dir'])
    geometry = parse_geometry(response.json())
    coords = LineString(geometry['coordinates'])
    return coords

from shapely.geometry import box

search_box = box(bbox[0][0], bbox[0][1], bbox[1][0], bbox[1][1])

inside = dict()
for deployment in response.json()['results']:
    try:
        coords = fetch_trajectory(deployment)
    except IOError:
        continue
    if search_box.intersects(coords):
        inside.update({deployment['name']: coords})

def plot_track(coords, name, color='orange'):
    x, y = coords.xy
    locations = list(zip(y.tolist(), x.tolist()))

    folium.CircleMarker(locations[0], fill_color='green', radius=10).add_to(m)
    folium.CircleMarker(locations[-1], fill_color='red', radius=10).add_to(m)

    folium.PolyLine(
        locations=locations,
        color=color,
        weight=8,
        opacity=0.2,
        popup=name
    ).add_to(m)

import folium


tiles = ('http://services.arcgisonline.com/arcgis/rest/services/'
         'World_Topo_Map/MapServer/MapServer/tile/{z}/{y}/{x}')

location = [search_box.centroid.y, search_box.centroid.x]

m = folium.Map(
    location=location,
    zoom_start=5,
    tiles=tiles,
    attr='ESRI'
)


for name, coords in inside.items():
    plot_track(coords, name, color='orange')

m

