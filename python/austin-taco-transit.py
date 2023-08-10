# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

get_ipython().magic('matplotlib inline')
import json
import os

import folium
from matplotlib import pyplot as plt
import numpy as np
from pandas.io import gbq
import pandas as pd
import sqlalchemy
import scipy.spatial

engine = sqlalchemy.create_engine('postgresql+psycopg2://{}:{}@localhost:5432/postgres'.format(os.environ['POSTGRES_USER'], os.environ['POSTGRES_PASSWORD']))

busstops = pd.read_sql_query(
    """
SELECT
  b.stop_id AS stop_id,
  MIN(ST_DistanceSphere(b.stop_geom, t.geom)) AS tacodist,
  b.schedule_deviation as schedule_deviation,
  b.stop_lat as stop_lat,
  b.stop_lon as stop_lon
FROM busstops b, tacos t
GROUP BY stop_id, schedule_deviation, stop_lat, stop_lon;""",
    con=engine)

tacos = pd.read_sql_query(
    "SELECT chain, latitude, longitude FROM tacos;",
    con=engine)

vor = scipy.spatial.Voronoi(busstops.as_matrix(['stop_lon', 'stop_lat']))

fig = scipy.spatial.voronoi_plot_2d(vor, show_vertices=False)
fig.set_size_inches(18.5, 18.5)

# Construct GeoJSON for the region.
# Based on 
# http://stackoverflow.com/a/20678647/101923
# to account for infinite regions.
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


new_regions, new_vertices = voronoi_finite_polygons_2d(vor)
stop_regions = []
regions_geojson = {'type': 'FeatureCollection', 'features': stop_regions}

for busi, _ in enumerate(vor.point_region):
    coords = []
    for vtx in new_regions[busi]:
        coords.append(new_vertices[vtx].tolist())
        
    region_json = {
        'type': 'Feature',
        'id': int(busstops.stop_id[busi]),
        'properties': {
            'stop_id': int(busstops.stop_id[busi]),
        },
        'geometry': {
            'type': 'Polygon',
            'coordinates': [coords],
        },
    }
    stop_regions.append(region_json)
    
rs = json.dumps(regions_geojson)

austin_loc = (30.2957147,-97.7472336)
busmap = folium.Map(location=austin_loc)
busmap.choropleth(
    geo_str=rs,
    data=busstops,
    columns=['stop_id', 'schedule_deviation'],
    threshold_scale=[300, 600, 900, 1800, 3600, 7200],
    key_on='feature.id',
    fill_color='BuPu', fill_opacity=0.8, line_opacity=0.25,
    legend_name='Schedule Deviation (s)',
    reset=True)
busmap

# One second of bus lateness equals one meter of distance of distance to tacos in my preferences.
busstops = busstops.assign(tacotransit=pd.Series(busstops.tacodist + busstops.schedule_deviation).values)

tacotransit = folium.Map(location=austin_loc)
tacotransit.choropleth(
    geo_str=rs,
    data=busstops,
    columns=['stop_id', 'tacotransit'],
    threshold_scale=[500, 1000, 1500, 2000, 4000, 8000],
    key_on='feature.id',
    fill_color='RdPu', fill_opacity=0.8, line_opacity=0.25,
    legend_name='Taco Transit (tims)',
    reset=True)
for _, taco in tacos.iterrows():
    folium.Marker([taco.latitude, taco.longitude], popup=taco.chain).add_to(tacotransit)
tacotransit



