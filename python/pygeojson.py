from pprint import * 

from geojson import Point

Point((-115.81, 37.24))  # doctest: +ELLIPSIS

from geojson import MultiPoint

MultiPoint([(-155.52, 19.61), (-156.22, 20.74), (-157.97, 21.46)])  # doctest: +ELLIPSIS
#{"coordinates": [[-155.5..., 19.6...], [-156.2..., 20.7...], [-157.9..., 21.4...]], "type": "MultiPoint"}

from geojson import LineString

lstring = LineString([(8.919, 44.4074), (8.923, 44.4075)])  # doctest: +ELLIPSIS
#{"coordinates": [[8.91..., 44.407...], [8.92..., 44.407...]], "type": "LineString"}

pprint(lstring)

from geojson import MultiLineString

mlstring = MultiLineString([
[(3.75, 9.25), (-130.95, 1.52)],
[(23.15, -34.25), (-1.35, -4.65), (3.45, 77.95)]
])  # doctest: +ELLIPSIS
#{"coordinates": [[[3.7..., 9.2...], [-130.9..., 1.52...]], [[23.1..., -34.2...],
#[-1.3..., -4.6...], [3.4..., 77.9...]]], "type": "MultiLineString"}

pprint(mlstring)

from geojson import Polygon

# no hole within polygon
polya = Polygon([[(2.38, 57.322), (23.194, -20.28), (-120.43, 19.15), (2.38, 57.322)]])  # doctest: +ELLIPSIS
#{"coordinates": [[[2.3..., 57.32...], [23.19..., -20.2...], [-120.4..., 19.1...]]], "type": "Polygon"}

pprint(polya)

# hole within polygon
polyb = Polygon([
[(2.38, 57.322), (23.194, -20.28), (-120.43, 19.15), (2.38, 57.322)],
[(-5.21, 23.51), (15.21, -10.81), (-20.51, 1.51), (-5.21, 23.51)]
])  # doctest: +ELLIPSIS
#{"coordinates": [[[2.3..., 57.32...], [23.19..., -20.2...], [-120.4..., 19.1...]], 
#[[-5.2..., 23.5...], [15.2..., -10.8...], [-20.5..., 1.5...], [-5.2..., 23.5...]]], "type": "Polygon"}

pprint(polyb)

from geojson import MultiPolygon

mp = MultiPolygon([
([(3.78, 9.28), (-130.91, 1.52), (35.12, 72.234), (3.78, 9.28)],),
([(23.18, -34.29), (-1.31, -4.61), (3.41, 77.91), (23.18, -34.29)],)
])  # doctest: +ELLIPSIS

#{"coordinates": [[[[3.7..., 9.2...], [-130.9..., 1.5...], [35.1..., 72.23...]]], 
#[[[23.1..., -34.2...], [-1.3..., #-4.6...], [3.4..., 77.9...]]]], "type": "MultiPolygon"}

pprint(mp)

from geojson import GeometryCollection, Point, LineString

my_point = Point((23.532, -63.12))

my_line = LineString([(-152.62, 51.21), (5.21, 10.69)])

gc = GeometryCollection([my_point, my_line])  # doctest: +ELLIPSIS
#{"geometries": [{"coordinates": [23.53..., -63.1...], "type": "Point"}, 
#{"coordinates": [[-152.6..., 51.2...], [5.2..., 10.6...]], "type": "LineString"}], "type": "GeometryCollection"}

pprint(gc)

from geojson import Feature, Point

my_point = Point((-3.68, 40.41))

f1 = Feature(geometry=my_point)  # doctest: +ELLIPSIS
#{"geometry": {"coordinates": [-3.68..., 40.4...], "type": "Point"}, "properties": {}, "type": "Feature"}
pprint(f1)

f2 = Feature(geometry=my_point, properties={"country": "Spain"})  # doctest: +ELLIPSIS
#{"geometry": {"coordinates": [-3.68..., 40.4...], "type": "Point"}, "properties": {"country": "Spain"}, 
#"type": "Feature"}
pprint(f2)

f3 = Feature(geometry=my_point, id=27)  # doctest: +ELLIPSIS
#{"geometry": {"coordinates": [-3.68..., 40.4...], "type": "Point"}, "id": 27, "properties": {}, "type": "Feature"}
pprint(f3)

from geojson import Feature, Point, FeatureCollection

my_feature = Feature(geometry=Point((1.6432, -19.123)))

my_other_feature = Feature(geometry=Point((-80.234, -22.532)))

fc = FeatureCollection([my_feature, my_other_feature])  # doctest: +ELLIPSIS
#{"features": [{"geometry": {"coordinates": [1.643..., -19.12...], "type": "Point"}, "properties": {}, "type": #"Feature"}, {"geometry": {"coordinates": [-80.23..., -22.53...], "type": "Point"}, "properties": {}, "type": #"Feature"}], "type": "FeatureCollection"}

pprint(fc)

import geojson

my_point = geojson.Point((43.24, -1.532))

pprint(my_point)  # doctest: +ELLIPSIS
#{"coordinates": [43.2..., -1.53...], "type": "Point"}

dump = geojson.dumps(my_point, sort_keys=True)

pprint(dump)  # doctest: +ELLIPSIS
#'{"coordinates": [43.2..., -1.53...], "type": "Point"}'

gj = geojson.loads(dump)  # doctest: +ELLIPSIS
#{"coordinates": [43.2..., -1.53...], "type": "Point"}
pprint(gj)

import geojson

class MyPoint():
     def __init__(self, x, y):
         self.x = x
         self.y = y

     @property
     def __geo_interface__(self):
         return {'type': 'Point', 'coordinates': (self.x, self.y)}

point_instance = MyPoint(52.235, -19.234)

geojson.dumps(point_instance, sort_keys=True)  # doctest: +ELLIPSIS
#'{"coordinates": [52.23..., -19.23...], "type": "Point"}'

import geojson

my_line = LineString([(-152.62, 51.21), (5.21, 10.69)])

my_feature = geojson.Feature(geometry=my_line)

list(geojson.utils.coords(my_feature))  # doctest: +ELLIPSIS
#[(-152.62..., 51.21...), (5.21..., 10.69...)]

import geojson

new_point = geojson.utils.map_coords(lambda x: x/2, geojson.Point((-115.81, 37.24)))

geojson.dumps(new_point, sort_keys=True)  # doctest: +ELLIPSIS
#'{"coordinates": [-57.905..., 18.62...], "type": "Point"}'

import geojson

validation = geojson.is_valid(geojson.Point((-3.68,40.41,25.14)))
print(validation['valid'])
#'no'

print(validation['message'])
#'the "coordinates" member must be a single position'

import geojson

geojson.utils.generate_random("LineString")  # doctest: +ELLIPSIS
#{"coordinates": [...], "type": "LineString"}



