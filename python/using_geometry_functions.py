from arcgis.gis import GIS
from arcgis.geocoding import geocode
from arcgis.geometry import lengths

gis = GIS()

map1 = gis.map()
map1.basemap = "satellite"

map1

map1.height = '650px'

location = geocode("Central Park, New York")[0]
map1.extent = location['extent']

map1.zoom = 14

# Define the callback function that computes the length.
def calc_dist(map1, g):
    print("Computing length of drawn polyline...")
    length = lengths(g['spatialReference'], [g], "", "geodesic")
    print("Length: " + str(length[0]) + " m.")

# Set calc_dist as the callback function to be invoked when a polyline is drawn on the map
map1.on_draw_end(calc_dist)

map1.draw("freehandpolyline")

map1.clear_graphics()

