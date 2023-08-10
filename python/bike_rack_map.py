get_ipython().system('ls')

get_ipython().system('cat ncle_cycle_parking1.csv')

get_ipython().system('cat ncle_cycle_parking.kml')

import folium
from fastkml.kml import KML

# source: https://ocefpaf.github.io/python4oceanographers/blog/2014/05/05/folium/
# version 1 ...
def read_kml(fname='ncle_cycle_parking.kml'):
    kml = KML()
    kml.from_string(open(fname).read())
    points = dict()
    placemarks = []
    for feature in kml.features():
        print(feature)
        for placemark in feature.features():
            placemarks.append(placemark)
#             if placemark.styleUrl.startswith('#hf'):
#                 points.update({placemark.name:
#                             (placemark.geometry.y, placemark.geometry.x, )})
    #return points
    return placemarks

placemarks = read_kml()

print(placemarks[0].to_string(prettyprint=True))

placemarks_list = list(placemarks[0].features())

for pm in placemarks_list:
    print(pm)

print(pm.geometry.x)

for el in pm.extended_data.elements:
    print(el.data)

len(pm.extended_data.elements)

print(pm.to_string())
print(pm.geometry)
print(pm.geometry.y, pm.geometry.x)

map_osm = folium.Map(location=[pm.geometry.y, pm.geometry.x],
                    zoom_start=15)

map_osm

folium.Marker([pm.geometry.y, pm.geometry.x], popup='Test point 1').add_to(map_osm)

map_osm

for pm in placemarks_list:
    folium.Marker([pm.geometry.y, pm.geometry.x], popup='Test point 1').add_to(map_osm)

map_osm

def get_details(p_marker):
    _str=""
    for d in p_marker.extended_data.elements[0].data:
        _str += " " + d['name'] + "=" + d['value']
    return _str

for i, pm in enumerate(placemarks_list):
    #folium.Marker([pm.geometry.y, pm.geometry.x], popup='Test point 1').add_to(map_osm)
    print(i)
    print(get_details(pm))

final_map_osm = folium.Map(location=[pm.geometry.y, pm.geometry.x],
                    zoom_start=15)
for i, pm in enumerate(placemarks_list):
    folium.Marker([pm.geometry.y, pm.geometry.x], popup=get_details(pm)).add_to(final_map_osm)

final_map_osm



