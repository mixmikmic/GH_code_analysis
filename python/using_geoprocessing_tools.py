from arcgis.gis import GIS
from arcgis.geocoding import geocode
from IPython.display import display

gis = GIS()
map1 = gis.map()
map1

# set the map's extent by geocoding the location
diegogarcia = geocode("Diego Garcia")[0]
map1.extent = diegogarcia['extent']

# geocode location of La Reunion island
lareunion = geocode("La Reunion")[0]

# Annotate the map by plotting Diego Garcia, and two other search locations
map1.draw(lareunion['location'], {"title": "Reunion Island", "content": "Debris found"})
map1.draw(diegogarcia['location'], {"title": "Diego Garcia", "content": "Naval Support Facility Diego Garcia"})
map1.draw([-43.5, 90.5], {"title":"Search Location", "content":"Predicted crash location"})

# Render a feature layer representing the search area
# Source: http://www.amsa.gov.au/media/incidents/images/DIGO_00718_01_14.jpg
map1.add_layer({"type":"FeatureLayer", 
                "url" : "http://services.arcgis.com/WQ9KVmV6xGGMnCiQ/arcgis/rest/services/MH370Search/FeatureServer/1"})

mh370items = gis.content.search("MH370", "feature service", max_items=6)
for item in mh370items:
    display(item)

map1.add_layer(mh370items[0])
map1.add_layer(mh370items[4])
map1.add_layer(mh370items[5])

map1.zoom = 6

toolbox_item = gis.content.search("Ocean Currents", item_type="geoprocessing toolbox", max_items=1)[0]
toolbox_item

from arcgis.geoprocessing import import_toolbox

ocean_currents = import_toolbox(toolbox_item)

help(ocean_currents.message_in_a_bottle)

from arcgis.features import FeatureSet, Feature

def do_analysis(m, g):
    print("Computing the path that debris would take...")
    
    # Parameter `g` contains the co-ordinates of the clicked location
    
    # Construct a FeatureSet object from the clicked locaiton
    my_feature_set = FeatureSet([Feature(g)])
    
    # Pass the input location as a FeatureSet
    ret = ocean_currents.message_in_a_bottle(my_feature_set, 150)
    
    # Render the resulting FeatureSet on the map using `draw()` method
    map1.draw(ret)
    
# Set the callback function that performs analysis. The `do_analysis` function is called whenever user clicks on the map.
map1.on_click(do_analysis)

