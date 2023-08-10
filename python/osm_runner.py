# Libraries for Creating Buttons & Handling Output
from IPython.display import display, clear_output
import ipywidgets as widgets

# OSM Runner & GIS Object
from osm_runner import gen_osm_sdf
from arcgis.gis import GIS

# Transformations & Geometries for ArcGIS Item Creation
from pyproj import Proj, transform
import shapefile

# Other Libraries
from collections import OrderedDict
import time

# Organization Login
gis = GIS('org_url', 'username', 'password')

# Set OSM Tags with Friendly Names
osm_tag_dict = {
    "Aerialway":        "aerialway",
    "Aeroway":          "aeroway",
    "Amenity":          "amenity",
    "Barrier":          "barrier",
    "Boundary":         "boundary",
    "Building":         "building",
    "Craft":            "craft",
    "Emergency":        "emergency",
    "Geological":       "geological",
    "Highway":          "highway",
    "Historic":         "historic",
    "Landuse":          "landuse",
    "Leisure":          "leisure",
    "Man Made":         "man_made",
    "Military":         "military",
    "Natural":          "natural",
    "Office":           "office",
    "Place":            "place",
    "Power":            "power",
    "Public Transport": "public transport",
    "Railway":          "railway",
    "Route":            "route",
    "Shop":             "shop",
    "Sport":            "sport",
    "Tourism":          "tourism",
    "Waterway":         "waterway"
}

# Converting Map Widget Extent into a Tuple for OSM Query
def collect_extent(e):
    
    # Strip Min/Max For Geometry Iterable
    min_set = {k[:1]: v for k, v in e.items() if k in ['ymin', 'xmin']}
    max_set = {k[:1]: v for k, v in e.items() if k in ['ymax', 'xmax']}

    box = []
    
    for geo_set in [min_set, max_set]:
        
        incoming_wkid = e.get('spatialReference')['latestWkid']
        
        if incoming_wkid == 4326:
            box.append(geo_set['y'])
            box.append(geo_set['x'])
            
        else:   
            p1 = Proj(init='epsg:{}'.format(incoming_wkid))
            p2 = Proj(proj='latlong',datum='WGS84')
            x, y = transform(p1, p2, geo_set['x'], geo_set['y'])
            box.append(y)
            box.append(x)

    return tuple(box)

# on_click() Logic
def running(button_object):

    global sdf
    
    clear_output()
    
    # Pull Values From Inputs
    geo_val = geo_sel.value
    osm_val = osm_sel.value
    bbox    = collect_extent(viz_map.extent)
    
    # Get Date YYYY-MM-DD From DatePicker
    t_1_val = str(t_1_sel.value)[:10] if t_1_sel.value else None
    t_2_val = str(t_2_sel.value)[:10] if t_2_sel.value else None
    
    try:
        print('Fetching Data From OpenStreetMap . . .')
        sdf = gen_osm_sdf(geo_val, bbox, osm_val, t_1_val, t_2_val)
         
    except Exception as e:
        print('Request Could Not Be Completed')
        print('{}'.format(str(e)))
        return
    
    else:
        print('Features Returned: {}'.format(len(sdf)))
        sdf_fs = sdf.to_featureset()
        
        for feature in sdf_fs:
            # Create Popup
            viz_map.draw(
                feature.geometry,
                popup={
                    'title': 'OSM ID: ' + feature.attributes['osm_id'] , 
                    'content': "{}".format(
                        '<br/>'.join([
                            '%s: %s' % (key.upper(), value) for (key, value) in feature.attributes.items()
                        ])
                    )
                }
            )

get_ipython().run_cell_magic('html', '', "<style>\n.intro {\n    padding: 10px; \n    color: #202020;\n    font-family: 'Helvetica'\n}\n.map {\n    border: solid;\n    height: 450px;\n}\n</style>")

# Create & Display Map
viz_map = gis.map('Smithsonian')
display(viz_map)

# Set Options For Return Geometry
geo_sel = widgets.Dropdown(
    options=['Point', 'Line', 'Polygon'],
    description='Geometry',
    value='Polygon'
)

# Set Options for OSM Tags
osm_sel = widgets.Dropdown(
    options=(sorted(osm_tag_dict.items(), key=lambda item: item[0])),
    description='Feature',
    value='building'
)

# Set Options for Time Selection
t_1_sel = widgets.DatePicker(description='Start Date')
t_2_sel = widgets.DatePicker(description='End Date')

# Create Submit Button & Set on_click
run_btn = widgets.Button(
    description='Fetch OSM',
    button_style='success',
    tooltip='Query OSM and View in Map Widget',
    layout=widgets.Layout(justify_content='center', margin='0px 0px 0px 10px')
)
run_btn.on_click(running)

# Handle Widget Layout
params = widgets.HBox(
    [geo_sel, osm_sel, t_1_sel, t_2_sel, run_btn], 
    layout=widgets.Layout(justify_content='center', margin='10px')
)
display(params)

def to_agol(button_object):
    
    clear_output()
    
    txt_val = txt_sel.value
    
    try:
        sdf
        
    except NameError:
        print('Please Collect Data with Fetch OSM Data First . . .')
        
    else:
        print('Creating Feature Layer in ArcGIS Online . . .')
        feat_lyr = sdf.to_featurelayer(
            '{}_{}'.format(txt_val, int(time.time())),
            gis=gis, 
            tags='OSM Runner'
        )

        display(feat_lyr)
        viz_map.add_layer(feat_lyr)

txt_sel = widgets.Text(description='Name', value='OSM Features')

add_btn = widgets.Button(
    description='Push OSM to ArcGIS',
    button_style='primary',
    tooltip='Create Content in ArcGIS Online'
)
add_btn.on_click(to_agol)

add_box = widgets.HBox([txt_sel, add_btn], layout=widgets.Layout(justify_content='center', margin='10px'))
display(add_box)

