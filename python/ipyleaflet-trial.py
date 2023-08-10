from ipyleaflet import (
    Map,
    Marker,
    TileLayer, ImageOverlay,
    Polyline, Polygon, Rectangle, Circle, CircleMarker,
    GeoJSON,
    DrawControl
)

center = [34.6252978589571, -77.34580993652344]
zoom = 10

m = Map(default_tiles=TileLayer(opacity=1.0), center=center, zoom=zoom)
m

m.interact(zoom=(5,10,1))

m.center = [46.86678, -96.45328]


m.remove_layer(m.default_tiles)

m.add_layer(m.default_tiles)

m.model_id

mark = Marker(location=m.center)

mark.visible

m += mark

mark.interact(opacity=(0.0,1.0,0.01))

cm = CircleMarker(location=m.center, radius=30, weight=2,
                  color='#F00', opacity=1.0, fill_opacity=1.0,
                  fill_color='#0F0')
m.add_layer(cm)

mark.location = m.center

import ipyleaflet as ipyl
import ipywidgets as ipyw
import json

# Map and label widgets
map = ipyl.Map(center=[53.88, 27.45], zoom=4)
label = ipyw.Label(layout=ipyw.Layout(width='100%'))

# geojson layer with hover handler
with open('./europe_110.geo.json') as f:
    data = json.load(f)
for feature in data['features']:
    feature['properties']['style'] = {
        'color': 'grey',
        'weight': 1,
        'fillColor': 'red',
        'fillOpacity': 1.0
    }
layer = ipyl.GeoJSON(data=data, hover_style={'fillColor': 'red'})

def hover_handler(event=None, id=None, properties=None):
    label.value = properties['geounit']

layer.on_hover(hover_handler)
map.add_layer(layer)


ipyw.VBox([map, label])



