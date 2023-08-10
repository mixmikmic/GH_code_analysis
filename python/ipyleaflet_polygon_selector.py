from IPython.display import display
import ipywidgets

from __future__ import print_function
import ipyleaflet

from ipyleaflet import (
    Map,
    Marker,
    TileLayer, ImageOverlay,
    Polyline, Polygon, Rectangle, Circle, CircleMarker,
    GeoJSON,
    DrawControl
)

#center = [48.355, -124.642] # Neah Bay
center = [41.75, -124.19] # Crescent City

zoom = 11
m = Map(center=center, zoom=zoom)
m

zoom = 13
c = ipywidgets.Box()

topo_background = False   # Use topo as background rather than map?
## NOTE: topo_background == True is broken -- see note above.

if topo_background:
    m = Map(width='1000px',height='600px', center=center, zoom=zoom,         default_tiles=TileLayer(url=u'http://otile1.mqcdn.com/tiles/1.0.0/sat/{z}/{x}/{y}.jpg'))
else:
    m = Map(width='1000px',height='600px', center=center, zoom=zoom)
    
c.children = [m]

# keep track of rectangles and polygons drawn on map:
def clear_m():
    global rects,polys
    rects = set()
    polys = set()
    
clear_m()
rect_color = '#a52a2a'
poly_color = '#00F'

myDrawControl = DrawControl(
rectangle={'shapeOptions':{'color':rect_color}},
        polygon={'shapeOptions':{'color':poly_color}}) #,polyline=None)

def handle_draw(self, action, geo_json):
    global rects,polys
    polygon=[]
    for coords in geo_json['geometry']['coordinates'][0][:-1][:]:
        polygon.append(tuple(coords))
    polygon = tuple(polygon)
    if geo_json['properties']['style']['color'] == '#00F':  # poly
        if action == 'created':
            polys.add(polygon)
        elif action == 'deleted':
            polys.discard(polygon)
    if geo_json['properties']['style']['color'] == '#a52a2a':  # rect
        if action == 'created':
            rects.add(polygon)
        elif action == 'deleted':
            rects.discard(polygon)
myDrawControl.on_draw(handle_draw)
m.add_control(myDrawControl)

clear_m()
display(m)

for r in polys: 
    print("\nPolygon vertices:")
    for c in r: print('%10.5f, %10.5f' % c)
        
for r in rects: 
    print("\nRectangle vertices:")
    for c in r: print('%10.5f, %10.5f' % c)

for r in rects:
    print("\nCoordinates of lower left and upper right corner of rectangle:")
    x1 = r[0][0]
    x2 = r[2][0]
    y1 = r[0][1]
    y2 = r[2][1]
    print("x1, y1, x2, y2 = %10.5f, %10.5f, %10.5f, %10.5f" % (x1,y1,x2,y2))

for r in rects:
    print("\nCoordinates of lower left and upper right corner of rectangle:")
    x1 = r[0][0]
    x2 = r[2][0]
    y1 = r[0][1]
    y2 = r[2][1]
    print("x = %10.5f, %10.5f" % (x1,x2))
    print("y = %10.5f, %10.5f" % (y1,y2))
    

for r in polys:
    print("\nCoordinates of distinct vertices of polygon:")
    sx = 'x = '
    sy = 'y = '
    for j in range(len(r)-1):
        sx = sx + ' %10.5f,' % r[j][0]
        sy = sy + ' %10.5f,' % r[j][1]
    print(sx)
    print(sy)
        

from clawpack.geoclaw import kmltools
reload(kmltools)
for i,r in enumerate(rects):
    x1 = r[0][0]
    x2 = r[2][0]
    y1 = r[0][1]
    y2 = r[2][1]
    name = "rect%i" % i
    kmltools.box2kml((x1,x2,y1,y2), name=name, verbose=True)

for i,r in enumerate(polys):
    x = [xy[0] for xy in r]
    y = [xy[1] for xy in r]
    kmltools.poly2kml((x,y), name="poly%i" % i, verbose=True)



