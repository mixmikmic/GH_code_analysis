import collections
from CesiumWidget import CesiumWidget
import czml

points = collections.OrderedDict()
points['p1'] = [18.07,59.33, 20]
points['p2'] = [19.07,59.33, 20]
points['p3'] = [20.07,59.33, 20]

points

doc = czml.CZML()

packet1 = czml.CZMLPacket(id='document',version='1.0')
doc.packets.append(packet1)

for i,v in enumerate(points):
    print(i,v)
    p = czml.CZMLPacket(id=i)
    p.position = czml.Position(cartographicDegrees = points[v])
    point = czml.Point(pixelSize=20, show=True)
    point.color = czml.Color(rgba=(223, 150, 47, 128))
    point.show = True
    p.point = point
    l = czml.Label(show=True, text=v)
    l.scale = 0.5
    p.label = l
    doc.packets.append(p)

cesiumExample = CesiumWidget(width="100%", czml=tuple(doc.data()))

cesiumExample

cesiumExample.zoom_to(points['p1'][0], points['p1'][1], 360000, 0 ,-90, 0)

doc.dumps()

