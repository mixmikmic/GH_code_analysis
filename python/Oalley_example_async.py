import folium
import grequests
import polyline
from urllib.parse import urlencode

baseurl = "https://api.oalley.fr/api/AppKeys/"
method = "/isochronous?"

appkey = "YOUR-API-KEY"

points = [
    {
        "lat":48.8738,
        "lng":2.295,
        "duration": 60 # 1 min by car
    },
    {
        "lat":48.8738,
        "lng":2.295,
        "duration":120 # 2 min
    },
    {
        "lat":48.8738,
        "lng":2.295,
        "duration":180 # 3 min
    }    
]

urls = [baseurl + appkey + method + urlencode(point) for point in points]

urls

zones = []

# Called when OAlley returned an isochrone
def callback(res, **kwargs):
    if(res.status_code != 200): 
        return print(res.json())
    
    body = res.json()
    zone = polyline.decode(body['polyline'])
    zones.append(zone)
    
def exception_handler(request, exception):
    print("Error : %s" % exception)
    
# Prepare all requests
reqs = [grequests.get(url, callback=callback) for url in urls]

# Execute all requests
grequests.map(reqs, exception_handler=exception_handler)

# Build output map
m = folium.Map(location=[46, 2], zoom_start=5)

# Draw all computed zones on the map
for points in zones:
    # Trick to close the polyline, until folium library supports it
    points.append(points[0])
    p = folium.PolyLine(locations=points,weight=3)
    m.add_children(p)

# Print the result
m.fit_bounds(m.get_bounds())

# IPython trick to display the map
m

