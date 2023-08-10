import folium
import grequests
import polyline
from urllib.parse import urlencode

# This should display a version > 0.3, otherwise PolygonMarker is not supported.
print(folium.__version__)

baseurl = "https://api.oalley.fr/api/AppKeys/"
method = "/isochronous?"

appkey = "YOUR-API-KEY"

durations = [ 60, 120, 180]
center =  {
    "lat":48.8738,
    "lng":2.295,
    "duration": 0.0
}

urls = [baseurl + appkey + method + urlencode(dict(center,duration=d)) for d in durations]

zones = []

# Called when OAlley returned a zone
def callback(res, **kwargs):
    if(res.status_code != 200): # Most likely, you don't have credits anymore. We give more for free. Contact us at contact@oalley.fr 
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
m = folium.Map(location=[46, 2])

# Contour color supports transparency. Here the last 00 means the contours is fully transparent
contour_color = "#ffffff00" 

# Fill color doesnt support transparency. The last 2 numbers for transparency must be removed.
# Otherwise it will default to black
fill_color    = "#002d5f"

# Draw all computed zones on the map
for points in zones:
    folium.features.PolygonMarker(locations=points, color=contour_color, fill_color=fill_color, weight=10).add_to(m)

# Print the result
m.fit_bounds(m.get_bounds())

# IPython trick to display the map
m

