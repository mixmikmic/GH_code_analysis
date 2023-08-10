from IPython.display import HTML
import folium
import requests
import polyline
from urllib.parse import urlencode

baseurl = "https://api.oalley.fr/api/AppKeys/"
method = "/isochronous?"

appkey = "YOUR-API-KEY"

point = {
    'lat': 48.729027,
    'lng': 2.369972,
    'duration' : 10 * 60
}
 
url = baseurl + appkey + method + urlencode(point)
url

res = requests.get(url)
body = res.json()

if res.status_code != 200:
    print(body)

# Convert the polyline to JSON 
isozone = polyline.decode(body['polyline'])

# Close the polyline
isozone.append(isozone[0])

isozone

mymap = folium.Map(location=[46, 2], zoom_start=5)

folium.PolyLine(locations=isozone,weight=5).add_to(mymap)

mymap.fit_bounds(mymap.get_bounds())

mymap

durations = [60 * 5, 60 * 10, 60 * 20, 60 * 30, 60 * 50]

point = {
    'lat': 48.729027,
    'lng': 2.369972,
    'duration': 0.0
}
 
# Build urls    
urls = [baseurl + appkey + method + urlencode(dict(point, duration=duration)) for duration in durations]

zones = []

# Execute them one at a time
for url in urls:
    r = requests.get(url)
    body = r.json()
    isozone = polyline.decode(body['polyline'])
    # Trick to close the polyline, until folium library supports it
    isozone.append(isozone[0])
    zones.append(isozone)

# Plot the result
mymap = folium.Map(location=[46, 2], zoom_start=5)
for zone in zones:
    folium.PolyLine(locations=zone,weight=5).add_to(mymap)

mymap.fit_bounds(mymap.get_bounds())
mymap

