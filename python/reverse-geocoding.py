from arcgis.geocoding import reverse_geocode

help(reverse_geocode)

from arcgis.gis import GIS
from arcgis.geocoding import reverse_geocode

gis = GIS("portal url", "username", "password")

results = reverse_geocode([2.2945, 48.8583])

results

from arcgis.geometry import Geometry

pt = Geometry({
    "x": 11563503,
    "y": 148410,
    "spatialReference": {
        "wkid": 3857
    }
})

results = reverse_geocode(pt)

results

result = reverse_geocode([4.366281,50.851994], lang_code="fr")

result

from arcgis.gis import GIS
from arcgis.geocoding import reverse_geocode
gis = GIS("portal url", "username", "password")

m = gis.map('Redlands, CA', 14)
m

def find_addr(m, g):
    try:
        geocoded = reverse_geocode(g)
        print(geocoded['address']['Match_addr'])
    except:
        print("Couldn't match address. Try another place...")

m.on_click(find_addr)

