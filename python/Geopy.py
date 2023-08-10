from geopy.geocoders import Nominatim
geolocator = Nominatim()

geolocator.geocode("Vidoe Smilevski Bato, Skopje, Macedonia")

geolocator.reverse("41.9801537, 21.4768662")

geolocator.geocode("Vidoe Smilevski Bato, Skopje, Macedonia").latitude

from geopy.distance import vincenty

skopje = (41.9794894, 21.4764231)
ohrid = (41.1127104, 20.7993744)

vincenty(skopje,ohrid).kilometers

geolocator.geocode("Amsterdam, Netherlands")

amsterdam = (52.374436, 4.8979956033677)

vincenty(skopje,amsterdam).kilometers

geolocator.geocode("Mae Sariang village;  Mae Hong Son")



