import folium

url_base = 'http://vignette3.wikia.nocookie.net/'
url_folder = 'the-adventures-of-the-gladiators-of-cybertron/'
url_file = 'images/6/67/Iron_Man.png/revision/latest?cb=20140720074813'
logo_url = url_base + url_folder + url_file
UCB_lat = 37.8719
UCB_lon = -122.2585

map_1 = folium.Map(location=[UCB_lat, UCB_lon], zoom_start=12,                      control_scale = True, tiles='Stamen Terrain')

icon = folium.features.CustomIcon(logo_url,                                  icon_size=(200, 200))

folium.Marker([UCB_lat, UCB_lon],
          popup='Iron man visit Berkeley',
          icon=icon).add_to(map_1)

map_1

from obspy.imaging.beachball import beachball

# latitude and longitude of the earthquake
evla = 38.215
evlo = -122.312

# South Napa EQ moment tensor
mt = [247, 82, 8]
beachball(mt, size=200, linewidth=2, facecolor='b', outfile= './beachball.png')

# Add the USGS style map
url_base = 'http://server.arcgisonline.com/'
tiles = 'ArcGIS/rest/services/NatGeo_World_Map/MapServer/tile/{z}/{y}/{x}'
tileset = url_base + tiles
map_1 = folium.Map(location=[38, -122], zoom_start=8,                      control_scale = True, tiles=tileset, attr='USGS style')

# Add the Moment Tensor as an icon
icon_url = './beachball.png'
icon = folium.features.CustomIcon('/beachball.png',icon_size=(45, 45))

folium.Marker([evla, evlo],
          popup='Napa Earthquake',
          icon=icon).add_to(map_1)

map_1.save('moment_tensors.html')

