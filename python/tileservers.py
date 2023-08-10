import folium
import json
import requests

basemap = 'Cartodb dark_matter' # "Mapbox Bright"

map = folium.Map(location=[28.29, -16.6], zoom_start=3, tiles=basemap)

#tileset = r"https://storage.googleapis.com/forma-public/Forma250/tiles/global_data/biweekly/forma_biweekly_2017_4/v1/{z}/{x}/{y}.png"
#tileset="https://storage.googleapis.com/gfw-climate-tiles/soil_carbon/{z}/{x}/{y}.png"
#tileset="https://storage.googleapis.com/gfw-climate-tiles/aboveground_carbon/{z}/{x}/{y}.png"
#tileset="https://storage.googleapis.com/gfw-climate-tiles/below_ground_carbon/{z}/{x}/{y}.png"
#tileset =r"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Aqua_CorrectedReflectance_TrueColor/default/2016-04-09/GoogleMapsCompatible_Level6/{level}/{row}/{col}.png"
#tileset=r"https://gibs.earthdata.nasa.gov/wmts/epsg4326/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/2012-07-09/250m/{z}/{x}/{y}.png"

tileset=r"https://storage.googleapis.com/landsat-cache/2016/{z}/{x}/{y}.png"

map.add_tile_layer(tiles=tileset, max_zoom=12,min_zoom=1,  attr='Custom tiles')

map

# Configuration
account = 'wri-rw'
urlCarto = 'https://'+account+'.carto.com/api/v1/map'
body = {
    "layers": [{
        "type": "cartodb",
        "options": {
            "sql": "select * from countries",
            "cartocss":"#layer {\n  polygon-fill: #374C70;\n  polygon-opacity: 0.9;\n  polygon-gamma: 0.5;\n  line-color: #FFF;\n  line-width: 1;\n  line-opacity: 0.5;\n  line-comp-op: soft-light;\n}",
            "cartocss_version": "2.1.1"
        }
    }]
}
# Get layer group id
r = requests.post(urlCarto, data=json.dumps(body), headers={'content-type': 'application/json; charset=UTF-8'})
tileUrl = 'https://'+account+'.carto.com/api/v1/map/' + r.json()['layergroupid'] + '/{z}/{x}/{y}.png32';

# Add layer to map
folium.TileLayer(
    tiles=tileUrl,
    attr='text',
    name='text',
    overlay=True
).add_to(map)
map

# to save the map to a html file, which you can share, execute the below:
map.save('landsat_2015.html')

