import os
import ee
import json

ee.Initialize()

def tile_url(image, viz_params=None):
    """Create a target url for tiles for an image.
    e.g.
    im = ee.Image("LE7_TOA_1YEAR/" + year).select("B3","B2","B1")
    viz = {'opacity': 1, 'gain':3.5, 'bias':4, 'gamma':1.5}
    url = tile_url(image=im),viz_params=viz)
    """
    if viz_params:
        d = image.getMapId(viz_params)
    else:
        d = image.getMapId()
    base_url = 'https://earthengine.googleapis.com'
    url = (base_url + '/map/' + d['mapid'] + '/{z}/{x}/{y}?token=' + d['token'])
    return url

collection = ee.ImageCollection('LANDSAT/LC8_L1T').filterDate('2016-01-01T00:00','2017-01-01T00:00')

composite = ee.Algorithms.SimpleLandsatComposite(collection=collection, percentile=50,
                                                 maxDepth=80, cloudScoreRange=1, asFloat=True)

hsv2 = composite.select(['B4', 'B3', 'B2']).rgbToHsv()

sharpened2 = ee.Image.cat([hsv2.select('hue'), hsv2.select('saturation'),
                           composite.select('B8')]).hsvToRgb().visualize(gain=1000, gamma= [1.15, 1.4, 1.15])


ee_tiles = tile_url(sharpened2)
ee_tiles

import requests

r = requests.get('https://staging-api.globalforestwatch.org/v1/landsat-tiles/2015')
print(r.status_code)
r.json()

ee_tiles= r.json().get('data').get('attributes').get('url')
print(ee_tiles)

pre_calculated_tileset="https://storage.googleapis.com/landsat-cache/2015/{z}/{x}/{y}.png"

import folium

map = folium.Map(location=[28.29, -16.6], zoom_start=2, tiles='Mapbox Bright' )

map.add_tile_layer(tiles=pre_calculated_tileset, max_zoom=11, min_zoom=0, attr='Earth Engine tiles by Vizzuality')
map.add_tile_layer(tiles=ee_tiles, max_zoom=20, min_zoom=13, attr="Live EE tiles")

map



