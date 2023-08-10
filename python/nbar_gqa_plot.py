import os
from os.path import join as pjoin, isdir, basename
from datetime import datetime
import argparse
import numpy
import pandas
import yaml
get_ipython().magic('matplotlib inline')
from descartes import PolygonPatch
from shapely.geometry import Polygon, MultiPolygon, mapping
import folium

METADATA_FNAME = 'ga-metadata.yaml'

def read_yaml(fname):
    """
    Read a yaml file.
    
    :param fname:
        A full filepath name to yaml document on disk
        
    :return:
        A `dict` containing the yaml document contents.
    """
    with open(fname) as src:
        try:
            yml = yaml.load(src)
            return yml
        except yaml.YAMLError as exc:
            print exc

def extract_scene_mb(yml):
    """
    Get the scene content structure in megabytes.
    
    :param yml:
        The dict containing the yaml document contents.
        
    :return:
        The total volume of the scene structure.
    """
    sz_bytes = yml['size_bytes']
    sz_mb = sz_bytes / 1024.0 / 1024.0
    return sz_mb

def calculate_storage(scenes, gb=True):
    """
    Given a list of scenes, compute the total storage volume.
    
    :param scenes:
        A list containing the scenes to process.
        
    :param gb:
        If set to True (default), then return the storage volume in gigabytes.
        
    :return:
        The total volume of all scenes.
    """
    storage = []

    for scene in scenes:
        fname = pjoin(scene, METADATA_FNAME)
        yml = read_yaml(fname)
        storage.append(extract_scene_mb(yml))

    # calculate GB or MB
    if gb:
        sz = numpy.array(storage).sum() / 1024.0
    else:
        sz = numpy.array(storage).sum()

    return sz

def extract_poly(yml):
    """
    Extract the scene polygon.
    
    :param yml:
        The dict containing the yaml document contents.
        
    :return:
        A `shapely.geometry.Polgon` instance.
    """
    coords = yml['extent']['coord']
    
    ul = (coords['ul']['lon'], coords['ul']['lat'])
    ur = (coords['ur']['lon'], coords['ur']['lat'])
    lr = (coords['lr']['lon'], coords['lr']['lat'])
    ll = (coords['ll']['lon'], coords['ll']['lat'])
    
    poly = Polygon([ul, ur, lr, ll])
    
    return poly

def extract_cep(yml):
    try:
        cep = yml['lineage']['source_datasets']['level1']['gqa']['cep90']
    except KeyError:
        cep = numpy.nan
    return cep

def plot_scenes(fmap, scenes):
    for scene in scenes:
        fname = pjoin(scene, METADATA_FNAME)
        yml = read_yaml(fname)
        poly = extract_poly(yml)
        cx, cy = poly.centroid.xy
        mp = mapping(poly)
        cep = extract_cep(yml)
        mp['properties'] = {'cep90': cep}
        folium.GeoJson(mp, style_function=map_colour).add_to(fmap)
        #folium.Marker([cy[0], cx[0]], popup=str(cep)).add_to(fmap)
        #fmap.simple_marker([cy[0], cx[0]], popup=str(cep))

def map_colour(poly):
    cep = poly['geometry']['properties']['cep90']
    if numpy.isfinite(cep):
        if cep > 10:
            return {'fillColor': 'red', 'color': 'red'}
        elif cep > 1:
            return {'fillColor': 'yellow', 'color': 'yellow'}
        else:
            return {'fillColor': 'green', 'color': 'green'}
    else:
        return {'fillColor': 'black', 'color': 'black'}

sensor = 'ls5'
year = 2004
month = 7

mnth= '{0:02d}'.format(7)
nbar_dir = '/g/data2/rs0/scenes/nbar-scenes-tmp/{sensor}/{year}/{month}/output/nbar'.format(sensor=sensor,
                                                                                            year=year, month=mnth)
scenes = [pjoin(nbar_dir, s) for s in os.listdir(nbar_dir)]
scenes = [s for s in scenes if not basename(s).startswith('.')]
scenes = [scene for scene in scenes if isdir(scene)]
print "Number of scenes: {}".format(len(scenes))

m = folium.Map(location=[-30,150], zoom_start=4)
plot_scenes(m, scenes)
m



