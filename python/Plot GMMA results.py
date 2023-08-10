get_ipython().magic('matplotlib inline')

import os
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

from mpl_toolkits.basemap import Basemap

import shapefile

# OGR/GDAL imports:
from osgeo import gdal
from osgeo import osr
from osgeo import ogr

from osgeo.gdalconst import GDT_Float32
gdal.UseExceptions()

def getProjection(shape_file):
    """
    Get the projection of a shape file
    
    :param str shape_file: Name of a valid shape file

    :returns: :class:`osgeo.osr.SpatialReference` of the shape file

    """

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataset = driver.Open(shape_file)
    layer = dataset.GetLayer()
    spatial_ref = layer.GetSpatialRef()
    
    return spatial_ref

def getExtent(inputShapeFile):
    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(inputShapeFile, 0)
    inLayer = inDataSource.GetLayer()
    extent = inLayer.GetExtent()
    return extent

def reproject(inputShapeFile, outputShapeFile, outEPSG=4326):
    """
    Reproject a shape file to a known projection. 
    
    :param str inputShapeFile: Source shape file to be reprojected.
    :param str outputShapeFile: Destination shape file.
    :param int outEPSG: EPSG code for the output projection. Default is 
                        4326 (WGS 1984)
                        
    """
    driver = ogr.GetDriverByName('ESRI Shapefile')
    insref = getProjection(inputShapeFile)
    outsref = osr.SpatialReference()
    outsref.ImportFromEPSG(outEPSG)
    
    coordTransform = osr.CoordinateTransformation(insref, outsref)
    
    inDataSet = driver.Open(inputShapeFile)
    inLayer = inDataSet.GetLayer()

    # create the output layer
    if os.path.exists(outputShapeFile):
        driver.DeleteDataSource(outputShapeFile)
    outDataSet = driver.CreateDataSource(outputShapeFile)
    outPrjFile = outputShapeFile.replace('.shp', '.prj')
        
    fh = open(outPrjFile, 'w')
    fh.write(outsref.ExportToWkt())
    fh.close()
    
    outLayer = outDataSet.CreateLayer("", geom_type=ogr.wkbMultiPolygon)

    # add fields
    inLayerDefn = inLayer.GetLayerDefn()
    for i in range(0, inLayerDefn.GetFieldCount()):
        fieldDefn = inLayerDefn.GetFieldDefn(i)
        outLayer.CreateField(fieldDefn)

    # get the output layer's feature definition
    outLayerDefn = outLayer.GetLayerDefn()

    # loop through the input features
    inFeature = inLayer.GetNextFeature()
    while inFeature:
        # get the input geometry
        geom = inFeature.GetGeometryRef()
        # reproject the geometry
        geom.Transform(coordTransform)
        # create a new feature
        outFeature = ogr.Feature(outLayerDefn)
        # set the geometry and attribute
        outFeature.SetGeometry(geom)
        for i in range(0, outLayerDefn.GetFieldCount()):
            outFeature.SetField(outLayerDefn.GetFieldDefn(i).GetNameRef(), inFeature.GetField(i))
        # add the feature to the shapefile
        outLayer.CreateFeature(outFeature)
        # destroy the features and get the next input feature
        outFeature.Destroy()
        inFeature.Destroy()
        inFeature = inLayer.GetNextFeature()

    # close the shapefiles
    inDataSet.Destroy()
    outDataSet.Destroy()

inshpfile = "C:/workspace/gmma/output/event_pop_affect.shp"
outshpfile = "C:/workspace/data/Glenda/BGY_impact_gcs.shp"
#reproject(inshpfile, outshpfile)

def drawmap(kwargs):
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(111)
    m = Basemap(**kwargs)
    m.drawcoastlines()
    m.drawmapboundary()
    m.drawcountries()
    m.fillcontinents('0.75')
    
    return m

extent = getExtent(outshpfile)
llcrnrlon = int(10*extent[0])/10.
urcrnrlon = int(10*extent[1] + 1)/10.
llcrnrlat = int(10*extent[2])/10.
urcrnrlat = int(10*extent[3] + 1)/10.
mapkwargs = dict(projection='cyl', llcrnrlon=llcrnrlon,
                  llcrnrlat=llcrnrlat, urcrnrlon=urcrnrlon,
                  urcrnrlat=urcrnrlat, resolution='f')
m = drawmap(mapkwargs)
m.readshapefile("C:/workspace/data/Glenda/BGY_impact_gcs", 'results')


r = shapefile.Reader(outshpfile)
shapes = r.shapes()
records = r.records()
print r.fields



