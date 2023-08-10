import pandas as pd
import numpy as np
import geopandas as gp
from fiona.crs import from_epsg
from geopandas.tools import sjoin, overlay
import pylab as pl
get_ipython().run_line_magic('pylab', 'inline')

#!curl -O http://egis3.lacounty.gov/dataportal/wp-content/uploads/2011/12/DPW-Supervisorial-District.zip

#!unzip DPW-Supervisorial-District.zip

district = gp.GeoDataFrame.from_file('sup_dist_2011.shp')

district.head()

ax = district.plot(edgecolor='k', figsize=(10,10))
ax.axis('off');

#!curl -O http://egis3.lacounty.gov/dataportal/wp-content/uploads/ShapefilePackages/CENSUS_TRACTS_2010.zip

#!unzip CENSUS_TRACTS_2010.zip

tract = gp.GeoDataFrame.from_file('CENSUS_TRACTS_2010.shp')

tract.head()

ax = tract.plot(edgecolor='k', figsize=(10,10))
ax.axis('off')

district = district.to_crs(epsg=4326)
tract = tract.to_crs(epsg=4326)

district.crs

tract.crs

district_tract = sjoin(district, tract, how='right', op='intersects')

district_tract.head()

district_tract.plot(color='w', edgecolor='k', figsize=(12,12))

test_sample = district_tract[district_tract['SUP_DIST_N'] == '1']

test_dis = district[district['SUP_DIST_N'] == '1']

test_dis = test_dis.to_crs(epsg = 4326)
test_sample = test_sample.to_crs(epsg = 4326)


ax = test_sample.plot( color='w',figsize=(12,12), edgecolor='k')
test_dis.plot(ax=ax,  color='w', edgecolor='r', alpha=0.5)
ax.axis('off');

get_ipython().run_line_magic('pinfo', 'test_sample.plot')

#intersect_shp = overlay(district, tract, how="intersection")

# https://github.com/geopandas/geopandas/pull/338
def spatial_overlays(df1, df2):
    '''Compute overlay intersection of two 
        GeoPandasDataFrames df1 and df2'''
    # Spatial Index to create intersections
    spatial_index = df2.sindex
    df1['bbox'] = df1.geometry.apply(lambda x: x.bounds)
    df1['histreg']=df1.bbox.apply(lambda x:list(spatial_index.intersection(x)))
    pairs = df1['histreg'].to_dict()
    nei = []
    for i,j in pairs.items():
        for k in j:
            nei.append([i,k])
    
    pairs = gp.GeoDataFrame(nei, columns=['idx1','idx2'], crs=df1.crs)
    pairs = pairs.merge(df1, left_on='idx1', right_index=True)
    pairs = pairs.merge(df2, left_on='idx2', right_index=True, suffixes=['_1','_2'])
    pairs['Intersection'] = pairs.apply(lambda x: (x['geometry_1'].intersection(x['geometry_2'])).buffer(0), axis=1)
    pairs = gp.GeoDataFrame(pairs, columns=pairs.columns, crs=df1.crs)
    cols = pairs.columns.tolist()
    cols.remove('geometry_1')
    cols.remove('geometry_2')
    cols.remove('histreg')
    cols.remove('bbox')
    cols.remove('Intersection')
    dfinter = pairs[cols+['Intersection']].copy()
    dfinter.rename(columns={'Intersection':'geometry'}, inplace=True)
    dfinter = gp.GeoDataFrame(dfinter, columns=dfinter.columns, crs=pairs.crs)
    dfinter = dfinter.loc[dfinter.geometry.is_empty==False]
    return dfinter

intersect_shp = spatial_overlays(district, tract)

intersect_shp.shape            ## results through overlay

district_tract.shape          ## results through sjoin

tract.shape

district.shape

test_sample = intersect_shp[intersect_shp['SUP_DIST_N'] == '1']

test_sample = test_sample.to_crs(epsg = 4326)

ax = test_sample.plot( color='w',figsize=(12,12), edgecolor='k')
test_dis.plot(ax=ax,  color='w', edgecolor='r', alpha=0.5)
ax.axis('off');

#!curl -O http://egis3.lacounty.gov/dataportal/wp-content/uploads/ShapefilePackages/FIRE_REGION_BOUNDARIES.zip

#!unzip FIRE_REGION_BOUNDARIES.zip

fire = gp.GeoDataFrame.from_file('FIRE_REGION_BOUNDARIES.shp')

fire.plot(color='w', edgecolor='k', figsize=(12,12))

#!curl -O http://egis3.lacounty.gov/dataportal/wp-content/uploads/ShapefilePackages/CENSUS_BLOCKS_2010.zip

#!unzip CENSUS_BLOCKS_2010.zip

blocks = gp.GeoDataFrame.from_file('CENSUS_BLOCKS_2010.shp')



blocks.plot(color='w', edgecolor='k', figsize=(12,12))

intersect_blocks = sjoin(fire, blocks, how='right', op='intersects')

intersect_blocks.head()

intersect_blocks.shape

blocks.shape

fire.shape

intersect_blocks.plot(color='w', edgecolor='k', figsize=(12,12))



