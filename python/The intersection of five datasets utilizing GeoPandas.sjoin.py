get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function, division
import pandas as pd
import geopandas as gpd
from matplotlib.pylab import plt
from fiona.crs import from_epsg

# Law Enforcement Reporting Districts: 
#http://egis3.lacounty.gov/dataportal/wp-content/uploads/ShapefilePackages/LACOUNTY_LAW_ENFORCEMENT_RDs.zip

filepath = '/Users/DXM/Documents/ARGO/ops'
law_enforcement = gpd.read_file(filepath + '/LACOUNTY_LAW_ENFORCEMENT_RDs/LACOUNTY_LAW_ENFORCEMENT_RDs.shp')
law = law_enforcement.to_crs(epsg=4269)
law = gpd.GeoDataFrame(law_enforcement)

law.plot()

# LAcounty_COMMUNITIES
#http://egis3.lacounty.gov/dataportal/wp-content/uploads/2010/10/Communities1.zip

community = gpd.read_file(filepath + '/Communities1/Communities.shp')
community = community.to_crs(epsg=4269)
co = gpd.GeoDataFrame(community)

co.plot()

# Registrar Recorder Precincts
#http://egis3.lacounty.gov/dataportal/wp-content/uploads/ShapefilePackages/RRCC_PRECINCTS.zip

registar = gpd.read_file(filepath + '/RRCC_PRECINCTS/RRCC_PRECINCTS.shp')
registar = registar.to_crs(epsg=4269)
rrp = gpd.GeoDataFrame(registar)

rrp.plot()

# Census Block (2010)
#http://egis3.lacounty.gov/dataportal/wp-content/uploads/ShapefilePackages/CENSUS_BLOCKS_2010.zip

blocks = gpd.read_file(filepath + '/CENSUS_BLOCKS_2010/CENSUS_BLOCKS_2010.shp')
blocks = registar.to_crs(epsg=4269)
bl = gpd.GeoDataFrame(blocks)

bl.plot()

# School District Boundaries (2011)
#http://egis3.lacounty.gov/dataportal/wp-content/uploads/2012/01/rrcc_school_districts1.zip

school = gpd.read_file(filepath + '/rrcc_school_districts1/rrcc_school_districts.shp')
school = registar.to_crs(epsg=4269)
sdb = gpd.GeoDataFrame(school)

sdb.plot()

co_law = gpd.sjoin(co, law, how='inner', op='intersects')

co_law.head()

rrp_co = gpd.sjoin(rrp, co, how='inner', op='intersects')

fig, ax = plt.subplots(figsize=(12,16))
rrp_co.plot(linewidth=0.8, alpha=1, axes=ax, edgecolor = 'k')
ax.set_title('Boundaries of LA County', fontsize=20)

bl_sdb = gpd.sjoin(bl, sdb, how='inner', op='intersects')

fig, ax = plt.subplots(figsize=(12,16))
bl_sdb.plot(linewidth=0.8, alpha=1, axes=ax, edgecolor = 'k')
ax.set_title('Boundaries of LA County', fontsize=20)

bl_sdb = bl_sdb.drop(['index_right'], axis=1)

rrp_co = rrp_co.drop(['index_right'], axis=1)

intersection = gpd.sjoin(bl_sdb, rrp_co, how='inner', op='intersects')

intersection.shape

fig, ax = plt.subplots(figsize=(12,16))
intersection.plot(linewidth=0.8, alpha=1, axes=ax, edgecolor = 'k')
ax.set_title('Intersected Boundaries of LA County', fontsize=20)



