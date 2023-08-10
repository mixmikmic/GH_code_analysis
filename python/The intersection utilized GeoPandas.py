get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function, division
import pandas as pd
import geopandas as gpd
from matplotlib.pylab import plt
from fiona.crs import from_epsg

# http://egis3.lacounty.gov/dataportal/2011/12/06/supervisorial-districts
filepath = '/Users/DXM/Documents/ARGO/ops'

supervisor = gpd.read_file(filepath + '/DPW-Supervisorial-District/sup_dist_2011.shp')
sup = supervisor.to_crs(epsg=4269)
sup = gpd.GeoDataFrame(supervisor)

# https://www.census.gov/geo/maps-data/data/cbf/cbf_puma.html

puma = gpd.read_file(filepath + '/cb_2016_06_puma10_500k/cb_2016_06_puma10_500k.shp')
pu = puma.to_crs(epsg=4269)
pu = gpd.GeoDataFrame(puma)

# http://egis3.lacounty.gov/dataportal/2012/03/01/health-districts-hd-2012/

health = gpd.read_file(filepath + '/HD_20121/Health_Districts_2012.shp')
hd = health.to_crs(epsg=4269)
hd = gpd.GeoDataFrame(health)

# http://egis3.lacounty.gov/dataportal/wp-content/uploads/2012/01/rrcc_school_districts1.zip

school = gpd.read_file(filepath + '/rrcc_school_districts1/rrcc_school_districts.shp')
sdb = school.to_crs(epsg=4269)
sdb = gpd.GeoDataFrame(school)

# http://egis3.lacounty.gov/dataportal/2011/11/08/california-state-senate-districts-2011/

senate = gpd.read_file(filepath + '/state-senate-2011/senate.shp')
se = senate.to_crs(epsg=4269)
se = gpd.GeoDataFrame(senate)

# http://egis3.lacounty.gov/dataportal/2010/01/14/us-congressional-districts/

congression = gpd.read_file(filepath + '/RRCC_CONGRESSIONAL_DISTRICTS/RRCC_CONGRESSIONAL_DISTRICTS.shp')
con = congression.to_crs(epsg=4269)
con = gpd.GeoDataFrame(congression)

# http://egis3.lacounty.gov/dataportal/2011/11/08/california-state-assembly-districts-2011/

assembly = gpd.read_file(filepath + '/state_assembly_districts/state_assembly_districts.shp')
asse = gpd.GeoDataFrame(assembly)

#http://egis3.lacounty.gov/dataportal/wp-content/uploads/ShapefilePackages/RRCC_PRECINCTS.zip

registar = gpd.read_file(filepath + '/RRCC_PRECINCTS/RRCC_PRECINCTS.shp')
registar = registar.to_crs(epsg=4269)
rrp = gpd.GeoDataFrame(registar)

#http://egis3.lacounty.gov/dataportal/wp-content/uploads/ShapefilePackages/LACOUNTY_LAW_ENFORCEMENT_RDs.zip

filepath = '/Users/DXM/Documents/ARGO/ops'
law_enforcement = gpd.read_file(filepath + '/LACOUNTY_LAW_ENFORCEMENT_RDs/LACOUNTY_LAW_ENFORCEMENT_RDs.shp')
law = law_enforcement.to_crs(epsg=4269)
law = gpd.GeoDataFrame(law_enforcement)

#http://egis3.lacounty.gov/dataportal/wp-content/uploads/2010/10/Communities1.zip

community = gpd.read_file(filepath + '/Communities1/Communities.shp')
community = community.to_crs(epsg=4269)
co = gpd.GeoDataFrame(community)

#http://egis3.lacounty.gov/dataportal/wp-content/uploads/ShapefilePackages/CENSUS_BLOCKS_2010.zip

blocks = gpd.read_file(filepath + '/CENSUS_BLOCKS_2010/CENSUS_BLOCKS_2010.shp')
blocks = registar.to_crs(epsg=4269)
bl = gpd.GeoDataFrame(blocks)

sup_pu = gpd.sjoin(sup, pu, how='inner', op='intersects')

sup_pu.head()

sdb_hd = gpd.sjoin(sdb, hd, how = 'inner', op='intersects')

sdb_hd.plot()

se_con = gpd.sjoin(con, se, how='inner', op='intersects')

se_con.plot()

rrp_law = gpd.sjoin(rrp, law, how = 'inner', op='intersects' )

rrp_law.head()

bl_co = gpd.sjoin(bl, co, how='inner', op='intersects')

bl_co.head(1)

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



