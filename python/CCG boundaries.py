import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

headers = [
    'code','name','postcode','status_code','ccg','setting']
epraccur = pd.read_csv("epraccur_may_2017.csv", names=headers, usecols=[0,1,9,12,23,25])

epraccur.head()

lsoa_lookup = pd.read_csv(
    "LSOA_2011_to_Clinical_Commissioning_Groups_to_Sustainability_and_Transformation_Partnerships_April_2017_Lookup_in_England.csv",
    names=['lsoa','ccg17cd', 'ccg','name'],
    header=0,
    usecols=[0,2,3,4]
)

lsoa_lookup.head()

boundaries = gpd.read_file('Clinical_Commissioning_Groups_April_2017_Full_Clipped_Boundaries_in_England_V4.shp')

boundaries.head()

boundaries.plot()

postcode_lookup = pd.read_csv(
    "NHS_Postcode_Directory_Latest_Centroids.csv",
    usecols=[0,1,2,3,4,19]
)

ccg_lookup = lsoa_lookup[['ccg17cd', 'ccg']].drop_duplicates().set_index('ccg17cd')
boundaries.set_index('ccg17cd', inplace=True)
boundaries_with_ccg = boundaries.join(ccg_lookup)

boundaries_with_ccg.head()

#postcode_lookup = postcode_lookup.set_index('pcd2')
epraccur_with_centroids = epraccur.merge(postcode_lookup, left_on='postcode', right_on='pcds', how='left')
#epraccur_with_centroids = postcode_lookup.join(epraccur, how='left', on='postcode', rsuffix='postcode')

epraccur_with_centroids.head()

epraccur_with_centroids = epraccur_with_centroids[epraccur_with_centroids.X.notnull()]
epraccur_with_centroids['geometry'] = epraccur_with_centroids.apply(lambda z: Point(z.X, z.Y), axis=1)
epraccur_with_centroids = gpd.GeoDataFrame(epraccur_with_centroids)
epraccur_with_centroids.head()

epraccur_with_centroids.crs = {'init' :'epsg:4326'}
boundaries_with_ccg.crs = {'init' :'epsg:27700'}
epraccur_with_centroids = epraccur_with_centroids.to_crs({'init': 'epsg:27700'})

practices_with_ccg = gpd.sjoin(epraccur_with_centroids, boundaries_with_ccg.reset_index(), how="left")

practices_with_ccg.head()

matching = practices_with_ccg[practices_with_ccg.ccg_x == practices_with_ccg.ccg]
non_matching = practices_with_ccg[practices_with_ccg.ccg_x != practices_with_ccg.ccg]
print "There are %s practices whose inferred CCG (based on location) does not match their stated one" % len(non_matching)

# discard uninteresting columns
non_matching = non_matching[['code', 'name', 'postcode', 'status_code', 'ccg_x', 'ccg']]
non_matching.head()

non_matching.groupby('status_code').code.agg('count')

non_matching[non_matching.status_code == 'A'].head(20)

epraccur[epraccur.ccg == '13Q'].count()

lsoa_lookup[lsoa_lookup.ccg == '13Q'].count()

for group, df in non_matching.groupby('ccg_x'):
    print group

non_matching[non_matching.ccg_x == '00X']

non_matching[non_matching.ccg_x == '00F']

non_matching[non_matching.ccg_x == '03V']

