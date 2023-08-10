get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function, division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import geopandas as gpd
from fiona.crs import from_epsg
from geopandas.tools import overlay

#Function for overlay
def spatial_overlays(df1, df2, how='intersection', reproject=True):
    """Perform spatial overlay between two polygons.

    Currently only supports data GeoDataFrames with polygons.
    Implements several methods that are all effectively subsets of
    the union.

    Parameters
    ----------
    df1 : GeoDataFrame with MultiPolygon or Polygon geometry column
    df2 : GeoDataFrame with MultiPolygon or Polygon geometry column
    how : string
        Method of spatial overlay: 'intersection', 'union',
        'identity', 'symmetric_difference' or 'difference'.
    use_sindex : boolean, default True
        Use the spatial index to speed up operation if available.

    Returns
    -------
    df : GeoDataFrame
        GeoDataFrame with new set of polygons and attributes
        resulting from the overlay

    """
    from functools import reduce
    df1 = df1.copy()
    df2 = df2.copy()
    df1['geometry'] = df1.geometry.buffer(0)
    df2['geometry'] = df2.geometry.buffer(0)
    if df1.crs!=df2.crs and reproject:
        print('Data has different projections.')
        print('Converted data to projection of first GeoPandas DatFrame')
        df2.to_crs(crs=df1.crs, inplace=True)
    if how=='intersection':
        # Spatial Index to create intersections
        spatial_index = df2.sindex
        df1['bbox'] = df1.geometry.apply(lambda x: x.bounds)
        df1['sidx']=df1.bbox.apply(lambda x:list(spatial_index.intersection(x)))
        pairs = df1['sidx'].to_dict()
        nei = []
        for i,j in pairs.items():
            for k in j:
                nei.append([i,k])
        pairs = gpd.GeoDataFrame(nei, columns=['idx1','idx2'], crs=df1.crs)
        pairs = pairs.merge(df1, left_on='idx1', right_index=True)
        pairs = pairs.merge(df2, left_on='idx2', right_index=True, suffixes=['_1','_2'])
        pairs['Intersection'] = pairs.apply(lambda x: (x['geometry_1'].intersection(x['geometry_2'])).buffer(0), axis=1)
        pairs = gpd.GeoDataFrame(pairs, columns=pairs.columns, crs=df1.crs)
        cols = pairs.columns.tolist()
        cols.remove('geometry_1')
        cols.remove('geometry_2')
        cols.remove('sidx')
        cols.remove('bbox')
        cols.remove('Intersection')
        dfinter = pairs[cols+['Intersection']].copy()
        dfinter.rename(columns={'Intersection':'geometry'}, inplace=True)
        dfinter = gpd.GeoDataFrame(dfinter, columns=dfinter.columns, crs=pairs.crs)
        dfinter = dfinter.loc[dfinter.geometry.is_empty==False]
        dfinter.drop(['idx1','idx2'], inplace=True, axis=1)
        return dfinter
    elif how=='difference':
        spatial_index = df2.sindex
        df1['bbox'] = df1.geometry.apply(lambda x: x.bounds)
        df1['sidx']=df1.bbox.apply(lambda x:list(spatial_index.intersection(x)))
        df1['new_g'] = df1.apply(lambda x: reduce(lambda x, y: x.difference(y).buffer(0), 
                                 [x.geometry]+list(df2.iloc[x.sidx].geometry)) , axis=1)
        df1.geometry = df1.new_g
        df1 = df1.loc[df1.geometry.is_empty==False].copy()
        df1.drop(['bbox', 'sidx', 'new_g'], axis=1, inplace=True)
        return df1
    elif how=='symmetric_difference':
        df1['idx1'] = df1.index.tolist()
        df2['idx2'] = df2.index.tolist()
        df1['idx2'] = np.nan
        df2['idx1'] = np.nan
        dfsym = df1.merge(df2, on=['idx1','idx2'], how='outer', suffixes=['_1','_2'])
        dfsym['geometry'] = dfsym.geometry_1
        dfsym.loc[dfsym.geometry_2.isnull()==False, 'geometry'] = dfsym.loc[dfsym.geometry_2.isnull()==False, 'geometry_2']
        dfsym.drop(['geometry_1', 'geometry_2'], axis=1, inplace=True)
        dfsym = gpd.GeoDataFrame(dfsym, columns=dfsym.columns, crs=df1.crs)
        spatial_index = dfsym.sindex
        dfsym['bbox'] = dfsym.geometry.apply(lambda x: x.bounds)
        dfsym['sidx'] = dfsym.bbox.apply(lambda x:list(spatial_index.intersection(x)))
        dfsym['idx'] = dfsym.index.values
        dfsym.apply(lambda x: x.sidx.remove(x.idx), axis=1)
        dfsym['new_g'] = dfsym.apply(lambda x: reduce(lambda x, y: x.difference(y).buffer(0), 
                         [x.geometry]+list(dfsym.iloc[x.sidx].geometry)) , axis=1)
        dfsym.geometry = dfsym.new_g
        dfsym = dfsym.loc[dfsym.geometry.is_empty==False].copy()
        dfsym.drop(['bbox', 'sidx', 'idx', 'idx1','idx2', 'new_g'], axis=1, inplace=True)
        return dfsym
    elif how=='union':
        dfinter = spatial_overlays(df1, df2, how='intersection')
        dfsym = spatial_overlays(df1, df2, how='symmetric_difference')
        dfunion = dfinter.append(dfsym)
        dfunion.reset_index(inplace=True, drop=True)
        return dfunion
    elif how=='identity':
        dfunion = spatial_overlays(df1, df2, how='union')
        cols1 = df1.columns.tolist()
        cols2 = df2.columns.tolist()
        cols1.remove('geometry')
        cols2.remove('geometry')
        cols2 = set(cols2).intersection(set(cols1))
        cols1 = list(set(cols1).difference(set(cols2)))
        cols2 = [col+'_1' for col in cols2]
        dfunion = dfunion[(dfunion[cols1+cols2].isnull()==False).values]
        return dfunion

#function for download data from egis3
def download_egis3(url, zipname, folder, shpname):
    #take in download url, downloaded zipfile name, folder name
    #containing unzipped shapefile, shapefile name
    #return the downloaded geodataframe
    os.system("curl -O " + url)
    os.system("unzip " + zipname + " -d " + folder)
    
    df = folder
    df = gpd.read_file(folder + "/" + shpname)

    df.crs = from_epsg(2229)
    df = df.to_crs(epsg=4326)
    return df

#function for reading downloaded census data from Census Bureau
def read_census(zipname, folder, shpname):
    os.system("curl -O " + url)
    os.system("unzip " + zipname + " -d " + folder)
    
    df = folder
    df = gpd.read_file(folder + "/" + shpname)

    return df 

#function to check the data quality
def checkdf(dfname):
    #take in geodataframe
    #return number of null values and unique values
    tota_poly = dfname.shape[0]
    print ('Total polygon number: {}'.format(tota_poly))    
    print ('number of null and unique values in each column:')
    
    for i in dfname.columns[:-1]:
        tota_valu = dfname.count()[i]
        null_valu = tota_poly - tota_valu
        uniq_valu = len(dfname[i].unique())
        print ('{}: null {}, unique {}'.format(i, null_valu, uniq_valu))
    return dfname.head(3)

#function to drop redundant columns
def cleandf(dfname, droplist):
    #take in geodataframe and list of column name redundant
    #return the clean df
    notneedlist = ['OBJECTID', 'Shape_area', 'Shape_len']
    
    for i in dfname.columns[:-1]:
        if i in notneedlist:
            dfname.drop([i], axis=1, inplace=True)
            
    for i in droplist:
        dfname.drop([i], axis=1, inplace=True)
    
    return dfname.head(3)

dfname_list = []

#Health Districts (HD) – 2012
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/2012/"       "02/HD_20121.zip"
health_districts = download_egis3(url, 'HD_20121.zip', 'health_districts', 'Health_Districts_2012.shp')
dfname_list.append('health_districts')

checkdf(health_districts)

droplist = ['SPA_NAME', 'SPA_2012']
cleandf(health_districts, droplist)

health_districts.plot()

#Law Enforcement Reporting Districts
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/"       "ShapefilePackages/LACOUNTY_LAW_ENFORCEMENT_RDs.zip"
law_enforcement = download_egis3(url, 'LACOUNTY_LAW_ENFORCEMENT_RDs.zip', 'law_enforcement', 'LACOUNTY_LAW_ENFORCEMENT_RDs.shp')
dfname_list.append('law_enforcement')
checkdf(law_enforcement)

droplist = ['Layer']
cleandf(law_enforcement, droplist)

law_enforcement.plot()

#School District Boundaries (2011)
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/2012/01/"       "rrcc_school_districts1.zip"
school_districts = download_egis3(url, 'rrcc_school_districts1.zip', 'school_districts', 'rrcc_school_districts.shp')
dfname_list.append('school_districts')
checkdf(school_districts)

droplist = ['UNIFIED', 'HIGH', 'ELEMENTARY', 'PH', 'ADDR', 'PH2', 'PH3',
       'STU', 'HI_ADDR', 'HI_PH', 'HI_STU', 'LABEL']
cleandf(school_districts, droplist)

school_districts.plot()

#California State Senate Districts (2011)
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/2011/11/"       "state-senate-2011.zip"
state_senate = download_egis3(url, 'state-senate-2011.zip', 'state_senate', 'senate.shp')
dfname_list.append('state_senate')
checkdf(state_senate)

droplist = ['LABEL']
cleandf(state_senate, droplist)

state_senate.plot()

#US Congressional Districts
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/"       "ShapefilePackages/RRCC_CONGRESSIONAL_DISTRICTS.zip"
congressional_districts = download_egis3(url, 'RRCC_CONGRESSIONAL_DISTRICTS.zip', 'congressional_districts', 'RRCC_CONGRESSIONAL_DISTRICTS.shp')
dfname_list.append('congressional_districts')
checkdf(congressional_districts)

droplist = []
cleandf(congressional_districts, droplist)

congressional_districts.plot()

#ZIP Code Boundaries
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/2011/01/"       "CAMS_ZIPCODE_PARCEL_SPECIFIC.zip"
zip_code = download_egis3(url, 'CAMS_ZIPCODE_PARCEL_SPECIFIC.zip', 'zip_code', 'CAMS_ZIPCODE_PARCEL_SPECIFIC.shp')
dfname_list.append('zip_code')
checkdf(zip_code)

droplist = []
cleandf(zip_code, droplist)

zip_code.plot()

#PUMA
get_ipython().system('wget http://www2.census.gov/geo/tiger/GENZ2016/shp/cb_2016_06_puma10_500k.zip')
puma = read_census('cb_2016_06_puma10_500k.zip', 'puma', 'cb_2016_06_puma10_500k.shp')
dfname_list.append('puma')
checkdf(puma)

droplist = ['STATEFP10', 'AFFGEOID10', 'GEOID10', 'LSAD10',
       'ALAND10', 'AWATER10']
cleandf(puma, droplist)

puma.plot()

# LAcounty_COMMUNITIES
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/2010/10/"       "Communities1.zip"
county_communities = download_egis3(url, 'Communities1.zip', 'county_communities', 'Communities.shp')
dfname_list.append('county_communities')
checkdf(county_communities)

droplist = ['COMMTYPE', 'COLOR', 'PO_NAME', 'STATNAME', 'X_CENTER',
       'Y_CENTER', 'ST_NAME', 'LABEL_CITY', 'LABEL_COMM', 'AREA_SQMI']
cleandf(county_communities, droplist)

county_communities.plot()

#Split 2010 Block Group/City – Community Statistical Area (formerly BASA)
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/2015/12/"       "EGIS_BG10FIPxx_CSA_20170118.zip"
community_stat_area = download_egis3(url, 'EGIS_BG10FIPxx_CSA_20170118.zip', 'community_stat_area', 'EGIS_BG10FIPxx_CSA_20170118.shp')
dfname_list.append('community_stat_area')
checkdf(community_stat_area)

droplist = ['BG10', 'CT10', 'FIP10', 'FIP11', 'FIP12', 'FIP13', 'FIP14',
       'FIP15', 'FIP16', 'CITY_TYPE', 'LCITY', 'LABEL', 'SOURCE', 'CT10FIP17',
            'DISTRICT', 'NOTES', 'PART', 'PARTS', 'MERGED', 'FIP17',
       'BG10FIP10', 'BG10FIP11', 'BG10FIP12', 'BG10FIP13', 'BG10FIP14',
       'BG10FIP15', 'BG10FIP16', 'CT10FIP10', 'CT10FIP11', 'CT10FIP12',
       'CT10FIP13', 'CT10FIP14', 'CT10FIP15', 'CT10FIP16', 'BG10FIP17', 
       'Shape_STAr', 'Shape_STLe']
cleandf(community_stat_area, droplist)

community_stat_area.plot()

#LA County TOWN_COUNCILS
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/2014/12/"       "DRP_TOWN_COUNCIL_AREAS.zip"
town_councils = download_egis3(url, 'DRP_TOWN_COUNCIL_AREAS.zip', 'town_councils', 'DRP_TOWN_COUNCIL_AREAS.shp')
dfname_list.append('town_councils')
checkdf(town_councils)

droplist = ['Id', 'SupDist', 'CertDate', 'SUB_REGION', 'TYPE',
       'DRP_NOTES', 'Shape_STAr', 'Shape_STLe']
cleandf(town_councils, droplist)

town_councils.plot()

#FIRE_DIVISION_BOUNDARIES
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/"       "ShapefilePackages/FIRE_DIVISION_BOUNDARIES.zip"
fire_division = download_egis3(url, 'FIRE_DIVISION_BOUNDARIES.zip', 'fire_division', 'FIRE_DIVISION_BOUNDARIES.shp')
dfname_list.append('fire_division')
checkdf(fire_division)

droplist = ['Shape_Leng', 'Shape_STAr', 'Shape_STLe']
cleandf(fire_division, droplist)

fire_division.plot()

#Los Angeles County Fire Department Battalion Boundaries
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/"       "ShapefilePackages/FIRE_BATTALION_BOUNDARIES.zip"
fire_battalion = download_egis3(url, 'FIRE_BATTALION_BOUNDARIES.zip', 'fire_battalion', 'FIRE_BATTALION_BOUNDARIES.shp')
dfname_list.append('fire_battalion')
checkdf(fire_battalion)

droplist = ['Shape_Leng', 'Shape_STAr', 'Shape_STLe']
cleandf(fire_battalion, droplist)

fire_battalion.plot()

#2011 Supervisorial District Boundaries (Official)
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/2011/12/"       "DPW-Supervisorial-District.zip"
supervisorial_district = download_egis3(url, 'DPW-Supervisorial-District.zip', 'supervisorial_district', 'sup_dist_2011.shp')
dfname_list.append('supervisorial_district')
checkdf(supervisorial_district)

droplist = ['SYMBOL', 'PERIMETER', 'AREA_SQ_MI', 'SHAPE_AREA', 'SHAPE_LEN']
cleandf(supervisorial_district, droplist)

supervisorial_district.plot()

#LA City Council Districts (2012)
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/2012/08/"       "CnclDist_July2012.zip"
LAcity_council_dist = download_egis3(url, 'CnclDist_July2012.zip', 'LAcity_council_dist', 'CnclDist_July2012.shp')
dfname_list.append('LAcity_council_dist')
checkdf(LAcity_council_dist)

droplist = ['AREA', 'PERIMETER', 'CDMEMBER', 'SQ_MI', 'SHADESYM',
       'Revised', 'Comments', 'SHAPE_Leng', 'SHAPE_Area']
cleandf(LAcity_council_dist, droplist)

LAcity_council_dist.plot()

#State Assembly Districts (State Legislative District – Lower Chamber)
get_ipython().system('wget https://www2.census.gov/geo/tiger/GENZ2016/shp/cb_2016_06_sldl_500k.zip')
legislative_lower = read_census('cb_2016_06_sldl_500k.zip', 'legislative_lower', 'cb_2016_06_sldl_500k.shp')
dfname_list.append('legislative_lower')
checkdf(legislative_lower)

droplist = ['STATEFP', 'SLDLST', 'AFFGEOID', 'GEOID', 'LSAD', 'LSY',
       'ALAND', 'AWATER']
cleandf(legislative_lower, droplist)

legislative_lower.plot()

#Registrar Recorder Precincts
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/"       "ShapefilePackages/RRCC_PRECINCTS.zip"
election_precinct = download_egis3(url, 'RRCC_PRECINCTS.zip', 'election_precinct', 'RRCC_PRECINCTS.shp')
dfname_list.append('election_precinct')
checkdf(election_precinct)

droplist = [ 'CITY_EST', 'SUBCODE', 'MAP1',
 'DIST_RES', 'DIST_MCRT', 'DIST_BEQ',
 'DST_CITY', 'DIV_CITY', 'DST_RES2', 'DIV_RES2',
 'DST_JRC', 'DIV_JRC', 'DST_USD', 'DIV_USD',
 'DST_HSD', 'DIV_HSD', 'DST_ESD', 'DIV_ESD',
 'DST_HOSP', 'DIV_HOSP', 'DST_PARK', 'DIV_PARK',
 'DST_WA', 'DIV_WA', 'DST_MWD', 'DIV_MWD',
 'DST_WR', 'DIV_WR', 'DST_WAG', 'DIV_WAG',
 'DST_CW', 'DIV_CW', 'DST_IRR', 'DIV_IRR',
 'DST_CS', 'DIV_CS', 'DST_LIB', 'DIV_LIB',
 'DST_RC', 'DIV_RC', 'DST_CAW', 'DIV_CAW',
 'DST_CEM', 'DIV_CEM', 'DST_MOS', 'DIV_MOS',
 'DST_SAN', 'DIV_SAN', 'DST_TRN', 'DIV_TRN',
 'DST_RES3', 'DIV_RES3', 'DST_FIR', 'DIV_FIR',
 'DST_FLD', 'DIV_FLD', 'DST_GARB', 'DIV_GARB',
 'DIST_OLDC', 'DIST_OLDS', 'DIST_OLDA', 'DST_CL', 'DIV_CL',
 'DST_SM', 'DIV_SM', 'DST_RD', 'DIV_RD',
 'DST_MISC1', 'DIV_MISC1', 'DST_MISC2', 'DIV_MISC2',
 'DST_MISC3', 'DIV_MISC3', 'DST_MISC4', 'DIV_MISC4',
 'DST_MISC5', 'DIV_MISC5', 'DST_MISC6', 'DIV_MISC6',
 'DST_ANX1', 'DIV_ANX1', 'DST_ANX2', 'DIV_ANX2',
 'DST_ANX3', 'DIV_ANX3', 'DST_ANX4', 'DIV_ANX4',
 'DST_ANX5', 'DIV_ANX5', 'CT_YEAR1', 'CT_YEAR2', 'CTRACT1',
 'CTRACT2', 'ESTAB', 'Shape_STAr', 'Shape_STLe']
cleandf(election_precinct, droplist)

election_precinct.plot()

#Comunity Plan Areas
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/2015/09/"       "LACITY_COMMUNITY_PLAN_AREAS.zip"
comunity_plan_area = download_egis3(url, 'LACITY_COMMUNITY_PLAN_AREAS.zip', 'comunity_plan_area', 'LACITY_COMMUNITY_PLAN_AREAS.shp')
dfname_list.append('comunity_plan_area')
checkdf(comunity_plan_area)

droplist = ['CPA', 'Shape_STAr', 'Shape_STLe']
cleandf(comunity_plan_area, droplist)

comunity_plan_area.plot()

#CDC Project Areas
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/"       "ShapefilePackages/CDC_PROJECT_AREAS.zip"
cdc_project = download_egis3(url, 'CDC_PROJECT_AREAS.zip', 'cdc_project', 'CDC_PROJECT_AREAS.shp')
dfname_list.append('cdc_project')
checkdf(cdc_project)

droplist = ['Division', 'Type', 'Active', 'Label', 'Shape_STAr', 'Shape_STLe']
cleandf(cdc_project, droplist)

cdc_project.plot()

#Census Tract
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/"       "ShapefilePackages/CENSUS_TRACTS_2010.zip"
census_tract = download_egis3(url, 'CENSUS_TRACTS_2010.zip', 'census_tract', 'CENSUS_TRACTS_2010.shp')
dfname_list.append('census_tract')
checkdf(census_tract)

droplist = ['LABEL', 'X_Center', 'Y_Center', 'Shape_STAr', 'Shape_STLe']
cleandf(census_tract, droplist)

census_tract.plot()

#Census Blocks
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/"       "ShapefilePackages/CENSUS_BLOCKS_2010.zip"
census_blocks = download_egis3(url, 'CENSUS_BLOCKS_2010.zip', 'census_blocks', 'CENSUS_BLOCKS_2010.shp')
dfname_list.append('census_blocks')
checkdf(census_blocks)

droplist = ['POP_2010', 'CT12', 'BG12', 'Shape_STAr', 'Shape_STLe',
           'HOUSING10', 'SUP_LABEL', 'SPA_NAME', 'CEN_FIP13', 'SPA_2012']
cleandf(census_blocks, droplist)

census_blocks.plot()

dfname_list

dfname_list = []

#Law Enforcement Reporting Districts
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/"       "ShapefilePackages/LACOUNTY_LAW_ENFORCEMENT_RDs.zip"
law_enforcement = download_egis3(url, 'LACOUNTY_LAW_ENFORCEMENT_RDs.zip', 'law_enforcement', 'LACOUNTY_LAW_ENFORCEMENT_RDs.shp')
dfname_list.append('law_enforcement')
checkdf(law_enforcement)

droplist = ['Layer']
cleandf(law_enforcement, droplist)

law_enforcement.columns = ['repo_dist_num', 'repo_dist_name',
                           'geometry']

#LAcounty_COMMUNITIES
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/2010/10/"       "Communities1.zip"
county_communities = download_egis3(url, 'Communities1.zip', 'county_communities', 'Communities.shp')
dfname_list.append('county_communities')
checkdf(county_communities)

droplist = ['COMMTYPE', 'COLOR', 'PO_NAME', 'STATNAME', 'X_CENTER',
       'Y_CENTER', 'ST_NAME', 'LABEL_CITY', 'LABEL_COMM', 'AREA_SQMI']
cleandf(county_communities, droplist)

county_communities.columns = ['coun_comm_name', 'geometry']

print ('overlay_shp {}'.format(law_enforcement.shape))
print ('county_communities {}'.format(county_communities.shape))

overlay_shp = spatial_overlays(law_enforcement, county_communities)

print ('After overlay: {}'.format(overlay_shp.shape))
#overlay_shp.plot()

#Registrar Recorder Precincts
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/"       "ShapefilePackages/RRCC_PRECINCTS.zip"
election_precinct = download_egis3(url, 'RRCC_PRECINCTS.zip', 'election_precinct', 'RRCC_PRECINCTS.shp')
dfname_list.append('election_precinct')
checkdf(election_precinct)

droplist = [ 'CITY_EST', 'SUBCODE', 'MAP1',
 'DIST_RES', 'DIST_MCRT', 'DIST_BEQ',
 'DST_CITY', 'DIV_CITY', 'DST_RES2', 'DIV_RES2',
 'DST_JRC', 'DIV_JRC', 'DST_USD', 'DIV_USD',
 'DST_HSD', 'DIV_HSD', 'DST_ESD', 'DIV_ESD',
 'DST_HOSP', 'DIV_HOSP', 'DST_PARK', 'DIV_PARK',
 'DST_WA', 'DIV_WA', 'DST_MWD', 'DIV_MWD',
 'DST_WR', 'DIV_WR', 'DST_WAG', 'DIV_WAG',
 'DST_CW', 'DIV_CW', 'DST_IRR', 'DIV_IRR',
 'DST_CS', 'DIV_CS', 'DST_LIB', 'DIV_LIB',
 'DST_RC', 'DIV_RC', 'DST_CAW', 'DIV_CAW',
 'DST_CEM', 'DIV_CEM', 'DST_MOS', 'DIV_MOS',
 'DST_SAN', 'DIV_SAN', 'DST_TRN', 'DIV_TRN',
 'DST_RES3', 'DIV_RES3', 'DST_FIR', 'DIV_FIR',
 'DST_FLD', 'DIV_FLD', 'DST_GARB', 'DIV_GARB',
 'DIST_OLDC', 'DIST_OLDS', 'DIST_OLDA', 'DST_CL', 'DIV_CL',
 'DST_SM', 'DIV_SM', 'DST_RD', 'DIV_RD',
 'DST_MISC1', 'DIV_MISC1', 'DST_MISC2', 'DIV_MISC2',
 'DST_MISC3', 'DIV_MISC3', 'DST_MISC4', 'DIV_MISC4',
 'DST_MISC5', 'DIV_MISC5', 'DST_MISC6', 'DIV_MISC6',
 'DST_ANX1', 'DIV_ANX1', 'DST_ANX2', 'DIV_ANX2',
 'DST_ANX3', 'DIV_ANX3', 'DST_ANX4', 'DIV_ANX4',
 'DST_ANX5', 'DIV_ANX5', 'CT_YEAR1', 'CT_YEAR2', 'CTRACT1',
 'CTRACT2', 'ESTAB', 'Shape_STAr', 'Shape_STLe',
           'DIST_SUP', 'CITY']
cleandf(election_precinct, droplist)

election_precinct.columns = ['precinct_num', 'congress_dist', 'senate_dist', 
                    'assembly_dist', 'community_num',
                    'community_name', 'area_num', 'area_name', 
                    'precinct_name', 'geometry']

print ('overlay_shp {}'.format(overlay_shp.shape))
print ('election_precinct {}'.format(election_precinct.shape))

overlay_shp = spatial_overlays(overlay_shp, election_precinct)

print ('After overlay: {}'.format(overlay_shp.shape))
#overlay_shp.plot()

#Census Blocks
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/"       "ShapefilePackages/CENSUS_BLOCKS_2010.zip"
census_blocks = download_egis3(url, 'CENSUS_BLOCKS_2010.zip', 'census_blocks', 'CENSUS_BLOCKS_2010.shp')
dfname_list.append('census_blocks')
checkdf(census_blocks)

droplist = ['POP_2010', 'CT12', 'BG12', 'Shape_STAr', 'Shape_STLe',
           'HOUSING10', 'SUP_LABEL', 'SPA_NAME', 'CEN_FIP13', 
            'SPA_2012', 'CTCB10', 'BG10FIP10', 'COMM', 'CITYCOM']
cleandf(census_blocks, droplist)

census_blocks.columns = ['census_tract', 'block_group', 
                         'census_block', 'FIP', 'city',
                         'zip_code', 'PUMA', 
                         'health_dist_num', 'health_districts_name',
                         'sup_dist', 'geometry']

print ('overlay_shp {}'.format(overlay_shp.shape))
print ('census_blocks {}'.format(census_blocks.shape))

overlay_shp = spatial_overlays(overlay_shp, census_blocks)

print ('After overlay: {}'.format(overlay_shp.shape))
#overlay_shp.plot()

overlay_shp.to_file("result.shp")

#School District Boundaries (2011)
url = "http://egis3.lacounty.gov/dataportal/wp-content/uploads/2012/01/"       "rrcc_school_districts1.zip"
school_districts = download_egis3(url, 'rrcc_school_districts1.zip', 'school_districts', 'rrcc_school_districts.shp')
dfname_list.append('school_districts')
checkdf(school_districts)

droplist = ['UNIFIED', 'HIGH', 'ELEMENTARY', 'PH', 'ADDR', 'PH2', 'PH3',
       'STU', 'HI_ADDR', 'HI_PH', 'HI_STU', 'LABEL']
cleandf(school_districts, droplist)

school_districts.columns = ['scho_dist_name', 'geometry']

print ('overlay_shp {}'.format(overlay_shp.shape))
print ('school_districts {}'.format(school_districts.shape))

overlay_shp = spatial_overlays(overlay_shp, school_districts)

print ('After overlay: {}'.format(overlay_shp.shape))
#overlay_shp.plot()

