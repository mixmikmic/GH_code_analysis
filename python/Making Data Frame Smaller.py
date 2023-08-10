get_ipython().run_line_magic('matplotlib', 'inline')
from shapely.geometry import Point, Polygon
import geopandas as gpd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
from geopandas import GeoSeries, GeoDataFrame

mpl.__version__, pd.__version__, gpd.__version__

data_path = "../big_data_leave"

file = "/Harvey_FEMA_HCAD_Damage.json"
filePath = data_path+file
df = gpd.read_file(filePath)

print(type(df))

df.head()

df.columns

type(df)

# one feature
{"type":"Feature","geometry":{"type":"Point","coordinates":[-95.73226883699994,29.735756641000023]},"properties":{"OBJECTID":2,"Join_Count":1,"TARGET_FID":5070,"DMG_LEVEL":"AFF","DMG_TYPE":"FL","ASMT_TYPE":"MOD","IN_DEPTH":0.18953704834,"WIND_SPEED":"UNK","PGA":0,"ACCESS":"UNK","COUNTY":"Fort Bend","STATE":"TX","FIPS":48157,"PROD_DATE":"2017-09-02T00:00:00.000Z","IMG_DATE":"1899-11-30T00:00:00.000Z","EVENT_NAME":"Hurricane Harvey","EVENT_DATE":"1899-11-30T00:00:00.000Z","SOURCE":"FEMA","DIS_NUMBER":"DR-4332","COMMENTS":"","LONGITUDE":-95.7323,"LATITUDE":29.7358,"USNG":"","HCAD_NUM":"1202010030003","BLK_NUM":"","LOT_NUM":"","CONDO_FLAG":"0","parcel_typ":"0","CurrOwner":"HARDESTY LISA A &","LocAddr":"20407 ARROW FIELD LN","city":"KATY","zip":"77450","LocNum":20407,"LocName":"ARROW FIELD","Shape_STAr":10231.328125,"Shape_STLe":416.719006112,"ACCOUNT":"1202010030003","TAX_YEAR":"2017","MAILTO":"HARDESTY LISA A &","MAIL_ADDR_":"20407 ARROW FIELD LN","MAIL_ADDR1":"","MAIL_CITY":"KATY","MAIL_STATE":"TX","MAIL_ZIP":"77450-7416","MAIL_COUNT":"","UNDELIVERA":"N","STR_PFX":"","STR_NUM":20407,"STR_NUM_SF":"","STR_NAME":"ARROW FIELD","STR_SFX":"LN","STR_SFX_DI":"","STR_UNIT":"","SITE_ADDR_":"20407 ARROW FIELD LN","SITE_ADDR1":"KATY","SITE_ADD_1":"77450","STATE_CLAS":"A1","SCHOOL_DIS":"19","MAP_FACET":"4556C","KEY_MAP":"486T","NEIGHBORHO":"2915.08","NEIGHBOR_1":"19015","MARKET_ARE":"341","MARKET_A_1":"ISD 19 - South of I-10 Katy Freeway","MARKET_A_2":"341","MARKET_A_3":"ISD 19 - South of I-10 Katy Freeway","ECON_AREA":"","ECON_BLD_C":"","CENTER_COD":"90","YR_IMPR":"2000","YR_ANNEXED":"","SPLT_DT":"","DSC_CD":"","NXT_BUILDI":"2","TOTAL_BUIL":"2161","TOTAL_LAND":"10298","ACREAGE":".2364","CAP_ACCOUN":"N","SHARED_CAD":"Y","LAND_VALUE":"41622","IMPROVEMEN":"209721","EXTRA_FEAT":"0","AG_VALUE":"0","ASSESSED_V":"251343","TOTAL_APPR":"251343","TOTAL_MARK":"251343","PRIOR_LND_":"41622","PRIOR_IMPR":"209721","PRIOR_X_FE":"0","PRIOR_AG_V":"0","PRIOR_TOTA":"251343","PRIOR_TO_1":"251343","NEW_CONSTR":"0","TOTAL_RCN_":"263198","VALUE_STAT":"Noticed","NOTICED":"Y","NOTICE_DAT":"2017-03-31 00:00:00.000","PROTESTED":"N","CERTIFIED_":"2017-08-11 00:00:00.000","LAST_INSPE":"2010-02-15 00:00:00.000","LAST_INS_1":"01348","NEW_OWNER_":"2001-02-23 00:00:00.000","LEGAL_DSCR":"LT 3 BLK 3","LEGAL_DS_1":"(HC* L 8% & I 0%)","LEGAL_DS_2":"CINCO RANCH EQUESTRIAN VILLAGE SEC 3","LEGAL_DS_3":"","JURS":"Split Account: See Split Jurs","ACCOUNT_1":"1202010030003","BUILDING_N":1,"CODE":"4","ADJ_CD":".810000","STRUCTURE_":"CDU","TYPE_DESCR":"Cond / Desir / Util","CATEGORY_D":"Average","STATE_CL_1":"A1","ACCOUNT_12":"1202010030003","LINE_NUMBE":1,"LAND_USE_C":"1001","LAND_USE_D":"Res Improved Table Value","SITE_CD":"SF1","SITE_CD_DS":"Primary SF","SITE_ADJ":"1.0000","UNIT_TYPE":"SF","UNITS":"7500.0000","SIZE_FACTO":"1.0000","SITE_FACT":"1.0000","APPR_OVERR":"1.0000","APPR_OVE_1":"","TOT_ADJ":"1.0000","UNIT_PRICE":"5.35","ADJ_UNIT_P":"5.3500","VALUE":"40125","OVERRIDE_V":""}},

### NOTE: This takes about 7-12 minutes to run.
df.plot()

df_reduced = df[['ACCOUNT','HCAD_NUM','KEY_MAP','ACREAGE','CONDO_FLAG','LAND_USE_C','COUNTY','LAND_USE_D','LAND_VALUE','DMG_LEVEL','FIPS','IMPROVEMEN','IN_DEPTH','LATITUDE','LONGITUDE','LocNum','LocName','LocAddr','MAIL_CITY','city','MAIL_STATE','MAIL_ZIP','SCHOOL_DIS','NEIGHBORHO','NEIGHBOR_1','NEW_CONSTR','TAX_YEAR','TOTAL_APPR','TOTAL_BUIL','TOTAL_LAND','TOTAL_MARK','TOTAL_RCN_','UNITS','UNIT_PRICE','UNIT_TYPE','YR_IMPR']]

df_reduced.head()

type(df_reduced)

### NOTE: This takes about 7-12 minutes to run.
df.plot()

df_reduced.city.unique()

# make a dataframe from reduced data frame that is only katy.
df_reduced_katy = df_reduced[df_reduced['city'].str.contains("KATY")]
df_reduced_katy.head()

df_reduced_katy.describe()

type(df_reduced_katy)

geometry = [Point(xy) for xy in zip(df_reduced_katy.LONGITUDE, df_reduced_katy.LATITUDE)]
crs = {'init':'epsg:4326'}
df_reduced_katy_gpd = gpd.GeoDataFrame(df_reduced_katy, crs=crs, geometry=geometry)
type(df_reduced_katy_gpd)

katy_out = data_path+"/Harvey_FEMA_HCAD_Damage_reduced_katy2.geojson"

df_reduced_katy_gpd.to_file(katy_out, driver='GeoJSON')

df_reduced_katy_gpd.plot()

# df_reduced
geometry = [Point(xy) for xy in zip(df_reduced.LONGITUDE, df_reduced.LATITUDE)]
crs = {'init':'epsg:4326'}
df_reduced_gpd = gpd.GeoDataFrame(df_reduced, crs=crs, geometry=geometry)
type(df_reduced_gpd)

reduced_out = data_path+"/Harvey_FEMA_HCAD_Damage_reduced.geojson"

df_reduced_gpd.to_file(reduced_out, driver='GeoJSON')



