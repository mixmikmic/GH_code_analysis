##library dependencies

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from numpy import loadtxt
from numpy import sort
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import bokeh
import sklearn
import xgboost as xgb
import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.mlab as mlab
import sqlite3
import matplotlib.pyplot as plt
from scipy import stats


##import properties 2016  data

prop2016= pd.read_csv('C:\\Users\\Dell\\Desktop\\inf552\\Zillow Prize\\properties_2016.csv')

##import train 2016  data
train2016=pd.read_csv('C:\\Users\\Dell\\Desktop\\inf552\\Zillow Prize\\train_2016.csv',parse_dates=["transactiondate"])

prop2016=prop2016.rename(index=str, columns={"pooltypeid7": "pool_with_spa_tub_no", 
                                             "pooltypeid2": "pool_with_spa_tub_yes"})

prop2016=prop2016.rename(index=str, columns={"hashottuborspa": "has_hottub_or_spa", 
                                             "pooltypeid10": "just_hottub_or_spa"})

prop2016=prop2016.rename(index=str, 
                         columns={"calculatedfinishedsquarefeet": "total_finished_living_area_sqft", 
                                  "finishedsquarefeet12": "finished_living_area_sqft",
                                  "lotsizesquarefeet": "lot_area_sqft",
                                  "finishedsquarefeet50": "finished_living_area_entryfloor_sqft1",
                                  "finishedfloor1squarefeet": "finished_living_area_entryfloor_sqft2",
                                  "finishedsquarefeet6": "base_unfinished_and_finished_area_sqft",
                                  "finishedsquarefeet15": "total_area_sqft",
                                  "finishedsquarefeet13": "preimeter_living_area_sqft",
                                  "taxvaluedollarcnt":"total_parcel_tax",
                                  "landtaxvaluedollarcnt":"land_tax",
                                  "taxamount":"total_property_tax_2016",
                                  "structuretaxvaluedollarcnt":"structure_tax",
                                  "garagetotalsqft":"garage_sqft",
                                  "fireplacecnt":"fireplace_count",
                                  "buildingqualitytypeid ":"building_quality_id",
                                  "heatingorsystemtypeid":"heating_system_id",
                                  "airconditioningtypeid":"ac_id"
                                 
                                 
                                 
                                 })
prop2016=prop2016.rename(index=str, columns={"decktypeid": "deck_flag"})

prop2016=prop2016.rename(index=str, columns={"yardbuildingsqft17": "patio_sqft",
                                             "yardbuildingsqft26":"storage_sqft"})

prop2016=prop2016.rename(index=str, columns={"bathroomcnt": "total_bath",
                                             "fullbathcnt": "full_bath",
                                             "threequarterbathnbr": "half_bath"})

prop2016=prop2016.rename(index=str, columns={"storytypeid": "basement_flag"})

prop2016=prop2016.rename(index=str, columns={"poolsizesum": "pool_sqft"})

             

prop2016.loc[prop2016.poolcnt.isnull(),'poolcnt']= 0

prop2016.loc[prop2016.pool_with_spa_tub_no.isnull(),'pool_with_spa_tub_no']=0
prop2016.loc[prop2016.pool_with_spa_tub_yes.isnull(),'pool_with_spa_tub_yes']=0

#when poolcnt=1 & has_hottub_or_spa=1 & just_hottub_or_spa is null then just_hottub_or_spa =0
#when poolcnt=0, has_hottub_or_spa=1, just_hottub_or_spa =1

prop2016.loc[ (prop2016.poolcnt==1) & (prop2016.has_hottub_or_spa==1) & (prop2016.just_hottub_or_spa.isnull()),'just_hottub_or_spa']=0
             
#has_hottub_or_spa is null and just_hottub_or_spa is null, both has to be zero

prop2016.loc[ (prop2016.has_hottub_or_spa.isnull()) & (prop2016.just_hottub_or_spa.isnull()),['has_hottub_or_spa','just_hottub_or_spa']]=0


prop2016.loc[ prop2016.poolcnt==0,'pool_sqft']=0
print(prop2016.pool_sqft.isnull().sum())

prop2016.loc[prop2016.basement_flag.isnull(),'basementsqft']=0
prop2016.loc[prop2016.basement_flag.isnull(),'basement_flag']=0
prop2016.loc[prop2016.basement_flag==7,'basement_flag']=1


#df_train.fireplaceflag.isnull().sum()
#df_train.fireplace_count.isnull().sum()
prop2016.loc[(prop2016.fireplace_count.isnull()) & (prop2016.fireplaceflag.isnull()),'fireplaceflag'] = False
prop2016.loc[(prop2016.fireplace_count.isnull()) & (prop2016.fireplaceflag==False),'fireplace_count'] = 0
prop2016.loc[prop2016['fireplace_count']>0,'fireplaceflag']= True

print("after",prop2016.fireplace_count.isnull().sum())
#print("after",df_train.fireplace_count.value_counts())


prop2016.loc[prop2016.garage_sqft.isnull() & prop2016.garagecarcnt.isnull(),['garagecarcnt','garage_sqft']]=0
prop2016.loc[(prop2016.garagecarcnt>0) & (prop2016.garage_sqft==0),'garage_sqft']=np.nan
print("after",prop2016.garagecarcnt.isnull().sum())
#print("after",df_train.garagecarcnt.value_counts())
print("after",prop2016.garage_sqft.isnull().sum())
#print("after",df_train.garage_sqft.value_counts())

#total_bath & calculatedbathnbr are duplicates , and calculatedbathnbr has more nulls, hence drop it

prop2016=prop2016.drop('calculatedbathnbr',axis=1)

# full_bath is null & half_bath is null & total_bath=0 (missing values)
prop2016.loc[(prop2016.full_bath.isnull()) & (prop2016.half_bath.isnull()) & (prop2016.total_bath==0),'total_bath']=np.nan


# when full_bath=total_bath, half_bath=0 

prop2016.loc[(prop2016.full_bath==prop2016.total_bath) ,'half_bath']=0

# when total_bath is present but full and half bath is null
# all 3 are null somemtimes

print(prop2016.total_bath.isnull().sum())
print(prop2016.half_bath.isnull().sum())
print(prop2016.full_bath.isnull().sum())

#yardbuildingsqft17-patio in yard
#yardbuildingsqft26- storage shed in yard

prop2016.loc[prop2016.patio_sqft.isnull() ,'patio_sqft']=0
prop2016.loc[prop2016.storage_sqft.isnull() ,'storage_sqft']=0

print(prop2016.patio_sqft.isnull().sum())
print(prop2016.storage_sqft.isnull().sum())

#fips
prop2016.loc[prop2016.fips==6037 ,'fips']="LA"
prop2016.loc[prop2016.fips==6059 ,'fips']="Orange_County"
prop2016.loc[prop2016.fips==6111 ,'fips']="Ventura"
print(prop2016.fips.isnull().sum())
print(prop2016.fips.value_counts())


prop2016['latitude']=prop2016['latitude'].divide(100000)
prop2016['longitude']=prop2016['longitude'].divide(100000)

prop2016.loc[prop2016.deck_flag==66 ,'deck_flag']=1
prop2016.loc[prop2016.deck_flag.isnull() ,'deck_flag']=0
print(prop2016.deck_flag.isnull().sum())

#numberofstories & unitcnt &roomcnt

prop2016.loc[prop2016.roomcnt==0 ,'roomcnt']=np.nan

print(prop2016.numberofstories.isnull().sum())
print(prop2016.roomcnt.isnull().sum())
print(prop2016.unitcnt.isnull().sum())

# propertylandusetypeid  and unitcnt is related 
#246 -Duplex (2 Units, Any Combination)
#247 -Triplex (3 Units, Any Combination)
#248 -Quadruplex (4 Units, Any Combination)
#260 -Residential General
#261 -Single Family Residential
#263 -Mobile Home
#264 -Townhouse
#266 -Condominium
#267 -Cooperative
#269 -Planned Unit Development
#275 -Residential Common Area 
#31 - Commercial/Office/Residential Mixed Used
#47 -Store/Office (Mixed Use)
#265 -Cluster Home

prop2016.loc[(prop2016.propertylandusetypeid==31) & (prop2016.unitcnt.isnull()),'unitcnt']=2
prop2016.loc[(prop2016.propertylandusetypeid==47) & (prop2016.unitcnt.isnull()),'unitcnt']=2
prop2016.loc[(prop2016.propertylandusetypeid==246) & (prop2016.unitcnt.isnull()),'unitcnt']=2
prop2016.loc[(prop2016.propertylandusetypeid==247) & (prop2016.unitcnt.isnull()),'unitcnt']=3
prop2016.loc[(prop2016.propertylandusetypeid==248) & (prop2016.unitcnt.isnull()),'unitcnt']=4
prop2016.loc[(prop2016.propertylandusetypeid==260) & (prop2016.unitcnt.isnull()),'unitcnt']=1
prop2016.loc[(prop2016.propertylandusetypeid==261) & (prop2016.unitcnt.isnull()),'unitcnt']=1
prop2016.loc[(prop2016.propertylandusetypeid==263) & (prop2016.unitcnt.isnull()),'unitcnt']=1
prop2016.loc[(prop2016.propertylandusetypeid==264) & (prop2016.unitcnt.isnull()),'unitcnt']=1
prop2016.loc[(prop2016.propertylandusetypeid==266) & (prop2016.unitcnt.isnull()),'unitcnt']=1
prop2016.loc[(prop2016.propertylandusetypeid==267) & (prop2016.unitcnt.isnull()),'unitcnt']=1
prop2016.loc[(prop2016.propertylandusetypeid==269) & (prop2016.unitcnt.isnull()),'unitcnt']=1
prop2016.loc[(prop2016.propertylandusetypeid==275) & (prop2016.unitcnt.isnull()),'unitcnt']=1
prop2016.loc[(prop2016.propertylandusetypeid==265) & (prop2016.unitcnt.isnull()),'unitcnt']=10
prop2016.loc[(prop2016.propertylandusetypeid==270) & (prop2016.unitcnt.isnull()),'unitcnt']=1


#typeconstructiontypeid (based on location and year of building)

print(prop2016.propertylandusetypeid.isnull().sum())
print(prop2016.unitcnt.isnull().sum())
#print(prop2016.propertylandusetypeid.value_counts())

prop2016.loc[prop2016.propertylandusetypeid==270,'unitcnt'].unique()

prop2016=prop2016.drop('preimeter_living_area_sqft', axis=1)
prop2016=prop2016.drop('total_area_sqft', axis=1)
prop2016=prop2016.drop('finished_living_area_sqft', axis=1)
prop2016=prop2016.drop('base_unfinished_and_finished_area_sqft', axis=1)

#calculatedfinishedsquarefeet": "total_finished_living_area_sqft", 
#"finishedsquarefeet12": "finished_living_area_sqft",
#"lotsizesquarefeet": "lot_area_sqft",
#"finishedsquarefeet50": "finished_living_area_entryfloor_sqft1",
#finishedfloor1squarefeet": "finished_living_area_entryfloor_sqft2",
#"finishedsquarefeet6": "base_unfinished_and_finished_area_sqft",
#"finishedsquarefeet15": "total_area_sqft",
#"finishedsquarefeet13": "preimeter_living_area_sqft"

#propertycountylandusecode
#print(df_train.propertycountylandusecode.isnull().sum())

prop2016=prop2016.drop(['propertycountylandusecode'], axis=1)

#propertyzoningdesc, same reason as above applies to this column as well

prop2016=prop2016.drop(['propertyzoningdesc'], axis=1)
#print(df_train.propertyzoningdesc.isnull().sum())

prop2016=prop2016.drop(['regionidzip','regionidcity','regionidneighborhood'], axis=1)

prop2016.assessmentyear.value_counts()
prop2016=prop2016.drop(['assessmentyear'], axis=1)

prop2016.loc[prop2016.taxdelinquencyflag.isnull(),'taxdelinquencyflag'] = "No"
prop2016.loc[prop2016.taxdelinquencyflag=='Y','taxdelinquencyflag'] = "Yes"
prop2016.loc[prop2016.taxdelinquencyflag=='No','taxdelinquencyyear'] = 0
#print(df_train.taxdelinquencyyear.value_counts())

print(prop2016.taxdelinquencyyear.value_counts())


prop2016.loc[prop2016.taxdelinquencyyear==79,'taxdelinquencyyear'] = 1979
prop2016.loc[prop2016.taxdelinquencyyear==82,'taxdelinquencyyear'] = 1982
prop2016.loc[prop2016.taxdelinquencyyear==84,'taxdelinquencyyear'] = 1984
prop2016.loc[prop2016.taxdelinquencyyear==86,'taxdelinquencyyear'] = 1986
prop2016.loc[prop2016.taxdelinquencyyear==87,'taxdelinquencyyear'] = 1987
prop2016.loc[prop2016.taxdelinquencyyear==88,'taxdelinquencyyear'] = 1988
prop2016.loc[prop2016.taxdelinquencyyear==90,'taxdelinquencyyear'] = 1990
prop2016.loc[prop2016.taxdelinquencyyear==91,'taxdelinquencyyear'] = 1991
prop2016.loc[prop2016.taxdelinquencyyear==92,'taxdelinquencyyear'] = 1992
prop2016.loc[prop2016.taxdelinquencyyear==93,'taxdelinquencyyear'] = 1993
prop2016.loc[prop2016.taxdelinquencyyear==94,'taxdelinquencyyear'] = 1994
prop2016.loc[prop2016.taxdelinquencyyear==95,'taxdelinquencyyear'] = 1995
prop2016.loc[prop2016.taxdelinquencyyear==96,'taxdelinquencyyear'] = 1996
prop2016.loc[prop2016.taxdelinquencyyear==97,'taxdelinquencyyear'] = 1997
prop2016.loc[prop2016.taxdelinquencyyear==98,'taxdelinquencyyear'] = 1998
prop2016.loc[prop2016.taxdelinquencyyear==99,'taxdelinquencyyear'] = 1999
prop2016.loc[prop2016.taxdelinquencyyear==1,'taxdelinquencyyear'] = 2001
prop2016.loc[prop2016.taxdelinquencyyear==2,'taxdelinquencyyear'] = 2002
prop2016.loc[prop2016.taxdelinquencyyear==3,'taxdelinquencyyear'] = 2003
prop2016.loc[prop2016.taxdelinquencyyear==4,'taxdelinquencyyear'] = 2004
prop2016.loc[prop2016.taxdelinquencyyear==5,'taxdelinquencyyear'] = 2005
prop2016.loc[prop2016.taxdelinquencyyear==6,'taxdelinquencyyear'] = 2006
prop2016.loc[prop2016.taxdelinquencyyear==7,'taxdelinquencyyear'] = 2007
prop2016.loc[prop2016.taxdelinquencyyear==8,'taxdelinquencyyear'] = 2008
prop2016.loc[prop2016.taxdelinquencyyear==9,'taxdelinquencyyear'] = 2009
prop2016.loc[prop2016.taxdelinquencyyear==10,'taxdelinquencyyear'] = 2010
prop2016.loc[prop2016.taxdelinquencyyear==11,'taxdelinquencyyear'] = 2011
prop2016.loc[prop2016.taxdelinquencyyear==12,'taxdelinquencyyear'] = 2012
prop2016.loc[prop2016.taxdelinquencyyear==13,'taxdelinquencyyear'] = 2013
prop2016.loc[prop2016.taxdelinquencyyear==14,'taxdelinquencyyear'] = 2014
prop2016.loc[prop2016.taxdelinquencyyear==15,'taxdelinquencyyear'] = 2015

prop2016.loc[prop2016.taxdelinquencyyear>0,'taxdelinquencyyear'] = 2016 - prop2016['taxdelinquencyyear']



#rawcensustractandblock
s_rawcensustractandblock=prop2016.rawcensustractandblock.apply(lambda x: str(x))

prop2016['census_tractnumber']=s_rawcensustractandblock.str.slice(4,11)
prop2016['block_number']=s_rawcensustractandblock.str.slice(start=11)
prop2016['block_number']=prop2016['block_number'].apply(lambda x: x[:4]+'.'+x[4:]+'0' )
prop2016['block_number']=prop2016['block_number'].apply(lambda x: int(round(float(x),0)) )
prop2016['block_number']=prop2016['block_number'].apply(lambda x: str(x).ljust(4,'0') )

#droping censustractandblock since this is just a duplicate of rawcensustractandblock
prop2016=prop2016.drop('censustractandblock', axis=1)

# drooping rawcensustractandblock, since it's already stored as substrings in different column names
prop2016=prop2016.drop('rawcensustractandblock', axis=1)

missingvalues_prop = (prop2016.isnull().sum()/len(prop2016)).reset_index()
missingvalues_prop.columns = ['field','percentage']
missingvalues_prop = missingvalues_prop.sort_values(by = 'percentage', ascending = False)
#print(missingvalues_prop)
missingvaluescols = missingvalues_prop[missingvalues_prop['percentage'] > 0.95].field.tolist()

print(missingvaluescols)

#analysing the columns with most missing values
#buildingclasstypeid', 'architecturalstyletypeid', 'typeconstructiontypeid
#not many related columns or correlated columns to impute with, since these columns miss more than 95% of missing rows, imputation
#will largely affect the accuracy of algorithms, hence droppping it

prop2016 = prop2016.drop(['buildingclasstypeid', 'architecturalstyletypeid', 'typeconstructiontypeid'], axis=1)


#pool related attribute , is related propertylandusetypeid, and only single or duplex houses have pools 

#pool_sqft  16932
poolsizesum_mean = prop2016.loc[prop2016['poolcnt'] > 0, 'pool_sqft'].mean()

prop2016.loc[prop2016.pool_sqft.isnull(),'pool_sqft']=poolsizesum_mean
print(prop2016.pool_sqft.isnull().sum())

#total_parcel_tax
#structure_tax
#land_tax
#total_property_tax_2016
#2)recalculate total_parcel_tax =structure_tax + land_tax

# total_parcel_tax =structure_tax + land_tax
#->structure_tax=total_parcel_tax -land_tax

prop2016.loc[(prop2016.total_parcel_tax>0) & (prop2016.land_tax>0),'structure_tax']=prop2016['total_parcel_tax']-prop2016['land_tax']

#structure_tax, i see a lot of structure tax is 0's, those must be NA's

prop2016.loc[prop2016.structure_tax==0,'structure_tax']=np.nan

print(prop2016.total_property_tax_2016.isnull().sum())
print(prop2016.structure_tax.isnull().sum())
print(prop2016.total_parcel_tax.isnull().sum())
print(prop2016.land_tax.isnull().sum())

#*******************************
#bedroomcnt #1421 zero bed room houses ??, observed it's missing all other room count also missing

prop2016.loc[prop2016.bedroomcnt==0,'bedroomcnt']= np.nan
print(prop2016.bedroomcnt.isnull().sum())

#caluculate full bath and half bath again from total bath as, it has few extra columns, (fixes 500 missing values in roomcnt )

prop2016.loc[(prop2016.full_bath.notnull()) 
             & (prop2016.half_bath.notnull()) 
             & (prop2016.bedroomcnt.notnull()) 
             & (prop2016.roomcnt.isnull()),['roomcnt']]=prop2016.full_bath + prop2016.half_bath + prop2016.bedroomcnt


# most bedroom count and roomcount null are in same place
# all column null count 1133 all columns are null

print(prop2016.total_bath.isnull().sum())
print(prop2016.full_bath.isnull().sum())
print(prop2016.half_bath.isnull().sum())
print(prop2016.bedroomcnt.isnull().sum())
print(prop2016.roomcnt.isnull().sum())


#numberofstories	69705
prop2016.loc[(prop2016.propertylandusetypeid==246) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==246) & (prop2016.numberofstories.isnull()),'numberofstories']=2
prop2016.loc[(prop2016.propertylandusetypeid==247) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==247) & (prop2016.numberofstories.isnull()),'numberofstories']=2
prop2016.loc[(prop2016.propertylandusetypeid==248) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==248) & (prop2016.numberofstories.isnull()),'numberofstories']=2
prop2016.loc[(prop2016.propertylandusetypeid==260) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==260) & (prop2016.numberofstories.isnull()),'numberofstories']=2
prop2016.loc[(prop2016.propertylandusetypeid==261) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==261) & (prop2016.numberofstories.isnull()),'numberofstories']=1
prop2016.loc[(prop2016.propertylandusetypeid==263) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==263) & (prop2016.numberofstories.isnull()),'numberofstories']=1
prop2016.loc[(prop2016.propertylandusetypeid==266) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==266) & (prop2016.numberofstories.isnull()),'numberofstories']=1
prop2016.loc[(prop2016.propertylandusetypeid==269) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==269) & (prop2016.numberofstories.isnull()),'numberofstories']=2
prop2016.loc[(prop2016.propertylandusetypeid==275) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==275) & (prop2016.numberofstories.isnull()),'numberofstories']=1
prop2016.loc[(prop2016.propertylandusetypeid==267) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==267) & (prop2016.numberofstories.isnull()),'numberofstories']=1
#https://en.wikipedia.org/wiki/Townhouse , typical town house are usualyy large, and has atleast 6 rooms
prop2016.loc[(prop2016.propertylandusetypeid==264) & (prop2016.numberofstories.isnull()),'numberofstories']=2
prop2016.loc[(prop2016.propertylandusetypeid==31) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==31) & (prop2016.numberofstories.isnull()),'numberofstories']=2
prop2016.loc[(prop2016.propertylandusetypeid==47) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==47) & (prop2016.numberofstories.isnull()),'numberofstories']=1
prop2016.loc[(prop2016.propertylandusetypeid==265) & (prop2016.numberofstories.notnull()),'numberofstories'].mode()
prop2016.loc[(prop2016.propertylandusetypeid==265) & (prop2016.numberofstories.isnull()),'numberofstories']=1
prop2016.loc[(prop2016.propertylandusetypeid==270) & (prop2016.numberofstories.isnull()),'numberofstories']=1
 

#fireplace_count
prop2016.loc[(prop2016.fireplaceflag==1) & (prop2016.fireplace_count.notnull()),'fireplace_count'].mode()
prop2016.loc[(prop2016.fireplaceflag==1) & (prop2016.fireplace_count.isnull()),'fireplace_count']=1

#location seems to be related to building quality, (knnclassifier)

def fillna_knn( df, base, target):
    data_colnames = [ target ] + base
    #print("data_colnames",data_colnames)
    missing_values_boolflag = df[target].isnull() #true for missing rows, false for columns with values
    #print("miss",missing_values_boolflag.head())
    not_missing_boolflag = ~missing_values_boolflag 
    #print("not miss",not_missing_boolflag.head())
    number_of_missing_val = missing_values_boolflag.sum()
    print("# of miss",number_of_missing_val)
    not_missing_rows = df.loc[ not_missing_boolflag, data_colnames ]
    #print(not_missing_rows.head())
    Y = not_missing_rows[target]
    X = not_missing_rows[base]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=3192,stratify=Y)
    metrics       = ['euclidean'] 
    weights       = ['distance'] 
    numNeighbors  = [10,15,20]
    param_grid    = dict(metric=metrics,weights=weights,n_neighbors=numNeighbors)
    cv            = StratifiedKFold(n_splits=2,random_state=3192,shuffle=False)
    grid = GridSearchCV(neighbors.KNeighborsClassifier(n_jobs=-1),param_grid=param_grid,cv=cv,scoring='f1_weighted',refit=True,return_train_score=True,verbose=1,n_jobs=-1,pre_dispatch='n_jobs')
    grid.fit(X_train ,Y_train)
    #print("grid.cv_results_",grid.cv_results_)
    print("grid.best_estimator_",grid.best_estimator_)
    print("grid.best_params_",grid.best_params_)
    print("grid.scorer_",grid.scorer_)
    #print("grid.n_splits_",grid.n_splits_)
    y_true, y_pred = Y_test, grid.predict(X_test)
    
    Z = grid.predict(df.loc[missing_values_boolflag, base])
    #df.loc[ missing_values_boolflag, target ]  = Z
    return Z
    


print(prop2016.buildingqualitytypeid.isnull().sum())
print(prop2016.shape)
temp=prop2016.copy()
temp['buildingqualitytypeid']=temp['buildingqualitytypeid'].fillna(-1)
temp=temp.groupby("buildingqualitytypeid").filter(lambda x: x.buildingqualitytypeid.size > 3)
temp['buildingqualitytypeid'] = temp['buildingqualitytypeid'].replace(-1,np.nan)
temp=temp.loc[(temp.latitude.notnull()) & (temp.longitude.notnull()),temp.columns]
print(temp.buildingqualitytypeid.isnull().sum())
print(temp.shape)

missing_values=fillna_knn(temp,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'buildingqualitytypeid')

print("predicted output shape",missing_values.shape)


missing_values_boolflag = prop2016.loc[(prop2016.latitude.notnull()) &(prop2016.longitude.notnull()),'buildingqualitytypeid'].isnull()

#missing_values.shape
prop2016.loc[ (prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) & (missing_values_boolflag), 'buildingqualitytypeid' ]  = missing_values

print(prop2016.buildingqualitytypeid.isnull().sum())



print(prop2016.heating_system_id.isnull().sum())
print(prop2016.shape)
temp=prop2016.copy()
temp['heating_system_id']=temp['heating_system_id'].fillna(-1)
temp=temp.groupby("heating_system_id").filter(lambda x: x.heating_system_id.size > 3)
temp['heating_system_id'] = temp['heating_system_id'].replace(-1,np.nan)
temp=temp.loc[(temp.latitude.notnull()) & (temp.longitude.notnull()),temp.columns]
print(temp.heating_system_id.isnull().sum())
print(temp.shape)

missing_values=fillna_knn(temp,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'heating_system_id')

print("predicted output shape",missing_values.shape)

missing_values_boolflag = prop2016.loc[(prop2016.latitude.notnull()) &(prop2016.longitude.notnull()),'heating_system_id'].isnull()

#missing_values.shape
prop2016.loc[ (prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) & (missing_values_boolflag), 'heating_system_id' ]  = missing_values

print(prop2016.heating_system_id.isnull().sum())


print(prop2016.ac_id.isnull().sum())
print(prop2016.shape)
temp=prop2016.copy()
temp['ac_id']=temp['ac_id'].fillna(-1)
temp=temp.groupby("ac_id").filter(lambda x: x.ac_id.size > 3)
temp['ac_id'] = temp['ac_id'].replace(-1,np.nan)
temp=temp.loc[(temp.latitude.notnull()) & (temp.longitude.notnull()),temp.columns]
print(temp.ac_id.isnull().sum())
print(temp.shape)

missing_values=fillna_knn(temp,
                  base = [ 'latitude', 'longitude' ] ,
                  target = 'ac_id')

print("predicted output shape",missing_values.shape)

missing_values_boolflag = prop2016.loc[(prop2016.latitude.notnull()) &(prop2016.longitude.notnull()),'ac_id'].isnull()

#missing_values.shape
prop2016.loc[ (prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) & (missing_values_boolflag), 'ac_id' ]  = missing_values

print(prop2016.ac_id.isnull().sum())


#location seems to be related to building quality, (knnregressor)
from sklearn.model_selection import KFold

def fillna_knnr( df, base, target):
    data_colnames = [ target ] + base
    #print("data_colnames",data_colnames)
    missing_values_boolflag = df[target].isnull() #true for missing rows, false for columns with values
    #print("miss",missing_values_boolflag.head())
    not_missing_boolflag = ~missing_values_boolflag 
    #print("not miss",not_missing_boolflag.head())
    number_of_missing_val = missing_values_boolflag.sum()
    print("# of miss",number_of_missing_val)
    not_missing_rows = df.loc[ not_missing_boolflag, data_colnames]
    #print(not_missing_rows.head())
    Y = not_missing_rows[target]
    X = not_missing_rows[base]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=3192)
    metrics       = ['euclidean'] 
    weights       = ['distance'] 
    numNeighbors  = [15,25]
    param_grid    = dict(metric=metrics,weights=weights,n_neighbors=numNeighbors)
    cv            = KFold(n_splits=2,random_state=3192,shuffle=False) 
    grid = GridSearchCV(neighbors.KNeighborsRegressor(n_jobs=-1),param_grid=param_grid,cv=cv,scoring='neg_mean_absolute_error',refit=True,return_train_score=True,verbose=1,n_jobs=-1,pre_dispatch='n_jobs')
    grid.fit(X_train ,Y_train)
    #print("grid.cv_results_",grid.cv_results_)
    print("grid.best_estimator_",grid.best_estimator_)
    print("grid.best_params_",grid.best_params_)
    print("grid.scorer_",grid.scorer_)
    #print("grid.n_splits_",grid.n_splits_)
    y_true, y_pred = Y_test, grid.predict(X_test) 
    Z = grid.predict(df.loc[missing_values_boolflag, base])
    #df.loc[ missing_values_boolflag, target ]  = Z
    return Z

#garage_sqft
print(prop2016.garage_sqft.isnull().sum())
print(prop2016.shape)


temp=prop2016.loc[prop2016.garagecarcnt>0,prop2016.columns].copy()
temp=temp.loc[(temp.latitude.notnull()) & (temp.longitude.notnull()),temp.columns]

print(temp.garage_sqft.isnull().sum())

print(temp.shape)

missing_values=fillna_knnr(temp,
                  base = [ 'latitude', 'longitude','garagecarcnt'] ,
                  target = 'garage_sqft')

print("predicted output shape",missing_values.shape)


missing_values_boolflag = prop2016.loc[(prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) &(prop2016.garagecarcnt>0),'garage_sqft'].isnull()

#missing_values.shape
prop2016.loc[ (prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) & (prop2016.garagecarcnt>0) & (missing_values_boolflag), 'garage_sqft' ]  = missing_values

print(prop2016.yearbuilt.isnull().sum())


#83 mins
#total_finished_living_area_sqft

temp=prop2016.copy()
temp=temp.loc[(temp.latitude.notnull()) & (temp.longitude.notnull()),temp.columns]
print("null values",temp.total_finished_living_area_sqft.isnull().sum())

missing_values=fillna_knnr(temp,
                  base = [ 'latitude', 'longitude','propertylandusetypeid'] ,
                  target = 'total_finished_living_area_sqft')

print("predicted output shape",missing_values.shape)


missing_values_boolflag = prop2016.loc[(prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) ,'total_finished_living_area_sqft'].isnull()
#missing_values.shape
prop2016.loc[ (prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) & (missing_values_boolflag), 'total_finished_living_area_sqft' ]  = missing_values
print(prop2016.total_finished_living_area_sqft.isnull().sum())

#yearbuilt

temp=prop2016.copy()
temp=temp.loc[(temp.latitude.notnull()) & (temp.longitude.notnull()),temp.columns]
temp=prop2016.copy()
temp['yearbuilt']=temp['yearbuilt'].fillna(-1)
temp=temp.groupby("yearbuilt").filter(lambda x: x.yearbuilt.size > 3)
temp['yearbuilt'] = temp['yearbuilt'].replace(-1,np.nan)
temp=temp.loc[(temp.latitude.notnull()) & (temp.longitude.notnull()),temp.columns]
print(temp.yearbuilt.isnull().sum())
print(temp.shape)

missing_values=fillna_knn(temp,
                  base = [ 'latitude', 'longitude','propertylandusetypeid' ] ,
                  target = 'yearbuilt')

print("predicted output shape",missing_values.shape)
missing_values_boolflag = prop2016.loc[(prop2016.latitude.notnull()) &(prop2016.longitude.notnull()),'yearbuilt'].isnull()

#missing_values.shape
prop2016.loc[ (prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) & (missing_values_boolflag), 'yearbuilt' ]  = missing_values

print(prop2016.yearbuilt.isnull().sum())

#total_finished_living_area_sqft
print(prop2016.total_bath.isnull().sum())
temp=prop2016.copy()
temp=temp.loc[(temp.latitude.notnull()) & (temp.longitude.notnull()),temp.columns]
print(temp.total_bath.isnull().sum())
print(temp.shape)

missing_values=fillna_knnr(temp,
                  base = ['propertylandusetypeid','total_finished_living_area_sqft' ] ,
                  target = 'total_bath')

print("predicted output shape",missing_values.shape)

missing_values_boolflag = prop2016.loc[(prop2016.latitude.notnull()) &(prop2016.longitude.notnull()),'total_bath'].isnull()

#missing_values.shape
prop2016.loc[ (prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) & (missing_values_boolflag), 'total_bath' ]  = missing_values

print(prop2016.total_bath.isnull().sum())

#bedroomcnt	1421

print(prop2016.bedroomcnt.isnull().sum())

temp=prop2016.copy()
temp=temp.loc[(temp.latitude.notnull()) & (temp.longitude.notnull()),temp.columns]

print(temp.bedroomcnt.isnull().sum())
print(temp.shape)
missing_values=fillna_knnr(temp,
                  base = ['propertylandusetypeid','total_finished_living_area_sqft','total_bath' ] ,
                  target = 'bedroomcnt')

print("predicted output shape",missing_values.shape)

missing_values_boolflag = prop2016.loc[(prop2016.latitude.notnull()) &(prop2016.longitude.notnull()),'bedroomcnt'].isnull()

#missing_values.shape
prop2016.loc[ (prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) & (missing_values_boolflag), 'bedroomcnt' ]  = missing_values

print(prop2016.bedroomcnt.isnull().sum())




prop2016['total_bath']=prop2016.total_bath.round(1)
prop2016['bedroomcnt']=prop2016.bedroomcnt.round(1)
#recalculate roomcnt	1416 as we have used imputation for total_bath and bedroomcnt

prop2016.loc[(prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) & (prop2016.roomcnt.isnull()),['roomcnt']]=prop2016.total_bath + prop2016.bedroomcnt

#lot_area_sqft
print(prop2016.lot_area_sqft.isnull().sum())

temp=prop2016.copy()
temp=temp.loc[(temp.latitude.notnull()) & (temp.longitude.notnull()),temp.columns]

print(temp.bedroomcnt.isnull().sum())
print(temp.shape)


missing_values=fillna_knnr(temp,
                  base = ['latitude','longitude','total_finished_living_area_sqft'],
                  target = 'lot_area_sqft')

print("predicted output shape",missing_values.shape)

missing_values_boolflag = prop2016.loc[(prop2016.latitude.notnull()) &(prop2016.longitude.notnull()),'lot_area_sqft'].isnull()

#missing_values.shape
prop2016.loc[ (prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) & (missing_values_boolflag), 'lot_area_sqft' ]  = missing_values

print(prop2016.lot_area_sqft.isnull().sum())


#prop2016.loc[(prop2016.latitude.notnull()) & (prop2016.longitude.notnull()),'total_finished_living_area_sqft'].isnull().sum()

# predict structure_tax and recalculate  total_parcel_tax = land_tax + structure_tax

print(prop2016.structure_tax.isnull().sum())

temp=prop2016.copy()
temp=temp.loc[(temp.latitude.notnull()) & (temp.longitude.notnull()),temp.columns]

print(temp.structure_tax.isnull().sum())
print(temp.shape)


missing_values=fillna_knnr(temp,
                  base = ['latitude','longitude','lot_area_sqft','total_finished_living_area_sqft' ] ,
                  target = 'structure_tax')

print("predicted output shape",missing_values.shape)

missing_values_boolflag = prop2016.loc[(prop2016.latitude.notnull()) &(prop2016.longitude.notnull()),'structure_tax'].isnull()

#missing_values.shape
prop2016.loc[ (prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) & (missing_values_boolflag), 'structure_tax' ]  = missing_values

print(prop2016.structure_tax.isnull().sum())


#36 total_property_tax_2016 

#total_parcel_tax = land_tax + structure_tax
    
prop2016['total_parcel_tax']=prop2016['structure_tax']+prop2016['land_tax']


#age of the property
prop2016['age'] = 2016 - prop2016['yearbuilt']
prop2016=prop2016.drop(['yearbuilt'],axis=1)

#prop2016.loc[(prop2016.latitude.notnull()) & (prop2016.longitude.notnull()),'lot_area_sqft'].isnull().sum()

#total_property_tax_2016


print(prop2016.total_property_tax_2016.isnull().sum())

temp=prop2016.copy()
temp=temp.loc[(temp.latitude.notnull()) & (temp.longitude.notnull()),temp.columns]

print(temp.total_property_tax_2016.isnull().sum())
print(temp.shape)



missing_values=fillna_knnr(temp,
                  base = ['latitude','longitude','lot_area_sqft','propertylandusetypeid','total_finished_living_area_sqft' ] ,
                  target = 'total_property_tax_2016')

print("predicted output shape",missing_values.shape)

missing_values_boolflag = prop2016.loc[(prop2016.latitude.notnull()) &(prop2016.longitude.notnull()),'total_property_tax_2016'].isnull()

#missing_values.shape
prop2016.loc[ (prop2016.latitude.notnull()) &(prop2016.longitude.notnull()) & (missing_values_boolflag), 'total_property_tax_2016' ]  = missing_values

print(prop2016.total_property_tax_2016.isnull().sum())




import MySQLdb
from sqlalchemy import create_engine
engineString = 'mysql+mysqldb://root:MyNewPass@localhost/sakila'
engine = create_engine(engineString)
con=engine.connect()

with engine.connect() as con, con.begin():
    prop2016.to_sql('prop2016_f2', engine, chunksize=10000, index =False,if_exists ='replace')

import pandas as pd
df_train = pd.DataFrame()
import MySQLdb
from sqlalchemy import create_engine
engineString = 'mysql+mysqldb://root:MyNewPass@localhost/sakila'
engine = create_engine(engineString)
#con=engine.connect()
prop2016=pd.read_sql_table('prop2016_f2',con=engine)

# missing count
missing_df = prop2016.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
print(missing_df)
print(missing_df.shape)

##import train 2016  data
train2016=pd.read_csv('C:\\Users\\Dell\\Desktop\\inf552\\Zillow Prize\\train_2016.csv',parse_dates=["transactiondate"])


df_train = train2016.merge(prop2016, how='left', on='parcelid')


df_train = df_train.drop(['full_bath','half_bath','finished_living_area_entryfloor_sqft2','finished_living_area_entryfloor_sqft1'], axis=1)

df_train = df_train.loc[df_train.total_parcel_tax.notnull(),df_train.columns]

df_train.head(5)

# missing count
missing_df = df_train.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
print(missing_df)
print(missing_df.shape)


df_train.columns


#proportion of living area
df_train['N-LivingAreaProp'] = df_train['total_finished_living_area_sqft']/df_train['lot_area_sqft']


#Amout of extra space
df_train['N-ExtraSpace'] = df_train['lot_area_sqft'] - df_train['total_finished_living_area_sqft'] 
#df_train['N-ExtraSpace-2'] = df_train['total_area_sqft'] - df_train['finished_living_area_sqft'] 

#Total number of rooms
df_train['N-TotalRooms'] = df_train['total_bath']*df_train['bedroomcnt']

# Number of Extra rooms
df_train['N-ExtraRooms'] = df_train['roomcnt'] - df_train['N-TotalRooms'] 

#Average room size N-AvRoomSizeze
df_train['N-AvRoomSize'] = df_train['total_finished_living_area_sqft']/df_train['roomcnt']


#Ratio of the built structure value to land area
df_train['N-ValueProp'] = df_train['structure_tax']/df_train['land_tax']

#Ratio of tax of property over parcel
df_train['N-ValueRatio'] = df_train['total_parcel_tax']/df_train['total_property_tax_2016']

#proportion of living area
df_train['N-LivingAreaProp'] = df_train['total_finished_living_area_sqft']/df_train['lot_area_sqft']

#Amout of extra space
df_train['N-ExtraSpace'] = df_train['lot_area_sqft'] - df_train['total_finished_living_area_sqft'] 

#TotalTaxScore
df_train['N-TaxScore'] = df_train['total_parcel_tax']*df_train['total_property_tax_2016']

#polnomials of the variable
df_train["N-structure_tax-2"] = df_train["structure_tax"] ** 2
df_train["N-structure_tax-3"] = df_train["structure_tax"] ** 3

#Indicator whether it has AC or not
df_train['N-ACInd'] = (df_train['ac_id']!=5)*1

#Indicator whether it has Heating or not 
df_train['N-HeatInd'] = (df_train['heating_system_id']!=13)*1

#There's 25 different property uses - let's compress them down to 4 categories
#df_train['N-PropType'] = df_train.propertylandusetypeid.replace({31 : "Mixed", 46 : "Other", 47 : "Mixed", 246 : "Mixed", 247 : "Mixed", 248 : "Mixed", 260 : "Home", 261 : "Home", 262 : "Home", 263 : "Home", 264 : "Home", 265 : "Home", 266 : "Home", 267 : "Home", 268 : "Home", 269 : "Not Built", 270 : "Home", 271 : "Home", 273 : "Home", 274 : "Other", 275 : "Home", 276 : "Home", 279 : "Home", 290 : "Not Built", 291 : "Not Built" })


df_train=df_train.drop(['ac_id','heating_system_id','propertylandusetypeid'], axis=1)


df_train.loc[df_train.has_hottub_or_spa==1,'has_hottub_or_spa']="Yes"
df_train.loc[df_train.has_hottub_or_spa==0,'has_hottub_or_spa']="No"

df_train.loc[df_train.just_hottub_or_spa==0,'just_hottub_or_spa']="No"
df_train.loc[df_train.just_hottub_or_spa==1,'just_hottub_or_spa']="Yes"

df_train.loc[df_train.deck_flag==0,'deck_flag']="No"
df_train.loc[df_train.deck_flag==1,'deck_flag']="Yes"

df_train.loc[df_train.basement_flag==0,'basement_flag']="No"
df_train.loc[df_train.basement_flag==1,'basement_flag']="Yes"

df_train.loc[df_train.fireplaceflag==False,'fireplaceflag']="No"
df_train.loc[df_train.fireplaceflag==True,'fireplaceflag']="Yes"
#

#ac_id,heating_system_id,propertylandusetypeid
dummieslist=['has_hottub_or_spa','just_hottub_or_spa','N-ACInd','N-HeatInd',
             'deck_flag','fips','basement_flag','fireplaceflag','taxdelinquencyflag']

df_train[dummieslist] = df_train[dummieslist].astype(object)
dummies = pd.get_dummies(df_train[dummieslist], prefix= dummieslist)

dummies.head(5)

df_train=df_train.drop(['has_hottub_or_spa','just_hottub_or_spa','N-ACInd','N-HeatInd',
             'deck_flag','fips','basement_flag','fireplaceflag','taxdelinquencyflag'], axis=1)

dummies2=['pool_with_spa_tub_no','pool_with_spa_tub_yes']
df_train[dummies2] = df_train[dummies2].astype(int)

numcols = set(list(df_train.columns))-set((list(dummies.columns)+dummies2))



#numcols2=['basementsqft','total_bath','bedroomcnt','total_finished_living_area_sqft','fireplace_count','garagecarcnt',
# 'garage_sqft','latitude','longitude','lot_area_sqft','poolcnt','pool_sqft','roomcnt','unitcnt','patio_sqft','storage_sqft',
# 'numberofstories','structure_tax','total_parcel_tax','land_tax','total_property_tax_2016','taxdelinquencyyear','transaction_month',
# 'census_tractnumber','block_number','age','N-LivingAreaProp','N-ExtraSpace','N-TotalRooms','N-ExtraRooms','N-AvRoomSize','N-ValueProp',
#'N-ValueRatio','N-LivingAreaProp','N-ExtraSpace','N-TaxScore','N-structure_tax-2','N-structure_tax-3']



numcols=['N-AvRoomSize','N-ExtraRooms','N-ExtraSpace','N-LivingAreaProp','N-TaxScore','N-TotalRooms','N-ValueProp','N-ValueRatio',
 'N-structure_tax-2','N-structure_tax-3','age','basementsqft','bedroomcnt','block_number','buildingqualitytypeid',
 'census_tractnumber','fireplace_count','garage_sqft','garagecarcnt','land_tax','latitude','longitude',
 'lot_area_sqft','numberofstories','patio_sqft','pool_sqft','poolcnt','roomcnt','storage_sqft','structure_tax',
 'taxdelinquencyyear','total_bath','total_finished_living_area_sqft','total_parcel_tax','total_property_tax_2016','transaction_month',
 'unitcnt']
        

Y=df_train['logerror']

df_train['transaction_month'] = df_train['transactiondate'].dt.month
df_train=df_train.drop('parcelid',axis=1)



df_train=df_train.drop(['transactiondate'], axis=1)


# 2 variables are in object datatype, coverting into numeric
df_train[['census_tractnumber','block_number']] = df_train[['census_tractnumber','block_number']].apply(pd.to_numeric)

X=pd.concat([dummies,df_train[dummies2],df_train[numcols]],axis=1)
X.columns

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=3192)

#pca selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
get_ipython().magic('matplotlib inline')
scaled_x = scale(X)
pca = PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
pca.fit(scaled_x)


#The amount of variance that each PC explains
var= pca.explained_variance_ratio_
#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
print(var1)
plt.plot(var1)

#Looking at above plot I'm taking 30 variables

pca = PCA(n_components=30, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
pca.fit(scaled_x)

pca1=pca.fit_transform(scaled_x)

pca = PCA(n_components=30, copy=True, whiten=True, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
pca.fit(scaled_x)
pca2=pca.fit_transform(scaled_x)


pcaX_train, pcaX_test, pcaY_train, pcaY_test = train_test_split(pca1, Y, test_size=0.10, random_state=3192)
pca2X_train, pca2X_test, pca2Y_train, pca2Y_test = train_test_split(pca2, Y, test_size=0.10, random_state=3192)

#  top features
import xgboost as xgb
xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1,
    'seed' : 0
}
dtrain = xgb.DMatrix(X_train, Y_train, feature_names=X_train.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=50)
# plot the important features #
fig, ax = plt.subplots(figsize=(12,18))
#max_num_features=50, error for no reason 
xgb.plot_importance(model, height=0.8, ax=ax)
plt.show()

#top features
#xgboost_selection=['total_finished_living_area_sqft','latitude','structure_tax','total_property_tax_2016',
#'total_parcel_tax','land_tax','longitude','lot_area_sqft','census_tractnumber','age','total_bath','bedroomcnt',
#'block_number','transaction_month','roomcnt','taxdelinquencyyear','unitcnt','taxdelinquencyflag_No',
#'fips_LA','garage_sqft','pool_with_spa_tub_no','has_hottub_or_spa_No','garagecarcnt','deck_flag_No',
#'poolcnt','pool_sqft'
#]

xgboost_selection=['N-AvRoomSize','N-ValueRatio','N-ExtraSpace','N-TaxScore','N-ValueProp','total_finished_living_area_sqft','N-structure_tax-2','latitude','census_tractnumber',
'total_property_tax_2016','N-LivingAreaProp','longitude','age','N-ExtraRooms','land_tax','block_number','total_parcel_tax','transaction_month',
'N-TotalRooms','bedroomcnt','fips_LA','taxdelinquencyflag_No','pool_with_spa_tub_no','total_bath','garage_sqft','roomcnt','buildingqualitytypeid','unitcnt',
'fips_Orange_County','has_hottub_or_spa_0','taxdelinquencyflag_Yes']

# feature selection
#c_id,heating_system_id,propertylandusetypeid
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectFromModel
reg = ExtraTreesRegressor(n_estimators=500, max_depth=8, max_features='sqrt',
                          min_samples_split=100 ,min_samples_leaf=10, bootstrap=True,n_jobs=-1, random_state=3192)
reg = reg.fit(X_train, Y_train)
#print("importance",reg.feature_importances_) 
model = SelectFromModel(reg, prefit=True)
# since we haven'tset any thresold , it applies l1 regularisation and picks 27 attributes
X_new = model.transform(X_train)
print(X_train.shape)
print(X_new.shape)  

feat_names = X.columns.values
importances = reg.feature_importances_
std = np.std([tree.feature_importances_ for tree in reg.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:36]
plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()

#tree_selection=[
#    'total_finished_living_area_sqft','structure_tax','total_property_tax_2016','total_bath','total_parcel_tax',
#    'age','latitude','census_tractnumber','bedroomcnt','longitude','land_tax','propertylandusetypeid','block_number',
#    'buildingqualitytypeid','numberofstories','heating_system_id','unitcnt','transaction_month','lot_area_sqft','roomcnt',
#    'garage_sqft','garagecarcnt','pool_with_spa_tub_no','poolcnt','fips_LA','taxdelinquencyyear','patio_sqft',
#    'taxdelinquencyflag_No','taxdelinquencyflag_Yes'
#]

tree_selection=['total_finished_living_area_sqft','structure_tax','N-TotalRooms','total_bath','roomcnt','bedroomcnt','total_property_tax_2016',
                'N-ExtraRooms',
'taxdelinquencyflag_No','longitude','age','transaction_month','N-LivingAreaProp','census_tractnumber','N-AvRoomSize','total_parcel_tax',
'taxdelinquencyflag_Yes','N-ValueRatio','latitude','land_tax','numberofstories','taxdelinquencyyear','buildingqualitytypeid','poolcnt',
'N-TaxScore','block_number','pool_with_spa_tub_no','pool_sqft','unitcnt','fips_Ventura','lot_area_sqft','N-ExtraSpace','fips_Orange_County',
'garagecarcnt']

import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge,Lasso
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,make_scorer

#model=Lasso(alpha=0.2, fit_intercept=True, normalize=True, precompute=False, copy_X=True,
 #                                max_iter=1000, 
  #                               tol=0.0001, warm_start=False, positive=False, random_state=3192, selection='cyclic')

#Ridge(random_state=3192,solver='auto',fit_intercept=True,normalize=True,alpha=0.1)
#LinearRegression(n_jobs=-1,fit_intercept=True, normalize=True, copy_X=True)


rfecv = RFECV(estimator=LinearRegression(n_jobs=-1,fit_intercept=True, normalize=True, copy_X=True), step=2, cv=KFold(4),scoring='neg_mean_absolute_error')
rfecv.fit(X_train, Y_train)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")

plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()

rfe_selection = [i for indx,i in enumerate(X.columns) if rfecv.support_[indx] == True]

#pcaX_train, pcaX_test, pcaY_train, pcaY_test

#rfe_selection, tree_selection, xgboost_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,make_scorer,mean_squared_error

# just to check whether normalized /not normalized data gives better results
parameters = {'fit_intercept':[True], 'normalize':[True,False], 'copy_X':[True]}
scoring = {'MAE':'neg_mean_absolute_error','MSE': make_scorer(mean_squared_error,greater_is_better=False)}

grid = GridSearchCV(LinearRegression(n_jobs=-1),param_grid=parameters, scoring=scoring,cv=5,refit='MAE',
                    return_train_score=True,
                    verbose=0,n_jobs=-1,pre_dispatch='n_jobs')

grid.fit(pcaX_train, pcaY_train)
#print("5. grid best_score_",abs(grid.best_score_))
pcaY_pred = grid.predict(pcaX_test)
print("MAE on test data",mean_absolute_error(pcaY_test,pcaY_pred))
print("MSE on test data",mean_squared_error(pcaY_test,pcaY_pred))
print("R Squared data ",r2_score(pcaY_test,pcaY_pred))

#Linear regression with rfe_selection selection
#rfe_selection, tree_selection, xgboost_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,make_scorer,mean_squared_error

# just to check whether normalized /not normalized data gives better results
parameters = {'fit_intercept':[True], 'normalize':[True,False], 'copy_X':[True]}
scoring = {'MAE':'neg_mean_absolute_error','MSE': make_scorer(mean_squared_error,greater_is_better=False)}

grid = GridSearchCV(LinearRegression(n_jobs=-1),param_grid=parameters, scoring=scoring,cv=5,refit='MAE',
                    return_train_score=True,
                    verbose=0,n_jobs=-1,pre_dispatch='n_jobs')

grid.fit(X_train[rfe_selection], Y_train)
#print("5. grid best_score_",abs(grid.best_score_))
Y_pred = grid.predict(X_test[rfe_selection])
print("MAE on test data",mean_absolute_error(Y_test,Y_pred))
print("MSE on test data",mean_squared_error(Y_test,Y_pred))
print("R Squared data ",r2_score(Y_test,Y_pred))

#tree_selection

#Linear regression with tree_selection selection
#rfe_selection, tree_selection, xgboost_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,make_scorer

# just to check whether normalized /not normalized data gives better results
parameters = {'fit_intercept':[True], 'normalize':[True,False], 'copy_X':[True]}
scoring = {'MAE':'neg_mean_absolute_error','MSE': make_scorer(mean_squared_error,greater_is_better=False)}

grid = GridSearchCV(LinearRegression(n_jobs=-1),param_grid=parameters, scoring=scoring,cv=5,refit='MAE',
                    return_train_score=True,
                    verbose=0,n_jobs=-1,pre_dispatch='n_jobs')

grid.fit(X_train[tree_selection], Y_train)
#print("5. grid best_score_",abs(grid.best_score_))
Y_pred = grid.predict(X_test[tree_selection])
print("MAE on test data",mean_absolute_error(Y_test,Y_pred))
print("MSE on test data",mean_squared_error(Y_test,Y_pred))
print("R Squared data ",r2_score(Y_test,Y_pred))

#xgboost_selection

#Linear regression with tree_selection selection
#rfe_selection, tree_selection, xgboost_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,make_scorer

# just to check whether normalized /not normalized data gives better results
parameters = {'fit_intercept':[True], 'normalize':[True,False], 'copy_X':[True]}
scoring = {'MAE':'neg_mean_absolute_error','MSE': make_scorer(mean_squared_error,greater_is_better=False)}

grid = GridSearchCV(LinearRegression(n_jobs=-1),param_grid=parameters, scoring=scoring,cv=5,refit='MAE',
                   return_train_score=True,
                   verbose=0,n_jobs=-1,pre_dispatch='n_jobs')

grid.fit(X_train[xgboost_selection], Y_train)
#print("5. grid best_score_",abs(grid.best_score_))
Y_pred = grid.predict(X_test[xgboost_selection])
print("MAE on test data",mean_absolute_error(Y_test,Y_pred))
print("MSE on test data",mean_squared_error(Y_test,Y_pred))
print("R Squared data ",r2_score(Y_test,Y_pred))

#Ridge
#rfe_selection, tree_selection, xgboost_selection
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_rand
from sklearn.metrics import r2_score,mean_absolute_error,make_scorer

# just to check whether normalized /not normalized data gives better results
parameters = {'fit_intercept':[True], 'normalize':[True], 'copy_X':[True],'alpha':[0.001,0.1,0.01,0.02,0.03,0.05,0.2,0.015,0.020]}
scoring = {'MAE':'neg_mean_absolute_error','MSE': make_scorer(mean_squared_error,greater_is_better=False)}

grid = GridSearchCV(Ridge(random_state=3192,solver='auto'),param_grid=parameters, scoring=scoring,cv=5,refit='MAE',
                    return_train_score=True,
                    verbose=0,n_jobs=-1,pre_dispatch='n_jobs')

grid.fit(X_train[rfe_selection], Y_train)
#print("5. grid best_score_",abs(grid.best_score_))
print("best params",grid.best_params_)
Y_pred = grid.predict(X_test[rfe_selection])
print("MAE on test data",mean_absolute_error(Y_test,Y_pred))
print("MSE on test data",mean_squared_error(Y_test,Y_pred))
print("R Squared data ",r2_score(Y_test,Y_pred))

#Ridge
#rfe_selection, tree_selection, xgboost_selection
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_rand
from sklearn.metrics import r2_score,mean_absolute_error,make_scorer

# just to check whether normalized /not normalized data gives better results
parameters = {'fit_intercept':[True], 'normalize':[True], 'copy_X':[True],'alpha':[0.001,0.1,0.01,0.02,0.03,0.05,0.2,0.015,0.020]}
scoring = {'MAE':'neg_mean_absolute_error','MSE': make_scorer(mean_squared_error,greater_is_better=False)}

grid = GridSearchCV(Ridge(random_state=3192,solver='auto'),param_grid=parameters, scoring=scoring,cv=5,refit='MAE',
                    return_train_score=True,
                    verbose=0,n_jobs=-1,pre_dispatch='n_jobs')

grid.fit(X_train[tree_selection], Y_train)
#print("5. grid best_score_",abs(grid.best_score_))
print("best params",grid.best_params_)
Y_pred = grid.predict(X_test[tree_selection])
print("MAE on test data",mean_absolute_error(Y_test,Y_pred))
print("MSE on test data",mean_squared_error(Y_test,Y_pred))
print("R Squared data ",r2_score(Y_test,Y_pred))

#Ridge
#rfe_selection, tree_selection, xgboost_selection
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform as sp_rand
from sklearn.metrics import r2_score,mean_absolute_error,make_scorer

# just to check whether normalized /not normalized data gives better results
parameters = {'fit_intercept':[True], 'normalize':[True], 'copy_X':[True],'alpha':[0.001,0.1,0.01,0.02,0.03,0.05,0.2,0.015]}
scoring = {'MAE':'neg_mean_absolute_error','MSE': make_scorer(mean_squared_error,greater_is_better=False)}

grid = GridSearchCV(Ridge(random_state=3192,solver='auto'),param_grid=parameters, scoring=scoring,cv=5,refit='MAE',
                    return_train_score=True,
                    verbose=0,n_jobs=-1,pre_dispatch='n_jobs')

grid.fit(X_train[xgboost_selection], Y_train)
#print("5. grid best_score_",abs(grid.best_score_))
print("best params",grid.best_params_)
Y_pred = grid.predict(X_test[xgboost_selection])
print("MAE on test data",mean_absolute_error(Y_test,Y_pred))
print("MSE on test data",mean_squared_error(Y_test,Y_pred))
print("R Squared data ",r2_score(Y_test,Y_pred))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,make_scorer
from sklearn.model_selection import GridSearchCV

# just to check whether normalized /not normalized data gives better results

 # 0.005 for 1200 trees.
#param_grid={'n_estimators':[1000,1200]}
param_grid={'n_estimators':[1200]}              
grid1 = GridSearchCV(GradientBoostingRegressor(subsample=0.8,max_features=11,min_samples_leaf=50,min_samples_split=50,
                                        
                                               max_depth=9,loss='ls',criterion='friedman_mse',learning_rate=0.005,
                                             random_state=3192),
                     param_grid=param_grid, cv=5,refit='MAE',
                    return_train_score=True,
                    verbose=0,n_jobs=-1,pre_dispatch='n_jobs')

grid1.fit(X_train[xgboost_selection], Y_train)
print("5. grid best_score_",abs(grid1.best_score_))
print("best params",grid1.best_params_)
print("best score",grid1.best_score_)
Y_pred = grid1.predict(X_test[xgboost_selection])
print("MAE on test data",mean_absolute_error(Y_test,Y_pred))
print("MSE on test data",mean_squared_error(Y_test,Y_pred))



from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,make_scorer
from sklearn.model_selection import GridSearchCV

# just to check whether normalized /not normalized data gives better results

 # 0.005 for 1200 trees.
param_grid={'n_estimators':[1200],'max_features':[22]}

              
grid1 = GridSearchCV(GradientBoostingRegressor(subsample=0.8,min_samples_leaf=50,min_samples_split=50,max_depth=9,loss='ls',criterion='friedman_mse',learning_rate=0.005,random_state=3192),
                     param_grid=param_grid, cv=5,refit='MAE',
                    return_train_score=True,
                    verbose=0,n_jobs=-1,pre_dispatch='n_jobs')

grid1.fit(X_train, Y_train)
print("5. grid best_score_",abs(grid1.best_score_))
print("best params",grid1.best_params_)
print("best score",grid1.best_score_)
Y_pred = grid1.predict(X_test)
print("MAE on test data",mean_absolute_error(Y_test,Y_pred))
print("MSE on test data",mean_squared_error(Y_test,Y_pred))


#pcaX_train, pcaX_test, pcaY_train, pcaY_test

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,make_scorer
from sklearn.model_selection import GridSearchCV

# just to check whether normalized /not normalized data gives better results

 # 0.005 for 1200 trees.
param_grid={'n_estimators':[1200],'max_features':[22]}

              
grid1 = GridSearchCV(GradientBoostingRegressor(subsample=0.8,min_samples_leaf=50,min_samples_split=50,max_depth=9,loss='ls',criterion='friedman_mse',learning_rate=0.005,random_state=3192),
                     param_grid=param_grid, cv=5,refit='MAE',
                    return_train_score=True,
                    verbose=0,n_jobs=-1,pre_dispatch='n_jobs')

grid1.fit(pcaX_train, pcaY_train)
print("5. grid best_score_",abs(grid1.best_score_))
print("best params",grid1.best_params_)
print("best score",grid1.best_score_)
pcaY_pred = grid1.predict(pcaX_test)
print("MAE on test data",mean_absolute_error(pcaY_test,pcaY_pred))
print("MSE on test data",mean_squared_error(pcaY_test,pcaY_pred))


def modelfit(model, X_train, Y_train,X_test,Y_test,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = model.get_xgb_params()
        
        xgtrain = xgb.DMatrix(X_train, Y_train)
        
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=model.get_params()['n_estimators'], nfold=cv_folds,
            metrics='mae', early_stopping_rounds=early_stopping_rounds, verbose_eval=1,seed=3192)
        
        model.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    model.fit(X_train,Y_train,eval_metric='mae')
        
    #Predict training set:
    y_pred = model.predict(X_test)
    print("MAE on test data",mean_absolute_error(Y_test, y_pred))

                    
    feat_imp = pd.Series(model.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    #print(" model.get_params()['n_estimators']",model.get_params()['n_estimators'])
    print("n_estimators",cvresult.shape[0])
    

param_test7 = {
 'reg_alpha':[0.001,0.005,0.01,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
}

gsearch7 = GridSearchCV(estimator = XGBRegressor(learning_rate =0.1,n_estimators=81,
                    max_depth=2,
                    min_child_weight=350,
                    gamma=0.4,
                    objective= 'reg:linear',
                    nthread=-1,scale_pos_weight=1,seed=3192,colsample_bytree=0.9,subsample=0.95),
param_grid = param_test7, scoring='neg_mean_absolute_error',n_jobs=-1, cv=5,verbose=2)

gsearch7.fit(X_train, Y_train)
print("5. grid best_score_",abs(gsearch7.best_score_))
print("best params",gsearch7.best_params_)
print("best score",gsearch7.best_score_)

Y_pred = gsearch7.predict(X_test)
print("MAE on test data",mean_absolute_error(Y_test,Y_pred))
print("MSE on test data",mean_squared_error(Y_test,Y_pred))

#0.05 0.0670037948915
#0.0670416222198

xgb24=XGBRegressor(learning_rate =0.1,n_estimators=1000,
                    max_depth=2,
                    min_child_weight=350,
                    gamma=0.4,
                    objective= 'reg:linear',
                    nthread=-1,scale_pos_weight=1,seed=3192,colsample_bytree=0.9,subsample=0.95,reg_alpha=0.05)

#recaliberate n_estimators with the values found above

modelfit(xgb24, X_train, Y_train,X_test,Y_test)

#pcaX_train, pcaX_test, pcaY_train, pcaY_test = train_test_split(pca1, Y, test_size=0.10, random_state=3192)


#0.05 0.0670037948915
#0.0670416222198

xgb24=XGBRegressor(learning_rate =0.1,n_estimators=1000,
                    max_depth=2,
                    min_child_weight=350,
                    gamma=0.4,
                    objective= 'reg:linear',
                    nthread=-1,scale_pos_weight=1,seed=3192,colsample_bytree=0.9,subsample=0.95,reg_alpha=0.05)

#recaliberate n_estimators with the values found above

modelfit(xgb24, pcaX_train,pcaY_train,pcaX_test,pcaY_test)

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,make_scorer,mean_squared_error


regr = RandomForestRegressor(n_jobs=-1, random_state=30, verbose=2)
parameters = { "n_estimators": [50],"criterion":["mae"]}


grid = GridSearchCV(regr,param_grid=parameters,cv=5,refit='MAE',verbose=0,n_jobs=-1,pre_dispatch='n_jobs')

grid.fit(X_train[xgboost_selection], Y_train)
print("5. grid best_score_",abs(grid.best_score_))
Y_pred = grid.predict(X_test[xgboost_selection])
print("MAE on test data",mean_absolute_error(Y_test,Y_pred))
print("MSE on test data",mean_squared_error(Y_test,Y_pred))
print("R Squared data ",r2_score(Y_test,Y_pred))

#Lets start by plotting a heatmap to determine if any variables are correlated
import seaborn as sns

numcols.remove('threequarterbathnbr')

numcols.remove('parcelid')
plt.figure(figsize = (12,8))
sns.heatmap(data=prop2016[numcols].corr())
plt.show()
plt.gcf().clear()

#unitcnt 996290
#regionidneighborhood 1817378
#buildingqualitytypeid 1035292

#from pandas.tools.plotting import scatter_matrix
#scatter_matrix(df_train[['buildingqualitytypeid','yearbuilt']])
#plt.show()
import matplotlib.pyplot as plt
plt.plot(prop2016.latitude, prop2016.longitude, 'ro')
plt.show()

