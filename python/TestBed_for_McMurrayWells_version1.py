# import some packages
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import welly
welly.__version__

import os
env = get_ipython().magic('env')

from welly import Well

# read one well log and take a look
w = Well.from_las('SPE_006_originalData/OilSandsDB/Logs/00-01-01-095-19W4-0.LAS')
w

w.data.keys()

w.data["CALI"]

tracks = ['CALI', 'GR', 'DPHI', 'NPHI', 'ILD']
w.plot(tracks=tracks)

import pandas as pd

picks_dicts = pd.read_table("SPE_006_originalData/OilSandsDB/PICKS_DIC.TXT", delimiter='\t')

picks_dicts

picks = pd.read_table("SPE_006_originalData/OilSandsDB/PICKS.TXT", delimiter='\t')

picks[picks['SitID']==102496]

wells = pd.read_table("SPE_006_originalData/OilSandsDB/WELLS.TXT", delimiter= '\t')

wells.shape

wells.head()

columns = ["DEPT","ILD", "DPHI", "GR", "NPHI"]

well_df = lasio.read("SPE_006_originalData/OilSandsDB/Logs/"+wellname+".LAS").df()
well_df.reset_index(level=0, inplace=True)
well_df=well_df[columns]
well_df['UWI']=wellname
sitid = wells[wells['UWI']==uwi]['SitID'].values[0]
picks_tgt = picks[picks['SitID']==sitid]
well_df['SitID']=sitid
well_df['HorID']=13000
well_df['Pick'] = int(picks_tgt[picks_tgt['HorID']==13000]["Pick"].values[0])
well_df.head()

picks.head()

wells.head()

sitid = wells[wells['UWI']==uwi]['SitID'].values[0]
print(sitid)
picks_tgt = picks[picks['SitID']==sitid]
pick_depth = picks_tgt[picks_tgt['HorID']==13000]["Pick"].values[0]
print(pick_depth)

well_df.head(10)

well_df.describe()

features = ['DEPT', 'ILD', 'DPHI','GR', 'NPHI']
label = 'Pick'

train_X = well_df[features]
train_y = well_df[label]
train_X.shape

train_X.head()

train_X.describe()

train_y.head()

train_y.describe()


from xgboost.sklearn import XGBRegressor

from xgboost.sklearn import XGBRegressor
model = XGBRegressor()

from xgboost.sklearn import XGBRegressor
model = XGBRegressor()
model.fit(train_X,train_y)
result = model.predict(train_X)
result

result = model.predict(train_X)
result

np.unique(result)

path = "SPE_006_originalData/OilSandsDB/Logs/"

import glob
file_list = []
for file in glob.glob(path+'*.LAS'):
    print(file[-23:-4])
    file_list.append(file[-23:-4])

file_list[:10]

UWIS = wells['UWI'].values

UWIS

well_data = pd.DataFrame()
for i in range(2):
    well_df = pd.DataFrame()
    uwi = UWIS[i]
    #print("UWI:{}".format(uwi))
    wellname = uwi.replace('/','-')
    well_df = lasio.read(path+wellname+".LAS").df()
    #print(well_df.head())
    well_df.reset_index(level=0, inplace=True)
    well_df=well_df[columns]
    well_df['UWI']=wellname
    sitid = wells[wells['UWI']==uwi]['SitID'].values[0]
    picks_tgt = picks[picks['SitID']==sitid]
    well_df['SitID']=sitid
    well_df['HorID']=13000
    well_df['Pick'] = 0
    pick_depth = int(picks_tgt[picks_tgt['HorID']==13000]["Pick"].values[0])
    well_df.loc[well_df['DEPT']==pick_depth,'Pick'] =1
    #print(well_df.head())
    well_data = pd.concat([well_data, well_df])
    

well_data

well_data[well_data['Pick']==1]

features = ['DEPT', 'ILD', 'DPHI','GR', 'NPHI', 'SitID']
label = 'Pick'
train_X = well_data[features]
train_y = well_data[label]
print(train_X.shape)
print(train_X.head())

from xgboost.sklearn import XGBRegressor
model = XGBRegressor()
model.fit(train_X,train_y)
result = model.predict(train_X)
result

well_data_pred = well_data.copy()

well_data_pred['Pick_pred'] = result

well_data_pred['Pick_pred'].max()

id = 102496
ix = well_data_pred['SitID']==102496

temp = well_data_pred.loc[ix]
temp

temp[temp['Pick_pred']==temp['Pick_pred'].max()]

for ids in well_data['SitID'].unique():
    ix = well_data_pred['SitID']==ids
    temp = well_data_pred.loc[ix]
    print(ids, temp[temp['Pick_pred']==temp['Pick_pred'].max()]['DEPT'])



