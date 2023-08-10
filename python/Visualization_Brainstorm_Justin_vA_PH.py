import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import welly
from welly import Well
import lasio
import glob
welly.__version__

get_ipython().run_cell_magic('timeit', '', 'import os\nenv = %env')

from IPython.display import display

pd.set_option('display.max_rows', 2000)
## pd.set_option('display.height', 2000)

## for finding similar geographic points
## scipy spatial kd tree - similar to quad tree ... CKD tree is a little faster as its the C version
## scipy.spatial
## spatial 
## asdfds
##  
## tree.query

## Modifying this slightly to bring in the base McMurray as well
## If we use that as a "pick we already have" it will make the training easier for the "top McMurray" pick.
picks_dic = pd.read_csv('./SPE_006_originalData/OilSandsDB/PICKS_DIC.TXT',delimiter='\t')
picks = pd.read_csv('./SPE_006_originalData/OilSandsDB/PICKS.TXT',delimiter='\t')
wells = pd.read_csv('./SPE_006_originalData/OilSandsDB/WELLS.TXT',delimiter='\t')
picks_new=picks[picks['HorID']==13000]
picks_paleoz=picks[picks['HorID']==14000]
df_new = pd.merge(wells, picks_new, on='SitID')
df_paleoz = pd.merge(wells, picks_paleoz, on='SitID')
df_new=pd.merge(df_paleoz, df_new, on='SitID')
df_new.head()

df_new.info()



df_new = df_new[["SitID","UWI (AGS)_x","UWI_x","HorID_x","Pick_x","Quality_x","HorID_y","Pick_y","Quality_y"]]
df_new["UWI (AGS)"] = df_new["UWI (AGS)_x"]
df_new["UWI"] = df_new["UWI_x"]
df_new["HorID"] = df_new["HorID_y"]
df_new["Pick"] = df_new["Pick_y"]
df_new["Quality"] = df_new["Quality_y"]
df_new["HorID_paleoz"] = df_new["HorID_x"]
df_new["Pick_paleoz"] = df_new["Pick_x"]
df_new["Quality_paleoz"] = df_new["Quality_x"]
df_new = df_new[["SitID","UWI (AGS)","UWI","HorID","Pick","Quality","HorID_paleoz","Pick_paleoz","Quality_paleoz"]]
df_new

#### Number of unique wells based on UWI
len(df_new.UWI.unique())



#### reading in an example well for testing
#  w_test_df = lasio.read(well_path+"00-01-04-075-23W4-0.LAS").df()
# w_test_df.df()

####
def addColWindowMean(df,col,windowSize,centered):
    featureName = col+"_mean_"+str(windowSize)+"winSize_"+"dir"+centered
    if(centered == "around"):
        df[featureName] = df[col].rolling(center=True,window=windowSize).mean() 
    elif(centered == "above"):
        df[featureName] = df[col].rolling(center=False,window=windowSize).mean() 
    elif(centered == "below"):
        #### reverse data frame
        #df = df.iloc[::-1]
        df = df.sort_index(ascending=False)
        df[featureName] = df[col].rolling(center=False,window=windowSize).mean() 
        #### unreverse
        df = df.sort_index(ascending=True)
    return df

####
def addColWindowMax(df,col,windowSize,centered):
    featureName = col+"_max_"+str(windowSize)+"winSize_"+"dir"+centered
    if(centered == "around"):
        df[featureName] = df[col].rolling(center=True,window=windowSize).max() 
    elif(centered == "above"):
        df[featureName] = df[col].rolling(center=False,window=windowSize).max() 
    elif(centered == "below"):
        #### reverse data frame
        #df = df.iloc[::-1]
        df = df.sort_index(ascending=False)
        df[featureName] = df[col].rolling(center=False,window=windowSize).max() 
        #### unreverse
        df = df.sort_index(ascending=True)
    return df

#### Returns a column with the min values of a window centered 
def addColWindowMin(df,col,windowSize,centered):
    featureName = col+"_min_"+str(windowSize)+"winSize_"+"dir"+centered
    if(centered == "around"):
        df[featureName] = df[col].rolling(center=True,window=windowSize).min() 
    elif(centered == "above"):
        df[featureName] = df[col].rolling(center=False,window=windowSize).min() 
    elif(centered == "below"):
        #### reverse data frame
        #df = df.iloc[::-1]
        df = df.sort_index(ascending=False)
        df[featureName] = df[col].rolling(center=False,window=windowSize).min() 
        #### unreverse
        
        df = df.sort_index(ascending=True)
    return df

#### helper function that takes in array and an integer for the number of highest values to find the mean of 
#### example: for an array = [1,3,6,44,33,22,452] and nValues = 2, the answer would be 44+452 / 2
def nLargest(array,nValues):
    answer = np.mean(array[np.argsort(array)[-nValues:]])  
    return answer

#### Returns a column with the average of the N largest values of a window 
def addColWindowAvgMaxNvalues(df,col,windowSize,centered,nValues):
    #df[featureName] = df[col].rolling(center=True,window=windowSize).nlargest(nValues).mean() 
    #return df
    featureName = col+"_min_"+str(windowSize)+"winSize_"+"dir"+centered+"_n"+str(nValues)
    if(centered == "around"):
        #df[featureName] = df[col].rolling(center=True,window=windowSize).nlargest(nValues).mean() 
        df[featureName] = df[col].rolling(center=True,window=windowSize).apply(lambda x: nLargest(x,nValues))
    elif(centered == "above"):
        df[featureName] = df[col].rolling(center=False,window=windowSize).apply(lambda x: nLargest(x,nValues))
    elif(centered == "below"):
        #### reverse data frame
        #df = df.iloc[::-1]
        df = df.sort_index(ascending=False)
        #   # df['new_column'] = df.apply(lambda x: your_function(x['High'],x['Low'],x['Close']), axis=1)
        df[featureName] = df[col].rolling(center=False,window=windowSize).apply(lambda x: nLargest(x,nValues))
        #df[featureName] = df[col].rolling(center=False,window=windowSize).nlargest(nValues).mean() 
        #### unreverse
        df = df.sort_index(ascending=True)
    return df

####
winVars = {"RangeOfCurves":['GR'],
                   "RangeOfWindows":[5,11,29],
                   "RangeOfWindowsCaution":[5],
                   "RangeOfDirection":['above','below','around'],
                   "MinOrMaxRange":['min','max'],
                   "NumbPtsRange":[1,5]}

def loadAndAddFeatures():
    count=0
    data_df=[]
    count_limit =1200
    list_of_failed_wells = []
    ### dictionary that holds every well as key:value or "UWI":df pair
    df_w_dict ={}
    while count < count_limit:
        for file in glob.glob('./SPE_006_originalData/OilSandsDB/Logs/*.LAS'):
            #### NOTE: limiting wells being read-in to 101 here !!!!!!!!!!!!!!!!
            count+=1
            if count > count_limit:
                print("hit limit of count below file for loop")
                answer = [df_w_dict,list_of_failed_wells]
                return answer
            else:
                l_df = lasio.read(file).df()
                #print(l_df)
                str_uwi= file[-23:-4].replace("-", "/",1)[:17]+file[-6:-4].replace("-", "/",1)
                #l_df.DEPT = l_df.DEPT.astype(float)
                ##bottom_well_depth = l_df['DEPT'].max()
                if df_new[df_new['UWI']==str_uwi]['Quality'].iloc[0] < 10:
                    l_df = l_df.reset_index()
                    print("got to UWI apppend")
                    l_df['UWI'] = str_uwi
                    print("UWI added is ",str_uwi," and type is ",type(str_uwi))
                    l_df['SitID']=df_new[df_new['UWI']==str_uwi]['SitID'].iloc[0]
                    l_df['UWI (AGS)']=df_new[df_new['UWI']==str_uwi]['UWI (AGS)'].iloc[0]
                    l_df['Pick']=df_new[df_new['UWI']==str_uwi]['Pick'].iloc[0] 
                    l_df['HorID']=df_new[df_new['UWI']==str_uwi]['HorID'].iloc[0]
                    l_df['Quality']=df_new[df_new['UWI']==str_uwi]['Quality'].iloc[0]
                    #### adding in paleozoic surface pick
                    l_df['Pick_paleoz']=df_new[df_new['UWI']==str_uwi]['Pick_paleoz'].iloc[0] 
                    l_df['HorID_paleoz']=df_new[df_new['UWI']==str_uwi]['HorID_paleoz'].iloc[0]
                    l_df['Quality_paleoz']=df_new[df_new['UWI']==str_uwi]['Quality_paleoz'].iloc[0]

                    print("got to end of col append & pick is ",l_df.Pick.unique()[0])  
                    try:
                        print("in first try statement, count = ",count)
                        float(l_df.Pick.unique()[0])
                        l_df.Pick = l_df.Pick.astype(float)
                        l_df.DEPT = l_df.DEPT.astype(float)
                        l_df['new_pick']=l_df['Pick']-l_df['DEPT']
                        l_df['new_pick2']=l_df['new_pick'].apply(lambda x: 1 if(x==0) else 0)
                        #### doing the same as below but for BASE mcMurray or Paleozoic surface pick
                        float(l_df.Pick_paleoz.unique()[0])
                        l_df.Pick_paleoz = l_df.Pick_paleoz.astype(float)
                        #l_df.DEPT = l_df.DEPT.astype(float)
                        l_df['new_pick_paleoz']=l_df['Pick_paleoz']-l_df['DEPT']
                        l_df['new_pick2_paleoz']=l_df['new_pick_paleoz'].apply(lambda x: 1 if(x==0) else 0)

                        print("got to below astype part")
                        #### instead of concat into a single dataframe, run functions & then add to dictionary   
                        ##### run functions to create features on array basis for each well in separate dataframe
                        ##### this makes certain things easier, compared to everything in a single dataframe, like making sure you don't accidentally grab data from next well up
                        ##### and will make it easier to write data back to LAS if we run into memory limitations later
                        curves = ['GR','ILD']
                        windows = [5,7,11,21]
                        directions = ["around","below","above"]
                        comboArg_A = [curves,windows,directions]
                        all_comboArgs_A = list(itertools.product(*comboArg_A))
                        for eachArgList in all_comboArgs_A:
                            try:
                                l_df_new = addColWindowMean(l_df,eachArgList[0],eachArgList[1],eachArgList[2])
                            except:
                                pass
                            try:
                                l_df_new = addColWindowMax(l_df,eachArgList[0],eachArgList[1],eachArgList[2])
                            except:
                                pass
                            try:
                                l_df_new = addColWindowMin(l_df,eachArgList[0],eachArgList[1],eachArgList[2])
                            except:
                                pass
                            try:
                                l_df_new = addColWindowAvgMaxNvalues(l_df,eachArgList[0],eachArgList[1],eachArgList[2],3)
                            except:
                                pass
                        #### add resultant dataframe to dictionary
                        if l_df['DEPT'].max() < 600:
                            df_w_dict[l_df_new['UWI'][0]]= l_df_new
                    except ValueError as e:
                        print("e = ",e)
                        print ('Error picking')
                        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                        message = template.format(type(e).__name__, e.args)
                        print("message = ",message)
                        print("file = ",file)
                        print("Got except, UWI added is ",str_uwi," and type is ",type(str_uwi))
                        list_of_failed_wells.append(str_uwi)
                        #continue;
                else:
                    pass
            #print("result = ",df_w_dict)
    #else: 
    #    return df_w_dict, list_of_failed_wells
    answer = [df_w_dict,list_of_failed_wells]
    
    return answer

## %timeit
answer = loadAndAddFeatures()

#answer=[df_w_dict,list_of_failed_wells]
df_w_dict = answer[0]
list_of_failed_wells = answer[1]





#print(df_w_dict)

print("list_of_failed_wells",list_of_failed_wells)

df_w_dict['00/04-13-077-05W4/0']

## testing one dataframe of one well in dictionary of all that were successfully read in
#df_w_dict['00/01-03-085-15W4/0'].shape

print(len(df_w_dict))

def turnDictofDFtoDF(dict_of_df):
    data_df = pd.DataFrame()
    list_of_df = []
    values = dict_of_df.values()
    for each in values:
        list_of_df.append(each)
    data_df = pd.concat(list_of_df)
    return data_df
        

data_df = turnDictofDFtoDF(df_w_dict)
data_df.shape

type(data_df)



paleozoic_pick_test = data_df['new_pick_paleoz'][1800:2000]
paleozoic_pick_test

keys = ['ILD','DPHI','GR','NPHI','CALI','COND','DELT','RHOB','PHIN','DT','ILM','SP','SFLU','IL','DEPTH','DEPH','MD']

keys2 = ['ILD','DPHI','GR','NPHI','CALI','RHOB']

all_col_names = list(df_w_dict['00/04-13-077-05W4/0'])
all_col_names

##old
OLDfeatures2 = ['DEPT',
 'DPHI',
 'NPHI',
 'GR',
 'ILD',
 'SitID',
 'new_pick_paleoz',
 'GR_mean_5winSize_diraround',
 'GR_max_5winSize_diraround',
 'GR_min_5winSize_diraround',
 'GR_min_5winSize_diraround_n3',
 'GR_mean_5winSize_dirabove',
 'GR_max_5winSize_dirabove',
 'GR_min_5winSize_dirabove',
 'GR_min_5winSize_dirabove_n3',
 'GR_mean_7winSize_diraround',
 'GR_max_7winSize_diraround',
 'GR_min_7winSize_diraround',
 'GR_min_7winSize_diraround_n3',
 'GR_mean_7winSize_dirabove',
 'GR_max_7winSize_dirabove',
 'GR_min_7winSize_dirabove',
 'GR_min_7winSize_dirabove_n3',
 'GR_mean_11winSize_diraround',
 'GR_max_11winSize_diraround',
 'GR_min_11winSize_diraround',
 'GR_min_11winSize_diraround_n3',
 'GR_mean_11winSize_dirabove',
 'GR_max_11winSize_dirabove',
 'GR_min_11winSize_dirabove',
 'GR_min_11winSize_dirabove_n3',
 'GR_mean_21winSize_diraround',
 'GR_max_21winSize_diraround',
 'GR_min_21winSize_diraround',
 'GR_min_21winSize_diraround_n3',
 'GR_mean_21winSize_dirabove',
 'GR_max_21winSize_dirabove',
 'GR_min_21winSize_dirabove',
 'GR_min_21winSize_dirabove_n3',
 'ILD_mean_5winSize_diraround',
 'ILD_max_5winSize_diraround',
 'ILD_min_5winSize_diraround',
 'ILD_min_5winSize_diraround_n3',
 'ILD_mean_5winSize_dirabove',
 'ILD_max_5winSize_dirabove',
 'ILD_min_5winSize_dirabove',
 'ILD_min_5winSize_dirabove_n3',
 'ILD_mean_7winSize_diraround',
 'ILD_max_7winSize_diraround',
 'ILD_min_7winSize_diraround',
 'ILD_min_7winSize_diraround_n3',
 'ILD_mean_7winSize_dirabove',
 'ILD_max_7winSize_dirabove',
 'ILD_min_7winSize_dirabove',
 'ILD_min_7winSize_dirabove_n3',
 'ILD_mean_11winSize_diraround',
 'ILD_max_11winSize_diraround',
 'ILD_min_11winSize_diraround',
 'ILD_min_11winSize_diraround_n3',
 'ILD_mean_11winSize_dirabove',
 'ILD_max_11winSize_dirabove',
 'ILD_min_11winSize_dirabove',
 'ILD_min_11winSize_dirabove_n3',
 'ILD_mean_21winSize_diraround',
 'ILD_max_21winSize_diraround',
 'ILD_min_21winSize_diraround',
 'ILD_min_21winSize_diraround_n3',
 'ILD_mean_21winSize_dirabove',
 'ILD_max_21winSize_dirabove',
 'ILD_min_21winSize_dirabove',
 'ILD_min_21winSize_dirabove_n3']

features2original = ['CALI','DEPT','DPHI','GR','ILD','NPHI', 'SitID','CALIder','DPHIder','GRder','ILDder']
features2 = [
    #'DEPT',
 'DPHI',
 'NPHI',
 'GR',
 'ILD',
 'SitID',
 'new_pick_paleoz',
 'GR_mean_5winSize_diraround',
 'GR_max_5winSize_diraround',
 'GR_min_5winSize_diraround',
 'GR_min_5winSize_diraround_n3',
 'GR_mean_5winSize_dirabove',
 'GR_max_5winSize_dirabove',
 'GR_min_5winSize_dirabove',
 'GR_min_5winSize_dirabove_n3',
 'GR_mean_7winSize_diraround',
 'GR_max_7winSize_diraround',
 'GR_min_7winSize_diraround',
 'GR_min_7winSize_diraround_n3',
 'GR_mean_7winSize_dirabove',
 'GR_max_7winSize_dirabove',
 'GR_min_7winSize_dirabove',
 'GR_min_7winSize_dirabove_n3',
 'GR_mean_11winSize_diraround',
 'GR_max_11winSize_diraround',
 'GR_min_11winSize_diraround',
 'GR_min_11winSize_diraround_n3',
 'GR_mean_11winSize_dirabove',
 'GR_max_11winSize_dirabove',
 'GR_min_11winSize_dirabove',
 'GR_min_11winSize_dirabove_n3',
 'GR_mean_21winSize_diraround',
 'GR_max_21winSize_diraround',
 'GR_min_21winSize_diraround',
 'GR_min_21winSize_diraround_n3',
 'GR_mean_21winSize_dirabove',
 'GR_max_21winSize_dirabove',
 'GR_min_21winSize_dirabove',
 'GR_min_21winSize_dirabove_n3',
 'ILD_mean_5winSize_diraround',
 'ILD_max_5winSize_diraround',
 'ILD_min_5winSize_diraround',
 'ILD_min_5winSize_diraround_n3',
 'ILD_mean_5winSize_dirabove',
 'ILD_max_5winSize_dirabove',
 'ILD_min_5winSize_dirabove',
 'ILD_min_5winSize_dirabove_n3',
 'ILD_mean_7winSize_diraround',
 'ILD_max_7winSize_diraround',
 'ILD_min_7winSize_diraround',
 'ILD_min_7winSize_diraround_n3',
 'ILD_mean_7winSize_dirabove',
 'ILD_max_7winSize_dirabove',
 'ILD_min_7winSize_dirabove',
 'ILD_min_7winSize_dirabove_n3',
 'ILD_mean_11winSize_diraround',
 'ILD_max_11winSize_diraround',
 'ILD_min_11winSize_diraround',
 'ILD_min_11winSize_diraround_n3',
 'ILD_mean_11winSize_dirabove',
 'ILD_max_11winSize_dirabove',
 'ILD_min_11winSize_dirabove',
 'ILD_min_11winSize_dirabove_n3',
 'ILD_mean_21winSize_diraround',
 'ILD_max_21winSize_diraround',
 'ILD_min_21winSize_diraround',
 'ILD_min_21winSize_diraround_n3',
 'ILD_mean_21winSize_dirabove',
 'ILD_max_21winSize_dirabove',
 'ILD_min_21winSize_dirabove',
 'ILD_min_21winSize_dirabove_n3']
label = 'new_pick2'
train_X2 = data_df[features2]
train_y = data_df[label]

train_X2.shape

from xgboost.sklearn import XGBRegressor
model2 = XGBRegressor()
model2.fit(train_X2, train_y)
result2= model2.predict(train_X2)
result2

well_data=data_df.copy()

well_data.shape

id_array = well_data['SitID'].unique()
id_array_permutation = np.random.permutation(id_array)
train_index = id_array_permutation[:int(len(id_array)*.8)]
test_index = id_array_permutation[int(len(id_array)*.8)+1:]
train_df = well_data.loc[well_data['SitID'].isin(train_index)]
test_df = well_data.loc[well_data['SitID'].isin(test_index)]

features_originalB = ['CALI','DEPT','DPHI','GR','ILD','NPHI']
features = [
    #'DEPT',
 'DPHI',
 'NPHI',
 'GR',
 'ILD',
 'SitID',
 'new_pick2_paleoz',
 'GR_mean_5winSize_diraround',
 'GR_max_5winSize_diraround',
 'GR_min_5winSize_diraround',
 'GR_min_5winSize_diraround_n3',
 'GR_mean_5winSize_dirabove',
 'GR_max_5winSize_dirabove',
 'GR_min_5winSize_dirabove',
 'GR_min_5winSize_dirabove_n3',
 'GR_mean_7winSize_diraround',
 'GR_max_7winSize_diraround',
 'GR_min_7winSize_diraround',
 'GR_min_7winSize_diraround_n3',
 'GR_mean_7winSize_dirabove',
 'GR_max_7winSize_dirabove',
 'GR_min_7winSize_dirabove',
 'GR_min_7winSize_dirabove_n3',
 'GR_mean_11winSize_diraround',
 'GR_max_11winSize_diraround',
 'GR_min_11winSize_diraround',
 'GR_min_11winSize_diraround_n3',
 'GR_mean_11winSize_dirabove',
 'GR_max_11winSize_dirabove',
 'GR_min_11winSize_dirabove',
 'GR_min_11winSize_dirabove_n3',
 'GR_mean_21winSize_diraround',
 'GR_max_21winSize_diraround',
 'GR_min_21winSize_diraround',
 'GR_min_21winSize_diraround_n3',
 'GR_mean_21winSize_dirabove',
 'GR_max_21winSize_dirabove',
 'GR_min_21winSize_dirabove',
 'GR_min_21winSize_dirabove_n3',
 'ILD_mean_5winSize_diraround',
 'ILD_max_5winSize_diraround',
 'ILD_min_5winSize_diraround',
 'ILD_min_5winSize_diraround_n3',
 'ILD_mean_5winSize_dirabove',
 'ILD_max_5winSize_dirabove',
 'ILD_min_5winSize_dirabove',
 'ILD_min_5winSize_dirabove_n3',
 'ILD_mean_7winSize_diraround',
 'ILD_max_7winSize_diraround',
 'ILD_min_7winSize_diraround',
 'ILD_min_7winSize_diraround_n3',
 'ILD_mean_7winSize_dirabove',
 'ILD_max_7winSize_dirabove',
 'ILD_min_7winSize_dirabove',
 'ILD_min_7winSize_dirabove_n3',
 'ILD_mean_11winSize_diraround',
 'ILD_max_11winSize_diraround',
 'ILD_min_11winSize_diraround',
 'ILD_min_11winSize_diraround_n3',
 'ILD_mean_11winSize_dirabove',
 'ILD_max_11winSize_dirabove',
 'ILD_min_11winSize_dirabove',
 'ILD_min_11winSize_dirabove_n3',
 'ILD_mean_21winSize_diraround',
 'ILD_max_21winSize_diraround',
 'ILD_min_21winSize_diraround',
 'ILD_min_21winSize_diraround_n3',
 'ILD_mean_21winSize_dirabove',
 'ILD_max_21winSize_dirabove',
 'ILD_min_21winSize_dirabove',
 'ILD_min_21winSize_dirabove_n3']

label = 'new_pick2'

seed = 123

from sklearn.metrics import mean_squared_error

from xgboost.sklearn import XGBRegressor
#params_final = (
#    gamma=0, 
#    alpha=0.2, 
#    maxdepth=3, 
#    subsample=0.8, 
#    colsamplebytree= 0.8, 
#    n_estimators= 100, 
#    learningrate= 0.1, 
#    minchildweight= 1
#)
train_X = train_df[features]
train_y = train_df[label]
test_X = test_df[features]
test_y = test_df[label]



model = XGBRegressor(
   gamma=0, 
   reg_alpha=0.2, 
   max_depth=3, 
   subsample=0.8, 
   colsample_bytree= 0.8, 
   n_estimators= 300, 
   learning_rate= 0.03, 
   min_child_weight= 3)
model.fit(train_X,train_y)
result = model.predict(test_X)
result

from xgboost import plot_importance
plot_imp = plot_importance(model)
fig = plot_imp.figure
fig.set_size_inches(12, 20)

test_df_pred = test_df.copy()
test_df_pred['Pick_pred'] = result
test_df_pred.head()

len(test_df_pred.UWI.unique())

idx = test_df_pred.groupby(['SitID'])['Pick_pred'].transform(max) == test_df_pred['Pick_pred']
test_df_pred2=test_df_pred[idx]
        
        
# Score
final_score = np.sqrt(mean_squared_error(test_df_pred2['DEPT'],test_df_pred2['Pick']))
print("Prediction RMSE: {}".format(final_score))

plt.plot(test_df_pred2['DEPT'],test_df_pred2['Pick'], 'ro')

plt.scatter(test_df_pred2['DEPT'],test_df_pred2['Pick'], s=2)
plt.plot(test_df_pred2['DEPT'],test_df_pred2['DEPT'], 'ro')
#plt.plot([Y_all.min(), Y_all.max()], [Y_all.min(), Y_all.max()], 'k--', lw=2)
plt.xlabel('True Measured')
plt.ylabel('Predicted Depth')

#test_df_pred2['new_pick_paleoz']

#plt.scatter(test_df_pred2['DEPT'],test_df_pred2['new_pick_paleoz'], s=2)
#plt.plot(test_df_pred2['DEPT'],test_df_pred2['DEPT'], 'ro')
##plt.plot([Y_all.min(), Y_all.max()], [Y_all.min(), Y_all.max()], 'k--', lw=2)
#plt.xlabel('base pick actual')
#plt.ylabel('Predicted Depth')

display= ['CALI','DEPT','DPHI','GR','NPHI', 'SitID','new_pick_paleoz','new_pick2_paleoz','Pick',
         'GR_mean_5winSize_diraround',
 'GR_mean_21winSize_diraround',
 'GR_max_21winSize_diraround',
 'GR_min_21winSize_diraround',
 'GR_min_21winSize_diraround_n3',
 'GR_mean_21winSize_dirabove',
 'GR_max_21winSize_dirabove',
 'GR_min_21winSize_dirabove',
 'GR_min_21winSize_dirabove_n3',
         ]
df_display=test_df_pred2[display]
df_display.hist(bins=50, figsize=(20,15))

#tracks = ['CALI', 'GR', 'DPHI', 'NPHI', 'ILD']
#w = Well.from_las(test_df_pred2)
#w.plot(tracks=tracks)

## this well exists in the dataframe used in training?   '00/04-13-077-05W4/0'

test_df_pred2.UWI.unique()

#'00/10-09-078-09W4/0'
#l_df = lasio.read(file).df()
#'./SPE_006_originalData/OilSandsDB/Logs/*.LAS'

#exp_l_df = lasio.read('./SPE_006_originalData/OilSandsDB/Logs/'+'00/10-09-078-09W4/0'+'.LAS').df()
#df_pred_well = test_df_pred2[test_df_pred2['UWI'] == '00/10-09-078-09W4/0']

df_pred_well = test_df_pred2[test_df_pred2['UWI'] == '00/10-09-078-09W4/0']
df_pred_well

well_path = 'SPE_006_originalData/OilSandsDB/Logs/'
w = Well.from_las(well_path+'00-10-09-078-09W4-0.LAS')
w

print(type(w))

tracks = ['GR', 'DPHI', 'NPHI', 'ILD','RHOB']
w.plot(tracks=tracks)

#w_df = w.df()
print(type(w))

#w_df = w.df()
#w_df

#w = Well.from_las(well_path+'00-10-09-078-09W4-0.LAS')
w_2 = lasio.read(well_path+'00-10-09-078-09W4-0.LAS')
w_df = w_2.df()
print(type(w_df))

merged_df_of_well =  pd.concat([w_df, df_pred_well], axis=1)
merged_df_of_well.head()

merged_df_of_well = merged_df_of_well['DEPT','GR','Pick']

w_2.set_data_from_df(merged_df_of_well, truncate=False)    # triggers LASFile to create a new CurveItem - otherwise it'll be silently ignored.


print(type(w_2))

w_2

fn = "scratch.las"
with open(fn, mode="w") as f: # Write LAS file to disk
    w_2.write(f)

#w_2asdfasdf = lasio.read("scratch.las")
#w_2asdfasdf

#w_new = Well.from_las("scratch.las")

w_new.plot(tracks=tracks)



