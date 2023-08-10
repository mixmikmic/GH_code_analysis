import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import welly
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
    ### dictionary that holds every well as key:value or "UWI":df pair
    df_w_dict ={}
    for file in glob.glob('./SPE_006_originalData/OilSandsDB/Logs/*.LAS'):
        #### NOTE: limiting wells being read-in to 101 here !!!!!!!!!!!!!!!!
        if count >101:
            break
        count+=1  
        l_df = lasio.read(file).df()
        str_uwi= file[-23:-4].replace("-", "/",1)[:17]+file[-6:-4].replace("-", "/",1)
        l_df = l_df.reset_index()
        l_df['UWI'] = str_uwi
        l_df['SitID']=df_new[df_new['UWI']==str_uwi]['SitID'].iloc[0]
        l_df['UWI (AGS)']=df_new[df_new['UWI']==str_uwi]['UWI (AGS)'].iloc[0]
        l_df['Pick']=df_new[df_new['UWI']==str_uwi]['Pick'].iloc[0] 
        l_df['HorID']=df_new[df_new['UWI']==str_uwi]['HorID'].iloc[0]
        l_df['Quality']=df_new[df_new['UWI']==str_uwi]['Quality'].iloc[0]
        print(l_df.Pick.unique()[0])  
        try:
            #print("count = ",count)
            float(l_df.Pick.unique()[0])
            l_df.Pick = l_df.Pick.astype(float)
            l_df.DEPT = l_df.DEPT.astype(float)
            l_df['new_pick']=l_df['Pick']-l_df['DEPT']
            l_df['new_pick2']=l_df['new_pick'].apply(lambda x: 1 if(x==0) else 0)
            #print("got to above count >= 2")
            #if count == 1:
             #   data_df=l_df
            if count >= 2:
                #print("got inside count >= 2")
                #### instead of concat into a single dataframe, run functions & then add to dictionary   
                ##### run functions to create features on array basis for each well in separate dataframe
                ##### this makes certain things easier, compared to everything in a single dataframe, like making sure you don't accidentally grab data from next well up
                ##### and will make it easier to write data back to LAS if we run into memory limitations later
                curves = ['ILD','GR']
                windows = [5,7,11,21]
                directions = ["around","below","above"]
                comboArg_A = [curves,windows,directions]
                all_comboArgs_A = list(itertools.product(*comboArg_A))
                for eachArgList in all_comboArgs_A:
                    l_df_new = addColWindowMean(l_df,eachArgList[0],eachArgList[1],eachArgList[2])
                    l_df_new = addColWindowMax(l_df,eachArgList[0],eachArgList[1],eachArgList[2])
                    l_df_new = addColWindowMin(l_df,eachArgList[0],eachArgList[1],eachArgList[2])
                    l_df_new = addColWindowAvgMaxNvalues(l_df,eachArgList[0],eachArgList[1],eachArgList[2],3)
                #print("type(l_df) = ",type(l_df_new))
                #print("l_df['avg_GR_windowCenter5'][0] = ",l_df_new['avg_GR_windowCenter5'][0])
                #### add resultant dataframe to dictionary
                df_w_dict[l_df_new['UWI'][0]]= l_df_new
        except ValueError as e:
            print("e = ",e)
            print ('Error picking')
            #continue;

get_ipython().magic('prun loadAndAddFeatures()')

df_w_dict['00/01-03-085-15W4/0']

## testing one dataframe of one well in dictionary of all that were successfully read in
df_w_dict['00/01-03-085-15W4/0'].shape

keys = ['ILD','DPHI','GR','NPHI','CALI','COND','DELT','RHOB','PHIN','DT','ILM','SP','SFLU','IL','DEPTH','DEPH','MD']

keys2 = ['ILD','DPHI','GR','NPHI','CALI','RHOB']

type(w_test_df)





















