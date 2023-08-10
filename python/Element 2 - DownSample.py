#imports
import pandas as pd
from math import radians, cos, sin, asin, sqrt
import multiprocessing
import numpy as np

#function to get the distance between two lat/long points
#return kilometers
def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    km = 6367 * c
    return km

#function to round a number to the nearest base
#Example: numRound(12, 10) = 10, numRound(18, 10) = 20
def numRound(x, base=5):
    return int(base * round(float(x)/base))

#Functions to support multiprocessing
#Source: https://gist.github.com/yong27/7869662
def _apply_df_rse(args):
    df = args
    return df.apply(lambda row: haversine(point153_lon, point153_lat, row[7]*10**-7, row[6]*10**-7), axis=1)

def apply_by_multiprocessing_rse(df):
    pool = multiprocessing.Pool(processes=8)
    result = pool.map(_apply_df_rse, [(d)
            for d in np.array_split(df, 8)])
    pool.close()
    return pd.concat(list(result),axis=0)

def _apply_df_p1(args):
    df = args
    return df.apply(lambda row: haversine(point153_lon, point153_lat, row[8], row[7]), axis=1)

def apply_by_multiprocessing_p1(df):
    pool = multiprocessing.Pool(processes=8)
    result = pool.map(_apply_df_p1, [(d)
            for d in np.array_split(df, 8)])
    pool.close()
    return pd.concat(list(result),axis=0)


point153_lon = -83.747333
point153_lat = 42.289141

path_to_rse_bsm = 'RSE_BSM.csv/RSE BSM.csv'
path_to_bsmp1 = 'mnt/win/RDE Development/Release 3/data environments/2 months safety pilot data/sent August 2015/April 2013/DAS_1_BSM_Data/april_BsmP1.csv'

get_ipython().run_cell_magic('time', '', "#ESTIMATED TIME 11s per chunk * 125 chunks = ~23mins\n#ACTUAL TIME 23min 40s\n#Chunk the csv\nrse_bsm_chunks = pd.read_csv(path_to_rse_bsm, header=None, chunksize=10**6)\nfor chunk in rse_bsm_chunks:\n    #Get dataframe from chunk and add distance column\n        #Note have to scale lon/lat since they are 1/10th microdegree\n    chunk['distance from 153'] = apply_by_multiprocessing_rse(chunk)\n    close_points = chunk.loc[chunk['distance from 153'] <= 0.5]\n\n    #If any rows exist within 1km of 153 then write to our file\n    if(len(close_points) > 0):\n        close_points.to_csv('data/rse_bsm_min.csv',index=False, header=False, mode='a')")

get_ipython().run_cell_magic('time', '', "#ESTIMATED TIME 5s per chunk * 1565 chunks = ~290mins\n#ACTUAL TIME 4h 58min 47s\n#Chunk the csv\nbsmp1_chunks = pd.read_csv(path_to_bsmp1, header=None, chunksize=10**6)\nfor chunk in bsmp1_chunks:\n    #Get dataframe from chunk and add distance column\n    chunk['distance from 153'] = apply_by_multiprocessing_p1(chunk)\n    close_points = chunk.loc[chunk['distance from 153'] <= 0.5]\n\n    #If any rows exist within 1km of 153 then write to our file\n    if(len(close_points) > 0):\n        close_points.to_csv('data/bsmp1_min.csv',index=False, header=False, mode='a')")

get_ipython().run_cell_magic('time', '', "#Downsample the data to just lat/long so that we can try plotting it on a map\n\n#Chunk the csv\nrse_chunks = pd.read_csv('data/rse_bsm_min.csv', header=None, chunksize=10**6)\nfor chunk in rse_chunks:\n    small_chunk = chunk.ix[:,6:7]\n    \n    small_chunk.to_csv('data/rse_latlong.csv', index=False, header=False, mode='a')")

get_ipython().run_cell_magic('time', '', "\n#Chunk the csv\np1_chunks = pd.read_csv('data/bsmp1_min.csv', header=None, chunksize=10**6)\nfor chunk in p1_chunks:\n    small_chunk = chunk.ix[:,7:8]\n    \n    small_chunk.to_csv('data/p1_latlong.csv', index=False, header=False, mode='a')")

#Lets down sample to 5 sigfigs for both p1 and rse
df = pd.read_csv('data/rse_latlong.csv', header=None)
df.columns = ['lat', 'lon']
df2 = pd.read_csv('data/p1_latlong.csv', header=None)
df2.columns = ['lat', 'lon']
def lower_precision_rse(x):
    return round(x*10**-7, 5)
def lower_precision_p1(x):
    return round(x, 5)

temp1 = df.applymap(lower_precision_rse).groupby(['lat','lon']).size().reset_index().rename(columns={0:'count'})
temp2 = df2.applymap(lower_precision_p1).groupby(['lat','lon']).size().reset_index().rename(columns={0:'count'})

temp1.to_csv('data/rse_latlon_min.csv', index=False)
temp2.to_csv('data/p1_latlon_min.csv', index=False)

#Read the previously downsampled files into pandas
p1_df = pd.read_csv('data/bsmp1_min.csv', header=None)
rse_df = pd.read_csv('data/rse_bsm_min.csv', header=None)

#Create speed column that is in mph binned to 10mph
p1_df['speed'] = p1_df[10].apply(lambda x: numRound(x*2.23694, 10))
rse_df['speed'] = rse_df[11].apply(lambda x: numRound(x*.02*2.23694, 10))

#Create distance column that is in meters binned to 50meters
p1_df['distance'] = p1_df[19].apply(lambda x: numRound(x*1000, 50))
rse_df['distance'] = rse_df[25].apply(lambda x: numRound(x*1000, 50))

#Find the count in each speed/distance bin and then create a column with that count
p1_temp = p1_df.groupby(['speed', 'distance']).size().reset_index(name = 'count')
rse_temp = rse_df.groupby(['speed', 'distance']).size().reset_index(name = 'count')

#Add an indicator of which file the count came from
p1_temp['file'] = 'p1'
rse_temp['file'] = 'rse'

#Write the two dataframes to a single csv file to be used in the viz
p1_temp.to_csv('data/heat.csv', index=False)
rse_temp.to_csv('data/heat.csv', index=False, header=False, mode='a')

