# Standard Imports
from itertools import chain
import cPickle as pickle
import pandas as pd
import numpy as np
from ast import literal_eval
import multiprocessing
from timeit import Timer
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# Imports from src directory
from Poisson import PoissonModel
from allocation import allocator
from hist_retrieval import hist_retriever
from allocation import allocator
from clustering import clusterer

# Create a Dataframe and drop extra column
df1 = pd.read_csv('../data/seattle_911_prepped_no_out1.csv', low_memory=False)
df1.drop(['Unnamed: 0'], axis=1, inplace=True)
df2 = pd.read_csv('../data/seattle_911_prepped_no_out2.csv', low_memory=False)
df2.drop(['Unnamed: 0'], axis=1, inplace=True)
df = pd.concat([df1, df2])
# Load pickled Poisson model
with open('PoissonModel.pkl', 'rb') as pkl_object:
    model = pickle.load(pkl_object)   

def get_history(df, query):
    # Set get home game info from user_data
    mariners, seahawks, sounders = 0,0,0
    if query['home_game'] == 'mariners':
        mariners = 1
    if query['home_game'] == 'seahawks':
        seahawks = 1
    if query['home_game'] == 'sounders':
        sounders = 1
    # Create History DataFrames for each zone
    df1 = hist_retriever(df, mariners, seahawks, sounders, query['date_input'],
                             query['time_range'], 'zone1')
    df2 = hist_retriever(df, mariners, seahawks, sounders, query['date_input'],
                             query['time_range'], 'zone2')
    df3 = hist_retriever(df, mariners, seahawks, sounders, query['date_input'],
                             query['time_range'], 'zone3')
    df4 =  hist_retriever(df, mariners, seahawks, sounders, query['date_input'],
                             query['time_range'], 'zone4')
    df5 =  hist_retriever(df, mariners, seahawks, sounders, query['date_input'],
                             query['time_range'], 'zone5')
    df6 = hist_retriever(df, mariners, seahawks, sounders, query['date_input'],
                             query['time_range'], 'zone6')
    df7 =  hist_retriever(df, mariners, seahawks, sounders, query['date_input'],
                             query['time_range'], 'zone7')
    return df1, df2, df3, df4, df5, df6, df7


def get_centroids(df1, df2, df3, df4, df5, df6, df7, alloc, query):
    limit = query['limit']
    # Count cores and create pool object for multiprocessing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cores)
    # Generate iterable arguments for pool
    datas = [[df1, int(alloc['zone1']), limit], [df2, int(alloc['zone2']), limit], 
             [df3, int(alloc['zone3']), limit], [df4, int(alloc['zone4']), limit], 
             [df5, int(alloc['zone5']), limit], [df6, int(alloc['zone6']), limit],
             [df7, int(alloc['zone7']), limit]]
    # Find centoids using clusterer function
    output = pool.map(clusterer, datas)
    centroids = list(chain(output[0][0], output[1][0], output[2][0],
                           output[3][0], output[4][0], output[5][0],
                           output[6][0]))
    return centroids


def make_centroid_df(centroids):
    # Convert list of tuples into DataFrame
    centroid_df = pd.DataFrame(centroids)
    centroid_df.columns = ['Latitude', 'Longitude']
    centroid_df.index += 1
    return centroid_df


def make_plot(df1, df2, df3, df4, df5, df6, df7, centroid_df):
    # Plot historical data against optimal unit placements
    plt.figure(figsize=(16.96,25))
    plt.scatter(x=df1.Longitude, y=df1.Latitude, color='m', s=30, alpha=0.4)
    plt.scatter(x=df2.Longitude, y=df2.Latitude, color='orange', s=30, alpha=0.4)
    plt.scatter(x=df3.Longitude, y=df3.Latitude, color='#38d159', s=30, alpha=0.4)
    plt.scatter(x=df4.Longitude, y=df4.Latitude, color='b', s=30, alpha=0.4)
    plt.scatter(x=df5.Longitude, y=df5.Latitude, color='r', s=30, alpha=0.4)
    plt.scatter(x=df6.Longitude, y=df6.Latitude, color='#53cfd6', s=30, alpha=0.4)
    plt.scatter(x=df7.Longitude, y=df7.Latitude, color='#868591', s=30, alpha=0.4)
    plt.scatter(centroid_df.Longitude, centroid_df.Latitude, s=300, color='k')
    plt.xlabel('Longitude', fontsize=28, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.ylabel('Latitude', fontsize=28, fontweight='bold')
    plt.yticks(fontsize=20)
    plt.title('Seattle 911 Responses by Zone', fontsize=36, fontweight='bold')
    plt.legend(['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 
                'Zone 5', 'Zone 6', 'Zone 7', 'Units'], fontsize=18)
    # To save plot, uncomment line below and create images directory in parent directory
    # plt.savefig('../images/plot911.png')

query = {'date_input': '2016-10-24', 'num_units': 24, 'home_game': 'no_game', 
         'time_range': 1, 'limit':None}

preds = model.predict(query)
preds

alloc = allocator(query["num_units"], preds)
alloc

get_ipython().run_cell_magic('time', '', 'df1,df2,df3,df4,df5,df6,df7 = get_history(df, query)\ncentroids = get_centroids(df1, df2, df3, df4, df5, df6, df7, alloc, query)\ncentroid_df = make_centroid_df(centroids)')

centroid_df

make_plot(df1, df2, df3, df4, df5, df6, df7, centroid_df)



