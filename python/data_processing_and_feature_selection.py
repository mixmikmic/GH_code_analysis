import glob
import json
import random
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tsfresh import extract_features, select_features
from tsfresh.utilities.dataframe_functions import impute

get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
random.seed(3)

table_dir = './MyShake_Training_Data/EQ/shake_table/'
table_files = glob.glob(table_dir + '*')
random.shuffle(table_files)


simulated_dir = './MyShake_Training_Data/EQ/simulated/'
simulated_files = glob.glob(simulated_dir + '*')
random.shuffle(simulated_files)

human_dir = './MyShake_Training_Data/Human/'
human_files = glob.glob(human_dir + '*')
random.shuffle(human_files)

def all_data(file):
    'Get dictionary from JSON file'
    with open(file, encoding='utf-8') as f:
        data=[json.loads(line) for line in f]
        data=data[0]
    return data

def get_data(data):
    '''
    Yuansi Chen's helper function to get the timestamp and 3-component acceleration data with small modifications. 
    '''
    # read in x, y, z data
    x = data['data']['x']
    y = data['data']['y']
    z = data['data']['z']
    
    size = min(len(x), len(y), len(z))
    x = x[:size]
    y = y[:size]
    z = z[:size]

    # calculate the timestamp
    # get the start time
    t0 = data['header']['starttime']
    npoints = len(x)
    sampling_rate = data['header']['sampling_rate']
    
    # get the end time 
    t1 = t0 + npoints / sampling_rate
    
    # form the timestamp
    t = np.arange(t0, t1, 1/sampling_rate)
    if len(t)>len(x):
        t = t[:len(x)] # Added this line to avoid rounding errors from generating extra point.
    elif len(t) < len(x):
        x = x[:len(t)]
        y = y[:len(t)]
        z = z[:len(t)]
    return t, x, y, z


def get_df(file, files_dir):
    'Function to transform file into pandas dataFrame'
    data = all_data(file)
    t, x, y, z = get_data(data)
    df = pd.DataFrame()
    df['t'] = t
    df['x'] = x
    df['y'] = y
    df['z'] = z
    if files_dir == human_dir:
        return df[:25*50] # If human, only get noise
    else:
        return df

mags = []
for file in simulated_files:
    simulated = all_data(file)
    mags.append(simulated['header']['mag'])


mags = np.asarray(mags)
print(scipy.stats.describe(mags))

triggertime = []
for file in human_files[:100]:
    human = all_data(file)
    triggertime.append(human['header']['triggertime'])

triggertime = np.asarray(triggertime)
print(scipy.stats.describe(triggertime))

def get_df_many_files(n_files, files, files_dir):
    file_id = 0
    'Get dataFrame with n samples from directory'
    df_all = pd.DataFrame()
    for file in files[:n_files]:
        df = get_df(file, files_dir)
        df['id'] = file_id
        df_all = df_all.append(df, ignore_index=True)
        file_id += 1
    return df_all

#    data = pandas.read_excel(infile)
#    appended_data.append(data) ## store dataframes in list
#appended_data = pd.concat(appended_data, axis=1) #

df_human = get_df_many_files(10, human_files, human_dir)
#df_table = get_df_many_files(10, table_files, table_dir)
df_simulated = get_df_many_files(10, simulated_files, simulated_dir)

df_simulated.describe()

df_human['id'] = df_human['id'] + 10

df_human.describe()

df_training = df_simulated.append(df_human, ignore_index=True)

df_training.describe()

X = extract_features(df_training, column_id='id', column_sort='t')

for column in list(X.keys()):
    print (column)
    plt.plot(X[column])
    plt.show()

Y = pd.DataFrame()
Y['id'] = np.arange(20)
Y['y'] = np.ones(20)
Y['y'][10:] = 0

y = np.ones(20)
y[10:] = 0
print (y)

#impute(X)
features_filtered = select_features(X, y)

