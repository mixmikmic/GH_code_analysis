get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
from utils import tutorial_pamap2

# datapath = '/media/sf_VBox_Shared/timeseries/actitiracker/WISDM_at_v2.0/'
# dfile = os.path.join(datapath, 'WISDM_at_v2.0_raw.txt')
datapath = '/media/sf_VBox_Shared/timeseries/actitiracker/WISDM_ar_v1.1/'
dfile = os.path.join(datapath, 'WISDM_ar_v1.1_raw_1.txt')

column_names = ['user','activity','timestamp', 'x-acc', 'y-acc', 'z-acc']
df_full = pd.read_csv(dfile, header=None, sep=',', names=column_names, na_values=';')

df_full['z-acc'] = [float(str(s).split(';')[0]) for s in df_full['z-acc']]
df_full = df_full.dropna()

#df_full['timestamp'] = df_full['timestamp'].astype('int')
#df_full = df_full[df_full['timestamp']>=0]
#df_full = df_full[df_full['timestamp']<=9e12]

df_full['datetime'] = pd.to_datetime(df_full.timestamp, unit='ns', errors='coerce')

df_full = df_full.sort_values(['user', 'timestamp'])

df_full.head()

df_full.shape

df_full['activity'].unique()

df_full.describe()

df_full['user'].nunique()

# A new block of data starts with a new user, or a leap in the time step
df_full['timediff'] = df_full['datetime'].diff()
df_full['newblock'] = False
df_full['newuser'] = False
df_full.loc[df_full['timediff'] > pd.Timedelta('1s'), 'newblock'] = True
df_full.loc[df_full['timediff'] < pd.Timedelta('20ms'), 'newblock'] = True
df_full.loc[df_full['user'].diff()!=0, 'newuser'] = True

# How many strange leaps do we have?
df_full['newblock'].sum()

# examples of leaps
df_full[df_full['newblock'] & ~df_full['newuser']].head()

X_dict = {}
for user in df_full['user'].unique():
    X_df =  df_full[df_full['user']==user]
    X = X_df[['x-acc', 'y-acc', 'z-acc']].as_matrix()
    labels = X_df['activity'].as_matrix()
    Xlist, ylist = tutorial_pamap2.split_activities(labels, 
                    X,
                    [], 
                    borders=0)
    X_dict[user] = (Xlist, ylist)

frame_length = 10 * 50 # 10 seconds
step = 10 * 50 # 1 second

sample_dict = {}
for user in X_dict:
    Xlist, ylist = X_dict[user]
    X_sample_list, y_sample_list = tutorial_pamap2.sliding_window(frame_length, step, Xlist, ylist)
    if len(X_sample_list) > 0:
        X = np.array(X_sample_list)
        y = np.array(y_sample_list)
        sample_dict[user] = X, y

userids = np.array(list(sample_dict.keys()))
nr_users = len(userids)
nr_users_test = int(nr_users*0.1)
nr_users_val = int(nr_users*0.1)

neworder = np.random.permutation(nr_users)
userids = userids[neworder]

train_userids = userids[:-(nr_users_test+nr_users_val)]
test_userids = userids[-(nr_users_test+nr_users_val):-nr_users_val]
val_userids = userids[-nr_users_val:]
print('train: {}, test: {}, val: {}'.format(len(train_userids), len(test_userids), len(val_userids)))

X_train = np.concatenate([sample_dict[userid][0] for userid in train_userids])
y_train = np.concatenate([sample_dict[userid][1] for userid in train_userids])
X_test = np.concatenate([sample_dict[userid][0] for userid in test_userids])
y_test = np.concatenate([sample_dict[userid][1] for userid in test_userids])
X_val = np.concatenate([sample_dict[userid][0] for userid in val_userids])
y_val = np.concatenate([sample_dict[userid][1] for userid in val_userids])

labels = list(df_full['activity'].unique().astype('unicode'))
mapclasses = {labels[i]: i for i in range(len(labels))}

y_train_binary = tutorial_pamap2.transform_y(y_train, mapclasses, len(labels))
y_test_binary = tutorial_pamap2.transform_y(y_test, mapclasses, len(labels))
y_val_binary = tutorial_pamap2.transform_y(y_val, mapclasses, len(labels))

X_train.shape, y_train_binary.shape, X_test.shape, y_test_binary.shape, X_val.shape, y_val_binary.shape

X_train.shape[0] + X_test.shape[0] + X_val.shape[0]

import json

outdatapath = os.path.join(datapath,'preprocessed')

tutorial_pamap2.numpify_and_store(X_train, y_train_binary, 'X_train', 'y_train', outdatapath, shuffle=True)
tutorial_pamap2.numpify_and_store(X_test, y_test_binary, 'X_test', 'y_test', outdatapath, shuffle=True)
tutorial_pamap2.numpify_and_store(X_val, y_val_binary, 'X_val', 'y_val', outdatapath, shuffle=True)

with open(os.path.join(outdatapath, 'labels.json'), 'w') as fp:
    json.dump(labels, fp)



