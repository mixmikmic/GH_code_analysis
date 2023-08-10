#analize models. 

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import donkeycar as dk
import requests
import pandas as pd
import tarfile
import numpy as np

import os

from PIL import Image

import keras

def get_record_from_df(df, data_path, record_transform=None, shuffle=False):
    
    index = list(df.index)
    if shuffle:
        random.shuffle(index)
    while True:
        for i in index:
            row = dict(df.iloc[i])
            img = Image.open(os.path.join(data_path, row['cam/image_array']))
            row['cam/image_array'] = np.array(img)
            if record_transform:
                row = record_transform(row)
            yield row

    
def get_batch_from_df(keys, df, record_transform=None, batch_size=128,
                      record_tranform=None, data_path=None, shuffle=False):
    
    record_gen = get_record_from_df(df, data_path, record_transform, shuffle=shuffle)
    
    if keys==None:
        keys = list(df.columns)
    while True:
        record_list = []
        for _ in range(batch_size):
            record_list.append(next(record_gen))

        batch_arrays = {}
        for i, k in enumerate(keys):
            arr = np.array([r[k] for r in record_list])
            #if len(arr.shape) == 1:
            #    arr = arr.reshape(arr.shape + (1,))
            batch_arrays[k] = arr

        yield batch_arrays
        
def train_gen(X_keys, Y_keys, df, batch_size=128, data_path=None, 
              record_transform=None):
    
    batch_gen = get_batch_from_df(X_keys+Y_keys, df, batch_size=128, record_transform=None, 
                                  data_path=data_path)
    while True:
        batch = next(batch_gen)
        X = [batch[k] for k in X_keys]
        Y = [batch[k] for k in Y_keys]
        yield X,Y

# Analyze model

tub_path = '/home/wroscoe/d2/data/alan/aw_medium/'
T = dk.parts.Tub(path=tub_path)

df = pd.DataFrame([T.get_json_record(i) for i in T.get_index(shuffled=False)])
print(df.shape)
df = df.dropna()
df.head()

img_gen = get_batch_from_df(['cam/image_array'], 
                            df, 
                            batch_size=len(df), 
                            data_path=tub_path)
data = next(img_gen)

m = keras.models.load_model('/home/wroscoe/d2/models/aw_ac_1')

predictions = m.predict(data['cam/image_array'])

psteering = predictions[0]
if len(psteering.shape) > 1:
    print('converting to linear output')
    psteering = dk.utils.unbin_Y(psteering)

df.head()

df['pilot/angle'] = psteering

df[['user/angle', 'pilot/angle']].hist()

print(df[['user/angle', 'pilot/angle']].std())

print(df[['user/angle', 'pilot/angle']].mean())

df[['user/angle', 'pilot/angle']][2180:2190].plot()

df[['user/angle', 'pilot/angle']][1100:1500].plot()

df['diff'] = df['user/angle'] - df['pilot/angle']

sdf = df.sort_values('diff', ascending=True)
sdf[:10]

sdf = sdf.reset_index(drop=True)
row = sdf.iloc[6]
print(row)
data_path = T.path
img = Image.open(os.path.join(data_path, row['cam/image_array']))
arr = np.array(img)

plt.imshow(arr)



