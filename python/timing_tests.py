from midifile import *
import numpy as np
import pandas as pd

fileNames = [fileName for fileName in os.listdir("midiIn") if 'Bach' in str(fileName)]
fileNames

mf = midiFile(fileNames[-1])

mf.df.note_length.value_counts()

sl=20
X,Y = mf.get_trainable_arrays(seq_length=sl,col='note_length')
data_seed = X[0]
data_seed = np.reshape(data_seed, (1, sl, 1))
target = mf.encode_target(Y,n_values=16)

print(Y[0],target[0])

####### For Training across all songs in the fileNames list
#######
from neuralnet import *

sl = 20
X = np.empty(shape=(10, sl, 1)) # shape[0] doesn't matter to the model
model = timing_network(X)
data_seeds = []
for track in fileNames:
    mf = midiFile(track)
    X,Y = mf.get_trainable_arrays(seq_length=sl,col='note_length')
    data_seed = X[0]
    data_seed = np.reshape(data_seed, (1, sl, 1))
    data_seeds.append(data_seed)
    target = mf.encode_target(Y,n_values=16)
    model.train(X,target,epochs=10,batch_size=5,filepath="recent_lstm_timing_model_weights.h5")

for x in model.model.predict(X):
    print(np.argmax(x)+1)

preds = model.model.predict(X)
plot_results = []
for p in preds:
    plot_results.append(np.argmax(p)+1)

yes = 0
count = 0
for i,j in zip(Y,preds):
    y = np.argmax(j)+1
    if i==y:
        yes+=1
    count+=1
print(float(yes/count))

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
plt.style.use("seaborn-poster")

lowx = 0
highx = 100
plt.plot(np.linspace(0,len(plot_results[lowx:highx]),len(plot_results[lowx:highx])),plot_results[lowx:highx],'b',label="Pred")
plt.plot(np.linspace(0,len(Y[lowx:highx]),len(Y[lowx:highx])),Y[lowx:highx],'r',label="True");
plt.xlabel("Note location in song")
plt.ylabel("Beat Length Class")
plt.title("Song map for %s"%mf.fname)
plt.legend()

pl_results = []
results = []
data_seed = X[0]
for i in range(2000):
    if i%500==0:
        print("Generating point ",i)
    data_seed = np.reshape(data_seed, (1, sl, 1))
    next_val = model.model.predict(data_seed)
    next_val = np.argmax(next_val)+1
    pl_results.append(next_val)
    next_val = np.reshape(next_val,(1))
    data_seed = data_seed[0].tolist()
    data_seed.append([next_val])
    data_seed = data_seed[1:len(data_seed)]

lowx = 0
highx = 100
data = data_seed
plt.plot(np.linspace(0,len(pl_results[lowx:highx]),len(pl_results[lowx:highx]))+sl,pl_results[lowx:highx],'k',label="Gen")
plt.plot(np.linspace(0,len(X[0][0:sl]),len(X[0][0:sl])),X[0][0:sl],'r',label="Seed");
plt.title("Generated Data From Seed")
plt.legend()

timing = mf.timing_df.sort_values(by=['note','timing'])
timing['length'] = timing.timing.diff()
timing['beat'] = timing.length.shift(-1)
timing

beats = timing[timing['cmd'].str.contains("Note_on")]
beats.beat.value_counts()

beats['note_length'] = beats.beat/mf.qn*4
beats['note_length'] = beats.note_length.astype(int)

beats.note_length.value_counts()

beats[beats.note_length == 7]

mf.df.join(beats['note_length'])



