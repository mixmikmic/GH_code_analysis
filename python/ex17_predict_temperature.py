import os
data_dir="jena_climate_2009_2016.csv"
fname=os.path.join(data_dir,"jena_climate_2009_2016.csv")

f=open(fname)
data=f.read()
f.close()

lines=data.split("\n")
header=lines[0].split(",")
values=lines[1:]

print(header)
print(values[0])

import numpy as np

float_data=np.zeros((len(values),len(header)-1))
for i,line in enumerate(values):
    v=[float(x) for x in line.split(",")[1:]]
    float_data[i,:]=v
    

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(np.arange(1,len(float_data)+1),float_data[:,1])

def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=6):
    if max_index is None:
        max_index=len(data)-1-delay
    i=min_index+lookback
    while 1:
        if shuffle:
            rows=np.random.randint(min_index+lookback,max_index,size=batch_size)
        else:
            if i+batch_size>=max_index:
                i=min_index+lookback
            rows=np.arange(i,min(i+batch_size,max_index))
            i+=len(rows)
        samples=np.zeros((len(rows),lookback//step,data.shape[-1]))
        targets=np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices=range(rows[j]-lookback,rows[j],step)
            samples[j]=data[indices]
            targets[j]=data[rows[j]+delay][1]
        yield samples,targets

lookback=1440
delay=144
batch_size=128
mean=float_data[:200000].mean(axis=0)
float_data-=mean
std=float_data[:200000].std(axis=0)
float_data/=std

train_gen=generator(float_data,lookback,delay,0,200000,shuffle=True)
val_gen=generator(float_data,lookback,delay,200001,300000)
test_gen=generator(float_data,lookback,delay,300001,None)

train_steps=(200000-lookback)//batch_size
val_steps=(300000-200001-lookback)//batch_size
test_steps=(len(data)-300001-lookback)//batch_size

print("One Epoch takes %s steps for training."%train_steps)
print("One Epoch takes %s steps for validation."%val_steps)
print("One Epoch takes %s steps for testing."%test_steps)

def evaluate_naive_method():
    batch_maes=[]
    for step in range(val_steps):
        samples,targets=next(val_gen)
        preds=samples[:,-1,1]
        mae=np.mean(np.abs(preds-targets))
        batch_maes.append(mae)
    return  np.mean(batch_maes)
var=evaluate_naive_method()
print("The variation of baseline temperature is %s"%(var*std[1]))
print(std[1])

from keras.layers import Dense,Flatten
from keras.models import Sequential

model=Sequential()
model.add(Flatten(input_shape=((lookback//6,float_data.shape[-1]))))
model.add(Dense(64,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(1))

model.compile(optimizer="rmsprop",
             loss="mae")

model.fit_generator(train_gen,
                   epochs=20,
                   steps_per_epoch=train_steps,
                   validation_data=val_gen,
                   validation_steps=val_steps)

std[1]*model.evaluate_generator(val_gen,steps=val_steps)

from keras.layers import GRU,Dense
from keras.models import Sequential
model=Sequential()
model.add(GRU(64,input_shape=(lookback//6,float_data.shape[-1])))
model.add(Dense(32,activation="relu"))
model.add(Dense(1))

model.compile(optimizer="rmsprop",
             loss="mae")

model.fit_generator(train_gen,
                  steps_per_epoch=train_steps,
                  epochs=20,
                  validation_data=val_gen,
                  validation_steps=val_steps)

std[1]*model.evaluate_generator(val_gen,steps=val_steps)

from keras.layers import GRU,Dense
from keras.models import Sequential
model=Sequential()
model.add(GRU(64,input_shape=(lookback//6,float_data.shape[-1])),return_sequences=True)
model.add(GRU(32,return_sequences=True))
model.add(GRU(1))

model.compile(optimizer="rmsprop",
             loss="mae")

model.fit_generator(train_gen,
                   steps_per_epoch=train_steps,
                   epochs=20,
                   validation_data=val_gen,
                   validation_steps=val_steps)

std[1]*model.evaluate_generator(val_gen,steps=val_steps)

from keras.layers import Bidirectional
from keras.layers import GRU,Dense
from keras.models import Sequential
model=Sequential()
model.add(Bidirecational(GRU(64),input_shape=(lookback//6,float_data.shape[-1])))
model.add(Dense(32,activation="relu"))
model.add(Dense(1))

model.compile(optimizer="rmsprop",
             loss="mae")

model.fit_generator(train_gen,
                   steps_per_epoch=train_steps,
                   epochs=20,
                   validation_data=val_gen,
                   validation_steps=val_steps)

std[1]*model.evaluate_generator(val_gen,steps=val_steps)

from keras.layers import Conv1D,MaxPooling1D,Dense,GRU
from keras.models import Sequential

model=Sequential()
model.add(Conv1D(64,5,input_shape=(lookback//6,float_data.shape[-1])))
model.add(MaxPooling1D(3))
model.add(GRU(32))
model.add(Dense(1))

model.compile(optimizer="rmsprop",
             loss="mae")

model.fit_generator(train_gen,
                   steps_per_epoch=train_steps,
                   epochs=20,
                   validation_data=val_gen,
                   validation_steps=val_steps)

std[1]*model.evaluate_generator(val_gen,steps=val_steps)

