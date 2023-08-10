from keras.datasets import boston_housing
(train_data,train_labels),(test_data,test_labels)=boston_housing.load_data()

import pandas as pd
pd.DataFrame(train_data).describe()

pd.DataFrame(test_data).describe()

mean=train_data.mean(axis=0)
std=train_data.std(axis=0)
train_data=(train_data-mean)/std
test_data=(test_data-mean)/std

from keras import models
from keras import layers

def build_model():
    model=models.Sequential()
    model.add(layers.Dense(64,activation="relu",input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="rmsprop",loss="mse",metrics=["mae"])
    return model

from sklearn.model_selection import KFold

folds=list(KFold(n_splits=4,shuffle=True,random_state=16).split(train_data,train_labels))
print (len(folds))

mae_scores=[]
maes=[]
for i in range(4):
    print("Fold :"+str(i))
    train_idxs,val_idxs=folds[i]
    part_train_data=train_data[train_idxs]
    part_train_labels=train_labels[train_idxs]
    val_data=train_data[val_idxs]
    val_labels=train_labels[val_idxs]
    model=build_model()
    history=model.fit(part_train_data,part_train_labels,epochs=100,batch_size=64,verbose=0)
    print(history.history.keys())
    maes.append(history.history["mean_absolute_error"])
    test_mse,test_mae=model.evaluate(test_data,test_labels)
    mae_scores.append(test_mae)
    

import numpy as np

print(np.mean(mae_scores))
print(mae_scores)

maes_avg=[np.mean([mae[i] for mae in maes]) for i in range(100)]

import matplotlib.pyplot as plt

epochs=range(1,len(maes_avg)+1)

plt.plot(epochs,maes_avg,label="MAE")
plt.legend(loc="best")
plt.show()

print("No Epoch %s got the minest mae."%np.argmin(maes_avg))



