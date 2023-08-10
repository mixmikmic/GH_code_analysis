import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D,Input,Dense,Flatten
from matplotlib import pyplot as plt

X1 = pd.read_csv('../../master_data/nilm/kettle_input.dat', header = 0, delim_whitespace = True,
                                index_col = 0)
X2 = pd.read_csv('../../master_data/nilm/syntethic_kettle_dae.dat', header = 0, delim_whitespace = True,
                                index_col = 0)
y1 = pd.read_csv('../../master_data/nilm/kettle_signatures.dat', header = 0, delim_whitespace = True,
                                index_col = 0)
y2 = pd.read_csv('../../master_data/nilm/syntethic_kettle_dae_response.dat', header = 0, delim_whitespace = True,
                                index_col = 0)

#Comparing the syntethic and the real data
means_real = X1.mean(axis=1).sort_values().reset_index(drop=True)
means_synth = X2[:means_real.size].mean(axis=1).sort_values().reset_index(drop=True)
plt.hist(means_real,alpha=0.5, bins=20, range=(0,4000))
plt.hist(means_synth,alpha=0.5,color='orange', bins=20, range=(0,4000))
plt.show()

stds_real = X1.std(axis=1).sort_values().reset_index(drop=True)
stds_synth = X2[:stds_real.size].std(axis=1).sort_values().reset_index(drop=True)
plt.hist(stds_real,alpha=0.5,bins=20, range=(0,4000))
plt.hist(stds_synth,alpha=0.5,color='orange',bins=20, range=(0,4000))
plt.show()

#real_data = pd.concat([training_set_kettle,response_kettle],axis=1)

col_dict = dict(zip(X1.columns.values,X2.columns.values))
X = X2.append(X1.rename(columns=col_dict))

y = y2.append(y1)

print(X.shape)
print(y.shape)

X_np = np.array(X,dtype=np.float64).reshape((X.shape[0],X.shape[1],1))
y_np = np.array(y,dtype=np.float64).reshape((y.shape[0],y.shape[1]))

#some cleaning (some of the syntethic data contained 0-length signals?)
dirty = []
for i in range(y_np.shape[0]):
    if np.isnan(y_np[i]).any() or y_np[i].mean() == 0:#assume signal present.
        dirty.append(i)
y_np = np.delete(y_np, dirty,axis=0)
X_np = np.delete(X_np, dirty,axis=0)

print("Removed " + str(len(dirty)) + " instances.")
#row_mean = syntethic_data.mean(axis=1)
#syntethic_data_n = syntethic_data.sub(row_mean.T,axis=0)
#rand_sd = syntethic_data.std(axis=1)
#rand_sd = rand_sd.sample(frac=1).reset_index(drop=True)
#syntethic_data_n = syntethic_data_n.div(rand_sd,axis=0)
#syntethic_data_n.head()


dice = np.random.randint(0,y_np.shape[0])

print(dice)
plt.plot(X_np[dice])
plt.plot(y_np[dice])
plt.show()

sample_length = X.shape[1]

mean = X_np.mean(axis=1).reshape(X_np.shape[0],1,1)
#mean = 0.0
X_np = X_np - mean
sd = X_np.std(axis=1).mean()
#rand_sd = rand_sd.sample(frac=1).reset_index(drop=True)
X_np /= sd
print("Mean: ", X_np.mean())
print("Std: ", X_np.std())

normalization_params = pd.DataFrame([[mean,sd]],columns=['mean','sd'])
normalization_params.to_csv('C:/Users/bfesc/Documents/Master_thesis/master_data/nilm/normalization_params_dAE.csv', sep=' ')

layer1 = Conv1D(filters=8, input_shape = (sample_length,1,),kernel_size=4,
                activation='linear',padding='same', strides=1)
layer1_b = Flatten()
layer2 = Dense(units=(sample_length-3)*8,activation='relu')
layer3 = Dense(units=128,activation='relu')
layer4 = Dense(units=(sample_length-3)*8,activation='relu')
layer5 = Conv1D(filters=1,kernel_size=4,activation='linear',padding='same', strides=1)

model = Sequential()

model.add(layer1)
#model.add(layer1_b)
model.add(layer2)
model.add(layer3)
model.add(layer4)
model.add(layer5)

y_np = y_np.reshape(y_np.shape[0],y_np.shape[1],1)
print(X_np.shape, y_np.shape)
model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['mae'])

training_history = model.fit(X_np, y_np, batch_size=64,verbose=1,epochs=10, validation_split=0.1)

model.save('C:\\Users\\bfesc\\Documents\\Master_thesis\\master_data\\nilm\\models\\model0129bdAE.h5')

plt.plot(training_history.history['loss'])
plt.plot(training_history.history['val_loss'])
plt.show()

