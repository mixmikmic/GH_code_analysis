import keras

import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from keras_tqdm import TQDMNotebookCallback

# Visualization
from IPython.display import SVG
from IPython.display import display
from keras.utils.vis_utils import model_to_dot

# Length of sequences to model
length = 10

# Generate a random string of 0s and 1s
x_train = np.round(np.random.uniform(0,1,[length])).reshape([1,length,1])
x_train

# Calculate parity (note this is the same algorithmic approach
# that we are trying to encourage our net to learn!)
def parity(x):
    temp = np.zeros(x.shape)
    mem = False
    # Iterate over the sequence
    for i in range(x.shape[0]):
        if x[i,0] > 0.5:
            current = True
        else:
            current= False
        mem = np.logical_xor(mem,current)
        if mem:
            temp[i,0] = 1.0
        else:
            temp[i,0] = 0.0
    return (temp.reshape(1,temp.shape[0],temp.shape[1])) # Tensor!

y_train = parity(x_train[0,:,:])
y_train

# Network creation
model = keras.Sequential()
model.add(keras.layers.SimpleRNN(int(length*2),activation='relu',return_sequences=True,input_shape=[length,1]))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(1,activation='sigmoid')))
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()

# Visualization - wish we could see the recurrent weights!
SVG(model_to_dot(model).create(prog='dot', format='svg'))

batch_size = 1   # only one pattern...
epochs = 200
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          callbacks=[TQDMNotebookCallback()])
print('Accuracy:',model.evaluate(x_train,y_train)[1]*100.0,'%')

plt.figure(1)  
   
# summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
   
# summarize history for loss  

plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  

plt.tight_layout()
plt.show()  

np.round(model.predict(x_train))

# Check against the y_train vector
y_train

# Generate a concatenation of sequences
n_seq = 100
X = np.concatenate([np.round(np.random.uniform(0,1,[length])).reshape([1,length,1]) for i in range(n_seq)])
X_test = np.concatenate([np.round(np.random.uniform(0,1,[length])).reshape([1,length,1]) for i in range(n_seq)])
X.shape

Y = np.concatenate([parity(X[i,:,:]) for i in range(n_seq)])
Y_test = np.concatenate([parity(X_test[i,:,:]) for i in range(n_seq)])
Y.shape

# Look at the first set for confirmation
X[0,:,:]

Y[0,:,:]

# Network creation
model = keras.Sequential()
model.add(keras.layers.SimpleRNN(int(length*2),activation='relu',return_sequences=True,input_shape=[length,1]))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(1,activation='sigmoid')))
model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
model.summary()

SVG(model_to_dot(model).create(prog='dot', format='svg'))

batch_size = min(20,X.shape[0]) # Either 20 or the number of patterns if fewer than 20...
epochs = 300
history = model.fit(X, Y,
          batch_size=batch_size,
          epochs=epochs,
          verbose=0,
          callbacks=[TQDMNotebookCallback()],
          validation_split=0.2) # 
print('Training Accuracy:',model.evaluate(X,Y)[1]*100.0,'%')
print('Testing Accuracy:',model.evaluate(X_test, Y_test)[1]*100.0,'%')

plt.figure(1)  
   
# summarize history for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')  
   
# summarize history for loss  

plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'test'], loc='upper left')

plt.tight_layout()
plt.show()  

# Pick a pattern, any pattern...
p = 0

# Easier to see when rounded...
np.round(model.predict(X[p:p+1,:,:]))

# Compare with corresponding result!
Y[p:p+1,:,:]



