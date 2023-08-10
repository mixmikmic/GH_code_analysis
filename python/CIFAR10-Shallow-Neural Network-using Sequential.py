import os    
#os.environ['THEANO_FLAGS'] = "device=gpu1"  
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32,lib.cnmem=1"
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
import theano
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.optimizers import *
from keras.utils.np_utils import *
from keras.regularizers import l2, activity_l2

from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

X_train=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/X_train.npy')
X_test=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/X_test.npy')
y_train=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/y_train.npy')
y_test=np.load('/media/mrafi123/UStore/Dimensionality-Reduction/data/CIFAR10/y_test.npy')

# Normalize the data: subtract the mean image
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image

X_train /= np.std(X_train, axis = 0) # normalize
X_test /= np.std(X_test, axis = 0) # normalize


print (X_train[0])

#x_train = X_train.astype('float32') / 255.
#x_test = X_test.astype('float32') / 255.
x_train = X_train.astype('float32') / 1.
x_test = X_test.astype('float32') / 1.
print (x_train[0])
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print (x_train.shape)
print (x_test.shape)

s_train, s_valid = train_test_split(x_train, test_size=0.1)
print (s_train.shape)
print (s_valid.shape)

label_train, label_valid = train_test_split(y_train, test_size=0.1)
print (label_train.shape)
print (label_valid.shape)

label_train=to_categorical(label_train)
label_valid=to_categorical(label_valid)

print (label_train.shape)
print (label_valid.shape)
print (label_train)

from keras import backend as K
import numpy as np

def my_init(shape, name=None):
    value = np.random.random(shape)
    print (value.shape)
    return K.variable(value, name=name)

#model.add(Dense(64, init=my_init))

my_init(100)

model = Sequential()
model.add(Dense(50, input_dim=3072,W_regularizer=l2(0.9),init='glorot_uniform', activation='relu'))
#model.add(Dense(50,W_regularizer=l2(0.5),init='glorot_uniform', activation='relu'))
model.add(Dense(10,W_regularizer=l2(0.9),init='glorot_uniform',activation='softmax'))

#autoencoder.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy') #around 35.47% test accuracy
model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy']) #around 40% test accuracy
#model.compile(optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='categorical_crossentropy', metrics=['accuracy']) #around 10% test accuracy
#model.compile(optimizer=Adam(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy']) #around 10% test accuracy
#model.compile(optimizer=SGD(lr=0.00001,decay=0.95,momentum=0.5), loss='categorical_crossentropy', metrics=['accuracy']) #around 10% test accuracy
#model.compile(optimizer=SGD(lr=0.00001, momentum=0.0, decay=0.0, nesterov=False), loss='categorical_crossentropy', metrics=['accuracy']) #around 10% test accuracy

history=model.fit(s_train, label_train,
                nb_epoch=60,
                batch_size=250,
                shuffle=True,
                validation_data=(s_valid, label_valid))


plt.plot(history.history['acc'])
print(max(history.history['acc']))

plt.plot(history.history['loss'])

plt.plot(history.history['val_acc'])

test_acc = model.predict_classes(x_test) 
print ('Test accuracy: ', test_acc)
print (test_acc.shape)
# round predictions
#rounded = [round(x) for x in test_acc]
#rounded = np.round(test_acc)
#print(rounded[1])



print (y_test.shape)
#y_test=to_categorical(y_test)
print (y_test.shape)

num=len(x_test)
r=0
w=0
for i in range(num):
        #print ('y_pred ',test_acc[i])
        #print ('labels ',y_test[i])
        #without the use of all() returns error truth value of an array with more than one element is ambiguous
        #if y_pred[i].all() == labels[i].all():
        if np.array_equal(test_acc[i],y_test[i]):
            r+=1
        else:
            w+=1
print ("tested ",  num, "digits")
print ("correct: ", r, "wrong: ", w, "error rate: ", float(w)*100/(r+w), "%")
print ("got correctly ", float(r)*100/(r+w), "%")



