from tensorflow.contrib import keras
from tensorflow.contrib.keras.python.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data() # will download the dataset the first time
print "Training data shape ", x_train.shape

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

from tensorflow.contrib.keras.python.keras.models import Sequential
from tensorflow.contrib.keras.python.keras.layers import Dense, Dropout

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(784,))) # must match input shape
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) # must match the number of classes in the dataset
print "Model size: %.2f"%(model.count_params()/1000000.0), "million parameters"

model.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=256,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_test, y_test))

model1 = Sequential() 
model1.add(Dense(256, activation='relu', input_shape=(28*28,))) 
print "One  Layer  with  28*28  BW  Image: ","%.2f"%(model1.count_params()/1000000.0), "million parameters"
model2 = Sequential() # For RGB colored 512*512 images
model2.add(Dense(256, activation='relu', input_shape=(512*512*3,))) 
print "One Layer with 512*512 colored Image: ", "%.2f"%(model2.count_params()/1000000.0), "million parameters"
model3 = Sequential() # For RGB colored 512*512 images
model3.add(Dense(256, activation='relu', input_shape=(1024*1024*3,))) 
print "One Layer with 1024*1024 colored Image: ", "%.2f"%(model3.count_params()/1000000.0), "million parameters"

from tensorflow.contrib.keras.python.keras.layers import Convolution2D, MaxPooling2D,Flatten
modelconv = Sequential()
modelconv.add(Convolution2D(256,3,3, padding='valid', input_shape=(28,28,1)))
modelconv.add(MaxPooling2D(pool_size=(2,2)))
modelconv.add(Dropout(0.2))
modelconv.add(Flatten())
modelconv.add(Dense(64, activation='relu')) # must match the number of classes in the dataset
modelconv.add(Dropout(0.5))
modelconv.add(Dense(10, activation='softmax')) # must match the number of classes in the dataset
print "Model size: %.2f"%(modelconv.count_params()/1000000.0), "million parameters"

x_train = x_train.reshape(x_train.shape[0], 28,28, 1)
x_test = x_test.reshape(x_test.shape[0], 28,28, 1)
modelconv.compile(loss='categorical_crossentropy',
              optimizer='Adam',
              metrics=['accuracy'])
historyconv = modelconv.fit(x_train, y_train,
                    batch_size=256,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_test, y_test))

