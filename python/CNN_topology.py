from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=4, strides=4, padding='valid', 
    activation='relu', input_shape=(200, 200, 1)))
model.summary()



from keras.models import Sequential
from keras.layers import Conv2D, Conv3D
model = Sequential()
model.add(Conv3D(filters=10, kernel_size=(5,5,3) ,strides=(5,5,3), padding='valid', 
    activation='relu', input_shape=(None,32, 32, 3)))
model.summary()

from keras.models import Sequential
from keras.layers import Conv2D, Conv3D
model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(5,3) ,strides=(5,3), padding='valid', 
    activation='relu', input_shape=(32, 32, 3)))
model.summary()



