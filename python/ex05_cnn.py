from keras.datasets import mnist
from keras.utils import to_categorical
(train_data,train_labels),(test_data,test_labels)=mnist.load_data()

train_images=train_data.reshape((len(train_data),28,28,1)).astype("float32")/255.
test_images=test_data.reshape((len(test_data),28,28,1)).astype("float32")/255.

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

from keras import layers
from keras import models

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation="relu"))
##Flatten
model.add(layers.Flatten())
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))
model.summary()

model.compile(optimizer="rmsprop",
             loss="categorical_crossentropy",
             metrics=["accuracy"])
model.fit(train_images,train_labels,
         epochs=5,batch_size=64)

model.evaluate(test_images,test_labels)



