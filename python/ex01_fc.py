from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')

(train_images,train_labels),(test_images,test_labels)=mnist.load_data()
print (str(train_images.shape)+str(train_labels.shape)+str(test_images.shape)+str(test_labels.shape))

show_idx=np.random.choice(train_images.shape[0],1)[0]
plt.imshow(train_images[show_idx])
plt.title("The label is %s"%train_labels[show_idx])
plt.show()

from keras import models
from keras import layers

network=models.Sequential()
network.add(layers.Dense(512,activation="relu",input_shape=(28*28,)))
network.add(layers.Dense(10,activation="softmax"))

from keras import layers
from keras import models
input_tensor=layers.Input(shape=(784,))
x=layers.Dense(512,activation="relu")(input_tensor)
output_tensor=layers.Dense(10,activation="softmax")(x)

network=models.Model(input=input_tensor,output=output_tensor)

network.compile(optimizer="rmsprop",
               loss="categorical_crossentropy",
               metrics=["accuracy"])

train_images=train_images.reshape((60000,28*28)).astype("float32")/255
test_images=test_images.reshape((10000,28*28)).astype("float32")/255

from keras.utils import to_categorical

train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

train_labels[show_idx]

network.fit(train_images,train_labels,epochs=5,batch_size=128)

test_loss,test_acc=network.evaluate(test_images,test_labels)
print("Test loss:"+str(test_loss)+"\tTest accuracy:"+str(test_acc))



