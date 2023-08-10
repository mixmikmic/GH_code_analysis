get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pylab as plt
import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
import numpy as np

#Loading the data using keras utility
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Printing the shape of the train and test sets
print x_train.shape
print x_test.shape

#Displaying a few samples of the training set
fig, ax = plt.subplots(nrows=5, ncols=5)
counter = 0
for row in ax:
    for col in row:
        col.imshow(x_train[counter], cmap = 'gray')
        col.axis("off")
        counter+=1
plt.show()

#Displaying a few samples of the test set
fig, ax = plt.subplots(nrows=5, ncols=5)
counter = 0
for row in ax:
    for col in row:
        col.imshow(x_test[counter], cmap = 'gray')
        col.axis("off")
        counter+=1
plt.show()

#Setting training parameters
batch_size = 128
num_classes = 10
epochs = 20

#Normalizing data between 0 and 1
x_train = x_train.reshape(60000, 784) #784 = 28*28
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# One hot encoding
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Defining the Neural Network architecture
def my_model():
    inputs = Input(shape=(784,))
    out1 = Dense(512, activation='relu')(inputs)
    out2 = Dropout(0.2)(out1)
    out3 = Dense(512, activation='relu')(out2)
    out4 = Dropout(0.2)(out3)
    out5 = Dense(10, activation='softmax')(out4)
    model = Model(inputs=inputs, outputs=out5)
    model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
    return model

#Building the model and showing a summary
model = my_model()
model.summary()

# Checkpoint callback to save model after each improvement along the epochs
checkpoint = ModelCheckpoint('mnist_keras.model', monitor='val_loss'
                             ,verbose=0,save_weights_only= True,
                             save_best_only=True)

#Training the model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split= 0.2,callbacks = [checkpoint])

#Plot accuracy history
plt.plot(history.history['acc'], label = "Train")
plt.plot(history.history['val_acc'],label = "Validation")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend()
plt.xlim(0,19)
plt.grid()
plt.show()

#Plot the loss history
plt.plot(history.history['loss'], label = "Train")
plt.plot(history.history['val_loss'],label = "Validation")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.xlim(0,19)
plt.grid()
plt.show()

#Load the best_model during training
best_model = my_model()
best_model.load_weights('mnist_keras.model')

#Evaluate the model on the test set
score = best_model.evaluate(x_test, y_test, verbose=0)
print 'Test loss: %f' %score[0]
print 'Test accuracy: %f' %score[1]

# Run predicition
ypred = best_model.predict(x_test).argmax(axis = 1)
y_test2 = y_test.argmax(axis = 1)

# See samples where prediction failed
indexes = np.where(ypred!=y_test2)[0][::-1]

#Display some of the errors
x_test = x_test.reshape(-1,28,28)
fig, ax = plt.subplots(nrows=5, ncols=5)
plt.tight_layout()
counter = 0
for row in ax:
    for col in row:
        aux = indexes[counter]
        col.imshow(x_test[aux], cmap = 'gray')
        col.axis("off")
        col.set_title("%d,%d" %(y_test2[aux],ypred[aux]))
        counter+=1
plt.show()

weights, biases = best_model.layers[1].get_weights()
print weights.shape
fig, ax = plt.subplots(nrows=5, ncols=5)
plt.tight_layout()
counter = 0
for row in ax:
    for col in row:
        aux = indexes[counter]
        col.imshow(weights[:,-counter].reshape(28,28), cmap = 'gray')
        col.axis("off")
        counter+=1
plt.show()

best_model.layers_by_depth

