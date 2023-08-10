from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.datasets import mnist

# Load MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)

nb_training_samples = X_train.shape[0]
nb_testing_samples = X_test.shape[0]
image_length = image_width = X_train.shape[1]
print("{} training samples").format(nb_training_samples)
print("{} testing samples").format(nb_testing_samples)
print("image width and image length is {} pixels").format(image_length)

input_dim = image_length*image_width
# Reshape images from 2D to 1D as NN imput 
X_train = X_train.reshape(nb_training_samples, input_dim )
X_test = X_test.reshape(nb_testing_samples, input_dim )
# Cast pixel data type to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# Convert each pixel from 0-255 to 0.0-1.0
X_train /= 255
X_test /= 255

nb_classes = len(set(y_train))
print("There are total of {} classes").format(nb_classes)

# In multiclass classification tasks with cross-entropy loss,
# we need to convert labels to one hot vector representation
# e.g. 0-->[0,0,0,0,0,0,0,0,0,0,1]
#      1-->[0,0,0,0,0,0,0,0,0,1,0]...
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# NN model 
model = Sequential()

# First hidden layer (input dim: 784, output dim: 800)
model.add(Dense(800, input_shape = (input_dim,), init = 'uniform'))
# Add activation fuction
model.add(Activation('relu'))
# Include dropout percentange
model.add(Dropout(0.5))

# Output layer (input dim: 800 (auto-induced), output dim: nb_classes = 10)
model.add(Dense(nb_classes, init='uniform'))
model.add(Activation('softmax'))

# Prints a summary of your model
model.summary()

# Compile model before training
model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

# Train on training samples and validate on testing samples. 
history = model.fit(X_train, Y_train,
                    batch_size=256, nb_epoch=10,
                    verbose=2, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, verbose=2)
print('Test score: {}').format(score[0])
print('Test accuracy: {}').format(score[1])



