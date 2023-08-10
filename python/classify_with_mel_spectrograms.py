

import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Read the images
# Generate dummy data
x_train = np.random.random((100, 100, 100, 3))
y_train = keras.utils.to_categorical(np.random.randint(2, size=(100, 1)), num_classes=2)
x_test = np.random.random((20, 100, 100, 3))
y_test = keras.utils.to_categorical(np.random.randint(2, size=(20, 1)), num_classes=2)

img_path = "/Volumes/EXTERNAL/MusicEngine/mel_spectrogram/cases/TARGET_Biz_Amulet.png"
img = image.load_img(img_path, target_size=(230, 345))


x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

image.img_to_array(img).shape

img

# Preprocess images
# Generate batches of tensor image data with real-time data augmentation. The data will be looped over
#     (in batches) indefinitely. In order to make the most of our few training examples, we will "augment"
#     them via a number of random transformations, so that our model would never see twice the exact same picture.
#     This helps prevent overfitting and helps the model generalize better ->> https://keras.io/preprocessing/image/

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

img_path = "/Users/valentin/Documents/MusicEngine/model/train/cases/"
image = "TARGET_Biz_Amulet.png"

img = load_img(img_path + image, target_size=(460, 690))           # This is a PIL image
x = img_to_array(img)                                              # Numpy array with shape 3, 460, 690
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)


# The .flow() generates batches of randomly transformed images and saves them to the `preview/` directory
img_generator = ImageDataGenerator(
   # rotation_range = 40,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest')


i = 0
for batch in img_generator.flow(x, batch_size = 1,
                               save_to_dir = "/Users/valentin/Documents/MusicEngine/model/",
                               save_format = "jpeg"):
    i += 1
    if i > 10:
        break









# Specify the model using LSTM

model = Sequential()
# input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))



# Specify the optimization algorithm for the backprop
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Specify the loss function and compile model
model.compile(loss='categorical_crossentropy', optimizer=sgd)

# Train model and plot accuracies
train_model = model.fit(x_train,
                        y_train,
                        batch_size=32,
                        epochs=20,
                        verbose=1,
                        validation_data = (x_test, y_test))

# print the training accuracy and validation loss at each epoch
# print the number of models of the network
print(train_model.history)
print(len(model.layers))

# Plot the errors of the epochs and MSE
plt.plot(train_model.history['loss'])
plt.plot(train_model.history['val_loss'])

#  plt.plot(modelEstimate.history['val_acc'])
plt.title('Model Error History')
plt.ylabel('Mean Squared Error')
plt.xlabel('Epochs')
plt.legend(['Training Error', 'Validation Error'])
plt.show()

# Predict on the validation data
trainPredict = model.predict(x_test)

# Assess accuracy
model.evaluate(x_test, y_test, batch_size=32)





# Add a model checkpoint (from https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d)

filepath="weights-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True,mode='max')
callbacks_list = [checkpoint]

model.fit_generator(
train_generator,
samples_per_epoch=nb_train_samples,
nb_epoch=nb_epoch,
validation_data=validation_generator,
nb_val_samples=nb_validation_samples,callbacks=callbacks_list)



