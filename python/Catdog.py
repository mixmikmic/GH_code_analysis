#Photo dataset originally from https://www.udemy.com/machinelearning/

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt  
import numpy as np

#Let's take a look at some of the images first:

from keras.preprocessing import image
from pathlib import Path
import random

def show_images(n=4, predict=False, patht='train'):

    while n>0:
        path = Path('C:/Users/Ryan/GitHub/CatDog/data/'+patht+'/'+random.choice(['cats','dogs']))
        rand_image = random.choice(list(path.glob('*.jpg')))
        rand_image = image.load_img(rand_image, target_size=(64, 64))
        
        if predict:
            test_image = image.img_to_array(rand_image)
            test_image = np.expand_dims(test_image, axis = 0)
            result = model.predict(test_image)
            labels = {0:'cat', 1:'dog'}
            print('Predicted: ',labels[round(result[0][0])])
            
        display(rand_image)
        
        n -= 1

show_images()

#Building the model

'''
Start with smaller filters, increase in later layers. 
Include a dense/fully connected layer before final sigmoid/softmax
maybe include dropout if it looks like overfitting
'''

model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", padding='same', input_shape=(64, 64, 3)))
model.add(Conv2D(32, (3, 3), padding='same', activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(Conv2D(64, (3, 3), padding='same', activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

# model.add(Conv2D(128, (3, 3), activation="relu"))
# model.add(Conv2D(128, (3, 3), activation="relu"))
# model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())

model.add(Dense(activation = 'relu', units = 128))
model.add(Dropout(0.5))
model.add(Dense(activation = 'relu', units = 128))
model.add(Dropout(0.5))

model.add(Dense(activation = 'sigmoid', units = 1))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

#Setting up the data for training and testing

'''including some data augmentation:'''

batch_size = 100

train_gen = ImageDataGenerator(rescale = 1/255, 
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_gen = ImageDataGenerator(rescale = 1/255)    #Don't augment the test data, only the training data

train_data = train_gen.flow_from_directory('data/train',
                                           target_size = (64, 64),
                                           batch_size = batch_size,
                                           class_mode = 'binary')

test_data = test_gen.flow_from_directory('data/test', 
                                         target_size = (64, 64),
                                         batch_size = batch_size,
                                         class_mode = 'binary')

#Training

'''
including earlystopping with excess epochs to prevent over/under fitting (probably reduce to patience=3 next time)
next time include model checkpoints callback too:

filepath = 'catdog.{epoch:02d}-{val_loss:.2f}.hdf5'
keras.callbacks.ModelCheckpoint(filepath, save_best_only=True, save_weights_only=False)
'''

train_size = train_data.n
test_size = test_data.n


history = model.fit_generator(train_data,
                             steps_per_epoch = train_size // batch_size,
                             epochs = 50,
                             validation_data = test_data,
                             validation_steps = test_size // batch_size,
                             callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
                             )

# save the model weights to disk, just in case

model.save("catdog.h5")
print("Saved model to disk")

# plotting the training history

fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(16,6))

# loss history
ax0.plot(history.history['loss'])
ax0.plot(history.history['val_loss'])
ax0.set_title('model loss')
ax0.set_ylabel('loss')
ax0.set_xlabel('epoch')
ax0.legend(['train', 'test'], loc='upper right')

#accuracy history
ax1.plot(history.history['acc'])
ax1.plot(history.history['val_acc'])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'test'], loc='upper left')

plt.show()

#Can see definite divergence in train vs test loss around epoch 35, signalling overfitting

#Let's visualise some predictions

show_images(n=4, predict=True, patht='test')

#possible improvements:

'''
Larger networksize, longer training
implement model checkpoints callback during training
Show confidence of predictions alongside images
Display low confidence images / incorrect classifications
'''

