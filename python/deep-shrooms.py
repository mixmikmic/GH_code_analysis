from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

DATASET_VERSION = 'mushroom_world_2017_16_10'
DATASET_LINK = 'https://s3.eu-central-1.amazonaws.com/deep-shrooms/{}.zip'.format(DATASET_VERSION)

with urlopen(DATASET_LINK) as zipresp:
    with ZipFile(BytesIO(zipresp.read())) as zfile:
        zfile.extractall('./data')
        

import pandas as pd
import numpy as np

DATASET_PATH = 'data/{}/'.format(DATASET_VERSION)

mushroom_classes = pd.read_json(DATASET_PATH + 'mushroom_classes.json', lines=True)
mushroom_imgs = pd.read_json(DATASET_PATH + 'mushroom_imgs.json', lines=True)
mushroom_info = mushroom_imgs.merge(mushroom_classes, how = "right", on = "name_latin")

# fin_names = pd.read_csv('test_labels.csv')

from scipy import misc

def load_mushroom_images(folder_path, img_df):
    img_dict = {}
    for index, path in enumerate(img_df['file_path']):
        img_dict[index] = misc.imread(folder_path + path)
    return img_dict
        
img_dict = load_mushroom_images(DATASET_PATH, mushroom_info)

#Quick look at the amount and shapes of images are not in the shape of the first element, 
#which is in the most common format (360,480,3)
i = 0
for img in img_dict:
    if img_dict[img].shape != img_dict[0].shape:
        i = i + 1
        print(img_dict[img].shape)
        print(img)
print(i)

#If this cell is run after the formatting is done, it should print 0 since all pictures should be in same format

import math
#Format the pictures to (480,480,3) by padding them with the edge values
for img in img_dict:
    height = 480 - img_dict[img].shape[0]
    width = 480 - img_dict[img].shape[1]

    if(height % 2 == 1 & width % 2 == 1):
        height1,height2 = math.floor(height/2), math.floor(height/2) + 1
        width1,width2 = math.floor(width/2), math.floor(width/2) +1
    elif(width % 2 == 1):
        width1,width2 = math.floor(width/2), math.floor(height/2) + 1
        height1,height2 = int(height/2), int(height/2)
    elif(height % 2 == 1):
        height1,height2 = math.floor(height/2), math.floor(height/2) + 1
        width1,width2 = int(width/2), int(width/2) 
    else:
        height1,height2 = int(height/2), int(height/2)
        width1,width2 = int(width/2), int(width/2)

    if(height == 0):
        img_dict[img] = np.lib.pad(img_dict[img], ((0,0),(width1, width2),(0,0)), 'edge')
    elif (width == 0):
        img_dict[img] = np.lib.pad(img_dict[img], ((height1, height2),(0,0),(0,0)), 'edge')
    else:
        img_dict[img] = np.lib.pad(img_dict[img], ((height1, height2),(width1, width2),(0,0)), 'edge')

import matplotlib.pyplot as plt

def draw_im(i, image_data=img_dict, label_data=mushroom_info):
    name = label_data.iloc[i].name_latin
    edibility = label_data.iloc[i].edibility
    url = label_data.iloc[i].img_url
    x = image_data[i]
    plt.imshow(x)
    plt.title( name + ": " + edibility + "\n(" + url + ")")
    plt.show()

draw_im(20)

mushroom_info.edibility.value_counts()

labels = mushroom_info.edibility.isin(("edible", "edible and good", "edible and excellent"))

X = []
y = []

for i in range(len(labels)):
    if(img_dict[i].shape == (480,480,3)):
        y.append(labels[i])
        X.append(img_dict[i])


X = np.stack(X)
y = pd.Series(y)

print(X.shape)
print(y.shape)

def draw_im2(i, X, y):
    plt.imshow(X[i])
    plt.title("edible: " + str(y[i]))
    plt.show()

draw_im2(2, X, y)

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

test_datagen = ImageDataGenerator()

# See how the augmentation looks like
x = X[1]
x = x.reshape((1,) + x.shape) 

f, ax = plt.subplots(3, 3)
ax[0,0].imshow(x[0])

i = 0
for batch in train_datagen.flow(x, batch_size=1):
    img = batch.astype(np.uint8) 
    fig = ax[int(i/3), i%3].imshow(img[0])
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    i += 1
    if i > 8:
        break  # otherwise the generator would loop indefinitely


plt.show()

y.value_counts()

# Train and test data

# rescale !!
X = X/255.0

N = len(X)
N_tr = int(0.8*N)

# shuffle the data
indx = np.arange(N)
np.random.shuffle(indx)
X = X[indx]
y = y[indx]

# split
X_tr = X[0:N_tr]
y_tr = y[0:N_tr]

X_te = X[N_tr:]
y_te = y[N_tr:]

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Dropout, Flatten

def my_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=X[0].shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(1))
    model.add(Activation('sigmoid')) # sigmoid is the binary case of softmax 
    return model


model = my_model()
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)

from keras.utils import plot_model
plot_model(model, show_shapes = True, to_file='docs/model.png')

# fits the model on batches with real-time data augmentation:
history = model.fit_generator(train_datagen.flow(X_tr, y_tr, batch_size = 32),
                    validation_data = test_datagen.flow(X_te, y_te, batch_size = 32),
                    steps_per_epoch= len(X_tr) / 32, 
                    validation_steps = len(X_te) /32,
                    epochs = 100, verbose = 1)

# save model weights and training history
model.save_weights('models/weights.h5')
import pickle
with open('models/history', 'wb') as f:
        pickle.dump(history.history, f)

# load the model training history
with open("models/history", "rb") as f:
    hist = pickle.load(f)
    
plt.plot(hist["acc"], label = "train accuracy", color = "green")
plt.plot(hist["val_acc"], label = "test accuracy", color = "blue")
plt.legend()
plt.savefig("docs/training_history.png")
plt.show()

# load the saved json model
from keras.models import model_from_json

json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/weights.h5")

X_imgs = X*255.0

# inputs
# i: integer defining which image in X to do prediction on
# X: ndarray of shape (N, 480, 480 3) containing image data
# y: array of shape (N) containing edibility labels
# model: a model which has a .predict() method that outputs probabilities
# save: save the image to the docs/ folders
def predict_and_draw(i, X, y, model, save = False):
    x = X[i]
    x = x.astype(np.uint8)
    plt.imshow(x)
    x = x/ 255.0
    x.shape = (1, ) + x.shape
    p = model.predict(x)[0,0].round(2)
    plt.title("P(edible): " + str(p) + " Actually edible: " + str(y[i]))
    if(save):
        plt.savefig("docs/prediction_example.png")
    plt.show()

predict_and_draw(8, X = X_imgs, y = y, model = loaded_model, save = True)

