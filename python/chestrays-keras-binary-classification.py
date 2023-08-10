import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pathlib
import shutil
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras import backend as K


get_ipython().run_line_magic('matplotlib', 'inline')

# Parameters
learning_rate = 0.01
#num_steps = 100
batch_size = 16
display_step = 100

# Images
toClassify = "Atelectasis"
IMG_HEIGHT = 250
IMG_WIDTH = 250
CH = 3
image_dir = "M:\\DataSets\\chestrays\\source\\" # XPS
rows = 4001 # number 
train_rows = 3000
test_rows = 1000

df = pd.read_csv("chestrays.csv", header=None, na_values="?")
df = df.iloc[1:rows]
df.head()

# Prepare train and test sets

# Factorize the labels and make the directories, convert all | to _'s, remove spaces
labels, names = pd.factorize(df[1])
image_names = image_dir + df.iloc[0:rows,0].values

# data mover function, also populates the dictionary so we can see the distribution of data
def copyImages(dataframe, idx, directory="train"):
    classification = dataframe.iloc[idx][1].replace(" ","").replace("|","_")
    source = image_dir + dataframe.iloc[idx][0]
    destination = directory + "/"
    
    if classification == "NoFinding":
        shutil.copy(source, destination + "NoFinding")
    elif classification.find(toClassify) >= 0:
        shutil.copy(source, destination + toClassify)

        
# Make classification directories
pathlib.Path("train/" + "NoFinding").mkdir(parents=True, exist_ok=True)
pathlib.Path("train/" + toClassify).mkdir(parents=True, exist_ok=True)
pathlib.Path("test/" + "NoFinding").mkdir(parents=True, exist_ok=True)
pathlib.Path("test/" + toClassify).mkdir(parents=True, exist_ok=True)


for r in range(train_rows):
    copyImages(df, r, "train")

for r in range(test_rows):
    copyImages(df, train_rows + r, "test")


num_classes = len(list(set(labels)))
print('Number of rows: {}'.format(len(labels))) 
print(names[:10])

img=mpimg.imread(image_names[0])

imgplot = plt.imshow(img, cmap='gray')
plt.title('Original - ' + image_names[0])
plt.show()

sess = tf.Session()
K.set_session(sess)
model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, CH), name="inputLayer"))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid', name="inferenceLayer"))

sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=["accuracy"])

# Visualize the model

SVG(model_to_dot(model).create(prog='dot', format='svg'))

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(rescale=1./255)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of './train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        'train',  # this is the target directory
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='binary')

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        'test',
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=batch_size,
        class_mode='binary')

print(validation_generator.class_indices)

# change epochs to a higher number, 3 is just for demo purposes
model.fit_generator(
        train_generator,
        steps_per_epoch=train_rows // batch_size,
        epochs=3,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

# Use TF to save the graph model instead of Keras save model to load it in Golang
builder = tf.saved_model.builder.SavedModelBuilder("myModel")
# Tag the model, required for Go
builder.add_meta_graph_and_variables(sess, ["myTag"])
builder.save()
sess.close()

# Example of running a prediction in Python
from keras.preprocessing import image
img = image.load_img(image_dir + "00001326_002.png", target_size=(IMG_WIDTH, IMG_HEIGHT))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = np.vstack([x]) # just append to this if we have more than one image.
classes = model.predict_classes(x)
print(classes)
sess.close()



