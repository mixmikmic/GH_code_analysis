from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
import keras
from sklearn.metrics import roc_auc_score
import numpy as np
from IPython.core.debugger import Tracer
from datetime import datetime

top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'Data/train'
validation_data_dir = 'Data/val'
nb_train_samples = 1800
nb_validation_samples = 200
epochs = 10 #50
batch_size = 25

class auc_roc_callback(keras.callbacks.Callback):
    def __init__(self, val_data_generator, val_labels, weight_file_path):
        self.val_data_generator = val_data_generator
        self.val_labels = val_labels
        self.val_samples = val_labels.shape[0]
        self.weight_file_path = weight_file_path
        self.best_AUC_Score = float("-inf")
    
    def on_train_begin(self, logs={}):
        self.auc_history = []
        self.loss = []
 
    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_generator(self.val_data_generator, self.val_samples)
        self.auc_history.append(roc_auc_score(self.val_labels, y_pred))
        print '\n AUC Score: ', self.auc_history[-1]
        
        #Saving the model weights if the AUC score is the best observed till now
        if self.best_AUC_Score < self.auc_history[-1]:
            dateTag = str(datetime.now().replace(second=0, microsecond=0)).replace(' ', '_').replace('-', '_').replace(':', '_')
            filepath = self.weight_file_path.format(str(round(self.auc_history[-1] * 100, 5)).replace('.', '_'), dateTag)
            print('Epoch %05d: AUC improved from %0.5f to %0.5f,'
                    ' saving model to %s'
                    % (epoch + 1, self.best_AUC_Score, self.auc_history[-1],
                       filepath))
            self.best_AUC_Score = self.auc_history[-1]
            self.model.save_weights(filepath, overwrite=True)
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        return

input = Input(shape=(3, img_width, img_height),name = 'image_input')
model = applications.VGG16(weights='imagenet', include_top=False, input_tensor = input)
print('Model loaded.', model.output_shape[1:])

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
top_model.load_weights(top_model_weights_path)

train_data = np.load(open('bottleneck_features_train.npy'))
train_data.shape

train_x = np.random.random_sample((24, 3,150,150))
train_y = (np.random.random_sample(24) > 0.8).astype(int)

val_x = np.random.random_sample((24, 3,150,150))
val_y = (np.random.random_sample(24) > 0.8).astype(int)

# add the model on top of the convolutional base
#Freezing the top 15 layers i.e. just before the last conv layer
for layer in model.layers[:15]:
    layer.trainable = False
    
mdl = Model(input= model.input, output= top_model(model.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)


# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
mdl.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, nesterov = True, decay = 0.05),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255)
'''
    rotation_range=95,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range = 0.2,
    height_shift_range = 0.2)
'''
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)


#Creating the callbacks
val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

val_labels = np.array([0]*167 + [1]*33)
#checkpointer = ModelCheckpoint(filepath="trainedmodel/weights_fineTuning_layersFreezed_AUC_{}_time_{}.hdf5", verbose=1, save_best_only=True)
auc_roc_hist = auc_roc_callback(val_generator, val_labels, "trainedmodel/weights_fineTuning_layersFreezed_AUC_{}_time_{}.hdf5")

# fine-tune the model
class_weight = {0 : 1.,
    1: 6.}
mdl.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples // batch_size,
    nb_epoch=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[auc_roc_hist],
    class_weight = class_weight)
'''

mdl.fit(
    train_x,
    train_y,
    #samples_per_epoch=nb_train_samples,
    batch_size = batch_size,
    nb_epoch=epochs,
    validation_data=(val_x, val_y),
    #nb_val_samples=nb_validation_samples,
    callbacks=[auc_roc_hist, checkpointer])
'''

input = Input(shape=(3, img_width, img_height),name = 'image_input')
model_adaptive = applications.VGG16(weights='imagenet', include_top=False, input_tensor = input)
print('Model loaded.', model_adaptive.output_shape[1:])

# add the model on top of the convolutional base
#Freezing the top 15 layers i.e. just before the last conv layer
for layer in model_adaptive.layers[:15]:
    layer.trainable = False
    
mdl_adap = Model(input= model_adaptive.input, output= top_model(model_adaptive.output))

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)


# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
mdl_adap.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1. / 255)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)


#Creating the callbacks
val_datagen = ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

val_labels = np.array([0]*167 + [1]*33)
#checkpointer = ModelCheckpoint(filepath="trainedmodel/weights_fineTuning_layersFreezed_AUC_{}_time_{}.hdf5", verbose=1, save_best_only=True)
auc_roc_hist_adap = auc_roc_callback(val_generator, val_labels, "trainedmodel/weights_fineTuning_AdamOpt_layersFreezed_AUC_{}_time_{}.hdf5")

# fine-tune the model

mdl_adap.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples // batch_size,
    nb_epoch=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples,
    callbacks=[auc_roc_hist_adap])

auc_roc_hist.auc_history

model.summary()

for i, layer in enumerate(model.layers[:25]):
    print i, layer

for i, layer in enumerate(model.layers[:15]):
    print i, layer

mdl.summary()



