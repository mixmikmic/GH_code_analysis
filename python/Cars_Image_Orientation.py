from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    cars_files = np.array(data['filenames'])
    cars_targets = np_utils.to_categorical(np.array(data['target']), 4)
    return cars_files, cars_targets

# load train, test, and validation datasets
train_files, train_targets = load_dataset('cars/train')
valid_files, valid_targets = load_dataset('cars/valid')
test_files, test_targets = load_dataset('cars/test')


#test_files9, test_targets9 = load_dataset('cars/test/2.90')


print(train_files)
print(train_targets)

# load list of dog names
cars_names = [item[13:-1] for item in sorted(glob("cars/train/*/"))]
print(cars_names)

# print statistics about the dataset
print('There are %d total cars image categories.' % len(cars_names))
print('There are %s total cars images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training cars images.' % len(train_files))
print('There are %d validation cars images.' % len(valid_files))
print('There are %d test cars images.'% len(test_files))

from keras.preprocessing import image                  
from tqdm import tqdm

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(299, 299))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True                 

# pre-process the data for Keras
train_tensors = paths_to_tensor(train_files).astype('float32')/255
valid_tensors = paths_to_tensor(valid_files).astype('float32')/255
test_tensors = paths_to_tensor(test_files).astype('float32')/255

import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization


IM_WIDTH, IM_HEIGHT = 299, 299 #fixed size for InceptionV3
NB_EPOCHS = 3
#BAT_SIZE = 32
FC_SIZE = 1024
NB_IV3_LAYERS_TO_FREEZE = 172
batch_size = 20

def setup_to_transfer_learn(model, base_model):
  #"""Freeze all layers and compile the model"""
  for layer in base_model.layers:
    layer.trainable = False    
  model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
  """Add last layer to the convnet
  Args:
    base_model: keras model excluding top
    nb_classes: # of classes
  Returns:
    new keras model with last layer
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  x = BatchNormalization()(x)
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model


def setup_to_finetune(model):
  """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  """
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
    
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  
  sgd = SGD(lr=1e-3,  momentum=0.9, nesterov=True) # decay=1e-10,
  model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


  # data prep
train_datagen =  ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )
test_datagen = ImageDataGenerator(
      preprocessing_function=preprocess_input,
      rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
  )

train_generator = train_datagen.flow_from_directory(
    'cars/train',
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

validation_generator = test_datagen.flow_from_directory(
    'cars/valid',
    target_size=(IM_WIDTH, IM_HEIGHT),
    batch_size=batch_size,
  )

  # setup model
base_model = InceptionV3(weights='imagenet', include_top=False) #include_top=False excludes final FC layer
model = add_new_last_layer(base_model, 4)

  # transfer learning
setup_to_transfer_learn(model, base_model)

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.InceptionV3.hdf5', 
                               verbose=1, save_best_only=True)

#model.load_weights('saved_models/weights.best.InceptionV3.hdf5')

#early = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')

#history_tl = model.fit_generator(
#    train_generator,
#    nb_epoch=10,
#    samples_per_epoch=6680,
#    validation_data=validation_generator,
#    nb_val_samples=835,
#    verbose=1,
#    callbacks=[checkpointer, early],
#    class_weight='auto')

  # fine-tuning
setup_to_finetune(model)

model.load_weights("saved_models/weights.best.InceptionV3.hdf5")

early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('inceptionv3-ft.model')

history_ft = model.fit_generator(
    train_generator,
    samples_per_epoch=3196,
    nb_epoch=10,
    verbose=1,
    validation_data=validation_generator,
    nb_val_samples=404 ,
    callbacks=[checkpointer, early],
    class_weight='auto')



from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/weights.best.InceptionV3.hdf5")

model= loaded_model

predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0))) for feature in test_tensors]

# report test accuracy
test_accuracy = 100*np.sum(np.array(predictions)==np.argmax(test_targets, axis=1))/len(predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

from keras.models import model_from_json

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("saved_models/weights.best.InceptionV3.hdf5")
model= loaded_model

#from extract_bottleneck_features import *

def predict_orientation(img_path):
    # extract bottleneck features
    bottleneck_feature = path_to_tensor(img_path).astype('float32')/255
    #print(bottleneck_feature)
    # obtain predicted vector
    
    predicted_vector = model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    #print(predicted_vector.shape)
    #print(predicted_vector)
    #print(np.argmax(predicted_vector))
    #print(predicted_vector[0,np.argmax(predicted_vector)])
    return int(cars_names[np.argmax(predicted_vector)]), predicted_vector[0,np.argmax(predicted_vector)] 

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

def make_prediction(path):   
    return predict_orientation(path)   
   

import os
import cv2
import numpy as np
import argparse

def rotateImage3(mat, angle):
    # angle in degrees
    #mat = cv2.imread(path)
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    #cv2.imwrite(dest, rotated_mat)
    
    return rotated_mat

def rotateImage(path, angle, dest):
    # angle in degrees
    mat = cv2.imread(path)
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0, 0])
    abs_sin = abs(rotation_mat[0, 1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w / 2 - image_center[0]
    rotation_mat[1, 2] += bound_h / 2 - image_center[1]

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    cv2.imwrite(dest, rotated_mat)
    
    
def rotateImage2(path, angle, dest, center = None, scale = 1.0):
    image = cv2.imread(path)
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    
    cv2.imwrite(dest, rotated)
    

f = 'C:\DL\image orientation\images'

from random import *

for filename in os.listdir(f):
    r = randint(1, 3)
    if r==1:
         rotateImage(f+'/'+filename, 90, f+'/'+filename)
    elif r==2:
        rotateImage(f+'/'+filename, 180, f+'/'+filename)
    else:
        rotateImage(f+'/'+filename, 270, f+'/'+filename)

def ManageRotation(path, angle):
    angle, confidence = make_prediction(path)
    rotateImage(path, int(angle), path)
    
    tempC = 0
    bestAngle = 0
    if angle==0:
        tempC = confidence
        bestAngle = 0
        
    trial = 1;     
    while trial<4:
        angle, confidence = make_prediction(path)
        if angle>0:
            rotateImage(path, 90, path)
            bestAngle += 90
        elif tempC<confidence:
            tempC = confidence   
            bestAngle = 0        
            
        trial = trial+1
    
    rotateImage(path, bestAngle, path)    


# import os
# import matplotlib.pyplot as plt
# f = 'C:\DL\image orientation\images'
# for filename in os.listdir(f):
#     img = mpimg.imread(f+'/'+filename)    
#     plt.imshow(img)
#     p,c = make_prediction(f+'/'+filename)
#     #print(filename +'-----' + str(p))
#     trial = 0;    
#     if int(p)>0:
#         while trial<10:
#             rotateImage(f+'/'+filename, int(p), f+'/'+filename)
#             trial = trial +1
#             p,c = make_prediction(f+'/'+filename)
#             #print(str(trial) + '-----'+str(p))
#             if int(p)==0:
#                 break           
                

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
def showImg(path):
    img = mpimg.imread(path)   
    plt.axis('off')
    plt.imshow(img)

print('before rotation')
showImg('images/1.jpg')   
angle, confidence = make_prediction('images/1.jpg')
print(str(angle) +' : '+ str(confidence * 100))  

angle, confidence = make_prediction('images/1.jpg')
ManageRotation('images/1.jpg', angle)
#rotateImage('images/1.jpg', int(angle), 'images/1.jpg')
print('after rotation')
showImg('images/1.jpg') 

print('before rotation')
showImg('images/2.jpg')   
angle, confidence = make_prediction('images/2.jpg')
print(str(angle) +' : '+ str(confidence * 100))  

angle, confidence = make_prediction('images/2.jpg')
ManageRotation('images/2.jpg', int(angle))
print('after rotation')
showImg('images/2.jpg') 

print('before rotation')
showImg('images/3.jpg')   
angle, confidence = make_prediction('images/3.jpg')
print(str(angle) +' : '+ str(confidence * 100))  

angle, confidence = make_prediction('images/3.jpg')
ManageRotation('images/3.jpg', int(angle))#, 'images/3.jpg')
print('after rotation')
showImg('images/3.jpg') 

print('before rotation')
showImg('images/4.jpg')   
angle, confidence = make_prediction('images/4.jpg')
print(str(angle) +' : '+ str(confidence * 100))  

angle, confidence = make_prediction('images/4.jpg')
ManageRotation('images/4.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/4.jpg') 

print('before rotation')
showImg('images/5.jpg')   
angle, confidence = make_prediction('images/5.jpg')
print(str(angle) +' : '+ str(confidence * 100))  

angle, confidence = make_prediction('images/5.jpg')
ManageRotation('images/5.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/5.jpg') 

print('before rotation')
showImg('images/6.jpg')   
angle, confidence = make_prediction('images/6.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/6.jpg')
ManageRotation('images/6.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/6.jpg') 

print('before rotation')
showImg('images/7.jpg')   
angle, confidence = make_prediction('images/7.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/7.jpg')
ManageRotation('images/7.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/7.jpg') 

print('before rotation')
showImg('images/8.jpg')   
angle, confidence = make_prediction('images/8.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/8.jpg')
ManageRotation('images/8.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/8.jpg') 

print('before rotation')
showImg('images/9.jpg')   
angle, confidence = make_prediction('images/9.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/9.jpg')
ManageRotation('images/9.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/9.jpg') 

print('before rotation')
showImg('images/11.jpg')   
angle, confidence = make_prediction('images/11.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/11.jpg')
ManageRotation('images/11.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/11.jpg') 

print('before rotation')
showImg('images/12.jpg')   
angle, confidence = make_prediction('images/12.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/12.jpg')
ManageRotation('images/12.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/12.jpg') 

print('before rotation')
showImg('images/14.jpg')   
angle, confidence = make_prediction('images/14.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/14.jpg')
ManageRotation('images/14.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/14.jpg') 

print('before rotation')
showImg('images/15.jpg')   
angle, confidence = make_prediction('images/15.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/15.jpg')
ManageRotation('images/15.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/15.jpg') 

print('before rotation')
showImg('images/16.jpg')   
angle, confidence = make_prediction('images/16.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/16.jpg')
ManageRotation('images/16.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/16.jpg') 

print('before rotation')
showImg('images/17.jpg')   
angle, confidence = make_prediction('images/17.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/17.jpg')
ManageRotation('images/17.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/17.jpg') 

print('before rotation')
showImg('images/18.jpg')   
angle, confidence = make_prediction('images/18.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/18.jpg')
ManageRotation('images/18.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/18.jpg') 

print('before rotation')
showImg('images/19.jpg')   
angle, confidence = make_prediction('images/19.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/19.jpg')
ManageRotation('images/19.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/19.jpg') 

print('before rotation')
showImg('images/20.jpg')   
angle, confidence = make_prediction('images/20.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/20.jpg')
ManageRotation('images/20.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/20.jpg') 

print('before rotation')
showImg('images/21.jpg')   
angle, confidence = make_prediction('images/21.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/21.jpg')
ManageRotation('images/21.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/21.jpg') 

print('before rotation')
showImg('images/greece.jpg')   
angle, confidence = make_prediction('images/greece.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/greece.jpg')
ManageRotation('images/greece.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/greece.jpg') 

print('before rotation')
showImg('images/22.jpg')   
angle, confidence = make_prediction('images/22.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/22.jpg')
ManageRotation('images/22.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/22.jpg') 

print('before rotation')
showImg('images/23.jpg')   
angle, confidence = make_prediction('images/23.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/23.jpg')
ManageRotation('images/23.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/23.jpg') 

print('before rotation')
showImg('images/24.jpg')   
angle, confidence = make_prediction('images/24.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/24.jpg')
ManageRotation('images/24.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/24.jpg') 

print('before rotation')
showImg('images/25.jpg')   
angle, confidence = make_prediction('images/25.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/25.jpg')
ManageRotation('images/25.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/25.jpg') 

print('before rotation')
showImg('images/26.jpg')   
angle, confidence = make_prediction('images/26.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/26.jpg')
ManageRotation('images/26.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/26.jpg') 

print('before rotation')
showImg('images/27.jpg')   
angle, confidence = make_prediction('images/27.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/27.jpg')
ManageRotation('images/27.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/27.jpg') 

print('before rotation')
showImg('images/28.jpg')   
angle, confidence = make_prediction('images/28.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/28.jpg')
ManageRotation('images/28.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/28.jpg') 

print('before rotation')
showImg('images/29.jpg')   
angle, confidence = make_prediction('images/29.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/29.jpg')
ManageRotation('images/29.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/29.jpg') 

print('before rotation')
showImg('images/30.jpg')   
angle, confidence = make_prediction('images/30.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/30.jpg')
ManageRotation('images/30.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/30.jpg') 

print('before rotation')
showImg('images/31.jpg')   
angle, confidence = make_prediction('images/31.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/31.jpg')
ManageRotation('images/31.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/31.jpg') 

print('before rotation')
showImg('images/32.jpg')   
angle, confidence = make_prediction('images/32.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/32.jpg')
ManageRotation('images/32.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/32.jpg') 

print('before rotation')
showImg('images/33.jpg')   
angle, confidence = make_prediction('images/33.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/33.jpg')
ManageRotation('images/33.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/33.jpg') 

print('before rotation')
showImg('images/34.jpg')   
angle, confidence = make_prediction('images/34.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/34.jpg')
ManageRotation('images/34.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/34.jpg') 

print('before rotation')
showImg('images/35.jpg')   
angle, confidence = make_prediction('images/35.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/35.jpg')
ManageRotation('images/35.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/35.jpg') 

print('before rotation')
showImg('images/36.jpg')   
angle, confidence = make_prediction('images/36.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/36.jpg')
ManageRotation('images/36.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/36.jpg') 

print('before rotation')
showImg('images/37.jpg')   
angle, confidence = make_prediction('images/37.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/37.jpg')
ManageRotation('images/37.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/37.jpg') 

print('before rotation')
showImg('images/38.jpg')   
angle, confidence = make_prediction('images/38.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/38.jpg')
ManageRotation('images/38.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/38.jpg') 

print('before rotation')
showImg('images/39.jpg')   
angle, confidence = make_prediction('images/39.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/39.jpg')
ManageRotation('images/39.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/39.jpg')

print('before rotation')
showImg('images/40.jpg')   
angle, confidence = make_prediction('images/40.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/40.jpg')
ManageRotation('images/40.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/40.jpg')

print('before rotation')
showImg('images/41.jpg')   
angle, confidence = make_prediction('images/41.jpg')
print(str(angle) +' : '+ str(confidence * 100)) 

angle, confidence = make_prediction('images/41.jpg')
ManageRotation('images/41.jpg', int(angle))#, 'images/4.jpg')
print('after rotation')
showImg('images/41.jpg')

