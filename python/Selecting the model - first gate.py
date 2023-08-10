from IPython.display import display,Image,clear_output
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import seaborn as sns
import random
import urllib.request
import h5py

from collections import Counter,defaultdict

import keras 
from keras.utils.data_utils import get_file  #?
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3  #?
from keras.applications.imagenet_utils import preprocess_input, decode_predictions  #?
from keras.preprocessing.image import ImageDataGenerator,img_to_array,array_to_img,load_img
from keras.models import save_model

from keras import backend as K
K.backend()

CLASS_INDEX = None
CLASS_INDEX_PATH = 'https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json'

# from Keras GitHub  
def get_predictions(preds, top=5):
    global CLASS_INDEX
    if len(preds.shape) != 2 or preds.shape[1] != 1000:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 1000)). '
                         'Found array with shape: ' + str(preds.shape))
    if CLASS_INDEX is None:
        fpath = get_file('imagenet_class_index.json',
                         CLASS_INDEX_PATH,
                         cache_subdir='models')
        CLASS_INDEX = json.load(open(fpath))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

vgg16 = VGG16(weights='imagenet')     # weights -> None,'imagenet' **None** implies random initialization while 'imagenet' species loading
                                      # weights pretrained on imagenet
resnet50 = ResNet50(weights='imagenet')
vgg19 = VGG19(weights='imagenet')
inception = InceptionV3(weights='imagenet')

dataset = './car-damage-dataset/data1a/training/00-damage'
images = os.listdir(dataset)
img = random.choice(images)
dest_path = os.path.join(dataset,img)
Image(dest_path,width=200)

def prepare_image(img_path):
    img = load_img(path=img_path,target_size=(224,224))
    x = img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x)         #?
    return x

y = prepare_image(dest_path)
preds = vgg16.predict(y)
print(get_predictions(preds,top=5))

preds = resnet50.predict(y)
print(get_predictions(preds,top=5))

preds = vgg19.predict(y)
print(get_predictions(preds,top=5))

vgg16.save('vgg16.h5')

img_list = os.listdir('./car-damage-dataset/data1a/training/')
# sab uttana h ki training
img_list

img_pathshs = []
for x in img_list:
    img_paths += os.listdir('./car-damage-dataset/data1a/training/' + x)
img_paths[:len(os.listdir('./car-damage-dataset/data1a/training/01-whole/'))] = ['./car-damage-dataset/data1a/training/01-whole/' + x for x in img_paths[:len(os.listdir('./car-damage-dataset/data1a/training/01-whole/'))]]
img_paths[len(os.listdir('./car-damage-dataset/data1a/training/00-damage//')):] = ['./car-damage-dataset/data1a/training/00-damage/' + x for x in img_paths[len(os.listdir('./car-damage-dataset/data1a/training/00-damage//')):]]
img_paths

def get_car_categories():
    d = defaultdict(float)
    # img_list = os.listdir('./car-damage-dataset/data1a/training/')    # cross-check the path
    for i,img_path in enumerate(img_paths):
        img = prepare_image(img_path)
        out = vgg16.predict(img)
        top = get_predictions(out,top=5)
        for j in top[0]:
            d[j[0:2]] += j[2]
        if i % 50 == 0:
            print(i, '/', len(img_paths), 'complete')
    return Counter(d)

cat_counter = get_car_categories()

cat_counter

cat_list = [k for k, v in cat_counter.most_common()[:48]]

cat_counter.most_common()[:48]

cat_list

with open('cat_counter.pk','wb') as f:
    pickle.dump(cat_counter,f,-1)

with open('cat_counter.pk','rb') as f:
    cat_counter = pickle.load(f)

cat_list = [k for k,v in cat_counter.most_common()[:48]]

cat_list

with open('vgg16_cat_list.pk','wb') as f:
    pickle.dump(cat_list,f,-1)

def get_car_categories_with_cat_list(cat_list):
    num = 0
    bad_list = []
    for i,img_path in enumerate(img_paths):
        img = prepare_image(img_path)
        out = vgg16.predict(img)
        top = get_predictions(out,top=5)
        for j in top[0]:
            if j[0:2] in cat_list:
                num += 1
                break # breaks out of for loop if one of top 50 categories is found
            bad_list.append(img_path)  # appends to "bad list" if none of the 50 are found
        if i % 100 == 0:
            print(i, '/', len(img_paths), 'complete')
    bad_list = [k for k, v in Counter(bad_list).iteritems() if v == 5]   #?
    return num, bad_list

number, bad_list = get_car_categories_with_cat_list(cat_list)

def view_images(img_paths):
    for img in img_paths:
        clear_output()
        display(Image(img,width=200))
        num = input("c to continue, q to quit")
        if num == 'c':
            pass
        else:
            return 'Finished for now.'

view_images(img_paths)

def car_categories_gate(img_url,cat_list):
    urllib.request.urlretrieve(img_url,'save.jpg')
    x = prepare_image('save.jpg')
    out = vgg16.predict(x)
    top = get_predictions(out,top=5)
    print("Validating that this is a picture of your car...")
    for j in top[0]:
        if j[0:2] in cat_list:
            print(j[0:2])
            return "Validation complete - proceed to damage evaluation"
    return "Are you sure this is a picture of your car? Please take another picture (try a different angle or lighting) and try again."

car_categories_gate('https://encrypted-tbn1.gstatic.com/images?q=tbn:ANd9GcSxhKhaSwPgdQkrDegC6sbUALBF9SiW6tDKg6dLDYj83e19krxy', cat_list)

car_categories_gate('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS7xHO3j12Xk4q4eaQUL1A02k1HrJ9G_RY6tj-4h-07EfdML6YL', cat_list)



