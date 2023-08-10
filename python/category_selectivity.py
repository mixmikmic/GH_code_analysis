import numpy as np
import pandas as pd

import glob
import os
from collections import defaultdict

import keras.backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg_preprocess_input

from helpers import *

# Load a model
model = VGG16(weights='imagenet', include_top=True)
model.summary()

LAYERS = [l.name for l in model.layers[7:-1] if 'pool' not in l.name]  # layer of interest
print(LAYERS)
CAT_FLD = 'images/'        # folder with images separated by category into subfolders
ACT_FLD = 'activations/'
SEL_FLD = 'selectivity'

def vgg_image(path, target_size=(224, 224)):
    """Tranform image for vgg with keras module"""
    img = image.load_img(path, target_size=target_size)
    x = image.img_to_array(img)
    x = np.expand_dims(img, axis=0).astype(np.float)
    x = vgg_preprocess_input(x)
    return x


def vgg_images_from_filenames(filenames):
    """Load and preprocess images from filenames"""
    images = [vgg_image(f) for f in filenames]
    images = [x for x in images if x is not None]
    images = np.squeeze(np.vstack(images))
    return images


def get_activations(model, images, layers_names):
    """Get activations for the input from specified layer"""
    
    inp = model.input                                          
    names_outputs = [(l.name, l.output) for l in model.layers if l.name in layers_names] 
    names = [x[0] for x in names_outputs]
    outputs = [x[1] for x in names_outputs]
    functor = K.function([inp] + [K.learning_phase()], outputs) 
    
    return dict(zip(names, functor([images])))

def get_activations_batches(model, filenames, layers, output_fld, img_loader_func, 
                            batch_size=128, crop_model_=True, **kwargs): 
    """Split input into batches and get activations"""
    
    file_gen = batch_generator(filenames, batch_size, equal_size=False)
    
    for l in layers:
        l_fld = os.path.join(output_fld, l)
        if not os.path.exists(l_fld):
            os.makedirs(l_fld)
    
    # get activations
    for batch_n, batch_files in enumerate(file_gen):
        print("batch", batch_n)
        images = img_loader_func(batch_files,  **kwargs)
        activations = get_activations(model, images, layers)
        
        for l_name, v in activations.items():
            for path, l_acts in zip(batch_files, v):
                name = os.path.basename(path).split('.')[0]            
                new_path = os.path.join(output_fld, l_name, name + '.npy')
                np.save(new_path, l_acts)

    print("All activations saved to folder", output_fld)

for cat in os.listdir(CAT_FLD):
    if cat == 'furniture':
        cat_files = glob.glob(os.path.join(CAT_FLD, cat, '*'))
        print('In category {} found {} files'.format(cat.upper(), len(cat_files)))
        print('Getting activations for files...')

        output_cat_fld = os.path.join(ACT_FLD, cat)
        get_activations_batches(model, cat_files, LAYERS, output_fld=output_cat_fld, 
                                img_loader_func=vgg_images_from_filenames, batch_size=128)
        print()

def get_cat_mask(categories, cat):
    return np.array([x == cat for x in categories])
    
def category_selectivity_index(activations, cat_mask, axis=0):
    
    act_norm = norm_values(activations)
    cat = act_norm[cat_mask]    
    noncat = act_norm[~cat_mask]
    selectivity = np.sum(cat, axis)/cat.shape[axis] - np.sum(noncat, axis)/noncat.shape[axis]
    return selectivity

def cat_selective_func(func, activations, labels, **kwargs):
    
    cat_selectivity = []
    
    for l in np.unique(labels):
        category_mask = get_cat_mask(labels, l)
        selectivity = func(activations, category_mask, **kwargs)
        cat_selectivity.append([selectivity])
    cat_selectivity = np.vstack(cat_selectivity)    
    return cat_selectivity

# Load all activations

def load_act_from_fld(fld, layer=''):
    
    label_dict = {}
    activations = []
    labels = []
    
    for (n, cat) in enumerate(os.listdir(fld)):
        label_dict[n] = cat
        cat_fld = os.path.join(fld, cat, layer)
        np_files = glob.glob(cat_fld + '/*.np[yz]')
        if len(np_files) > 0:
            labels.extend([n] * len(np_files))
            cat_activations = []
            print('Loading {} files from {}'.format(len(np_files), cat_fld))
            for (i, f) in enumerate(np_files):
                activations.append(np.load(f))

    return np.asarray(activations), labels, label_dict

ACTIVATIONS = defaultdict(lambda : defaultdict())

for l in LAYERS:
    acts, labels, label_dict = load_act_from_fld(ACT_FLD, layer=l)
    ACTIVATIONS[l]['activations'] = acts
    ACTIVATIONS[l]['label_dict'] = label_dict
    ACTIVATIONS[l]['labels'] = labels

# Find selectivity for filters in different layers

for l, v in ACTIVATIONS.items():  
    print(l)
    cat_sel = cat_selective_func(category_selectivity_index, v['activations'], v['labels'])
    # select by mean filter activation
    cat_sel_filter = np.mean(cat_sel, axis=(1, 2))

    # # select by activation of filter's cantral unit
    # central_unit = np.ceil(cat_sel.shape[1]/2).astype(int)
    # cat_sel_filter = cat_sel[:, central_unit, central_unit, :]
    ACTIVATIONS[l]['filter_selectivity'] = cat_sel_filter

# Convert selectivity index into dataframe

if not os.path.exists(SEL_FLD):
    os.makedirs(SEL_FLD)
        
for l, v in ACTIVATIONS.items():  
    print(l)
    
    df = pd.DataFrame(v['filter_selectivity'].T)
    df['filter'] = df.index
    df = df.melt(id_vars=df.columns[-1], value_vars=df.columns[:-1], 
                 var_name='category', value_name='selectivity')

    df['label'] = df['category'].map(v['label_dict'])
    df['layer'] = l
    df.to_csv(os.path.join(SEL_FLD, '{}.csv'.format(l)), index=False)
    df.head()

# Optional
# Join all files into one dataframe
df = pd.concat((pd.read_csv(f) for f in os.path.join(SEL_FLD, "*.csv")))
df.to_csv('selectivity_vgg16.csv', index=False)

