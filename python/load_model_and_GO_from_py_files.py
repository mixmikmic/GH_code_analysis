get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#from PIL import Image
import numpy as np

#from keras.models import load_model

import src.picture_stuff as pix

import load_and_go as ld

# This cell copied from main in load_and_go.py

use_gray = True
X,y_list = ld.load_examples("data/",use_gray)
filepath = 'saved_models/'
model = ld.load_keras_model(filepath, use_gray)
label_dic = ld.load_label_dictionary(filepath, use_gray)

image_dims = 299
border_fraction = .3

picture_index_lookup = pix.picture_index_function(y_list)

camera = pix.initialize_camera()

# Input a file name = brick shape: e.g. 3021
pic_label = raw_input('Type label (integer as file name):')

extension, filename = pix.increment_filename(pic_label,extension=1)

one_pic_X = pix.keep_shooting_until_acceptable(camera,filename)
del(camera)

predict_gen = model.predict_on_batch(np.expand_dims(one_pic_X,axis=0))

preds, weights = pix.make_one_prediction_list(
    predict_gen,label_dic,n_match=10)

idx_preds = [picture_index_lookup[pred] for pred in preds]

ld.plot_top_8(one_pic_X,pic_label,X,idx_preds,preds,weights);

