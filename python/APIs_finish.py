import glob

glob.glob('my_dir/*')

import tensorflow as tf

glob = tf.matching_files('my_dir/*')

session = tf.Session()
session.run(glob)

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

weight_initial = tf.truncated_normal([5, 5, 1, 32], stddev=0.1)
weight = tf.Variable(weight_initial)
bias_initial = tf.constant(0.1, shape=[32])
bias = tf.Variable(bias_initial)

convolved = tf.nn.conv2d(x, weight, strides=[1, 1, 1, 1], padding='SAME')
convolved_biased = convolved + bias
y = tf.nn.relu(convolved_biased)

x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])

y = tf.layers.conv2d(inputs=x, 
                     filters=32, kernel_size=[5, 5], padding='same', 
                     activation=tf.nn.relu)

x = tf.contrib.keras.layers.Input(shape=[28, 28, 1])

y = tf.contrib.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same',
                                   activation='relu')(x)

model = tf.contrib.keras.models.Sequential()

model.add(tf.contrib.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding='same',
                                         activation='relu', 
                                         input_shape=[28, 28, 1]))

x = tf.contrib.keras.layers.Input(shape=[28, 28, 1])
residual = x
x = tf.contrib.keras.layers.Activation('relu')(x)
x = tf.contrib.keras.layers.SeparableConv2D(filters=728, kernel_size=[3, 3], padding='same')(x)
x = tf.contrib.keras.layers.BatchNormalization()(x)
y = tf.contrib.keras.layers.add([x, residual])

# estimator = keras_model.get_estimator()

get_ipython().system('head -3 president_gdp.csv')

import numpy as np
import pandas as pd

data = pd.read_csv('president_gdp.csv')
party = data.party == 'D'
party = np.expand_dims(party, axis=1)
growth = data.growth

import sklearn.linear_model

model = sklearn.linear_model.LinearRegression()
model.fit(X=party, y=growth)
model.predict([[0], [1]])

# clear old run...
import shutil
shutil.rmtree('tflinreg', ignore_errors=True)

party_col = tf.contrib.layers.real_valued_column(column_name='')

model = tf.contrib.learn.LinearRegressor(feature_columns=[party_col],
                                         model_dir='tflinreg')

model.fit(x=party, y=growth, steps=1000)
list(model.predict(np.array([[0], [1]])))

get_ipython().system('tensorboard --logdir tflinreg')

dataset = tf.contrib.data.TextLineDataset('president_gdp.csv')
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

session = tf.Session()
session.run(next_element)

session.run(next_element)

session.run(next_element)

model = tf.contrib.keras.applications.InceptionV3()

filename = 'n01882714_4157_koala_bear.jpg'
image = tf.contrib.keras.preprocessing.image.load_img(
    filename, target_size=(299, 299))

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.imshow(image)

array = tf.contrib.keras.preprocessing.image.img_to_array(image)
array = np.expand_dims(array, axis=0)
array = tf.contrib.keras.applications.inception_v3.preprocess_input(array)

probabilities = model.predict(array)

tf.contrib.keras.applications.inception_v3.decode_predictions(probabilities)



