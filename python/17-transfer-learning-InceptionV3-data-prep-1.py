reset -fs

import os
import re
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import pandas as pd
import sklearn
from sklearn import cross_validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import pickle
get_ipython().magic('matplotlib inline')

model_dir = '/Users/carles/Desktop/data/tutorial/imagenet'

def create_graph():
    with gfile.FastGFile(os.path.join(
        model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def extract_features(list_images):
    # The number of features correspond to the output of the specified layer below.
    nb_features = 2048
    features = np.empty((len(list_images), nb_features))
    # We don't need to use the labels provided by InceptionV3.
    # labels = []

    create_graph()

    with tf.Session() as sess:

        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')

        for ind, image in enumerate(list_images):
            if (ind%100 == 0):
                print('Processing %s...' % (image))
            if not gfile.Exists(image):
                tf.logging.fatal('File does not exist %s', image)

            image_data = gfile.FastGFile(image, 'rb').read()
            predictions = sess.run(next_to_last_tensor,
            {'DecodeJpeg/contents:0': image_data})
            
        features[ind,:] = np.squeeze(predictions)
        # We don't need to use the labels provided by InceptionV3.
        # labels.append(re.split('_\d+',image.split('/')[1])[0])

    # return features, labels
    return features

images_ok_dir = '/Users/carles/Desktop/data/tutorial/images/ok/'

list_images_ok = [images_ok_dir+f for f in os.listdir(images_ok_dir) if re.search('jpg|JPG', f)]

images_nok_dir = '/Users/carles/Desktop/data/tutorial/images/nok/'

list_images_nok = [images_nok_dir+f for f in os.listdir(images_nok_dir) if re.search('jpg|JPG', f)]

list_images = list_images_ok + list_images_nok

features = extract_features(list_images)

pickle.dump(features, open('/Users/carles/Desktop/data/tutorial/X', 'wb'))

y = np.vstack((np.array([1]*(len(features)/2)).reshape((len(features)/2), 1), np.array([0]*(len(features)/2)).reshape((len(features)/2), 1)))

pickle.dump(y, open('/Users/carles/Desktop/data/tutorial/y', 'wb'))



