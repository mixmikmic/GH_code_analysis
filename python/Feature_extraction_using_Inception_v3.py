import tensorflow as tf
import time
import numpy as np
import myutils
import sys
import os

data_training, data_testing = myutils.load_CIFAR_dataset(shuffle=False)

# # One can download classify_image.py from 
# # https://github.com/tensorflow/models/blob/master/tutorials/image/imagenet/classify_image.py
# # Next, it is enough to run this code
#
# from classify_image import *
# FLAGS.model_dir = 'model/'       # path to save the file inception-2015-12-05.tgz
# maybe_download_and_extract()
# create_graph()

#
# Please download the file with the Inception model
# http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# Extract the model to the file 'model/classify_image_graph_def.pb'
#
graph_def = tf.GraphDef()
with open('model/classify_image_graph_def.pb', "rb") as f:
    graph_def.ParseFromString(f.read())
tf.import_graph_def(graph_def, name='Incv3')

# number of samples to extract from 

nsamples_training = 50000    # at most 50000
nsamples_testing  = 10000    # at most 10000

# nsamples_training = len(data_training)
# nsamples_testing  = len(data_testing)

nsamples = nsamples_training + nsamples_testing

X_data = [ data_training[i][0] for i in range(nsamples_training) ] +          [ data_testing[i][0]  for i in range(nsamples_testing)  ]

y_training = np.array( [ data_training[i][1] for i in range(nsamples_training) ] )
y_testing  = np.array( [ data_testing[i][1]  for i in range(nsamples_testing)  ] )

# Running tensorflow session in order to extract features
def _progress(count, start, time):
    percent = 100.0*(count+1)/nsamples;
    sys.stdout.write('\r>> Extracting features %4d/%d  %6.2f%%                         ETA %8.1f seconds' % (count+1, nsamples, percent, (time-start)*(100.0-percent)/percent) )
    sys.stdout.flush()

with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    representation_tensor = sess.graph.get_tensor_by_name('pool_3:0')
    predictions = np.zeros((nsamples, 1008), dtype='float32')
    representations = np.zeros((nsamples, 2048), dtype='float32')

    print('nsamples = ', nsamples)
    start = time.time()
    for i in range(nsamples):
        [reps, preds] = sess.run([representation_tensor, softmax_tensor], {'DecodeJpeg:0': X_data[i]})      
        predictions[i] = np.squeeze(preds)
        representations[i] = np.squeeze(reps)
        if i+1==nsamples or not (i%10): _progress(i, start, time.time())
    print('\nElapsed time %.1f seconds' % (time.time()-start))

predictions.shape

representations.shape

print( predictions[0,:3] )
print( representations[0,:3] )

# Finally we can save our work to the file

np.savez_compressed("features/CIFAR10_Inception_v3_features.npz",                          features_training=representations[:nsamples_training],                      features_testing=representations[-nsamples_testing:],                       labels_training=y_training,                                                 labels_testing=y_testing )

import numpy as np
import myutils

data = np.load('features/CIFAR10_Inception_v3_features.npz')

X_training = data['features_training']
y_training = data['labels_training']

X_testing = data['features_testing']
y_testing = data['labels_testing']

# data_training, data_testing = myutils.load_CIFAR_dataset(shuffle=False)
# assert( (np.array( [data_training[i][1] for i in range(len(data_training))] ) == y_training).all() )
# assert( (np.array( [data_testing[i][1] for i in range(len(data_testing))] ) == y_testing).all() )
print( 'X_training size = {}'.format(X_training.shape))

from sklearn import decomposition

pca = decomposition.PCA(n_components=2)
pca.fit(X_training)

print(pca.explained_variance_ratio_)

X = pca.transform(X_training)

X.shape

from matplotlib import pyplot as plt
plt.figure( figsize=(15,15) )
plt.scatter( X[:, 0], X[:, 1], c=y_training, cmap='tab10' )
# plt.colorbar()
plt.show()

from sklearn.manifold import TSNE
pca = decomposition.PCA(n_components=60)
X_training_reduced = pca.fit_transform(X_training)

np.sum( pca.explained_variance_ratio_ )

tsne = TSNE(n_components=2)

X_training_reduced_tsne = tsne.fit_transform(X_training_reduced)

X_training_reduced_tsne.shape

plt.figure( figsize=(15,15) )
plt.scatter( X_training_reduced_tsne[:, 0], X_training_reduced_tsne[:, 1], c=y_training, cmap='tab10' )
plt.show()

