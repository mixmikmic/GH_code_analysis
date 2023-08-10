import numpy as np

X = np.array([np.random.normal(0.3, 5, 1000), np.random.normal(-3.5, 1.1, 1000)]).T

y = np.zeros(1000)
y[(X[:,0] < -3) | (X[:,0] > 10)] = 1

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

plt.scatter(X[:,0], X[:,1], c=y)
plt.show()

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint

model = Sequential()
model.add(Dense(2, input_shape= (2,)))
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))
print 'Compiling model...'
model.compile('adam', 'binary_crossentropy')
model.summary()

MODEL_FILE = 'dummynet'

try:
    model.fit(X_train, y_train, batch_size=16,
        callbacks = [
            EarlyStopping(verbose=True, patience=20, monitor='val_loss'),
            ModelCheckpoint(MODEL_FILE + '-progress', monitor='val_loss', verbose=True, save_best_only=True)
        ],
    nb_epoch=100,
    validation_split = 0.2,
    show_accuracy=True)

except KeyboardInterrupt:
    print 'Training ended early.'

# -- load in best network                                                                                                                                                                                      
model.load_weights(MODEL_FILE + '-progress')

# -- save model and weights to protobufs
import tensorflow as tf                                                                                                                                                                                        
import keras.backend.tensorflow_backend as tfbe                                                                                                                                                                

sess = tfbe._SESSION                                                                                                                                                                                           
saver = tf.train.Saver()                                                                                                                                                                                       
tf.train.write_graph(sess.graph_def, 'models/', 'graph.pb', as_text=False)                                                                                                                                     
save_path = saver.save(sess, "./model-weights.ckpt")                                                                                                                                                           
print "Model saved in file: %s" % save_path                                                                                                                                                                    

# -- print out this information: it will be useful when freezing the graph
print 'filename_tensor_name = ' + saver.as_saver_def().filename_tensor_name                                                                                                                                                                
print 'restore_op_name = ' + saver.as_saver_def().restore_op_name                                                                                                                                                                     

# -- find out the name of your output node: it will be useful when freezing the graph and evaluating
model.get_output()

# -- find out the name of your output node: it will be useful when evaluating
model.get_input()

# -- testing
yhat = model.predict(X_test, verbose = True, batch_size = 516)

# -- plot classifier output
_ = plt.hist(yhat[y_test == 1], normed = True, histtype = 'stepfilled', color = 'green', alpha = 0.5)
_ = plt.hist(yhat[y_test == 0], normed = True, histtype = 'stepfilled', color = 'red', alpha = 0.5)

# -- real flavor of test set examples
plt.scatter(X_test[:,0], X_test[:,1], c=y_test)
plt.show()

# -- net predictions on test set examples
plt.scatter(X_test[:,0], X_test[:,1], c=yhat)
plt.show()

# -- inputs
X_test[0]

# -- predicted output (using Keras)
yhat[0]

from tensorflow.core.framework import graph_pb2

# -- read in the graph
f = open("models/graph.pb", "rb")
graph_def = graph_pb2.GraphDef()
graph_def.ParseFromString(f.read())

import tensorflow as tf
# -- actually import the graph described by graph_def
tf.import_graph_def(graph_def, name = '')

for node in graph_def.node:
    print node.name



