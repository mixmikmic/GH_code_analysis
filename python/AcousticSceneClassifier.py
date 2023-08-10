from __future__ import absolute_import, division, print_function

import os
import sys
sys.path.append(os.path.join('.', '..'))
import utils
import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import fully_connected, batch_norm, dropout
from tensorflow.python.ops.nn import relu, softmax
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

(e1_train,l1_train) =  utils.tfRead('train1')
print("tfRecord train1 uploaded!")
(e2_train,l2_train) =  utils.tfRead('train2')
print("tfRecord train2 uploaded!")
(e3_train,l3_train) =  utils.tfRead('train3')
print("tfRecord train3 uploaded!")
(e4_train,l4_train) =  utils.tfRead('train4')
print("tfRecord train4 uploaded!")
(e5_train,l5_train) =  utils.tfRead('train5')
print("tfRecord train5 uploaded!")
(e6_train,l6_train) =  utils.tfRead('train6')
print("tfRecord train6 uploaded!")
(e7_train,l7_train) =  utils.tfRead('test1')
print("tfRecord train7 uploaded!")
(e8_train,l8_train) =  utils.tfRead('test2')
print("tfRecord train8 uploaded!")
(e9_train,l9_train) =  utils.tfRead('val1')
print("tfRecord train9 uploaded!")
(e10_train,l10_train) =  utils.tfRead('val2')
print("tfRecord train10 uploaded!")
embedding_train= np.concatenate((e1_train, e2_train, e3_train, e4_train, e5_train, e6_train, e7_train, e8_train, e9_train, e10_train), axis=0)
print("Train embedding shape: ",embedding_train.shape)

embedding_labels_train = np.concatenate((l1_train, l2_train, l3_train, l4_train, l5_train, l6_train, l7_train, l8_train, l9_train, l10_train), axis=0)
print(embedding_labels_train.shape)

(e1_val,l1_val) =  utils.tfRead('Evalval1')
print("tfRecord val1 uploaded!")
(e2_val,l2_val) =  utils.tfRead('Evalval2')
print("tfRecord val2 uploaded!")

embedding_val= np.concatenate((e1_val, e2_val), axis=0)
print("Val embedding shape: ",embedding_val.shape)

embedding_labels_val = np.concatenate((l1_val,l2_val), axis=0)
print(embedding_labels_val.shape)

#Apply if you want to run the model for the 3 class problem: 
embedding_labels_train = utils.labelMinimizer(embedding_labels_train)
embedding_labels_val = utils.labelMinimizer(embedding_labels_val)


#One hot encoding
embedding_labels_train = utils.OnehotEnc(embedding_labels_train)
embedding_labels_val = utils.OnehotEnc(embedding_labels_val)

weight_initializer = tf.truncated_normal_initializer(stddev=0.1)


num_features = embedding_train[0].shape[0]
num_classes = embedding_labels_train[0].shape[0]
print('number of features: ', num_features)
print('number of classes: ', num_classes)

reuse_flag = True

## Define placeholders
x_pl = tf.placeholder(tf.float32, shape=[None, num_features], name='xPlaceholder')
y_pl = tf.placeholder(tf.float32, shape=[None, num_classes], name='yPlaceholder')


## Define initializer for the weigths
num_hidden1 = 128
num_hidden2 = 128
num_hidden2 = 128

l1 =  fully_connected(x_pl, num_hidden1, activation_fn=relu)

#l2 = fully_connected(l1, num_hidden2, activation_fn=relu,weights_regularizer=regularizer)

#l3 = fully_connected(l2, num_hidden3, activation_fn=relu,weights_regularizer=regularizer)

lout = fully_connected(l1, num_classes, activation_fn=None)

prediction = tf.nn.softmax(lout,name="op_to_restore")


### Implement training ops

LEARNING_RATE = 0.1;

# 1) Define cross entropy loss
cross_entropy = -tf.reduce_sum(y_pl * tf.log(prediction), reduction_indices=[1])
cross_entropy = tf.reduce_mean(cross_entropy)

# 2) Define the training op
#train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

# 3) Define accuracy op
correct_prediction = tf.equal(tf.argmax(prediction, axis=1), tf.argmax(y_pl, axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

max_epochs = 100
batch_size = 128

epochs_completed_train = 0
epochs_completed_val = 0

idx_epochs_train = 0
idx_epochs_val = 0

old_epochs_completed = 0
# restricting memory usage, TensorFlow is greedy and will use all memory otherwise
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)

train_cost, val_cost, train_acc, val_acc = [],[],[],[]
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
    sess.run(tf.global_variables_initializer())
    try:

        while epochs_completed_train < max_epochs:
            train_cost, train_acc = [],[]
            (x_tr,y_tr,epochs_completed_train, idx_epochs_train, embedding_train,embedding_labels_train) = utils.next_batch(batch_size,idx_epochs_train,epochs_completed_train,embedding_train,embedding_labels_train)
            (x_val,y_val,epochs_completed_val, idx_epochs_val, embedding_val,embedding_labels_val) = utils.next_batch(batch_size,idx_epochs_val,epochs_completed_val,embedding_val,embedding_labels_val)
            
                # Traning optimizer
            feed_dict_train = {x_pl: x_tr, y_pl: y_tr}

                # running the train_op
            res = sess.run( [train_op, cross_entropy, accuracy], feed_dict=feed_dict_train)

            train_cost += [res[1]]
            train_acc += [res[2]]

                # Validation:
            feed_dict_valid = {x_pl: x_val, y_pl: y_val}

            res = sess.run([cross_entropy, accuracy], feed_dict=feed_dict_valid)
            val_cost += [res[0]]
            val_acc += [res[1]]
            
            if old_epochs_completed != epochs_completed_train:
                print("Epoch %i, Train Cost: %0.3f\tVal Cost: %0.3f\t Val acc: %0.3f"                       %(epochs_completed_train, train_cost[-1],val_cost[-1],val_acc[-1]))
                
            old_epochs_completed = epochs_completed_train
            
            
            # Save the output of the network to a local place
        saver = tf.train.Saver()
        saver.save(sess, "C:/tmp/audio_classifier")

    except KeyboardInterrupt:
        print('KeyboardInterrupt')

print('Done')

from sklearn.metrics import confusion_matrix

(embedding_test,embedding_labels_test) =  utils.tfRead('test')
print("tfRecord test uploaded!")

embedding_labels_test = utils.labelMinimizer(embedding_labels_test)
embedding_list = embedding_test

gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
 # load the trained network from a local drive
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
#First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph("C:/tmp/audio_classifier.meta")
    saver.restore(sess,tf.train.latest_checkpoint('C:/tmp/'))
# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

    graph = tf.get_default_graph()
    x_pl = graph.get_tensor_by_name("xPlaceholder:0")
    feed_dict = {x_pl: embedding_list}

        #Now, access the op that you want to run.
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

    y_pred = sess.run(op_to_restore, feed_dict)

    pred = sess.run(tf.argmax(y_pred, axis=1))
    #print("class predicion embedding 1:", pred)
    #print("real label: ",embedding_labels_test[0:100])

correct_pred = 0;
for i in range(0,len(pred)):
    if pred[i] == embedding_labels_test[i]:
        correct_pred+=1

acc = correct_pred/len(pred)
print("Test accuracy: ", acc)



conf_mat = confusion_matrix(embedding_labels_test,pred)
np.set_printoptions(precision=2)
conf_norm = conf_mat.astype('float')/conf_mat.sum(axis=1)[:,np.newaxis]
print(conf_norm*100)
print(sum(conf_mat))
print(sum(conf_mat)[1])

# 0 outdoor, 1 indoor, 2 vehicle
className = ["Outdoor","Indoor","Vehicle"]
# Plot normalized confusion matrix
plt.figure()
utils.plot_confusion_matrix(conf_mat, classes=className, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig('myfig')
plt.show()



from collections import Counter

correct_pred = 0;
idx = 0;
for n in range(1,round(len(pred)/10)):
    majorLabel= embedding_labels_test[idx:idx+9]
    majorPred = pred[idx:idx+9]
    cntLabel = Counter(majorLabel)
    cntPred = Counter(majorPred)

    idx = idx+10
    cLabel = cntLabel.most_common(1)[0]
    cPred = cntPred.most_common(1)[0]
    #print(cnt.most_common(1)[0])
    #print(cLabel[0])
    #print(cPred[0])
    if cLabel[0] == cPred[0]:
        correct_pred+=1

acc = correct_pred/round(len(pred)/10)
print("Test accuracy (major): ", acc)


