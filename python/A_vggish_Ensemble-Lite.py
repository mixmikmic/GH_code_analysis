from __future__ import print_function

import random
from sklearn.utils import shuffle
from sklearn.utils import resample
import numpy as np
import tensorflow as tf

import vggish_input 
import vggish_params as params
import vggish_slim
import pickle
import math
from sklearn.model_selection import StratifiedShuffleSplit
import datetime
import pandas as pd
import time
from sklearn.utils import shuffle

slim = tf.contrib.slim
training = True
_NUM_CLASSES = 15

# Finding unique recording names to ensure that the same recording is only in train or validation set:
with open('soundname1.pickle','rb') as f:  # Python 3: open(..., 'rb')
    name_1,label_1 = pickle.load(f) # name_1 contains the names of the recordings in subset 1, label_1 contains the corresponding onehot encoded labels
# The following outcommented lines should be used if the entire dataset is used. However, due to limits in the size of files that can be uploaded to github, the entire dataset is not uploaded.
# with open('soundname2.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     name_2,label_2 = pickle.load(f)
# with open('soundname3.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     name_3,label_3 = pickle.load(f)
# with open('soundname4.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     name_4,label_4 = pickle.load(f) 
# with open('soundname5.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     name_5,label_5 = pickle.load(f)
# with open('soundname6.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     name_6,label_6 = pickle.load(f)
# with open('soundname7.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     name_7,label_7 = pickle.load(f)
# with open('soundname8.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     name_8,label_8 = pickle.load(f)
# with open('soundname9.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     name_9,label_9 = pickle.load(f)
# with open('soundname10.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     name_10,label_10 = pickle.load(f)

# my_names = np.concatenate((name_1,name_2,name_3,name_4,name_5,name_6,name_7,name_8,name_9,name_10))
# my_labels = np.concatenate((label_1,label_2,label_3,label_4,label_5,label_6,label_7,label_8,label_9,label_10))
my_names = name_1
my_labels = label_1
    
uni_names, uni_index = np.unique(my_names, return_index = True)

# Finding unique labels
uni_labels = []
for w in range(0,int(len(uni_index))):
    c =uni_index[w]
    uni_labels.append(my_labels[c])

# Splitting unique recordings in two sets, training and validation
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
sss.get_n_splits(uni_names, uni_labels)

train = []
test = []
for train_index, test_index in sss.split(uni_names, uni_labels):
    train.append(train_index) # Recordings for training set
    test.append(test_index) # Recordings for validation set


# Finding index of the files from the recordings for training set
train_idx = train[0]
train_index = [i for i,x in enumerate(my_names) if x == uni_names[train_idx[0]]]
for f in range(1,len(train_idx)):
    indices = [i for i,x in enumerate(my_names) if x == uni_names[train_idx[f]]]
    train_index = np.concatenate((train_index,indices))
    


# Finding index of the files from the recordings for validation set
test_idx = test[0]
test_index = [i for i,x in enumerate(my_names) if x == uni_names[test_idx[0]]]
for f in range(1,len(test_idx)):
    indices = [i for i,x in enumerate(my_names) if x == uni_names[test_idx[f]]]
    test_index = np.concatenate((test_index,indices))
    

# Load the mono log mel spectrograms and labels for the development set 

# Download here : wget https://www.dropbox.com/s/x7t82y3kzmzhdc2/logmel_subset1.pickle?dl=0
with open('logmel_subset1.pickle','rb') as f:  # Python 3: open(..., 'rb')
    my_features_sub1,my_labels_sub1 = pickle.load(f) 
    
# The following outcommented lines should be used if the entire dataset is used. However, due to limits in the size of files that can be uploaded to github, the entire dataset is not uploaded.
# with open('logmel_subset2.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub2,my_labels_sub2 = pickle.load(f)
# with open('logmel_subset3.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub3,my_labels_sub3 = pickle.load(f)
# with open('logmel_subset4.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub4,my_labels_sub4 = pickle.load(f)
# with open('logmel_subset5.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub5,my_labels_sub5 = pickle.load(f)
# with open('logmel_subset6.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub6,my_labels_sub6 = pickle.load(f)
# with open('logmel_subset7.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub7,my_labels_sub7 = pickle.load(f)
# with open('logmel_subset8.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub8,my_labels_sub8 = pickle.load(f)
# with open('logmel_subset9.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub9,my_labels_sub9 = pickle.load(f)
# with open('logmel_subset10.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub10,my_labels_sub10 = pickle.load(f)

# my_features_sub = np.concatenate((my_features_sub1,my_features_sub2,my_features_sub3,my_features_sub4,my_features_sub5,my_features_sub6,my_features_sub7,my_features_sub8,my_features_sub9,my_features_sub10))
# my_labels_sub = np.concatenate((my_labels_sub1,my_labels_sub2,my_labels_sub3,my_labels_sub4,my_labels_sub5,my_labels_sub6,my_labels_sub7,my_labels_sub8,my_labels_sub9,my_labels_sub10))
my_features_sub = my_features_sub1
my_labels_sub = my_labels_sub1

my_labels_s = []
for w in range(0,int(len(my_labels_sub)/10)):
    my_labels_s.append(my_labels_sub[w*10])

# Load the right log mel spectrograms and labels for the development set 
with open('../logmel_subset1_right.pickle','rb') as f:  # Python 3: open(..., 'rb')
    my_features_sub1,my_labels_sub1 = pickle.load(f)
# Download here : wget https://www.dropbox.com/s/1gd6gmyjrzo6auj/logmel_subset1_right.pickle?dl=0

# The following outcommented lines should be used if the entire dataset is used. However, due to limits in the size of files that can be uploaded to github, the entire dataset is not uploaded.
# with open('../logmel_subset2_right.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub2,my_labels_sub2 = pickle.load(f)
# with open('../logmel_subset3_right.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub3,my_labels_sub3 = pickle.load(f)
# with open('../logmel_subset4_right.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub4,my_labels_sub4 = pickle.load(f)
# with open('../logmel_subset5_right.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub5,my_labels_sub5 = pickle.load(f)
# with open('../logmel_subset6_right.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub6,my_labels_sub6 = pickle.load(f)
# with open('../logmel_subset7_right.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub7,my_labels_sub7 = pickle.load(f)
# with open('../logmel_subset8_right.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub8,my_labels_sub8 = pickle.load(f)
# with open('../logmel_subset9_right.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub9,my_labels_sub9 = pickle.load(f)
# with open('../logmel_subset10_right.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub10,my_labels_sub10 = pickle.load(f)

# my_features_sub_MD = np.concatenate((my_features_sub1,my_features_sub2,my_features_sub3,my_features_sub4,my_features_sub5,my_features_sub6,my_features_sub7,my_features_sub8,my_features_sub9,my_features_sub10))
# my_labels_sub_MD = np.concatenate((my_labels_sub1,my_labels_sub2,my_labels_sub3,my_labels_sub4,my_labels_sub5,my_labels_sub6,my_labels_sub7,my_labels_sub8,my_labels_sub9,my_labels_sub10))
my_features_sub_MD = my_features_sub1
my_labels_sub_MD = my_labels_sub1

# Load the right log-mel spectrograms and labels for the evaluation/test set 
with open('../Eval1_RIGHT.pickle','rb') as f:  # Python 3: open(..., 'rb')
    my_features_sub1,my_labels_sub1 = pickle.load(f)
# Download here : wget https://www.dropbox.com/s/9lnnuabmy6jqi4d/Eval1_RIGHT.pickle?dl=0

# The following outcommented lines should be used if the entire dataset is used. However, due to limits in the size of files that can be uploaded to github, the entire dataset is not uploaded.
# with open('../Eval2_RIGHT.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub2,my_labels_sub2 = pickle.load(f)
# with open('../Eval3_RIGHT.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub3,my_labels_sub3 = pickle.load(f)
# with open('../Eval4_RIGHT.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub4,my_labels_sub4 = pickle.load(f)

# my_features_eval_MD = np.concatenate((my_features_sub1,my_features_sub2,my_features_sub3,my_features_sub4))
# my_labels_eval_MD = np.concatenate((my_labels_sub1,my_labels_sub2,my_labels_sub3,my_labels_sub4))
my_features_eval_MD = my_features_sub1
my_labels_eval_MD = my_labels_sub1


my_features_test_MD = []

for n in range(0,int(len(my_features_eval_MD)/10)):
    my_features_test_MD.append(my_features_eval_MD[n*10:n*10+10])    

my_features_test_MD = list(np.reshape(my_features_test_MD,[len(my_features_test_MD)*10,96,64]))

# Load the mono log-mel spectrograms and labels for the evaluation/test set 
with open('logmel_eval1.pickle','rb') as f:  # Python 3: open(..., 'rb')
    my_features_sub1,my_labels_sub1 = pickle.load(f)
# Download here : wget https://www.dropbox.com/s/6g2ujcajfrrsxja/Eval1.pickle?dl=0
    
# The following outcommented lines should be used if the entire dataset is used. However, due to limits in the size of files that can be uploaded to github, the entire dataset is not uploaded.
# with open('logmel_eval2.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub2,my_labels_sub2 = pickle.load(f)
# with open('logmel_eval3.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub3,my_labels_sub3 = pickle.load(f)
# with open('logmel_eval4.pickle','rb') as f:  # Python 3: open(..., 'rb')
#     my_features_sub4,my_labels_sub4 = pickle.load(f)

# my_features_eval = np.concatenate((my_features_sub1,my_features_sub2,my_features_sub3,my_features_sub4))
# my_labels_eval = np.concatenate((my_labels_sub1,my_labels_sub2,my_labels_sub3,my_labels_sub4))
my_features_eval = my_features_sub1
my_labels_eval = my_labels_sub1

my_labels_s_eval = []
for w in range(0,int(len(my_labels_eval)/10)):
    my_labels_s_eval.append(my_labels_eval[w*10])

my_features_test = []
my_labels_test = []
for n in range(0,int(len(my_features_eval)/10)):
    my_features_test.append(my_features_eval[n*10:n*10+10])    
    my_labels_test.append(my_labels_eval[n*10:n*10+10])

my_features_test = list(np.reshape(my_features_test,[len(my_features_test)*10,96,64]))
my_labels_test = list(np.reshape(my_labels_test,[len(my_labels_test)*10,15]))

'''Final model in training mode, if you wish to only run the test, please proceed to next cell'''

# Used for saving weights
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summaries_path = "tensorboard/%s" % (timestamp)

def main(_):
    
    # Defining empty array to save results
    loss_mean_tr = []
    acc_mean_tr = []
    acc_major_tr = []

    lss_mean_val = []
    acc_mean_val = []
    acc_major_val = []
    
    lss_mean_test = []
    acc_mean_test = []
    acc_major_test = []
    
    accuracy_mean_test = []
    accuracy_major_test = []
    loss_mean_test = []
    
    with tf.Graph().as_default(), tf.Session() as sess:
        is_training = tf.placeholder(dtype=tf.bool, shape=None) # Define boelean for using batch norm
    
        # Define VGGish.
        
        # VGGish for mono logmel spectrograms 
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_initializer=tf.truncated_normal_initializer(
                           stddev=params.INIT_STDDEV),
                       biases_initializer=tf.zeros_initializer(),
                       activation_fn=tf.nn.relu,
                       trainable=training), \
                 slim.arg_scope([slim.conv2d],
                       kernel_size=[3, 3], stride=1, padding='SAME'), \
                 slim.arg_scope([slim.max_pool2d],
                       kernel_size=[2, 2], stride=2, padding='SAME'), \
            tf.variable_scope('vggish'):
             # Input: a batch of 2-D log-mel-spectrogram patches.
            features_ph = tf.placeholder(
               tf.float32, shape=(None, params.NUM_FRAMES, params.NUM_BANDS),
               name='input_features0')
   
            # Reshape to 4-D so that we can convolve a batch with conv2d().
            net01 = tf.reshape(features_ph, [-1, params.NUM_FRAMES, params.NUM_BANDS, 1])
            print(tf.shape(net01))
  
            # The VGG stack of alternating convolutions and max-pools.
            net1 = slim.conv2d(net01, 64, scope='conv1')
            net1 = slim.max_pool2d(net1, scope='pool1')
            net1 = slim.conv2d(net1, 128, scope='conv2')
            net1 = slim.max_pool2d(net1, scope='pool2')
            net1 = slim.repeat(net1, 2, slim.conv2d, 256, scope='conv3')
            net1 = slim.max_pool2d(net1, scope='pool3')
            net1 = slim.repeat(net1, 2, slim.conv2d, 512, scope='conv4')
            net1 = slim.max_pool2d(net1, scope='pool4')
            vggOut1 = net1
  
            # Flatten before entering fully-connected layers
            net1 = slim.flatten(net1)
            netFlat1 = net1
            net = slim.repeat(net1, 2, slim.fully_connected, 4096, scope='fc1')
        
            # The embedding layer.
            embeddings1 = slim.fully_connected(net, params.EMBEDDING_SIZE, scope='fc2')
            embeddings12 = slim.fully_connected(net1, params.EMBEDDING_SIZE, scope='fc2_1')


        
        
        # VGGish for right logmel spectrograms      
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                       weights_initializer=tf.truncated_normal_initializer(
                           stddev=params.INIT_STDDEV),
                       biases_initializer=tf.zeros_initializer(),
                       activation_fn=tf.nn.relu,
                       trainable=training), \
                 slim.arg_scope([slim.conv2d],
                       kernel_size=[3, 3], stride=1, padding='SAME'), \
                 slim.arg_scope([slim.max_pool2d],
                       kernel_size=[2, 2], stride=2, padding='SAME'), \
            tf.variable_scope('vggish'):
            # Input: a batch of 2-D log-mel-spectrogram patches.
            features2_ph = tf.placeholder(
                tf.float32, shape=(None, params.NUM_FRAMES, params.NUM_BANDS),
                name='input_features2')
    
            # Reshape to 4-D so that we can convolve a batch with conv2d().
            net0 = tf.reshape(features2_ph, [-1, params.NUM_FRAMES, params.NUM_BANDS, 1])
            print(tf.shape(net0))

            # The VGG stack of alternating convolutions and max-pools.
            net = slim.conv2d(net0, 64, scope='conv1_MD')
            net = slim.max_pool2d(net, scope='pool1_MD')
            net = slim.conv2d(net, 128, scope='conv2_MD')
            net = slim.max_pool2d(net, scope='pool2_MD')
            net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3_MD')
            net = slim.max_pool2d(net, scope='pool3_MD')
            net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4_MD')
            net = slim.max_pool2d(net, scope='pool4_MD')
            vggOut2 = net

            # Flatten before entering fully-connected layers
            net1 = slim.flatten(net)
            #netFlat = net1
            net = slim.repeat(net1, 2, slim.fully_connected, 4096, scope='fc1_MD')
            # The embedding layer.
            embeddings2 = slim.fully_connected(net, params.EMBEDDING_SIZE, scope='fc2_MD')
            embeddings22 = slim.fully_connected(net1, params.EMBEDDING_SIZE, scope='fc2_MD_1')


        # Concatenate the two embedding 
        embeddings = tf.concat(values=[embeddings12, embeddings22], axis=1, name="embeddings")
        vggEmbeddings = tf.concat(values= [vggOut1, vggOut2], axis=1, name = 'vggEmbeddings')
        #print(tf.shape(embeddings12), tf.shape(embeddings22))

        with tf.variable_scope('mymodel'):
            '''Baseline model'''
            # The following outcommented lines should be used to run the baseline model
#             Add a fully connected layer with 100 units.
#             num_units = 100
#             fc = slim.fully_connected(embeddings12, num_units)
           

#            # Add a classifier layer at the end, consisting of parallel logistic
#            # classifiers, one per class. This allows for multi-class tasks.
#             logits = slim.fully_connected(
#              fc, _NUM_CLASSES, activation_fn=None, scope='logits') 
#             preds = tf.sigmoid(logits, name='prediction')
            
            

            '''Final model'''
            l2 = tf.contrib.layers.l2_regularizer(0.1, scope=None) # Adding weight regularizer
            num_units1 = 192
            fc = tf.layers.dense(embeddings, num_units1, activation=tf.nn.relu, kernel_regularizer=l2) # Dense layer, with ReLU and L2 reguraliser
            h_fc1_drop = tf.layers.dropout(fc, rate=0.2, training = is_training) # Adding dropout
#             h_fc1_drop = tf.layers.batch_normalization(fc, training=is_training) # Batch norm
            
            num_units2 = 100
            fc2 = tf.layers.dense(h_fc1_drop, num_units2, activation=tf.nn.relu, kernel_regularizer=l2) # Dense layer, with ReLU and L2 reguraliser
            h_fc2_drop = tf.layers.dropout(fc2, rate=0.5, training = is_training) # Adding dropout
#             h_fc2_drop = tf.layers.batch_normalization(fc2, training=is_training) # Batch norm
            
            num_units3 = 35
            fc3 = tf.layers.dense(h_fc2_drop, num_units3, activation=tf.nn.relu, kernel_regularizer=l2) # Dense layer, with ReLU and L2 reguraliser
            h_fc3_drop = tf.layers.dropout(fc3, rate=0.5, training = is_training) # Adding dropout
#             h_fc3_drop = tf.layers.batch_normalization(fc3, training=is_training) # Batch norm

            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.
            logits = slim.fully_connected(
              h_fc3_drop, _NUM_CLASSES, activation_fn=None, scope='logits')

            preds = tf.nn.softmax(logits, name='prediction')

            # Add training ops.
            with tf.variable_scope('train'):
                global_step = tf.Variable(
                    0, name='global_step', trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                 tf.GraphKeys.GLOBAL_STEP])

                # Labels are assumed to be fed as a batch multi-hot vectors, with
                # a 1 in the position of each positive class label, and 0 elsewhere.
                labels = tf.placeholder(
                    tf.float32, shape=(None, _NUM_CLASSES), name='labels')

                # Cross-entropy label loss.
                xent = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=labels, name='xent')
                loss = tf.reduce_mean(xent, name='loss_op')
                tf.summary.scalar('loss', loss)

                # Accuracy
                argmax_y = tf.to_int32(tf.argmax(preds, axis=1))
                argmax_t = tf.to_int32(tf.argmax(labels, axis=1))
                correct = tf.to_float(tf.equal(argmax_y,argmax_t))
                accuracy1 = tf.reduce_mean(correct, name='accuracy_op')
                tf.summary.scalar('accuracy1', accuracy1)
                
                '''Uncomment for decaying learning rate'''
#                 starter_learning_rate = 0.1
#                 learningRate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                            100000, 0.96, staircase=True)
                
                learningRate = params.LEARNING_RATE # 1e-4, change to set custom learning rate
                
                # We use the same optimizer and hyperparameters as used to train VGGish.
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=learningRate, # 1e-4
                    epsilon=params.ADAM_EPSILON) # 1e-8
                train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')
                
                saver = tf.train.Saver()

        # Initialize all variables in the model, and then load the pre-trained
        # VGGish checkpoint.
        sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, './vggish_model.ckpt') # Load VGGish weights

        # Define a writer for the tensorboard    
        writer = tf.summary.FileWriter('log/train/'+summaries_path, sess.graph)
        writer_val = tf.summary.FileWriter('log/val/'+summaries_path, sess.graph)

        # Locate all the tensors and ops we need for the training loop.
        labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
        global_step_tensor = sess.graph.get_tensor_by_name(
            'mymodel/train/global_step:0')
        loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
        prediction_op = sess.graph.get_operation_by_name('mymodel/prediction')

        # Training loop over epochs and batches
        num_epochs = 15
        batch_size_train = 16
        batch_size_val = 16

        # Merging all summaries for the tensorboard
        merged = tf.summary.merge_all()

        start = time.perf_counter() # Measure time start
        cur_epoch = 0
        
        # Start training of the model
        for i in range(0,num_epochs):
            startEpoch = time.perf_counter() # time per epoch
            num_batch_train = math.ceil(len(train_index)/batch_size_train)

            # Taking out the spectrograms and labels corresponding to the training set 
            my_features_train = []
            my_features_train_MD = []
            idx_train = shuffle(train_index)
            
            my_labels_train = []
            for h in range(0,len(idx_train)):
                my_features_train.append(my_features_sub[idx_train[h]*10:idx_train[h]*10+10])
                my_features_train_MD.append(my_features_sub_MD[idx_train[h]*10:idx_train[h]*10+10])
                my_labels_train.append(my_labels_sub[idx_train[h]*10:idx_train[h]*10+10])

            my_features_train = list(np.reshape(my_features_train,[len(my_features_train)*10,96,64]))
            my_features_train_MD = list(np.reshape(my_features_train_MD,[len(my_features_train_MD)*10,96,64]))
            my_labels_train = list(np.reshape(my_labels_train,[len(my_labels_train)*10,15]))
            
            # Defining empty arrays of measures to append
            accuracy_mean = []
            accuracy_major_mean = []
            loss_mean = []

            my_features_train, my_features_train_MD, my_labels_train = shuffle(my_features_train,my_features_train_MD,my_labels_train) # Shuffle to not use majority vote
            
            # Run the training set
            for j in range(0,num_batch_train):
                # Taking out specrograms and labels corresponding to one batch
                features = my_features_train[j*batch_size_train*10:(j+1)*batch_size_train*10]
                features2 = my_features_train_MD[j*batch_size_train*10:(j+1)*batch_size_train*10]
                labels = my_labels_train[j*batch_size_train*10:(j+1)*batch_size_train*10]

                # Training on one batch
                [num_steps, loss, acc1, _, pred, summ] = sess.run(
                    [global_step_tensor, loss_tensor, accuracy1, train_op, preds, merged],
                    feed_dict={features_ph: features, features2_ph: features2, labels_tensor: labels, is_training: True})
                
                # Majority vote - not used for training in the final model
                pred10 = []
                label10 = []
                for l in range(0,pred.shape[0],10):
                    pred10.append([l0+l1+l2+l3+l4+l5+l6+l7+l8+l9 for l0,l1,l2,l3,l4,l5,l6,l7,l8,l9 in zip(pred[l+0],pred[l+1],pred[l+2],pred[l+3],pred[l+4],pred[l+5],pred[l+6],pred[l+7],pred[l+8],pred[l+9],)]) 
                    label10.append(labels[l])
                argmax_y = np.argmax(pred10,axis=1)
                argmax_t = np.argmax(label10,axis=1)
                correct = np.float32(np.equal(argmax_y,argmax_t))
                accuracy10 = np.mean(correct)

                # Appending measures
                accuracy_mean.append(acc1)
                accuracy_major_mean.append(accuracy10)
                loss_mean.append(loss)

                writer.add_summary(summ, num_steps)
#                 print('Epoch %d, batch %d: loss %g, accuracy1 %f, acc major vote %f' % (i+1, j+1, loss, acc1, accuracy10))

            # Finding mean for all batches in current epoch
            loss_mean_tr.append(sum(loss_mean)/len(loss_mean))
            acc_mean_tr.append(sum(accuracy_mean)/len(accuracy_mean))
            acc_major_tr.append(sum(accuracy_major_mean)/len(accuracy_major_mean))
            
            print('Mean values train: loss %g, accuracy %f, accuracy major %f' % (sum(loss_mean)/len(loss_mean), sum(accuracy_mean)/len(accuracy_mean), sum(accuracy_major_mean)/len(accuracy_major_mean)))

            # Running the validation set
            idx_val = shuffle(test_index)
            my_features_val = []
            my_features_val_MD = []
            my_labels_val = []

            for k in range(0,len(idx_val)):
                # Taking out spectrograms and labels corresponding to validation set
                my_features_val.append(my_features_sub[idx_val[k]*10:idx_val[k]*10+10])
                my_features_val_MD.append(my_features_sub_MD[idx_val[k]*10:idx_val[k]*10+10])
                my_labels_val.append(my_labels_sub[idx_val[k]*10:idx_val[k]*10+10])

            my_features_val = list(np.reshape(my_features_val,[len(my_features_val)*10,96,64]))
            my_features_val_MD = list(np.reshape(my_features_val_MD,[len(my_features_val_MD)*10,96,64]))
            
            my_labels_val = list(np.reshape(my_labels_val,[len(my_labels_val)*10,15]))
            
            num_batch_val = math.ceil(len(test_index)/batch_size_val)

            # Empty arrays for measures to append
            accuracy_mean_val = []
            accuracy_major_val = []
            loss_mean_val = []

            for j in range(0,num_batch_val):
                # Taking out spectrograms and labels corresponding to one batch
                features_val = my_features_val[j*batch_size_val*10:(j+1)*batch_size_val*10]
                features_val_MD = my_features_val_MD[j*batch_size_val*10:(j+1)*batch_size_val*10]
                labels_val = my_labels_val[j*batch_size_val*10:(j+1)*batch_size_val*10]
                
                # Running validation set without training
                [num_steps2, valloss, valacc1,valpred, summ2] = sess.run(
                    [global_step_tensor, loss_tensor, accuracy1,preds, merged],
                    feed_dict={features_ph:features_val,features2_ph:features_val_MD, labels_tensor: labels_val, is_training: False})
                
                # Majority vote
                pred10_val = []
                label10_val = []
                for l in range(0,valpred.shape[0],10):
                    pred10_val.append([l0+l1+l2+l3+l4+l5+l6+l7+l8+l9 for l0,l1,l2,l3,l4,l5,l6,l7,l8,l9 in zip(valpred[l+0],valpred[l+1],valpred[l+2],valpred[l+3],valpred[l+4],valpred[l+5],valpred[l+6],valpred[l+7],valpred[l+8],valpred[l+9],)]) 
                    label10_val.append(labels_val[l])
                argmax_y_val = np.argmax(pred10_val,axis=1)
                argmax_t_val = np.argmax(label10_val,axis=1)
                correct_val = np.float32(np.equal(argmax_y_val,argmax_t_val))
                accuracy10_val = np.mean(correct_val)
                
                # Appending measures
                accuracy_mean_val.append(valacc1)
                accuracy_major_val.append(accuracy10_val)
                loss_mean_val.append(valloss)

                writer.add_summary(summ2, num_steps2)
#                 print('Epoch %d, val_batch %d: loss %g, accuracy1 %f, acc major vote %f' % (i+1, j+1, valloss, valacc1, accuracy10_val))
            
            # Mean values for all batches in current epoch
            lss_mean_val.append(sum(loss_mean_val)/len(loss_mean_val))
            acc_mean_val.append(sum(accuracy_mean_val)/len(accuracy_mean_val))
            acc_major_val.append(sum(accuracy_major_val)/len(accuracy_major_val))
            
            print('Mean values val: loss %g, accuracy %f, accuracy major %f' % (sum(loss_mean_val)/len(loss_mean_val), sum(accuracy_mean_val)/len(accuracy_mean_val), sum(accuracy_major_val)/len(accuracy_major_val)))

            cur_epoch += 1
            elapsedEpoch = (time.perf_counter() - startEpoch)/60
            print('Elapsed time for Epoch %d: %.3f minutes.' % (cur_epoch, elapsedEpoch))
            
        # After training the model, run the test
        batch_size_test = batch_size_val # change to custom batch size, as default same as validation
        num_batch_test = math.ceil(len(my_features_eval)/(batch_size_test*10))
        
        for i in range(0,num_batch_test):
            # Taking out spectrograms and labels corresponding to one batch
            features_test = my_features_test[j*batch_size_test*10:(j+1)*batch_size_test*10]
            features_test_MD = my_features_test_MD[j*batch_size_val*10:(j+1)*batch_size_val*10]
            labels_test = my_labels_test[j*batch_size_test*10:(j+1)*batch_size_test*10]
            
            # Running without training
            [num_steps3, testloss, testacc1,testpred, summ3] = sess.run(
                [global_step_tensor, loss_tensor, accuracy1,preds, merged],
                feed_dict={features_ph:features_test, features2_ph:features_test_MD, labels_tensor: labels_test, is_training: False})

            # Majority vote
            pred10_test = []
            label10_test = []
            for l in range(0,testpred.shape[0],10):
                pred10_test.append([l0+l1+l2+l3+l4+l5+l6+l7+l8+l9 for l0,l1,l2,l3,l4,l5,l6,l7,l8,l9 in zip(testpred[l+0],testpred[l+1],testpred[l+2],testpred[l+3],testpred[l+4],testpred[l+5],testpred[l+6],testpred[l+7],testpred[l+8],testpred[l+9],)]) 
                label10_test.append(labels_test[l])
            argmax_y_test = np.argmax(pred10_test,axis=1)
            argmax_t_test = np.argmax(label10_test,axis=1)
            correct_test = np.float32(np.equal(argmax_y_test,argmax_t_test))
            accuracy10_test = np.mean(correct_test)

            # Appending measures
            accuracy_mean_test.append(testacc1)
            accuracy_major_test.append(accuracy10_test)
            loss_mean_test.append(testloss)

            writer.add_summary(summ3, num_steps3)
#                 print('test_batch %d: loss %g, accuracy1 %f, acc major vote %f' % (j+1, testloss, testacc1, accuracy10_test))

            # Average
            lss_mean_test.append(sum(loss_mean_test)/len(loss_mean_test))
            acc_mean_test.append(sum(accuracy_mean_test)/len(accuracy_mean_test))
            acc_major_test.append(sum(accuracy_major_test)/len(accuracy_major_test))
        print('\n','****', '\n')
        print('Mean values test: loss %g, accuracy %f, accuracy major %f' % (sum(loss_mean_test)/len(loss_mean_test), sum(accuracy_mean_test)/len(accuracy_mean_test), sum(accuracy_major_test)/len(accuracy_major_test)))
        print('\n','****', '\n')
        elapsed = (time.perf_counter() - start)/60
        print('Elapsed total time %.3f minutes.' % elapsed) 
        print('Success!')
        
        # Save all mean value to dataframe, then save to CSV
        mean_results = pd.DataFrame({'lossTr': loss_mean_tr,
                                     'accTr': acc_mean_tr,
                                     'accMajorTr': acc_major_tr,
                                     'lossVal': lss_mean_val,
                                     'accVal': acc_mean_val,
                                     'accMajorVal': acc_major_val
                                    })
        mean_results.to_csv('VGGish_final.csv', index = False)
    
#         save_path = saver.save(sess, "./final.ckpt") # Uncomment to save model

if __name__ == '__main__':
    tf.app.run()

'''Run the following script to test a pretrained model, by loading weights.'''

def main(_):
    
    
    lss_mean_test = []
    acc_mean_test = []
    acc_major_test = []
    
    with tf.Graph().as_default(), tf.Session() as sess:
        is_training = tf.placeholder(dtype=tf.bool, shape=None) # Define boelean for using batch norm
        # Define VGGish.
        
        # VGGish for mono logmel spectrograms 
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_initializer=tf.truncated_normal_initializer(
                           stddev=params.INIT_STDDEV),
                       biases_initializer=tf.zeros_initializer(),
                       activation_fn=tf.nn.relu,
                       trainable=training), \
                 slim.arg_scope([slim.conv2d],
                       kernel_size=[3, 3], stride=1, padding='SAME'), \
                 slim.arg_scope([slim.max_pool2d],
                       kernel_size=[2, 2], stride=2, padding='SAME'), \
            tf.variable_scope('vggish'):
             # Input: a batch of 2-D log-mel-spectrogram patches.
            features_ph = tf.placeholder(
               tf.float32, shape=(None, params.NUM_FRAMES, params.NUM_BANDS),
               name='input_features0')
   
            # Reshape to 4-D so that we can convolve a batch with conv2d().
            net01 = tf.reshape(features_ph, [-1, params.NUM_FRAMES, params.NUM_BANDS, 1])
            print(tf.shape(net01))
  
            # The VGG stack of alternating convolutions and max-pools.
            net1 = slim.conv2d(net01, 64, scope='conv1')
            net1 = slim.max_pool2d(net1, scope='pool1')
            net1 = slim.conv2d(net1, 128, scope='conv2')
            net1 = slim.max_pool2d(net1, scope='pool2')
            net1 = slim.repeat(net1, 2, slim.conv2d, 256, scope='conv3')
            net1 = slim.max_pool2d(net1, scope='pool3')
            net1 = slim.repeat(net1, 2, slim.conv2d, 512, scope='conv4')
            net1 = slim.max_pool2d(net1, scope='pool4')
            vggOut1 = net1
  
            # Flatten before entering fully-connected layers
            net1 = slim.flatten(net1)
            netFlat1 = net1
            net = slim.repeat(net1, 2, slim.fully_connected, 4096, scope='fc1')
        
            # The embedding layer.
            embeddings1 = slim.fully_connected(net, params.EMBEDDING_SIZE, scope='fc2')
            embeddings12 = slim.fully_connected(net1, params.EMBEDDING_SIZE, scope='fc2_1')


        
        
        # VGGish for right logmel spectrograms      
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                       weights_initializer=tf.truncated_normal_initializer(
                           stddev=params.INIT_STDDEV),
                       biases_initializer=tf.zeros_initializer(),
                       activation_fn=tf.nn.relu,
                       trainable=training), \
                 slim.arg_scope([slim.conv2d],
                       kernel_size=[3, 3], stride=1, padding='SAME'), \
                 slim.arg_scope([slim.max_pool2d],
                       kernel_size=[2, 2], stride=2, padding='SAME'), \
            tf.variable_scope('vggish'):
            # Input: a batch of 2-D log-mel-spectrogram patches.
            features2_ph = tf.placeholder(
                tf.float32, shape=(None, params.NUM_FRAMES, params.NUM_BANDS),
                name='input_features2')
    
            # Reshape to 4-D so that we can convolve a batch with conv2d().
            net0 = tf.reshape(features2_ph, [-1, params.NUM_FRAMES, params.NUM_BANDS, 1])
            print(tf.shape(net0))

            # The VGG stack of alternating convolutions and max-pools.
            net = slim.conv2d(net0, 64, scope='conv1_MD')
            net = slim.max_pool2d(net, scope='pool1_MD')
            net = slim.conv2d(net, 128, scope='conv2_MD')
            net = slim.max_pool2d(net, scope='pool2_MD')
            net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3_MD')
            net = slim.max_pool2d(net, scope='pool3_MD')
            net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4_MD')
            net = slim.max_pool2d(net, scope='pool4_MD')
            vggOut2 = net

            # Flatten before entering fully-connected layers
            net1 = slim.flatten(net)
            #netFlat = net1
            net = slim.repeat(net1, 2, slim.fully_connected, 4096, scope='fc1_MD')
            # The embedding layer.
            embeddings2 = slim.fully_connected(net, params.EMBEDDING_SIZE, scope='fc2_MD')
            embeddings22 = slim.fully_connected(net1, params.EMBEDDING_SIZE, scope='fc2_MD_1')


        # Concatenate the two embedding 
        embeddings = tf.concat(values=[embeddings12, embeddings22], axis=1, name="embeddings")
        vggEmbeddings = tf.concat(values= [vggOut1, vggOut2], axis=1, name = 'vggEmbeddings')
        #print(tf.shape(embeddings12), tf.shape(embeddings22))

        with tf.variable_scope('mymodel'):
            '''Baseline model'''
              # Use the outcommented lines to use baseline model
#             Add a fully connected layer with 100 units.
#             num_units = 100
#             fc = slim.fully_connected(embeddings12, num_units)
           
#            # Add a classifier layer at the end, consisting of parallel logistic
#            # classifiers, one per class. This allows for multi-class tasks.
#             logits = slim.fully_connected(
#              fc, _NUM_CLASSES, activation_fn=None, scope='logits') 
#             preds = tf.sigmoid(logits, name='prediction')
            
            

            '''Final model'''
            l2 = tf.contrib.layers.l2_regularizer(0.1, scope=None) # Adding weight regularizer
            num_units1 = 192
            fc = tf.layers.dense(embeddings, num_units1, activation=tf.nn.relu, kernel_regularizer=l2) # Dense layer, with ReLU and L2 reguraliser
            h_fc1_drop = tf.layers.dropout(fc, rate=0.2, training = is_training) # Adding dropout layer
#             h_fc1_drop = tf.layers.batch_normalization(fc, training=is_training) # Batch norm
            
            num_units2 = 100
            fc2 = tf.layers.dense(h_fc1_drop, num_units2, activation=tf.nn.relu, kernel_regularizer=l2) # Dense layer, with ReLU and L2 reguraliser
            h_fc2_drop = tf.layers.dropout(fc2, rate=0.5, training = is_training) # Adding dropout layer
#             h_fc2_drop = tf.layers.batch_normalization(fc2, training=is_training) # Batch norm
            
            num_units3 = 35
            fc3 = tf.layers.dense(h_fc2_drop, num_units3, activation=tf.nn.relu, kernel_regularizer=l2) # Dense layer, with ReLU and L2 reguraliser
            h_fc3_drop = tf.layers.dropout(fc3, rate=0.5, training = is_training) # Adding dropout layer
#             h_fc3_drop = tf.layers.batch_normalization(fc3, training=is_training) # Batch norm

            # Add a classifier layer at the end, consisting of parallel logistic
            # classifiers, one per class. This allows for multi-class tasks.
            logits = slim.fully_connected(
              h_fc3_drop, _NUM_CLASSES, activation_fn=None, scope='logits')

            preds = tf.nn.softmax(logits, name='prediction')

            # Add training ops.
            with tf.variable_scope('train'):
                global_step = tf.Variable(
                    0, name='global_step', trainable=False,
                    collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                 tf.GraphKeys.GLOBAL_STEP])

                # Labels are assumed to be fed as a batch multi-hot vectors, with
                # a 1 in the position of each positive class label, and 0 elsewhere.
                labels = tf.placeholder(
                    tf.float32, shape=(None, _NUM_CLASSES), name='labels')

                # Cross-entropy label loss.
                xent = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits, labels=labels, name='xent')
                loss = tf.reduce_mean(xent, name='loss_op')
                tf.summary.scalar('loss', loss)

                # Accuracy
                argmax_y = tf.to_int32(tf.argmax(preds, axis=1))
                argmax_t = tf.to_int32(tf.argmax(labels, axis=1))
                correct = tf.to_float(tf.equal(argmax_y,argmax_t))
                accuracy1 = tf.reduce_mean(correct, name='accuracy_op')
                tf.summary.scalar('accuracy1', accuracy1)

                # We use the same optimizer and hyperparameters as used to train VGGish.
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=params.LEARNING_RATE,
                    epsilon=params.ADAM_EPSILON)
                train_op = optimizer.minimize(loss, global_step=global_step, name='train_op')
                
                saver = tf.train.Saver()

        # Initialize all variables in the model, and then load the pre-trained
        # VGGish checkpoint.
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, './VGGish_concate2.ckpt')

        writer = tf.summary.FileWriter('log/train/'+summaries_path, sess.graph)

        # All tensors and ops
        labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
        global_step_tensor = sess.graph.get_tensor_by_name(
            'mymodel/train/global_step:0')
        loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
        prediction_op = sess.graph.get_operation_by_name('mymodel/prediction')

        
        merged = tf.summary.merge_all()
        
        
        # Running test set in batches
        batch_size_test = 30
        num_batch_test = math.ceil(len(my_features_eval)/(batch_size_test*10))
        
        # Defining measures to append
        accuracy_mean_test = []
        accuracy_major_test = []
        loss_mean_test = []

        for j in range(0,num_batch_test):
            # Taking out spectrograms and labels corresponding to one batch
            features_test = my_features_test[j*batch_size_test*10:(j+1)*batch_size_test*10]
            features_test_MD = my_features_test_MD[j*batch_size_val*10:(j+1)*batch_size_val*10]
            labels_test = my_labels_test[j*batch_size_test*10:(j+1)*batch_size_test*10]
            
            # Running 
            [num_steps3, testloss, testacc1,testpred, summ3] = sess.run(
                [global_step_tensor, loss_tensor, accuracy1,preds, merged],
                feed_dict={features_ph:features_test, features2_ph:features_test_MD, labels_tensor: labels_test, is_training: False})
            
            # Majority vote
            pred10_test = []
            label10_test = []
            for l in range(0,testpred.shape[0],10):
                pred10_test.append([l0+l1+l2+l3+l4+l5+l6+l7+l8+l9 for l0,l1,l2,l3,l4,l5,l6,l7,l8,l9 in zip(testpred[l+0],testpred[l+1],testpred[l+2],testpred[l+3],testpred[l+4],testpred[l+5],testpred[l+6],testpred[l+7],testpred[l+8],testpred[l+9],)])
                label10_test.append(labels_test[l])
            argmax_y_test = np.argmax(pred10_test,axis=1)
            argmax_t_test = np.argmax(label10_test,axis=1)
            correct_test = np.float32(np.equal(argmax_y_test,argmax_t_test))
            accuracy10_test = np.mean(correct_test)
            
            # Appending measures
            accuracy_mean_test.append(testacc1)
            accuracy_major_test.append(accuracy10_test)
            loss_mean_test.append(testloss)

            writer.add_summary(summ3, num_steps3)
            print('test_batch %d: loss %g, accuracy1 %f, acc major vote %f' % (j+1, testloss, testacc1, accuracy10_test))
        
        # Average for all batches
        lss_mean_test.append(sum(loss_mean_test)/len(loss_mean_test))
        acc_mean_test.append(sum(accuracy_mean_test)/len(accuracy_mean_test))
        acc_major_test.append(sum(accuracy_major_test)/len(accuracy_major_test))
        
        print('Mean values test: loss %g, accuracy %f, accuracy major %f' % (sum(loss_mean_test)/len(loss_mean_test), sum(accuracy_mean_test)/len(accuracy_mean_test), sum(accuracy_major_test)/len(accuracy_major_test)))

    print('Succes!')
    
    mean_results = pd.DataFrame({'lossTest': lss_mean_test,
                                 'accTest': acc_mean_test,
                                 'accMajorTest': acc_major_test
                                })
    mean_results.to_csv('results_test.csv', index = False)

if __name__ == '__main__':
    tf.app.run()



