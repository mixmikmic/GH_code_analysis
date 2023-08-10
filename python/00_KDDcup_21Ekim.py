# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle

# Config the matplotlib backend as plotting inline in IPython
get_ipython().magic('matplotlib inline')

import csv
from array import *
from collections import Counter

    
        

with open('/home/isl-eyup/tensorflow/KDDcup_datasets/kddcup_data_corrected_uniq.csv', 'rb') as f:
    reader = csv.reader(f)
    data_as_list = list(reader)
def file_len(adress):
  f = open(adress)
  nr_of_lines = sum(1 for line in f)
  f.close()
  return nr_of_lines

number_of_lines = file_len('/home/isl-eyup/tensorflow/KDDcup_datasets/kddcup_data_corrected_uniq.csv')
print('Toplam satir sayisi : ', number_of_lines)



def label_index(str):
    indx = 4
    if str == 'back.':
        indx = 0
    elif str == 'buffer_overflow.':
        indx = 1
    elif str =='ftp_write.':
        indx = 2
    elif str == 'guess_password.':
        indx = 2
    elif str == 'imap.':
        indx = 2
    elif str == 'ipsweep.':
        indx = 3
    elif str == 'land.':
        indx = 0
    elif str == 'loadmodule.':
        indx = 1
    elif str == 'multihop.':
        indx = 0
    elif str == 'neptune.':
        indx = 0
    elif str == 'nmap.':
        indx = 3
    elif str == 'normal.':
        indx = 4
    elif str == 'perl.':
        indx = 1
    elif str =='phf.':
        indx = 2
    elif str == 'pod.':
        indx = 0
    elif str == 'portsweep.':
        indx = 3
    elif str == 'rootkit.':
        indx = 1
    elif str == 'satan.':
        indx = 3
    elif str == 'smurf.':
        indx = 0
    elif str == 'spy.':
        indx = 2
    elif str == 'teardrop.':
        indx = 0
    elif str == 'warezclient.':
        indx == 2
    elif str == 'warezmaster.':
        indx = 2
    else :
        indx = 4
    return indx
    
    
attack_types = ['dos','u2r','r2l','probe','normal']    
print("Classes are : ",attack_types)

#LABEL ARRAY 
#label_array = np.empty(number_of_lines, dtype=object)
#attack_matrix = np.ndarray(shape=(number_of_lines,5), dtype='int')

#for i in range(number_of_lines):
#    for j in range(5):
#        attack_matrix[i][j]=0

#FROM LABEL TO ATTACK MATRIX        
#for i in range(number_of_lines):
#    label_array[i] = data_as_list[i][41] 
#    attack_matrix[i][label_index(label_array[i])] = 1

"""Protocols : ['icmp', 'tcp', 'udp']"""
def protocol_index(str):
    indx = 0
    if str == 'icmp':
        indx = 0
    elif str == 'tcp':
        indx = 1
    elif str =='udp':
        indx = 2
    return indx

"""FLAGS : ['SF', 'REJ', 'RSTO', 'RSTR', 'S0', 'SH', 'S1', 'OTH', 'S2', 'S3', 'RSTOS0']"""
def flag_index(str):
    indx = 0
    if str == 'SF':
        indx = 0
    elif str == 'REJ':
        indx = 1
    elif str =='RSTO':
        indx = 2
    elif str == 'RSTR':
        indx = 3
    elif str == 'S0':
        indx = 4
    elif str == 'SH':
        indx = 5
    elif str == 'S1':
        indx = 6
    elif str == 'OTH':
        indx = 7
    elif str == 'S2':
        indx = 8
    elif str == 'S3':
        indx = 9
    elif str == 'nmap.':
        indx = 10
    elif str == 'RSTOS0':
        indx = 11
    return indx


"""THERE ARE 70 SERVICES """
def service_index(str):
    indx = 0
    if str == 'eco_i':
        indx = 0
    elif str == 'ecr_i':
        indx = 1
    elif str =='red_i':
        indx = 2
    elif str == 'tim_i':
        indx = 3
    elif str == 'urh_i':
        indx = 4
    elif str == 'urp_i':
        indx = 5
    elif str == 'aol':
        indx = 6
    elif str == 'auth':
        indx = 7
    elif str == 'bgp':
        indx = 8
    elif str == 'courier':
        indx = 9
    elif str == 'csnet_ns':
        indx = 10
    elif str == 'ctf':
        indx = 11
    elif str == 'daytime':
        indx = 12
    elif str == 'discard':
        indx = 13
    elif str == 'domain':
        indx = 14
    elif str == 'echo':
        indx = 15
    elif str == 'efs':
        indx = 16
    elif str == 'exec':
        indx = 17
    elif str == 'finger':
        indx = 18
    elif str == 'ftp_data':
        indx = 19
    elif str == 'ftp':
        indx = 20
    elif str == 'gopher':
        indx = 21
    elif str == 'harvest':
        indx = 22
    elif str == 'hostnames':
        indx = 23
    elif str == 'http_2784':
        indx = 24
    elif str == 'http_443':
        indx = 25
    elif str == 'http_8001':
        indx = 26
    elif str == 'http':
        indx = 27
    elif str == 'imap4':
        indx = 28
    elif str == 'IRC':
        indx = 29
    elif str == 'iso_tsap':
        indx = 30
    elif str == 'klogin':
        indx = 31
    elif str == 'kshell':
        indx = 32        
    elif str == 'ldap':
        indx = 33
    elif str == 'link':
        indx = 34
    elif str == 'login':
        indx = 35
    elif str == 'mtp':
        indx = 36
    elif str == 'name':
        indx = 37
    elif str == 'netbios_dgm':
        indx = 38
    elif str == 'netbios_ns':
        indx = 39
    elif str == 'netbios_ssn':
        indx = 40
    elif str == 'netstat':
        indx = 41
    elif str == 'nnsp':
        indx = 42
    elif str == 'nntp':
        indx = 43
    elif str == 'other':
        indx = 44
    elif str == 'pm_dump':
        indx = 45
    elif str == 'pop_2':
        indx = 46
    elif str == 'pop_3':
        indx = 47
    elif str == 'printer':
        indx = 48
    elif str == 'private':
        indx = 49
    elif str == 'remote_job':
        indx = 50
    elif str == 'rje':
        indx = 51
    elif str == 'shell':
        indx = 52
    elif str == 'smtp':
        indx = 53
    elif str == 'sql_net':
        indx = 54
    elif str == 'ssh':
        indx = 55
    elif str == 'sunrpc':
        indx = 56
    elif str == 'supdup':
        indx = 57
    elif str == 'systat':
        indx = 58
    elif str == 'telnet':
        indx = 59
    elif str == 'time':
        indx = 60
    elif str == 'uucp_path':
        indx = 61
    elif str == 'uucp':
        indx = 62
    elif str == 'vmnet':
        indx = 63
    elif str == 'whois':
        indx = 64
    elif str == 'X11':
        indx = 65
    elif str == 'Z39_50':
        indx = 66
    elif str == 'domain_u':
        indx = 67
    elif str == 'ntp_u':
        indx = 68
    elif str == 'tftp_u':
        indx = 69
    return indx


protocol = []
sayac =0
for i in range(number_of_lines):
    if data_as_list[i][1] in protocol:
        sayac = sayac
    else:
        protocol.append(data_as_list[i][1]) 
        sayac += 1
print("There are %d protocol types " %sayac)
print("Protocols : %s" %protocol)


service = []
sayac =0
for i in range(number_of_lines):
    if data_as_list[i][2] in service:
        sayac = sayac
    else:
        service.append(data_as_list[i][2])
        sayac += 1
print("There are %d service " %sayac)   
print("Services : %s" %service)



flag = []
sayac =0
for i in range(number_of_lines):
    if data_as_list[i][3] in flag:
        sayac = sayac
    else:
        flag.append(data_as_list[i][3])
        sayac += 1
print("There are %d flag " %sayac)     
print("Flags : %s" %flag)

attack_types[label_index(data_as_list[1][41])]



data_uniq = np.empty(shape=[number_of_lines, 42])
label_array = np.empty(number_of_lines, dtype=object)

for i in range(number_of_lines):
    label_array[i] = attack_types[label_index(data_as_list[i][41])]
print("Label Array generated ")

for i in range(number_of_lines):
    data_uniq[i][0] = data_as_list[i][0]
    data_uniq[i][1] = protocol_index(data_as_list[i][1])
    data_uniq[i][2] = service_index(data_as_list[i][2])
    data_uniq[i][3] = flag_index(data_as_list[i][3])
print("Protocol, service and flag names change with int value ")        
        
for i in range(number_of_lines):
    for j in range(4,41):
        data_uniq[i][j] = data_as_list[i][j]


for i in range(number_of_lines):
    data_uniq[i][41] = attack_types.index(label_array[i])

print("Dataset reorganize as data_uniq ... ")    

#"""Array save to csv file """
#np.savetxt(
#    'Dataset_uniq_corrected.csv', # file name
#    data_uniq,              # array to save
#    fmt='%.5f',             # formatting, 2 digits in this case
#    delimiter=',',          # column delimiter
#    newline='\n',           # new line character
#    footer='end of file',   # file footer
#    comments='# ',          # character to use for comments
#    header='Data generated by numpy')      # file header

from sklearn.utils import shuffle
data = np.empty(shape=[300000, 42], dtype = float)
data = data_uniq[:300000][:].astype(float)

data = shuffle(data)
print("Data shuffled ! ")
print(data.shape, data.dtype)


sayac = np.zeros(5, dtype=int)
for i in range(300000):
    sayac[data[i][41]] +=  1
    
for i in range(5):
    print('Size of data set for class ' + str(i) + ': ' + str(sayac[i]))

    

#new_data_uniq = np.empty(shape=[300000, 42], dtype = int)
#say1 = 0
#say2 = 0 

#for i in range(number_of_lines):
#    for j in range(42):
#        if say2 < 300000:
#            if data[i][41] == 9 :
#                if say1 < 35000:
#                    new_data_uniq[say2][j] = data[i][j]
#                    say1 += 1
#            elif data[i][41] == 11 :
#                if say2 < 300000:
#                    new_data_uniq[say2][j] = data[i][j]
#            else:
#                new_data_uniq[say2][j] = data[i][j]
#    say2 +=1
#print(new_data_uniq.shape)    



#sayac = np.zeros(5, dtype=int)
#for i in range(300000):
#    sayac[new_data_uniq[i][41]] +=  1
#    
#for i in range(5):
#    print('number of insance for class ' + str(i) + ': ' + str(sayac[i]))



"""Dataset: new_data_uniq matrix """
"""Train_set : 80% of dataset """
"""Test set : 10 % of dataset """

dataset = np.empty(shape=[300000, 42], dtype = float)
for i in range(300000):
    for j in range(42):
        dataset[i][j] = int(data[i][j])

dataset = np.float32(dataset)  

train_set = np.empty(shape=[240000, 42] ,dtype = float)
train_set = np.float32(train_set)
train_labels = np.empty(shape=[240000] ,dtype = float)
train_labels = np.float32(train_labels)
test_set = np.empty(shape=[30000, 42] ,dtype = float)
test_set = np.float32(test_set)
test_labels = np.empty(shape=[30000] ,dtype = float)
test_labels = np.float32(test_labels)
valid_set = np.empty(shape=[30000, 42] ,dtype = float)
valid_set = np.float32(valid_set)
valid_labels = np.empty(shape=[30000] ,dtype = float)
valid_labels = np.float32(valid_labels)
print("Done ! ")


train_set.dtype

for i in range(240000):
    for j in range(42):
        train_set[i][j] = int(dataset[i][j])
print("Train set created . . ", train_set.shape, train_set.dtype)
        
k = 0
for i in range(240000,270000):
    l = 0
    for j in range(42):
        test_set[k][l] = int(dataset[i][j])
        l +=1
    k+=1    
print("Test set created . . ", test_set.shape, test_set.dtype)
    
k = 0
for i in range(270000,300000):
    l = 0
    for j in range(42):
        valid_set[k][l] = int(dataset[i][j])
        l +=1
    k +=1
print("Validation set created . . ", valid_set.shape, valid_set.dtype)
   
print("Done ! ")



for i in range(240000):
    train_labels[i] = dataset[i][41]
print("training labels array created :   ", train_labels.shape, train_labels.dtype) 

k=0    
for i in range(240000,270000):
    test_labels[k] = dataset[i][41]    
    k += 1
print("test labels array created :  ", test_labels.shape, test_labels.dtype)

m=0    
for i in range(270000,300000):
    valid_labels[m] = dataset[i][41]
    m += 1
print("validation labels array created :  ", valid_labels.shape, valid_labels.dtype)

print("Done ! ")



sayac = np.zeros(5, dtype=int)
for i in range(300000):
    sayac[dataset[i][41]] +=  1
    
for i in range(5):
    print('number of insance for class ' + str(i) + ': ' + str(sayac[i]))









print('Full dataset tensor:', dataset.shape)

"""Mean & Standart deviation of columns """
print("Means of each columns  : ")
print(dataset.mean(0))
print("Standart deviations of each columns : ")
print(dataset.std(0))




train_size = len(train_set)
test_size = len(test_set)
valid_size = len(valid_set)


print('Training:', train_set.shape)
print('Validation:', valid_set.shape)
print('Testing:', test_set.shape)






#lets use md5 hasing to measure the overlap between something
from hashlib import md5
#prepare image hashes
get_ipython().magic('time set_valid_set = set([ md5(x).hexdigest() for x in valid_set])')
get_ipython().magic('time set_test_set = set([ md5(x).hexdigest() for x in test_set])')
get_ipython().magic('time set_train_set = set([ md5(x).hexdigest() for x in train_set])')

#measure overlaps and print them
overlap_test_valid = set_test_set - set_valid_set
print('overlap test valid: ' + str(len(overlap_test_valid)))

overlap_train_valid = set_train_set - set_valid_set
print ('overlap train valid: ' + str(len(overlap_train_valid)))

overlap_train_test = set_train_set - set_test_set
print ('overlap train test: ' + str(len(overlap_train_test)))
print('done')




data.dtype

#en başta tanımladığımız LogisticRegression u kullanacağız
logReg = LogisticRegression();

fittedmodel = logReg.fit(train_set,train_labels)         
test_score = logReg.score(test_set,test_labels)

print(test_score)
print("Done !")



biases.dtype

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000
num_labels = 5
data_size = 42 
graph = tf.Graph()
with graph.as_default():

  # Input data.
  # Load the training, validation and test data into constants that are
  # attached to the graph.
  tf_train_dataset = tf.constant(train_set[:train_subset, :])
  tf_train_labels = tf.constant(train_labels[:train_subset])
  tf_valid_dataset = tf.constant(valid_set)
  tf_test_dataset = tf.constant(test_set)
  beta_regul = tf.placeholder(tf.float32)

  
  # Variables.
  # These are the parameters that we are going to be training. The weight
  # matrix will be initialized using random values following a (truncated)
  # normal distribution. The biases get initialized to zero.
  weights = tf.Variable(
    tf.truncated_normal([data_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  # We multiply the inputs with the weight matrix, and add biases. We compute
  # the softmax and cross-entropy (it's one operation in TensorFlow, because
  # it's very common, and it can be optimized). We take the average of this
  # cross-entropy across all training examples: that's our loss.
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + beta_regul * tf.nn.l2_loss(weights)
  
  # Optimizer.
  # We are going to find the minimum of this loss using gradient descent.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  # These are not part of training, but merely here so that we can report
  # accuracy figures as we train.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(
    tf.matmul(tf_valid_dataset, weights) + biases)
  test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

num_steps = 3001

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    feed_dict = {beta_regul : 1e-3}
    _, l, predictions = session.run([optimizer, loss, train_prediction],feed_dict=feed_dict)
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, train_labels[:train_subset, :]))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

batch_size = 128
num_hidden_nodes = 1024

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  beta_regul = tf.placeholder(tf.float32)
# Variables.
  weights1 = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
  biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
  weights2 = tf.Variable(
    tf.truncated_normal([num_hidden_nodes, num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
  
  # Training computation.
  lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
  logits = tf.matmul(lay1_train, weights2) + biases2
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + \
      beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
 # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits2)
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1), weights2) + biases2) 
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1), weights2) + biases2)

batch_size = 128
num_hidden_nodes = 1024
graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  beta_regul = tf.placeholder(tf.float32)
  # Variables.
  weights = tf.Variable(
    tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
  biases = tf.Variable(tf.zeros([num_hidden_nodes]))
  weights2 = tf.Variable(
    tf.truncated_normal([num_hidden_nodes, 10]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
    
  
  # Training computation.
  lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights) + biases)
  logits = tf.matmul(lay1_train, weights2) + biases2
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + \
      beta_regul * (tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights2))
  
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits2)
  valid_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_valid_dataset, weights) + biases), weights2) + biases2) 
  test_prediction = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(tf_test_dataset, weights) + biases), weights2) + biases2)









num_steps = 801

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  # This is a one-time operation which ensures the parameters get initialized as
  # we described in the graph: random weights for the matrix, zeros for the
  # biases. 
  tf.initialize_all_variables().run()
  print('Initialized')
  for step in range(num_steps):
    # Run the computations. We tell .run() that we want to run the optimizer,
    # and get the loss value and the training predictions returned as numpy
    # arrays.
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, train_labels[:train_subset, :]))
      # Calling .eval() on valid_prediction is basically like calling run(), but
      # just to get that one numpy array. Note that it recomputes all its graph
      # dependencies.
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))



num_steps = 3001

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))











