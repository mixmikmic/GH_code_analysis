# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

import numpy as np
import tensorflow as tf
import csv

    
        

with open('trainingSetNormalized.csv', 'rb') as f1:
    reader = csv.reader(f1)
    trainList = list(reader)
print("trainingSetNormalized.csv file opened")
    
with open('testSetNormalized.csv', 'rb') as f2:
    reader = csv.reader(f2)
    testList = list(reader)    
print("testSetNormalized.csv file opened")
print("OK!")

def file_len(adress):
    f = open(adress)
    nr_of_lines = sum(1 for line in f)
    f.close()
    return nr_of_lines

trainLen = file_len('trainingSetNormalized.csv')
print("Total length of train data : ", trainLen)

testLen = file_len('testSetNormalized.csv')
print("Total length of test data : ", testLen)

trainingSet = np.empty(shape=[trainLen, 41])
testSet = np.empty(shape=[testLen, 41]) 

for i in range(trainLen):
    for j in range(41):
        trainingSet[i][j] = trainList[i][j]
print("TrainingSet : ", trainingSet.shape)     
for i in range(trainLen):
    trainingSet[i][19] = 0

for i in range(testLen):
    for j in range(41):
        testSet[i][j] = testList[i][j] 
for i in range(testLen):
    testSet[i][19] = 0
print("TestSet : ", testSet.shape)        
#index 19 is nan because std is 0,,, so in matrix' 19th index element = 0

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 200
display_step = 1


# Network Parameters
n_input = 41
n_hidden_1 = 40 # 1st layer num features
n_hidden_2 = 35 # 2nd layer num features


# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
# Using InteractiveSession (more convenient while using Notebooks)
sess = tf.InteractiveSession()
sess.run(init)
display_step = 300
num_steps=3001
total_batch = int(trainingSet.shape[0]/batch_size)
print("Total batch = ",total_batch)
# Training cycle
for step in range(num_steps):
# Pick an offset within the training data, which has been randomized.
# Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (trainingSet.shape[0])
# Generate a minibatch.
    batch_data = trainingSet[offset:(offset + batch_size), :]
    _, c = sess.run([optimizer, cost], feed_dict={X: batch_data})
    if step % display_step == 0:
        print("Epoch:", '%04d' % (step+1),"cost=", "{:.9f}".format(c))
        print("Step Count :" ,step);
        
print("Optimization Finished!")

#encodeDecode = sess.run(y_pred, feed_dict={X: testData[:10000]})
encode_decode = sess.run(
    y_pred, feed_dict={X: testSet[:100]})

encode_decode.shape

testSet[13]

encode_decode[13]

tempCost = 0
totalCost= 0
step = 0

for i in range(encode_decode.shape[0]):
    tempCost = np.mean(pow(testSet[i] - encode_decode[i], 2))
    if step % 10 == 0 :
        print("STEP :", '%04d' % (step+1),"cost = ", "{:.9f}".format(tempCost))
    step +=1
    totalCost+=tempCost
averageCost=float(totalCost/100)
print("Total cost : ", totalCost)
print("Average cost : ", averageCost)



