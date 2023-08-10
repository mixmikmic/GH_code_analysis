# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.

import numpy as np
import tensorflow as tf
import csv

    
        

with open('data_uniq.csv', 'rb') as f:
    reader = csv.reader(f)
    data_as_list = list(reader)

def file_len(adress):
  f = open(adress)
  nr_of_lines = sum(1 for line in f)
  f.close()
  return nr_of_lines

number_of_lines = file_len('data_uniq.csv')
print('Toplam satir sayisi : ', number_of_lines)

fullData = np.empty(shape=[number_of_lines, 42])

for i in range(number_of_lines):
    for j in range(42):
        fullData[i][j] = data_as_list[i][j]

sayac = np.zeros(5, dtype=int)
for i in range(number_of_lines):
    sayac[fullData[i][41]] +=  1
    
for i in range(5):
    print('Size of data set for class ' + str(i) + ': ' + str(sayac[i]))

    

meanArray = np.mean(fullData, axis=0)
stdArray = np.std(fullData, axis=0)
fullDataNormalized = np.empty(shape=[fullData.shape[0], fullData.shape[1]])
print(fullDataNormalized.shape)


for i in range(fullData.shape[0]):
    for j in range(fullData.shape[1]-1):
        fullDataNormalized[i][j] = (fullData[i][j] - meanArray[j]) / stdArray[j]

for i in range(fullData.shape[0]):
    fullDataNormalized[i][41] = fullData[i][41]

"""Array save to csv file """
np.savetxt(
    'fullDataNormalized.csv', # file name
    fullDataNormalized,    # array to save
    fmt='%.5f',             # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    comments='# ') 


print("Saved fullDataNormalized csv file  ")



trainingSet = np.empty(shape=[sayac[4], fullDataNormalized.shape[1]])
testSet = np.empty(shape=[sayac[0], fullDataNormalized.shape[1]]) # we choosed dos attack type for testing  

count = 0
tcount= 0
for i in range(fullDataNormalized.shape[0]):
    if fullDataNormalized[i][41] == 4 :
        for k in range(fullDataNormalized.shape[1]):
            trainingSet[count][k] = fullDataNormalized[i][k]
        count += 1
        
print("Training set created with normal class data")    
print(trainingSet.shape) 

for i in range(fullDataNormalized.shape[0]):
    if fullDataNormalized[i][41] == 0:
        for k in range(fullDataNormalized.shape[1]):
            testSet[tcount][k] = fullDataNormalized[i][k]
        tcount += 1
print("Test set created with normal class data")    
print(testSet.shape)      

np.savetxt(
    'trainingSetNormalized.csv', # file name
    trainingSet,    # array to save
    fmt='%.5f',             # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    comments='# ') 
print("Saved trainingSetNormalized csv file  ")

np.savetxt(
    'testSetNormalized.csv', # file name
    testSet,    # array to save
    fmt='%.5f',             # formatting, 2 digits in this case
    delimiter=',',          # column delimiter
    newline='\n',           # new line character
    comments='# ') 
print("Saved testSetNormalized csv file  ")

# Parameters
learning_rate = 0.01
training_epochs = 20
batch_size = 200
display_step = 1


# Network Parameters
n_input = 42
n_hidden_1 = 36 # 1st layer num features
n_hidden_2 = 32 # 2nd layer num features


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
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
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





