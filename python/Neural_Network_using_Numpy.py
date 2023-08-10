# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

from sklearn import metrics # to calculate metrics
import numpy as np 
import time # to calculate time 

from numpy_neural_box import neuralNetwork

mnist.train.images.shape

mnist.train.labels.shape

model = neuralNetwork(784, 256, 10, 0.001)

epochs = 20
batch_size = 32

start = time.time()
for i in range(epochs):
    batches = int(len(mnist.train.images)/batch_size)
    for j in range(batches):
        model.train(mnist.train.images[j*batch_size:(j+1)*batch_size,:],                     mnist.train.labels[j*batch_size:(j+1)*batch_size,:])
        if j % 400 == 0:
            t_acc = model.query(mnist.train.images)
            t_val = model.query(mnist.validation.images)
            train_acc = metrics.accuracy_score(np.argmax(mnist.train.labels, axis=1), np.argmax(t_acc, axis=0))
            valid_acc = metrics.accuracy_score(np.argmax(mnist.validation.labels, axis=1), np.argmax(t_val, axis=0))
            print ("epoch: ", i , "Train_Accuracy: ", train_acc, "Validation_Accuracy: ", valid_acc, "time_taken:", (time.time() - start))
    print ("total time taken: ", time.time()-start)

print ("Total_time_taken_to_train:" , time.time() - start)

pred = model.query(mnist.test.images)

print(metrics.classification_report(np.argmax(mnist.test.labels, axis=1),np.argmax(pred, axis=0) ))

test_accuracy = metrics.accuracy_score(np.argmax(mnist.test.labels, axis=1), np.argmax(pred, axis=0))

print(test_accuracy)

