"""Initialization (see "00 Basic solver usage")."""
import os
import numpy as np
import math

# Silence caffe network loading output. Must be set before importing caffe
os.environ["GLOG_minloglevel"] = '2'
import caffe
CAFFE_ROOT="/caffe"
os.chdir(CAFFE_ROOT) # change the current directory to the caffe root, to help
                     # with the relative paths
USE_GPU = True
if USE_GPU:
    caffe.set_device(0)
    caffe.set_mode_gpu()
else:
    caffe.set_mode_cpu()
# For reproducible results
caffe.set_random_seed(0) # recent modification, remove if it doesn't work
np.random.seed(0)

import lmdb

def image_generator(db_path):
    """A generator that yields all the images in the database, normalized."""
    db_handle = lmdb.open(db_path, readonly=True) # We don't need to write in there
    with db_handle.begin() as db:
        cur = db.cursor() # Points to an element in the database
        for _, value in cur: # Iterate over all the images
            # Read the LMDB and transform the protobuf into a numpy array
            datum = caffe.proto.caffe_pb2.Datum()
            datum.ParseFromString(value) # String -> Protobuf
            int_x = caffe.io.datum_to_array(datum) # parse the datum into a nparray
            x = np.asfarray(int_x, dtype=np.float32) # Convert to float
            yield x - 128 # Normalize by removing the mean
            
def batch_generator(shape, db_path):
    """A generator that yield all the images in the database by batches"""
    gen = image_generator(db_path)
    res = np.zeros(shape) # Result array
    while True: # It will stop when next(gen) finishes
        for i in range(shape[0]):
            res[i] = next(gen) # Set by slices
        yield res

def test_network(test_net, db_path_test):
    # Average the accuracy and loss over the number of batches
    accuracy = 0
    loss = 0
    test_batches = 0
    input_shape = test_net.blobs["data"].data.shape
    for test_batch in batch_generator(input_shape, db_path_test):
        test_batches += 1
        # Run the forward step
        test_net.blobs["data"].data[...] = test_batch
        test_net.forward()
        # Collect the outputs
        accuracy += test_net.blobs["accuracy"].data
        loss += test_net.blobs["loss"].data
    return (accuracy / test_batches, loss / test_batches)

net_path = "examples/mnist/lenet_train_test.prototxt"
net = caffe.Net(net_path, caffe.TRAIN)
test_net = caffe.Net(net_path, caffe.TEST) # Testing version
net.share_with(test_net) # Share the weights between the two networks

num_epochs = 2 # How many times we are going to run through the database
iter_num = 0 # Current iteration number

# Training and testing examples
db_path = "examples/mnist/mnist_train_lmdb"
db_path_test = "examples/mnist/mnist_test_lmdb"

# Learning rate. We are using the lr_policy "inv", here, with no momentum
base_lr = 0.01
# Parameters with which to update the learning rate
gamma = 1e-4
power = 0.75

for epoch in range(num_epochs):
    print("Starting epoch {}".format(epoch))
    # At each epoch, iterate over the whole database
    input_shape = net.blobs["data"].data.shape
    for batch in batch_generator(input_shape, db_path):
        iter_num += 1
        
        # Run the forward step
        net.blobs["data"].data[...] = batch
        net.forward()
        
        # Clear the diffs, then run the backward step
        for name, l in zip(net._layer_names, net.layers):
            for b in l.blobs:
                b.diff[...] = net.blob_loss_weights[name]
        net.backward()
        
        # Update the learning rate, with the "inv" lr_policy
        learning_rate = base_lr * math.pow(1 + gamma * iter_num, - power)
        
        # Apply the diffs, with the learning rate
        for l in net.layers:
            for b in l.blobs:
                b.data[...] -= learning_rate * b.diff
        
        # Display the loss every 50 iterations
        if iter_num % 50 == 0:
            print("Iter {}: loss={}".format(iter_num, net.blobs["loss"].data))
            
        # Test the network every 200 iterations
        if iter_num % 200 == 0:
            print("Testing network: accuracy={}, loss={}".format(*test_network(test_net, db_path_test)))

print("Training finished after {} iterations".format(iter_num))
print("Final performance: accuracy={}, loss={}".format(*test_network(test_net, db_path_test)))
# Save the weights
net.save("examples/mnist/lenet_iter_{}.caffemodel".format(iter_num))

