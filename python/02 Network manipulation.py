"""Initialization (see "00 Basic solver usage")."""
import os
import numpy as np

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

net = caffe.Net("examples/mnist/lenet_train_test.prototxt", caffe.TRAIN)

print("Network layers:")
for name, layer in zip(net._layer_names, net.layers):
    print("{:<7}: {:17s}({} blobs)".format(name, layer.type, len(layer.blobs)))

len(net.params["ip1"])

print("Blobs:")
for name, blob in net.blobs.iteritems():
    print("{:<5}:  {}".format(name, blob.data.shape))

print("Blob attributes:")
[e for e in dir(net.blobs["label"]) if not e.startswith("__")]

net.top_names["mnist"]

net.top_names["ip1"]

net.bottom_names["loss"]

net.inputs # No inputs, since our input layer is of type "Data", not "Input"

net.outputs # In testing mode, we would also have 'accuracy'

batch = np.random.randn(*net.blobs["data"].shape) * 50 # normal distribution(0, 50), in the shape of the input batch
labels = np.random.randint(0, 10, net.blobs["label"].shape) # random labels

net.blobs["data"].data[...] = batch
net.blobs["label"].data[...] = labels

net.forward()

res = net.forward(start="mnist", end="conv1")

net.blobs["ip2"].data[0]

net.blobs["loss"].data

net.blob_loss_weights

net.backward()
net.layers[list(net._layer_names).index("ip2")].blobs[0].diff # Gradient for the parameters of the ip2 layer

# We need to clear the previously computed diffs from the layers, otherwise they are just added
for l in net.layers:
    for b in l.blobs:
        b.diff[...] = 0
        
d = net.blobs["ip2"].diff # Top of the ip2 layer
d[...] = 0 # Clear the diff
d[:, 0] = 1 # Optimize for each element of the batch, for class 1 (indexes are 0-based)
net.backward(start="ip2") # Start the backpropagation at the ip2 layer, working down
net.layers[list(net._layer_names).index("ip2")].blobs[0].diff

lr = 0.01
for l in net.layers:
    for b in l.blobs:
        b.data[...] -= lr * b.diff

