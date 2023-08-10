import theano, theano.tensor as T
import numpy as np, matplotlib.pyplot as plt
import os, time, random, sys
notebook_start_time = time.time()

import lasagne
from lasagne.layers.normalization import BatchNormLayer
from lasagne.layers import Conv2DLayer, InputLayer, ConcatLayer, DenseLayer, Pool2DLayer, FlattenLayer

print "theano",theano.version.full_version
print "lasagne",lasagne.__version__

#Set seed for random numbers:
np.random.seed(1234)
lasagne.random.set_rng(np.random.RandomState(1234))

if sys.version_info[0] == 2:
    from urllib import urlretrieve
else:
    from urllib.request import urlretrieve

def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
    print("Downloading %s" % filename)
    urlretrieve(source + filename, filename)

import gzip
def load_mnist_images(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    data = data.reshape(-1, 1, 28, 28).transpose(0,1,3,2)
    data = np.asarray([np.rot90(np.fliplr(x[0])) for x in data])
    data = data.reshape(-1, 1, 28, 28)
    return data / np.float32(255)

def load_mnist_labels(filename):
    if not os.path.exists(filename):
        download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

x_train = load_mnist_images('train-images-idx3-ubyte.gz')
t_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
x_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
t_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
x_train, x_val = x_train[:-10000], x_train[-10000:]
t_train, t_val = t_train[:-10000], t_train[-10000:]

print "x_train", x_train.shape
print "t_train", t_train.shape
print "x_test", x_test.shape
print "t_test", t_test.shape



get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 5)
plt.imshow(x_train[0][0],interpolation='none', cmap='gray');







input_var = T.tensor4('inputs')

input_shape = (None, 1, 28, 28)
net = InputLayer(shape=input_shape, input_var=input_var)


net = BatchNormLayer(Conv2DLayer(net, num_filters=20,
        filter_size=5,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(gain='relu')))
print net.output_shape


net = BatchNormLayer(Conv2DLayer(net, num_filters=20,
        filter_size=10,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(gain='relu')))
print net.output_shape


net = FlattenLayer(net)  
net = DenseLayer(net, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)
print net.output_shape

output_shape = lasagne.layers.get_output_shape(net)
print "input_shape:",input_shape,"-> output_shape:",output_shape
sys.stdout.flush()



#Output functions
output = lasagne.layers.get_output(net)
y = T.argmax(output, axis=1)
f_predict = theano.function([input_var], y)
print "DONE building output functions"

#Training function
target_var = T.ivector('target')
target_one_hot = T.extra_ops.to_one_hot(target_var, 10)
loss = lasagne.objectives.categorical_crossentropy(output, target_one_hot)
loss = loss.mean()

#create lr variable so we can adjust the learning rate without compiling
lr = theano.shared(np.array(0.001, dtype=theano.config.floatX))

params = lasagne.layers.get_all_params(net, trainable=True)
updates = lasagne.updates.adam(loss, params, learning_rate=lr)
#updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=lr, momentum=0.9)

f_train = theano.function([input_var, target_var], loss, updates=updates)
print "DONE building training functions"



#test to seee it the train function works
f_train(x_train[:10],t_train[:10])



#Test out a prediction
print "predicted", f_predict(x_train[:10])
print "true val ",t_train[:10]



# train model
batch_size = 2**9
print "batch_size:",batch_size

stats = []
for i in range(0,20):
    start_time = time.time()
    trainerror = []

    #Shuffle batches!
    order = range(0,len(x_train))
    random.shuffle(order)
    x_train = x_train[order]
    t_train = t_train[order]
    
    #Start batch training!
    for start in range(0, len(x_train), batch_size):
        x_batch = x_train[start:start + batch_size]
        t_batch = t_train[start:start + batch_size]
        cost = f_train(x_batch, t_batch)
        trainerror.append(cost)
        #print "batch error: %.5f" % cost
    elapsed_time = time.time() - start_time
    
    ## Calculate test error (just for logging)
    pred = np.array([])
    for start in range(0, len(x_test), batch_size):
        pred = np.append(pred,f_predict(x_test[start:start + batch_size]))
    testacc = np.mean(pred == t_test)
    
    epocherror = np.mean(trainerror)
    stats.append((testacc,epocherror))
    
    print "iteration: %d, trainerror: %.5f, accuracy: %.5f, seconds: %d" % (i, epocherror,testacc, elapsed_time)



#Test out a prediction
print "predicted", f_predict(x_train[:10])
print "true val ",t_train[:10]



statsnp = np.rollaxis(np.asarray(stats),0,2)
plt.plot(statsnp[1])

plt.plot(statsnp[0])

print "Max Testing Accuracy", max(statsnp[0])





notebook_elapsed_time = time.time() - notebook_start_time
print "Notebook took", notebook_elapsed_time/60.0, "minutes"









