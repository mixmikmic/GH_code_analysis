import mxnet as mx
import os
import urllib
def getData(fname):
    data=[]
    label=[]
    firstline=True
    for line in file(fname):
        if (firstline):
            firstline=False
            continue
        tks = line.strip().split(',')
        data.append([float(i) for i in tks[1:-2]]) # omit the first column of user ID.
        label.append(int(tks[-1]))
    return mx.nd.array(data),mx.nd.array(label)
if not os.path.exists('train.csv'):
    urllib.urlretrieve('https://www.kaggle.com/c/santander-customer-satisfaction/download/train.csv.zip', 'train.csv')
if not os.path.exists('test.csv'):
    urllib.urlretrieve('https://www.kaggle.com/c/santander-customer-satisfaction/download/test.csv.zip', 'test.csv')
X_train,y_train=getData('train.csv')

n=int(X_train.shape[0])
n_val=int(n/10)
X_val=X_train[:n_val]
y_val=y_train[:n_val]
X_train=X_train[n_val:]
y_train=y_train[n_val:]
print X_train.shape
batch_size = 100
train_iter=mx.io.NDArrayIter(X_train,y_train, batch_size, shuffle=True)
val_iter=mx.io.NDArrayIter(X_val,y_val,batch_size)

# Create a place holder variable for the input data
data = mx.sym.Variable('data')

# The first fully-connected layer
fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
# Apply relu to the output of the first fully-connnected layer
act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

# The second fully-connected layer and the according activation function
fc2  = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")

# The thrid fully-connected layer, note that the hidden size should be 2, which is the number of classes
fc3  = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=2)
# The softmax and loss layer
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

# We visualize the network structure with output size (the batch_size is ignored.)
shape = {"data" : (batch_size, 368)}
mx.viz.plot_network(symbol=mlp, shape=shape)

import logging
logging.getLogger().setLevel(logging.DEBUG)

model = mx.model.FeedForward(
    symbol = mlp,       # network structure
    num_epoch = 3,     # number of data passes for training 
    learning_rate = 0.1 # learning rate of SGD 
)
model.fit(
    X=train_iter,       # training data
    eval_data=val_iter, # validation data
    batch_end_callback = mx.callback.Speedometer(batch_size, 200) # output progress for each 1000 data batches
)

valid_acc = model.score(val_iter)
print 'Validation accuracy: %f%%' % (valid_acc *100,)
assert valid_acc > 0.95, "Low validation accuracy."

