import warnings
warnings.filterwarnings("ignore")
import datetime
import pandas as pd
# import pandas.io.data
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sys
import sompylib.sompy as SOM# from pandas import Series, DataFrame

from ipywidgets import interact, HTML, FloatSlider
import tensorflow as tf
get_ipython().magic('matplotlib inline')

# Two signals have the same frequencies but with a shift in time
N = 500
t = np.arange(N)
a = np.random.rand(4)*.61

x = np.sin(a[0]*(t)) #+ np.cos(a[1]*(t)) #+ np.cos(a[2]*(t))#+np.cos(a[3]*(t))+ .1*np.random.rand(N)
plt.plot(x)

N = 1000
t = np.arange(N)
Waves = []
for i in range(2000):
    
    a = np.random.rand(4)*.6
    x = np.sin(a[0]*(t)) #+ np.cos(a[1]*(t)) #+ np.cos(a[2]*(t))#+np.cos(a[3]*(t))+ .1*np.random.rand(N)
    Waves.append(x[:,np.newaxis])
Waves = np.asarray(Waves)
Waves.shape

import random
time_lag = 20
train_test_row=1000
train_data = []
test_data= []

for r in range(train_test_row):
    for t in range(0,Waves.shape[1]-time_lag-1):
        train_data.append(Waves[r,range(t,t+time_lag+1),:])
        
train_data = np.asarray(train_data)    
    
random.shuffle(train_data)
# train_data = np.transpose(train_data,[1,0,2]) #time,batch,inputdim

for r in range(train_test_row,train_test_row+1000):
    for t in range(0,Waves.shape[1]-time_lag-1):
        test_data.append(Waves[r,range(t,t+time_lag+1),:])
        
test_data = np.asarray(test_data)    

# random.shuffle(test_data)
# test_data = np.transpose(test_data,[1,0,2]) #time,batch,inputdim

print train_data.shape
print test_data.shape

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
from random import shuffle
import tensorflow as tf

NUM_EXAMPLES = 20000
INPUT_SIZE    = 1       # 1 bits per timestep
RNN_HIDDEN    = 20
OUTPUT_SIZE   = 1       # 1 bit per timestep
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01

tf.reset_default_graph()

test_input = test_data[:NUM_EXAMPLES,:time_lag,:]
test_output = test_data[:NUM_EXAMPLES,time_lag,:]
train_input = train_data[:NUM_EXAMPLES,:time_lag,:]
train_output = train_data[:NUM_EXAMPLES,time_lag,:]

print "test and training data loaded"



data = tf.placeholder(tf.float32, [None, time_lag,INPUT_SIZE]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

num_hidden = 24
num_layers=1
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32,)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

# prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

prediction = tf.matmul(last, weight) + bias

error = tf.reduce_sum(tf.pow(target-prediction, 2))
# error = tf.reduce_mean(error)

# cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(error)

# mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
# error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

accuracy = tf.abs(1-(target -prediction)/target)*100

print test_input.shape,test_output.shape, train_input.shape,train_output.shape

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

batch_size = 100
no_of_batches = int(len(train_input)) / batch_size
epoch = 50
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    if i%2 ==0:
        incorrect = sess.run(error,{data: inp, target: out})
        print "Epoch {} error: {}".format(i,incorrect*100)

test_preds = sess.run(prediction,{data: test_input, target: test_output})

# for i in range(1):
plt.subplot(111)
plt.plot(test_preds[:,0],test_output[:,0],'.')

fig = plt.figure(figsize=(15,5))
plt.plot(test_output[:150,0],'.-')
plt.plot(test_preds[:150,0],'or')


N = 1000
t = np.arange(N)
a = np.random.rand(4)*.6
test_wave = np.sin(a[0]*(t)) #+ np.cos(a[1]*(t)) #+ np.cos(a[2]*(t))#+np.cos(a[3]*(t))+ .1*np.random.rand(N)
test_wave_rnn = []
for t in range(0,test_wave.shape[0]-time_lag-1):
    test_wave_rnn.append(test_wave[range(t,t+time_lag+1)])
        
test_wave_rnn = np.asarray(test_wave_rnn)    
test_wave_rnn.shape

r = 0
inp = test_input[r:r+1]
inp = test_wave_rnn[:1,:time_lag][:,:,np.newaxis]
preds = []
for step in range(1,500):
    
    pred = sess.run(prediction,{data:inp})
    preds.append(pred[0])
    pred_len = len(preds)
    if pred_len<time_lag:
        x = test_wave_rnn[step:step+1,:time_lag]
        x[0,-pred_len:] = preds[-pred_len:]
        inp = np.asarray(x)[:,:,np.newaxis]
    else:
        x = np.asarray(preds[-time_lag:])
        inp = np.asarray(x)[np.newaxis,:,:]
preds = np.asarray(preds)

fig = plt.figure(figsize=(10,10))
plt.subplot(2,2,1)

plt.plot(preds[:50,0],'.-')
plt.plot(test_wave_rnn[:50,time_lag],'.-r')

# plt.plot(Waves[r:r+1,range(t+time_lag,t+time_lag+3),0].T,'.-r')
plt.grid()

from IPython.display import HTML
HTML("""
<video width="600" height="400" controls>
  <source src="files/Images/lorentz_attractor.mp4" type="video/mp4">
</video>
""")

#Code from: https://jakevdp.github.io/blog/2013/02/16/animating-the-lorentz-system-in-3d/

import numpy as np
from scipy import integrate

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import cnames
from matplotlib import animation
get_ipython().magic('matplotlib inline')
N_trajectories = 2000


#dx/dt = sigma(y-x)
#dy/dt = x(rho-z)-y
#dz/dt = xy-beta*z

def lorentz_deriv((x, y, z), t0, sigma=10., beta=8./3, rho=28.0):
    """Compute the time-derivative of a Lorentz system."""
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


# Choose random starting points, uniformly distributed from -15 to 15
np.random.seed(1)
x0 = -15 + 30 * np.random.random((N_trajectories, 3))

# Solve for the trajectories
t = np.linspace(0, 20, 2000)
x_t = np.asarray([integrate.odeint(lorentz_deriv, x0i, t)
                  for x0i in x0])

x_t.shape

Data = x_t[:,range(0,2000,6)]
Data.shape

# No regularity in the behavior 
# The effect of initial value
figure =plt.figure(figsize=(10,10))
for i in range(20):
    plt.subplot(5,4,i+1);
    plt.plot(Data[i,:,1]);
    plt.xlabel('time')
    plt.ylabel('x')
plt.tight_layout();

import random
time_lag = 1
train_test_row=1000
train_data = []
test_data= []

for r in range(train_test_row):
    for t in range(0,Data.shape[1]-time_lag-1):
        train_data.append(Data[r,range(t,t+time_lag+1),:])
        
train_data = np.asarray(train_data)    
    
random.shuffle(train_data)
# train_data = np.transpose(train_data,[1,0,2]) #time,batch,inputdim

for r in range(train_test_row,train_test_row+1000):
    for t in range(0,Data.shape[1]-time_lag-1):
        test_data.append(Data[r,range(t,t+time_lag+1),:])
        

# We don't need to shuffle it and later easily will use it for predictions        
test_data = np.asarray(test_data)    

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
from random import shuffle
import tensorflow as tf
tf.reset_default_graph()

NUM_EXAMPLES = 20000
INPUT_SIZE    = 3       # 2 bits per timestep
RNN_HIDDEN    = 20
OUTPUT_SIZE   = 3       # 1 bit per timestep
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01



test_input = test_data[:NUM_EXAMPLES,:time_lag,:]
test_output = test_data[:NUM_EXAMPLES,time_lag,:]
train_input = train_data[:NUM_EXAMPLES,:time_lag,:]
train_output = train_data[:NUM_EXAMPLES,time_lag,:]

print "test and training data loaded"



data = tf.placeholder(tf.float32, [None, time_lag,INPUT_SIZE]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

num_hidden = 24
num_layers=2
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

# prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

prediction = tf.matmul(last, weight) + bias

error = tf.reduce_sum(tf.pow(target-prediction, 2))
# error = tf.reduce_mean(error)

# cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(error)

# mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
# error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

accuracy = tf.abs(1-(target -prediction)/target)*100


print test_input.shape,test_output.shape, train_input.shape,train_output.shape

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

batch_size = 200
no_of_batches = int(len(train_input)) / batch_size
epoch = 100
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    if i%3 ==0:
        incorrect = sess.run(error,{data: inp, target: out})
        print "Epoch {} error: {}".format(i,incorrect*100)
        

test_preds = sess.run(prediction,{data: test_input, target: test_output})

ac = sess.run(accuracy,{data:test_input, target: test_output})

for i in range(3):
    plt.subplot(3,3,i+1)
    plt.plot(test_preds[:,i],test_output[:,i],'.')

fig = plt.figure(figsize=(15,5))
plt.plot(test_output[:150,0],'.-')
plt.plot(test_preds[:150,0],'or')

r = train_test_row+900
t = 80
inp = Data[r:r+1,range(t,t+time_lag),:]
preds = []
for step in range(1800):
    
    pred = sess.run(prediction,{data: inp})
    preds.append(pred[0])
    pred_len = len(preds)
    pred_len = np.minimum(pred_len,time_lag)
    x = list(inp[0,:,:])
    x = x[step+1:]+preds[-pred_len:]
    inp = np.asarray(x)[np.newaxis,:,:]
    
    
preds = np.asarray(preds)

# preds1 = preds.copy()
# preds2 = preds.copy()
# preds3 = preds.copy()

fig = plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.plot(preds[:100,0],'.-r')
plt.plot(Data[r:r+1,range(t+time_lag,t+time_lag+100),0].T)

plt.subplot(2,2,2)
plt.plot(preds[:100,1],'.-r')
plt.plot(Data[r:r+1,range(t+time_lag,t+time_lag+100),1].T)

plt.subplot(2,2,3)
plt.plot(preds[:100,2],'.-r')
plt.plot(Data[r:r+1,range(t+time_lag,t+time_lag+100),2].T)

pred_trajs = preds[np.newaxis,:,:]
# pred_trajs = np.concatenate((preds3[np.newaxis,:Data.shape[1]-(t+time_lag),:],preds2[np.newaxis,:Data.shape[1]-(t+time_lag),:],preds1[np.newaxis,:Data.shape[1]-(t+time_lag),:],preds[np.newaxis,:Data.shape[1]-(t+time_lag),:]),axis=0)

# pred_trajs = np.concatenate((Data[r:r+1,t+time_lag:,:],preds[np.newaxis,:Data.shape[1]-(t+time_lag),:]),axis=0)
pred_trajs.shape

# Set up figure & 3D axis for animation
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
ax.axis('off')
plt.set_cmap(plt.cm.YlOrRd_r)
# plt.set_cmap(plt.cm.hot)
# choose a different color for each trajectory
N_trajectories = pred_trajs.shape[0]
colors = plt.cm.jet_r(np.linspace(0, 1, N_trajectories))


# set up lines and points
lines = sum([ax.plot([], [], [], '-', c=c)
             for c in colors], [])
pts = sum([ax.plot([], [], [], 'o', c=c)
           for c in colors], [])

# prepare the axes limits
ax.set_xlim((-25, 25))
ax.set_ylim((-35, 35))
ax.set_zlim((5, 55))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
ax.view_init(30, 0)

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts

# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]
    
    
    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts

# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=500, interval=10, blit=True)

# Save as mp4. This requires mplayer or ffmpeg to be installed
anim.save('./Images/lorenz_preds.mp4', fps=15, extra_args=['-vcodec', 'libx264'],dpi=200)

plt.close()

from IPython.display import HTML
HTML("""
<video width="600" height="400" controls>
  <source src="files/Images/lorenz_preds.mp4" type="video/mp4">
</video>
""")

import glob
import os
path = './Data/Forex_10m/'
path =  './Data/Forex_hourly/'
all_pairs = []
counter = 1
for filename in glob.glob(os.path.join(path, '*.csv')):
    all_pairs.append(filename)

print len(all_pairs)

all_opens = pd.DataFrame()
for pair in all_pairs[:]:
    DF = pd.read_csv(pair,index_col=0)
    Ticker = pair.replace(path,'').replace('.csv','')
    DF[Ticker] = DF[Ticker].fillna(method='backfill',limit=1,axis=0)
    all_opens[Ticker] = DF[Ticker]
    print DF.index[0],DF.index[-1], Ticker, DF.shape

all_opens.head()

DF  = all_opens.ix[:50]
print DF.shape
DF = DF.fillna(method='backfill',limit=1,axis=0)
DF[DF.columns[1:2]].plot(logy=False,legend=False,rot=45,style='.-',grid=True)

import random
time_lag = 20
train_data = []
maxdlen = min(all_opens.shape[0],250000)
for t in range(0,maxdlen-time_lag-1):
        train_data.append(all_opens.values[:][range(t,t+time_lag+1),:])
train_data = np.asarray(train_data)

indnan = np.isnan(train_data).sum(axis=2).sum(axis=1)
ind = indnan==0
train_data = train_data[ind]
train_data.shape
print train_data.shape
NUM_EXAMPLES = 60000
test_data = train_data[NUM_EXAMPLES:]
train_data = train_data[:NUM_EXAMPLES]
print train_data.shape
print test_data.shape
random.shuffle(train_data)

pd.DataFrame(data=all_opens.columns).T

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
from random import shuffle
import tensorflow as tf
tf.reset_default_graph()



# Target = sel_cols
Target = [12]
sel_cols = [7,10,11,13,12]


NUM_EXAMPLES = 20000
INPUT_SIZE    = len(sel_cols)           
RNN_HIDDEN    = 20
OUTPUT_SIZE   = len(Target)      
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01



test_input = train_data[:NUM_EXAMPLES,:time_lag,sel_cols]
if len(Target)>1:
    test_output = train_data[:NUM_EXAMPLES,time_lag,Target]
    train_output = train_data[:NUM_EXAMPLES,time_lag,Target]
else:
    test_output = train_data[:NUM_EXAMPLES,time_lag,Target[0]:Target[0]+1]
    train_output = train_data[:NUM_EXAMPLES,time_lag,Target[0]:Target[0]+1]
train_input = train_data[:NUM_EXAMPLES,:time_lag,sel_cols]
print "test and training data loaded"



data = tf.placeholder(tf.float32, [None, time_lag,INPUT_SIZE]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

num_hidden = 24
num_layers=2
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

# prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

prediction = tf.matmul(last, weight) + bias

error = tf.reduce_sum(tf.pow(target-prediction, 2))
# error = tf.reduce_mean(error)

# cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(error)

# mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
# error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

accuracy = tf.abs(1-(target -prediction)/target)*100


print test_input.shape,test_output.shape, train_input.shape,train_output.shape

init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

batch_size = 200
no_of_batches = int(len(train_input)) / batch_size
epoch = 50
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    if i%3 ==0:
        SE = sess.run(error,{data: inp, target: out})
        print "Epoch {} error: {}".format(i,SE)
        

test_preds = sess.run(prediction,{data: test_input, target: test_output})



for i in range(len(Target)):
    plt.subplot(3,3,i+1)
    plt.plot(test_preds[:,i],test_output[:,i],'.')

fig = plt.figure(figsize=(15,5))
plt.plot(test_output[:50,0],'.-')
plt.plot(test_preds[:50,0],'.-')

diff_preds = test_input[:,-1,-1]-test_preds[:,-1]
diff_preds[diff_preds>=0]=1
# diff_preds[diff_preds==0]=0
diff_preds[diff_preds<0]=-1

diff_real = test_input[:,-1,-1]-test_output[:,-1]
diff_real[diff_real>=0]=1
# diff_real[diff_real==0]=0
diff_real[diff_real<0]=-1

diff_real

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

get_ipython().magic('matplotlib inline')
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="green" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    font = {'size'   : 8}
    plt.rc('font', **font)

from sklearn.metrics import confusion_matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(diff_real, diff_preds)

np.set_printoptions(precision=1)
class_names = [str(i) for i in np.unique(diff_real)]
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,normalize=False, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()

import random
time_lag = 5
train_data = []
maxdlen = min(all_opens.shape[0],250000)
for t in range(0,maxdlen-time_lag-1):
        train_data.append(all_opens.values[:][range(t,t+time_lag+1),:])
train_data = np.asarray(train_data)

indnan = np.isnan(train_data).sum(axis=2).sum(axis=1)
ind = indnan==0
train_data = train_data[ind]
train_data.shape
print train_data.shape
NUM_EXAMPLES = 40000
test_data = train_data[NUM_EXAMPLES:]
train_data = train_data[:NUM_EXAMPLES]
print train_data.shape
print test_data.shape
# random.shuffle(train_data)

pd.DataFrame(data=all_opens.columns).T

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import random
from random import shuffle
import tensorflow as tf
tf.reset_default_graph()

sel_cols = [7,10,12]


Target = 12
print 'predicting {}'.format(all_opens.columns[Target])

n_ahead = 1
# train_output = train_data[:,time_lag,Target].copy()
train_output = 100*np.diff(train_data[:,time_lag,Target])/train_data[n_ahead::,time_lag-1,Target]
train_output = np.around(train_output,decimals=2)

train_data[n_ahead:,time_lag,Target]= train_output

#no empty
train_data= train_data[n_ahead:]


random.shuffle(train_data)


train_input = train_data[:,:time_lag,sel_cols].copy()
train_output = train_data[:,time_lag,Target].copy()


print train_output.shape, train_input.shape






test_output = test_data[:,time_lag,Target].copy()
test_output = 100*np.diff(test_output)/test_data[n_ahead::,time_lag-1,Target]
test_data[n_ahead:,time_lag,Target]= test_output

#no empty
test_data= test_data[n_ahead:]

# No need to shuffle
test_input = test_data[:,:time_lag,sel_cols]
test_output = np.around(test_output,decimals=2)





# test_input = test_data[n_ahead:,:time_lag,sel_cols]
# test_output =100*np.diff(test_data[:,time_lag,Target],n=n_ahead)/test_data[:-n_ahead,time_lag,Target]
# test_output = np.around(test_output,decimals=2)
print test_output.shape, test_input.shape



print "test and training data loaded"

plt.plot(train_output[20000:21000])

plt.plot(test_output[20000:21000])

DF = pd.DataFrame(data=test_output)
stat = DF.describe(percentiles=[.1,.3,.5,.7,.9]).T
stat

DF = pd.DataFrame(data=train_output)
stat = DF.describe(percentiles=[.1,.3,.5,.7,.9]).T
stat

ind_10 = train_output<stat['30%'].values[:]
ind_30 = (train_output>= stat['30%'].values[:])&(train_output< stat['70%'].values[:])
# ind_50 = (train_output>= stat['30%'].values[:])&(train_output< stat['50%'].values[:])
# ind_70 = (train_output>= stat['50%'].values[:])&(train_output< stat['70%'].values[:])
# ind_70p = (train_output>= stat['70%'].values[:])&(train_output< stat['90%'].values[:])
ind_90 = (train_output>= stat['70%'].values[:])

train_output[ind_10] = 0
train_output[ind_30] = 1
# train_output[ind_50] = 2
# train_output[ind_70] = 3
# train_output[ind_70p] = 4
train_output[ind_90] = 2



ind_10 = test_output<stat['30%'].values[:]
ind_30 = (test_output>= stat['30%'].values[:])&(test_output< stat['70%'].values[:])
# ind_50 = (test_output>= stat['30%'].values[:])&(test_output< stat['50%'].values[:])
# ind_70 = (test_output>= stat['50%'].values[:])&(test_output< stat['70%'].values[:])
# ind_70p = (test_output>= stat['70%'].values[:])&(test_output< stat['90%'].values[:])
ind_90 = (test_output>= stat['70%'].values[:])

test_output[ind_10] = 0
test_output[ind_30] = 1
# test_output[ind_50] = 2
# test_output[ind_70] = 3
# test_output[ind_70p] = 4
test_output[ind_90] = 2





# test_output[test_output>= 0]=0 #Long Position
# test_output[test_output<0]=1 #Short Position
# outputs[outputs==0]=0 #Hold
with tf.Session():
    test_Label= tf.one_hot(test_output,len(np.unique(test_output))).eval()

    
    
# train_output[train_output>= 0]=0 #Long Position
# train_output[train_output<0]=1#Short Position
# # outputs[outputs==0]=0 #Hold

with tf.Session():
    train_Label= tf.one_hot(train_output,len(np.unique(train_output))).eval()

plt.hist(train_output,bins=100);

print train_output[:10]
train_Label[:10]

# test_input = inputs[NUM_EXAMPLES:]
# test_output = Labels[NUM_EXAMPLES:]
# train_input = inputs[:NUM_EXAMPLES]
# train_output = Labels[:NUM_EXAMPLES]
train_output = train_Label
test_output = test_Label
print train_input.shape, test_input.shape,train_output.shape
print "test and training data loaded"

plt.plot(train_input[10,:,0])
plt.plot(train_input[11,:,0])
plt.plot(train_input[42,:,0])
plt.plot(train_input[43,:,0])

train_output.shape

INPUT_SIZE    = len(sel_cols)      
RNN_HIDDEN    = 20
OUTPUT_SIZE   =train_output.shape[1]       
TINY          = 1e-6    # to avoid NaNs in logs
LEARNING_RATE = 0.01

tf.reset_default_graph()
data = tf.placeholder(tf.float32, [None, time_lag,INPUT_SIZE]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])

num_hidden = 20
num_layers=2
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=.8)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))

# prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

# prediction = tf.sigmoid(tf.matmul(last, weight) + bias)

# # prediction = tf.nn.xw_plus_b(last, weight, bias)
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, target))
# optimizer = tf.train.AdamOptimizer()
# minimize = optimizer.minimize(cost)


prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)

mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))




# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction,1), tf.argmax(target,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)

batch_size = 50
no_of_batches = int(len(train_input)) / batch_size
epoch = 30
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    if i%3 ==0:
        acc = sess.run(accuracy,{data: inp, target: out})
        print "Epoch {} training accuracy: {}".format(i,acc*100)
        ind = np.random.randint(0,high=test_input.shape[0],size=1000)
        acc = sess.run(accuracy,{data: test_input[:1000], target: test_output[:1000]})
        print "Epoch {} test accuracy: {}".format(i,acc*100)

howmany = 1000
test_preds = sess.run(prediction,{data: test_input[:howmany]})
np.argmax(test_output[:],axis=1).shape
acc = sess.run(accuracy,{data: test_input[:howmany], target: test_output[:howmany]})
print "Epoch {} accuracy: {}".format(i,acc*100)

test_input.shape

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

get_ipython().magic('matplotlib inline')
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    font = {'size'   : 8}
    plt.rc('font', **font)


from sklearn.metrics import confusion_matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(np.argmax(test_output[:howmany],axis=1), np.argmax(test_preds,axis=1))
# cnf_matrix = confusion_matrix(np.argmax(test_output[:],axis=1), np.zeros(len(test_preds)))

np.set_printoptions(precision=1)
class_names = [str(i) for i in range(train_output.shape[1])]
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix,normalize=True, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()

