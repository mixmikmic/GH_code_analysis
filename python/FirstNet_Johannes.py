get_ipython().magic('ls -rtlh')

import pickle
import gzip
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import time

import lasagne
import nolearn.lasagne
lasagne, nolearn.lasagne

start = time.time()
#Inselspital
with gzip.open('GBM_tumors.pickle.gz') as f:
    Names,X,y = pickle.load(f)
print ("1) Loaded data in " + str(time.time() - start))
Names = np.asarray(Names)
print ("   " + str(X.shape) + " y " + str(y.shape) + " Names " + str(len(Names)))


gbm_test = 408

y_train = y[:gbm_test]
y_test  = y[gbm_test:]

X_train = X[:gbm_test,:,:,:]
X_test  = X[gbm_test:,:,:]
Names_test = Names[gbm_test:]
len(Names_test), X_test.shape, y_test.shape
start = time.time()
#Inselspital
with gzip.open('META_tumors.pickle.gz') as f:
    Names,X,y = pickle.load(f)
print ("2) Loaded data in " + str(time.time() - start))
Names = np.asarray(Names)
print ("   " + str(X.shape) + " y " + str(y.shape) + " Names " + str(len(Names)))



meta_test = 340

y_train = np.append(y_train, y[:meta_test])
y_test = np.append(y_test, y[meta_test:])

X_train = np.concatenate((X_train, X[:meta_test,:,:,:]), axis=0)
X_test = np.concatenate((X_test, X[meta_test:,:,:,:]), axis=0)

Names_test = np.append(Names_test,Names[meta_test:])

Y_test = np.asarray(y_test - 1,dtype='int32')
X_test.shape, y_test.shape, len(Names_test)

perm = np.random.permutation(len(y_train))
X_train = X_train[perm,:,:,:]
y_train = y_train[perm]

y = np.asarray(y_train - 2,dtype='int32')
X = np.asanyarray(X_train,dtype='float32')

print(str(np.shape(X)) + " " + str(np.shape(y)))
print(str(np.shape(X_test)) + " " + str(np.shape(Y_test)))
y[0:20], Y_test[0:10], Y_test[-10:]

## Training
Xmean = X_train.mean(axis = 0)
XStd = np.sqrt(X_train.var(axis=0))
X = (X-Xmean)/(XStd + 0.01)

## Testing
Xmean = X_test.mean(axis = 0)
XStd = np.sqrt(X_test.var(axis=0))
X_test = (X_test-Xmean)/(XStd + 0.01)

X = np.asarray(X, dtype='float32')
X_test = np.asarray(X_test, dtype='float32')

np.mean(X[:,0,2,0])

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import numpy as np
from skimage import transform as tf

#rots = np.deg2rad(np.asarray((90,180,0,5,-5,10,-10)))
rots = np.deg2rad(range(0,359))


def manipulateTrainingData(Xb):
    retX = np.zeros((Xb.shape[0], Xb.shape[1], Xb.shape[2], Xb.shape[3]), dtype='float32')
    for i in range(len(Xb)):
        rot = rots[np.random.randint(0, len(rots))]
        tf_rotate = tf.SimilarityTransform(rotation=rot)
        shift_y, shift_x = np.array((X.shape[2], X.shape[3])) / 2.
        tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])
        tform_rot = (tf_shift + (tf_rotate + tf_shift_inv))

        ## TODO add the transformations
        scale = np.random.uniform(0.9,1.10)
        d = tf.SimilarityTransform(scale=scale, translation=(np.random.randint(5),np.random.randint(5)))
        tform_other = (tform_rot + d)
        
        c = 0
        retX[i,c,:,:] = tf.warp(Xb[i,c,:,:], tform_other, preserve_range = True) # "Float Images" are only allowed to have values between -1 and 1
    return retX

Xb = np.copy(X[0:100,:,:,:])
Xb = manipulateTrainingData(Xb)

fig = plt.figure(figsize=(10,10))
for i in range(18):
    a=fig.add_subplot(6,6,2*i+1,xticks=[], yticks=[])
    plt.imshow(X[i,0,:,:], cmap=plt.get_cmap('cubehelix'))
    a=fig.add_subplot(6,6,2*i+2,xticks=[], yticks=[])
    plt.imshow(Xb[i,0,:,:], cmap=plt.get_cmap('cubehelix'))
    print('before {0} after {1}'.format(np.mean(X[i,0,:,:]), np.mean(Xb[i,0,:,:])))

PIXELS = 48
COLORS =  1 #The number of layers of the input image 1 (BW), 3 rgb, 5 for HCS data

from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet

conv = (3,3)
stride = (1,1)
pool = (2,2)

num1 = 32
num2 = 64
num3 = 128
num4 = 256
num5 = 256

net_bigger = NeuralNet(
    # Geometry of the network
    layers=[
        ('input', layers.InputLayer),
        
        ('conv1', layers.Conv2DLayer),
        ('conv11', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        
        ('conv2', layers.Conv2DLayer),
        ('conv22', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        
        ('conv3', layers.Conv2DLayer),
        ('conv33', layers.Conv2DLayer),
#        ('pool3', layers.MaxPool2DLayer),
        
#         ('conv4', layers.Conv2DLayer),
#         ('conv44', layers.Conv2DLayer),
#         ('pool4', layers.MaxPool2DLayer),
        
#         ('conv5', layers.Conv2DLayer),
#         ('conv55', layers.Conv2DLayer),
#         ('pool5', layers.MaxPool2DLayer),
                
              
        ('hidden1', layers.DenseLayer),
        ('dropout1', layers.DropoutLayer),
        
        ('hidden2', layers.DenseLayer),
        ('dropout2', layers.DropoutLayer),
        
        ('hidden3', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer),
        
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, COLORS, PIXELS, PIXELS), #None in the first axis indicates that the batch size can be set later
    
    conv1_num_filters=num1, conv1_filter_size=conv, conv1_stride=stride, 
    conv11_num_filters=num1, conv11_filter_size=conv, conv11_stride=stride,
    pool1_pool_size=pool, #pool_size used to be called ds in old versions of lasagne
    
    conv2_num_filters=num2, conv2_filter_size=conv, conv2_stride=stride, 
    conv22_num_filters=num2, conv22_filter_size=conv, conv22_stride=stride,
    pool2_pool_size=pool, #pool_size used to be called ds in old versions of lasagne
    
    conv3_num_filters=num3, conv3_filter_size=conv, conv3_stride=stride, 
    conv33_num_filters=num3, conv33_filter_size=conv, conv33_stride=stride,
#    pool3_pool_size=pool, #pool_size used to be called ds in old versions of lasagne
    
#     conv4_num_filters=num4, conv4_filter_size=conv, conv4_stride=stride, 
#     conv44_num_filters=num4, conv44_filter_size=conv, conv44_stride=stride,
#     pool4_pool_size=pool, #pool_size used to be called ds in old versions of lasagne
    
#     conv5_num_filters=num5, conv5_filter_size=conv, conv5_stride=stride, 
#     conv55_num_filters=num5, conv55_filter_size=conv, conv55_stride=stride,
#     pool5_pool_size=pool, #pool_size used to be called ds in old versions of lasagne
    
    hidden1_num_units=200,
    dropout1_p=0.3,
    
    hidden2_num_units=200,
    dropout2_p=0.3,
    
    hidden3_num_units=50,
    dropout3_p=0.3,
    
    output_num_units=2, output_nonlinearity=nonlinearities.softmax,
    # learning rate parameters
    update_learning_rate=0.01,
    update_momentum=0.9,
    regression=False,
    # We only train for 10 epochs
    max_epochs=200,
    verbose=1,

    # Training test-set split
    eval_size = 0.2
    )

y[0:10], X[0,0,:,:], X[1,0,:,:], np.std(X[:,0,33,0]), np.min(y), np.max(y)

from nolearn.lasagne import BatchIterator

class SimpleBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(SimpleBatchIterator, self).transform(Xb, yb)
        #return Xb[:,:,:,::-1], yb #<--- Here we do the flipping of the images
        return manipulateTrainingData(Xb), yb
    
# Setting the new batch iterator
net1Aug = net_bigger
net1Aug.max_epochs = 200
net1Aug.batch_iterator_train = SimpleBatchIterator(50)
net1Aug = net1Aug.fit(X,y)

get_ipython().magic('matplotlib inline')
import pandas as pd
dfNoAug = pd.DataFrame(net1Aug.train_history_)
dfNoAug[['train_loss','valid_loss','valid_accuracy']].plot(title='With Augmentation', ylim=(0,1))

X_test = np.asanyarray(X_test,dtype='float32')
pred = net1Aug.predict(X_test)
np.sum(pred  == np.asanyarray(y_test - 2, dtype='int32')) / (1.0*len(y_test))

pred

import pandas as pd
df = pd.DataFrame(columns = ('Name','y_True','y_pred','p_0','p_1'))
probs = net1Aug.predict_proba(X_test)

for i, name in enumerate(Names_test):
    df.loc[i,] = (Names_test[i].split('=')[0], y_test[i]-2, pred[i], probs[i][0], probs[i][1])

df.head()
df.to_csv('FirstNetJohannes_pred.txt')

# http://bconnelly.net/2013/10/summarizing-data-in-python-with-pandas/
res = df.groupby('Name')
np.max((res['y_True'].max() - res['y_True'].min())) #All Names have the same group (Sanity Check should be 0)

y_pred_cond = res['p_1'].aggregate(np.average)
y_true_cond = res['y_True'].aggregate(np.average)
y_pred_cond, y_true_cond

y_true_cond.values, y_pred_cond.values

from sklearn import metrics
fpr, tpr, thresholds = metrics.roc_curve(y_true_cond, y_pred_cond)
plt.plot(fpr, tpr)
plt.ylim(0.0,1.0)
metrics.roc_auc_score(y_true_cond, y_pred_cond), metrics.accuracy_score(y_true_cond, y_pred_cond > 0.5)

np.sum(np.asarray(y_true_cond) == np.asarray(y_pred_cond_num)) / float(len(y_pred_cond_num))

np.mean(y_pred_cond_num)

