import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_data=pd.read_csv('data_twente_wrist.csv')
df_data.shape
#The 17 columns are: 1: Ts + 3 x 4 (sensors)+ 2x (u_id,act_id) (Numerical and categorical)

col_sensors=[u'ax', u'ay', u'az', u'lx', u'ly', u'lz', u'gx', u'gy',  u'gz', u'mx', u'my', u'mz']
X_train_test=df_data.ix[:,col_sensors].as_matrix().reshape(12600, 1, 50,12)
X_train_test.shape

y_train_test=df_data.ix[:,'act_codes'].as_matrix().reshape(12600, 1, 50,1)
y_train_test.shape

N_train=12600.0*0.8 # 70% train and 30% test
X_train=X_train_test[0:N_train,:,:,:]
X_test=X_train_test[N_train:,:,:,:]
print(X_train.shape)
print(X_test.shape)

y_train=y_train_test[0:N_train,0,0,0]
print(np.bincount(y_train))

y_test=y_train_test[N_train:,0,0,0]
print(np.bincount(y_test))

nb_classes=7

from keras.utils import np_utils
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print(Y_train.shape)
print(Y_test.shape)

X_train/=X_train.max()
X_test/=X_test.max()

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D , AveragePooling2D
from keras.utils import np_utils

# Refrom keras.models import load_model
model = load_model('mnist_cnn_keras.h5')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

y_pred=model.predict_classes(X_test)
(y_pred==y_test).sum()/float(len(y_test))

import seaborn as sns
df_results=pd.DataFrame(data=np.column_stack((y_test,y_pred)),columns=['True','Predicted'])
CT_results=pd.crosstab(df_results.True,df_results.Predicted)
CM_results=CT_results/360.0
sns.heatmap(CM_results, annot=True, fmt=".2f", linewidths=.5)

model.summary()

def mnist_cnn_keras(weights_path=None, img_rows=50, img_cols=12):
    batch_size = 128
    nb_classes = 7
    nb_epoch = 32
    # input image dimensions
    # img_rows, img_cols = 50, 12
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    kernel_size = (3, 3)
    ###############################################################
    ###############################################################
    model = Sequential()
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(Activation('relu'))
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    return model

model = load_model('mnist_cnn_keras.h5')
from keras.utils.visualize_util import plot
plot(model, to_file='model_cnn.png',show_layer_names=False,show_shapes=True)
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))

Image(filename='prd.png') 
from sktensor import dtensor, cp_als
# Create dense tensor from numpy array
T0 = np.random.rand(3, 50,4 )
T = dtensor(T0)
P, fit, itr, exectimes = cp_als(T,5, init='random')
print(P.U[0].shape)
print(P.U[1].shape)
print(P.U[2].shape)
print(P.lmbda.shape)
T_est=P.totensor()
print(T_est.shape)
print(np.sqrt(((T-T_est)**2).sum()/(T**2).sum())*100.0)
### Option-2
#The quality of the reconstructed signal is measured as the 
# percent-root mean square distortion (PRD):
T1=P.toarray()
prd=np.sqrt(((T0-T1)**2).sum()/(T0**2).sum())*100.0
##
cmpr_ratio=100-100.0*(P.U[0].size+P.U[1].size+P.U[2].size+P.lmbda.size)/T0.size
print(cmpr_ratio)
X=np.array(T1)
X.shape

Image(filename='prd.png') 

# Create dense tensor from numpy array
def compress_tensor(T_in, n_cp):
    '''Input is a 3-way numpy.ndarray
    ex: T_in = np.random.rand(3, 50,4 )
    '''
    T = dtensor(T_in)
    P, fit, itr, exectimes = cp_als(T,n_cp, init='random')
    #The quality of the reconstructed signal is measured as the 
    # percent-root mean square distortion (PRD):
    T1=P.toarray()
    prd=np.sqrt(((T_in-T1)**2).sum()/(T_in**2).sum())*100.0
    cmpr_ratio=100-100.0*(P.U[0].size+P.U[1].size+P.U[2].size+P.lmbda.size)/T_in.size
    T_out=np.array(T1)
    return T_out,prd,cmpr_ratio

X_4Dcuboid=df_12.as_matrix().reshape(12600,50,4,3)
t=0
tensor_t=X_4Dcuboid[t,:,:,:] # Size: (50L, 4L, 3L)
tensor_t.shape

tensor_t[0,:,:]

T_out,prd,cmpr_ratio=compress_tensor(tensor_t,5)
print(prd)
cmpr_ratio

df_compr=pd.DataFrame(T_out[:,0,:],columns=['ax-c','ay-c','az-c'])
df_compare = pd.concat([df_compr, df_12.ix[0:50,['ax','ay','az']]], axis=1)

df_compr=pd.DataFrame(T_out[:,0,:],columns=['ax-c','ay-c','az-c'])
df_compare = pd.concat([df_compr, df_12.ix[0:50,['ax','ay','az']]], axis=1)
fig=plt.figure(figsize=(5, 10), dpi=80, facecolor='k', edgecolor='k')
ax=df_compare.plot()
ax.set_xlabel('Time indices')
ax.set_ylabel('Accelerometer readings')
ax.set_title('')
ax.grid(False)
ax.set_title("Compression Ratio: %.2f | Percent-root mean square distortion(PRD): %.2f  " % (cmpr_ratio,prd))
ax.spines['left'].set_color('red')
ax.set_axis_bgcolor('white')
ax.set_axis_on()
plt.savefig('acc_compare.pdf', format='pdf',dpi=100)

X_4Dcuboid=df_12.as_matrix().reshape(12600,50,4,3)
t=0
tensor_t=X_4Dcuboid[t,:,:,:] # Size: (50L, 4L, 3L)
tensor_t.shape
print(X_4Dcuboid.shape)

n_cp=5
cr_vec=np.zeros((12600,1),dtype='float')
prd_vec=np.zeros((12600,1),dtype='float')
first_time=1
for t_ind in range(12600):
    if first_time:
        first_time=0
        tensor_t=X_4Dcuboid[t_ind,:,:,:]
        T_out,prd,cmpr_ratio=compress_tensor(tensor_t,n_cp)
        df_12_compress=pd.DataFrame(data=T_out.reshape((50,12)),columns=[u'ax', u'ay', u'az', u'lx', u'ly', u'lz', u'gx', u'gy',  u'gz', u'mx', u'my', u'mz']) 
        prd_vec[t_ind]=prd
        cr_vec[t_ind]=cmpr_ratio
        continue
    tensor_t=X_4Dcuboid[t_ind,:,:,:]
    T_out,prd,cmpr_ratio=compress_tensor(tensor_t,5)
    df_12_compress_t=pd.DataFrame(data=T_out.reshape((50,12)),columns=[u'ax', u'ay', u'az', u'lx', u'ly', u'lz', u'gx', u'gy',  u'gz', u'mx', u'my', u'mz'])
    df_12_compress=df_12_compress.append(df_12_compress_t,ignore_index=True)
    prd_vec[t_ind]=prd
    cr_vec[t_ind]=cmpr_ratio
df_12_compress.to_csv('df_12_compress.csv',index=False)

from scipy import stats

ax = sns.distplot(prd_vec, bins=20, kde=False, fit=stats.norm);
# Get the fitted parameters used by sns
(mu, sigma) = stats.norm.fit(prd_vec)
print("mu={0}, sigma={1}".format(mu, sigma))

# Legend and labels 
plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma)])
plt.ylabel('Frequency')

# Cross-check this is indeed the case - should be overlaid over black curve
x_dummy = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)
ax.plot(x_dummy, stats.norm.pdf(x_dummy, mu, sigma))
plt.legend(["Normal dist. fit ($\mu=${0:.3f}, $\sigma=${1:.2f})".format(mu, sigma),
           "cross-check"])
ax.set_xlabel('Percent-root mean square distortion(PRD)')
ax.grid(False)
ax.set_title("Compression Ratio: %.2f | $n_{cp}=5$  " % (cmpr_ratio))
ax.spines['left'].set_color('red')
ax.set_axis_bgcolor('0.95')
ax.set_axis_on()
ax.set_xlim([0,20])
plt.savefig('prd_dist.pdf', format='pdf',dpi=100)

df_12_compress=pd.read_csv('df_12_compress.csv')
# df_36_compress=df_12_compress.ix[:,index_vec]
X_train_test_compress=df_12_compress.as_matrix().reshape(12600, 1, 50,12)
print(X_train_test_compress.shape)

y_train_test_compress=df_data.ix[:,'act_codes'].as_matrix().reshape(12600, 1, 50,1)
print(y_train_test_compress.shape)


N_train=12600.0*0.8 # 80% train and rest test
X_train_compress=X_train_test_compress[0:N_train,:,:,:]
X_test_compress=X_train_test_compress[N_train:,:,:,:]
print(X_train_compress.shape)
print(X_test_compress.shape)

y_train_compress=y_train_test_compress[0:N_train,0,0,0]
print(np.bincount(y_train_compress))

y_test_compress=y_train_test_compress[N_train:,0,0,0]
print(np.bincount(y_test_compress))

nb_classes=7
# convert class vectors to binary class matrices
Y_train_compress= np_utils.to_categorical(y_train_compress, nb_classes)
Y_test_compress= np_utils.to_categorical(y_test_compress, nb_classes)


X_train_compress/=X_train_compress.max()
X_test_compress/=X_test_compress.max()

from keras.models import load_model
model = load_model('mnist_cnn_keras.h5')
score = model.evaluate(X_test_compress, Y_test_compress, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model = load_model('cnn_compressed_5.h5')
score = model.evaluate(X_test_compress, Y_test_compress, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print( "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

X_4Dcuboid=df_12.as_matrix().reshape(12600,50,4,3)
t=0
tensor_t=X_4Dcuboid[t,:,:,:] # Size: (50L, 4L, 3L)
tensor_t.shape
print(X_4Dcuboid.shape)
n_cp=4
cr_vec_4=np.zeros((12600,1),dtype='float')
prd_vec_4=np.zeros((12600,1),dtype='float')
first_time=1
tic()
for t_ind in range(12600):
    if first_time:
        first_time=0
        tensor_t=X_4Dcuboid[t_ind,:,:,:]
        T_out,prd,cmpr_ratio=compress_tensor(tensor_t,n_cp)
        df_12_compress_4=pd.DataFrame(data=T_out.reshape((50,12)),columns=[u'ax', u'ay', u'az', u'lx', u'ly', u'lz', u'gx', u'gy',  u'gz', u'mx', u'my', u'mz']) 
        prd_vec_4[t_ind]=prd
        cr_vec_4[t_ind]=cmpr_ratio
        print(cmpr_ratio)
        continue
    tensor_t=X_4Dcuboid[t_ind,:,:,:]
    T_out,prd,cmpr_ratio=compress_tensor(tensor_t,n_cp)
    df_12_compress_4_t=pd.DataFrame(data=T_out.reshape((50,12)),columns=[u'ax', u'ay', u'az', u'lx', u'ly', u'lz', u'gx', u'gy',  u'gz', u'mx', u'my', u'mz'])
    df_12_compress_4=df_12_compress_4.append(df_12_compress_4_t,ignore_index=True)
    prd_vec_4[t_ind]=prd
    cr_vec_4[t_ind]=cmpr_ratio
toc()
df_12_compress_4.to_csv('df_12_compress_4.csv',index=False)



# df_12_compress_4=pd.read_csv('df_12_compress_4.csv')
X_train_test_compress_4=df_12_compress_4.as_matrix().reshape(12600, 1, 50,12)
y_train_test_compress_4=df_data.ix[:,'act_codes'].as_matrix().reshape(12600, 1, 50,1)
N_train=12600.0*0.8 # 80% train and rest test
X_train_compress_4=X_train_test_compress_4[0:N_train,:,:,:]
X_test_compress_4=X_train_test_compress_4[N_train:,:,:,:]

y_train_compress_4=y_train_test_compress_4[0:N_train,0,0,0]

y_test_compress_4=y_train_test_compress_4[N_train:,0,0,0]

nb_classes=7
# convert class vectors to binary class matrices
Y_train_compress_4= np_utils.to_categorical(y_train_compress_4, nb_classes)
Y_test_compress_4= np_utils.to_categorical(y_test_compress_4, nb_classes)


X_train_compress_4/=X_train_compress_4.max()
X_test_compress_4/=X_test_compress_4.max()

# Train: UNcompressed Test: Compressed
from keras.models import load_model
model = load_model('mnist_cnn_keras.h5')
score = model.evaluate(X_test_compress_4, Y_test_compress_4, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model = load_model('cnn_compressed_4.h5')
score = model.evaluate(X_test_compress_4, Y_test_compress_4, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model = load_model('cnn_compressed_4.h5')
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

fig=plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
ax = sns.distplot(prd_vec, bins=20, kde=False, fit=stats.norm);
# Get the fitted parameters used by sns
(mu, sigma) = stats.norm.fit(prd_vec)
print("mu={0}, sigma={1}".format(mu, sigma))

# Legend and labels 
plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma)])
plt.ylabel('Frequency')

# Cross-check this is indeed the case - should be overlaid over black curve
x_dummy = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)
ax.plot(x_dummy, stats.norm.pdf(x_dummy, mu, sigma))
plt.legend(["Normal dist. fit ($\mu=${0:.3f}, $\sigma=${1:.2f})".format(mu, sigma),
           "cross-check"])
ax.set_xlabel('Percent-root mean square distortion(PRD)')
ax.grid(False)
ax.set_title("Compression Ratio: 51.67 | $n_{cp}=5$  ")
ax.spines['left'].set_color('red')
ax.set_axis_bgcolor('0.95')
ax.set_axis_on()
ax.set_xlim([0,20])
plt.savefig('prd_dist.pdf', format='pdf',dpi=100)

plt.subplot(1,2,2)
ax = sns.distplot(prd_vec_4, bins=20, kde=False, fit=stats.norm);
# Get the fitted parameters used by sns
(mu, sigma) = stats.norm.fit(prd_vec_4)
print("mu={0}, sigma={1}".format(mu, sigma))

# Legend and labels 
plt.legend(["normal dist. fit ($\mu=${0:.2g}, $\sigma=${1:.2f})".format(mu, sigma)])
plt.ylabel('Frequency')

# Cross-check this is indeed the case - should be overlaid over black curve
x_dummy = np.linspace(stats.norm.ppf(0.01), stats.norm.ppf(0.99), 100)
ax.plot(x_dummy, stats.norm.pdf(x_dummy, mu, sigma))
plt.legend(["Normal dist. fit ($\mu=${0:.3f}, $\sigma=${1:.2f})".format(mu, sigma),
           "cross-check"])
ax.set_xlabel('Percent-root mean square distortion(PRD)')
ax.grid(False)
ax.set_title("Compression Ratio: %.2f | $n_{cp}=4$  " % (cmpr_ratio))
ax.spines['left'].set_color('red')
ax.set_axis_bgcolor('0.95')
ax.set_axis_on()
ax.set_xlim([0,20])
plt.savefig('prd_dist.pdf', format='pdf',dpi=100)

y_pred_compress_4=model.predict_classes(X_test_compress_4)
print((y_pred_compress_4==y_test_compress_4).sum()/float(len(y_test_compress_4)))
df_results_compress_4=pd.DataFrame(data=np.column_stack((y_test_compress_4,y_pred_compress_4)),columns=['True','Predicted'])
CT_results_compress_4=pd.crosstab(df_results_compress_4.True,df_results_compress_4.Predicted)
CM_results_compress_4=CT_results_compress_4/360.0
ax=sns.heatmap(CM_results_compress_4, annot=True, fmt=".2f", linewidths=.5)
ax.set_title('Confusion matrix for compressed data - $n_{cp}=4$')
plt.savefig('cmat_ncp4.pdf', format='pdf',dpi=100)

fig=plt.figure(figsize=(8,4), dpi=80)

df_compr5=df_12_compress.ix[0:50,['ax','ay','az']]*df_12.max().max()
df_compr5.columns=['$ax_{c5}$','$ay_{c5}$','$az_{c5}$']

df_compr4=df_12_compress_4.ix[0:50,['ax','ay','az']]*df_12.max().max()
df_compr4.columns=['$ax_{c4}$','$ay_{c4}$','$az_{c4}$']

df_compare = pd.concat([df_compr4,df_compr5,df_12.ix[0:50,['ax','ay','az']]], axis=1)
ax=df_compare.plot(ax=fig.gca())
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.xlabel('Time indices')
plt.ylabel('Accelerometer readings')
ax.grid(True)
#ax.set_title("Compression Ratio: %.2f | Percent-root mean square distortion(PRD): %.2f  " % (cmpr_ratio,prd))
ax.spines['left'].set_color('red')
ax.set_axis_bgcolor('0.95')
ax.set_axis_on()

plt.savefig('acc_compare.pdf', format='pdf',dpi=100)

