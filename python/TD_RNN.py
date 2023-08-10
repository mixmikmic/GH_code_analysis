import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.font_manager import FontProperties
get_ipython().magic('matplotlib inline')

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras import optimizers
import utils

DATA_PATH = ''
test_fname_start = 'test_linear'
train_fname_start = 'train_linear'
no_files = 1
train_X, train_y_true = utils.load_data(DATA_PATH, train_fname_start, no_files)
test_X, test_y_true = utils.load_data(DATA_PATH, test_fname_start, no_files)
print(train_X.shape, test_X.shape, train_y_true.shape, test_y_true.shape)

font0 = FontProperties();
font1 = font0.copy();
font1.set_size('xx-large');
font1.set_weight('bold');

fig = plt.figure(figsize=(30,20));
cmap = colors.ListedColormap(['white', 'black']);
#rect = l,b,w,h
rect1 = 0.2, 0.1, 0.1, 0.2
rect2 = 0.4, 0.1, 0.3, 0.2
start = 2*3 
ax1= fig.add_axes(rect1);
ax2 = fig.add_axes(rect2);
im = test_X[start,:,:].copy()
ax1.imshow(im.transpose(),origin='lower', cmap=cmap, interpolation = 'none',aspect='auto');
ax1.set_title('Example of noise image',fontproperties=font1);
ax1.set_xlabel('non-dim time',fontproperties=font1);
ax1.set_ylabel('non-dim range',fontproperties=font1);
ims = test_X[start:start+3,:,:].copy()
im = np.reshape(ims, (ims.shape[0]*ims.shape[1],ims.shape[2]));
ax2.imshow(im.transpose(),origin='lower', cmap=cmap, interpolation = 'none',aspect='auto');
ax2.set_title('Example of three stacked images: noise, noise+track, noise+track',fontproperties=font1);
ax2.set_xlabel('non-dim time',fontproperties=font1);
ax2.set_ylabel('non-dim range',fontproperties=font1);
ax2.set_xlim(0,30);
ax2.set_ylim(0.30);
for i in range(0,30,10):
    ax2.plot([i, i],[0, 30],'r-');

fig = plt.figure(figsize=(30,20));
cmap = colors.ListedColormap(['white', 'black']);
#rect = l,b,w,h
rect1 = 0.2, 0.1, 0.1, 0.2
rect2 = 0.22, 0.11, 0.1, 0.2
rect3 = 0.25, 0.12, 0.1, 0.2

ax1= fig.add_axes(rect3);
im = test_X[start+2,:,:].copy()
ax1.imshow(im.transpose(),origin='lower', cmap=cmap, interpolation = 'none',aspect='auto');
ax2= fig.add_axes(rect2);
im = test_X[start+1,:,:].copy()
ax2.imshow(im.transpose(),origin='lower', cmap=cmap, interpolation = 'none',aspect='auto');
ax3= fig.add_axes(rect1);
im = test_X[start,:,:].copy()
ax3.imshow(im.transpose(),origin='lower', cmap=cmap, interpolation = 'none',aspect='auto');
ax3.set_xlabel('non-dim time',fontproperties=font1);
ax3.set_ylabel('non-dim range',fontproperties=font1);

keras_model_load = True # False, True
batch_size = 3

if keras_model_load:
    model_name = 'keras_3k_dat_linmodel'
    model_lin = utils.load_keras_model(model_name)
else:
    np.random.seed(17)
    input_shape = (train_X.shape[1],train_X.shape[2])
    hidden_size = 16
    model_lin = Sequential()
    model_lin.add(LSTM(input_shape=input_shape, output_dim=hidden_size, return_sequences=True))
    model_lin.add(Dense(hidden_size))
    model_lin.add(Activation('relu'))
    model_lin.add(Dense(output_dim=1, activation="relu"))
    optimizer = optimizers.Adam(clipnorm=2)
    model_lin.compile(optimizer=optimizer, loss='binary_crossentropy')
    model_lin.summary()

if not keras_model_load:
    y3D = utils.track_y_3D(train_y_true, n = dxn)
    model_lin.fit(train_X, y3D, epochs = 100, batch_size = batch_size, verbose = 1, shuffle=True)

Y_estim_train = model_lin.predict(train_X, batch_size = batch_size)
Y_estim_test = model_lin.predict(test_X, batch_size = batch_size)
print(Y_estim_train.shape, Y_estim_test.shape)

Y_estim_train=Y_estim_train.sum(axis=1)/Y_estim_train.shape[1]
Y_estim_test=Y_estim_test.sum(axis=1)/Y_estim_test.shape[1]

Y_estim_test[Y_estim_test < 0.5]=0
Y_estim_test[Y_estim_test >= 0.5]=1
Y_estim_train[Y_estim_train < 0.5]=0
Y_estim_train[Y_estim_train >= 0.5]=1

row1_train = 60
row2_train = 90
row1_test = 100
row2_test = 150
dxn = 10

utils.plot_results(test_y_true, Y_estim_test, train_X, train_y_true, Y_estim_train, test_X,
                 dxn, row1_train, row2_train, row1_test, row2_test, N_plots = 7)

utils.roc_dat(Y_estim_test, test_y_true, 0.5)

utils.roc_dat(Y_estim_train, train_y_true, 0.5)



