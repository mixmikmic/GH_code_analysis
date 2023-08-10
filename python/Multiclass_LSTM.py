import numpy as np

from get_labels import get_labels
from sklearn.externals import joblib
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import ShuffleSplit

import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, Dropout
from keras.layers import LSTM
from keras.optimizers import *;

from keras.callbacks import ModelCheckpoint, Callback

import urllib2;
import json;

'''
Helper function to send notifications on Slack via Post CURL.
'''
def notify_slack(message):
    url = 'https://hooks.slack.com/services/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';
    post_fields = json.dumps({'channel' : '#random', 'username': 'webhookbot', 'text': message});

    request = urllib2.Request(url, post_fields);
    response = urllib2.urlopen(request)
    read_val = response.read()

labels = get_labels()
notify_slack('Got labels');

labels_array = np.array([x for x in labels])
labels_reshaped = labels_array.reshape(1851243, 1, 1070)
notify_slack('Labeles Reshaped');

train_x = joblib.load("/mnt/cleaned_tfidf_reduced_420_morning")
notify_slack('Loaded Train X');

print (train_x.shape)
train_x_reshaped = train_x.reshape(1851243,1,1000)
print (train_x_reshaped.shape)

x_train, x_test, y_train, y_test = train_test_split(train_x_reshaped, labels_reshaped, test_size=0.20, random_state=1024)
notify_slack('Obtained Train Test Split');

del train_x
del labels_array
del labels_reshaped

print ("\n New shapes:")
print("x_train shape", x_train.shape)
print("x_test shape", x_test.shape)

print("y_train shape", y_train.shape)
print("y_test shape", y_test.shape)

'''
A custom Loss function for Multi-Class Prediction.
'''
def multiclass_loss(y_true, y_pred):
    EPS = 1e-5
    y_pred = K.clip(y_pred, EPS, 1 - EPS)
    return -K.mean((1 - y_true) * K.log(1 - y_pred) + y_true * K.log(y_pred))

shape = x_train.shape[2]
num_classes = 1070

'''
A Class that acts as a Callback between each Epoch, used to monitor progress.
'''
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.num = 0;

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.num = self.num + 1;
        notify_slack('Finished epoch ' + str(self.num) + ' with ' + str(logs));

'''
Returns a Model Object instantiated with a LSTM Layer and a Dense Layer, along with an Adam optimizer.
'''
def create_model(shape,num_classes):
    print (type(shape), type(num_classes))
    print (shape, num_classes) # (None,shape)
    
    model = Sequential()
    
    '''
    model.add(LSTM(output_dim=128, input_shape=(None, shape), return_sequences=True));
    '''
    
    model.add(LSTM(output_dim=32, input_shape=(None, shape), return_sequences=True));
    model.add(Dropout(rate=0.5));
    model.add(Dense(num_classes, kernel_initializer='normal', activation='sigmoid'));
    
    '''
    opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=0.1, decay=0.05);
    '''
    
    filepath = 'model_checkpoint'
    history = LossHistory()
    
    model.compile(loss=multiclass_loss, optimizer='adam', metrics=['accuracy', 'mse', 'mae']);
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    
    callbacks_list = [checkpoint, history];

    return model, callbacks_list

model, callbacks_list = create_model(shape, num_classes)
notify_slack('Obtained LSTM Model and starting training');
history = model.fit(x_train, y_train,
              batch_size=512, epochs=50,
              verbose = 1, callbacks=callbacks_list, validation_split=0.125)
notify_slack('Completed LSTM Model on 50 epochs');

from keras.models import load_model
model.save('khot_LSTM_216.h5') 
notify_slack('Saved LSTM Model');

