from keras.layers import Input, Dense, Lambda, Layer
from keras.models import Model
from keras import regularizers
import keras
import pandas as pd
import numpy as np
from keras import backend as K
from keras import metrics
from collections import namedtuple
pd.set_option("display.max_rows",35)
get_ipython().magic('matplotlib inline')

kdd_train_2labels = pd.read_pickle("dataset/kdd_train_2labels.pkl")
kdd_test_2labels = pd.read_pickle("dataset/kdd_test_2labels.pkl")

#y_train_labels = pd.read_pickle("dataset/kdd_train_2labels_y.pkl")
#y_train_labels = pd.read_pickle("dataset/kdd_train_2labels.pkl")
#y_test_labels = pd.read_pickle("dataset/kdd_test_2labels_y.pkl")

output_columns_2labels = ['is_Attack','is_Normal']

from sklearn import model_selection as ms
from sklearn import preprocessing as pp

x_input = kdd_train_2labels.drop(output_columns_2labels, axis = 1)
y_output = kdd_train_2labels.loc[:,output_columns_2labels]

ss = pp.StandardScaler()
x_input = ss.fit_transform(x_input)

#le = pp.LabelEncoder()
#y_train = le.fit_transform(y_train_labels).reshape(-1, 1)
#y_test = le.transform(y_test_labels).reshape(-1, 1)

y_train = kdd_train_2labels.loc[:,output_columns_2labels]

x_train, x_valid, y_train, y_valid = ms.train_test_split(x_input, 
                              y_train, 
                              test_size=0.1)
#x_valid, x_test, y_valid, y_test = ms.train_test_split(x_valid, y_valid, test_size = 0.4)

x_test = kdd_test_2labels.drop(output_columns_2labels, axis = 1)
y_test = kdd_test_2labels.loc[:,output_columns_2labels]

x_test = ss.transform(x_test)

#x_train = np.hstack((x_train, y_train))
#x_valid = np.hstack((x_valid, y_valid))

#x_test = np.hstack((x_test, np.random.normal(loc = 0, scale = 0.01, size = y_test.shape)))

input_dim = 122
intermediate_dim = 122
latent_dim = 32
batch_size = 1409
epochs = 5
hidden_layers = 8

class Train:
    def train():
        Train.x = Input(shape=(input_dim,))
        
        hidden_encoder = Train.x
        for i in range(hidden_layers):
            hidden_encoder = Dense(intermediate_dim, activation='relu')(hidden_encoder)

        Train.mean_encoder = Dense(latent_dim, activation=None)(hidden_encoder)
        Train.logvar_encoder = Dense(latent_dim, activation=None)(hidden_encoder)

        def get_distrib(args):

            m_e, l_e = args

            # Sample epsilon
            epsilon = np.random.normal(loc=0.0, scale=0.05, size = (batch_size, latent_dim))

            # Sample latent variable
            z = m_e + K.exp(l_e / 2) * epsilon
            return z

        z = Lambda(get_distrib)([Train.mean_encoder, Train.logvar_encoder])

        hidden_decoder = z
        for i in range(hidden_layers):
            hidden_decoder = Dense(intermediate_dim, activation="relu")(hidden_decoder)

        Train.x_ = Dense(input_dim, activation=None)(hidden_decoder)

def vae_loss(x, x_decoded_mean):
    xent_loss = input_dim * keras.losses.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.sum(1 + Train.logvar_encoder - K.square(Train.mean_encoder) - K.exp(Train.logvar_encoder), axis=-1)
    return K.abs(K.mean(xent_loss + kl_loss))

import itertools
#features_arr = [4, 16, 32, 256, 1024]
#hidden_layers_arr = [2, 6, 10, 100]

features_arr = [2, 4,  8, 16, 32]
hidden_layers_arr = [2, 6, 10]

epoch_arr = [50]

score = namedtuple("score", ['epoch', 'no_of_features','hidden_layers','train_score', 'test_score'])
scores = []
predictions = pd.DataFrame()

for e, h, f in itertools.product(epoch_arr, hidden_layers_arr, features_arr):
    
    print(" \n Current Layer Attributes - epochs:{} hidden layers:{} features count:{}".format(e,h,f))
    latent_dim = f
    epochs = e
    hidden_layers = h

    Train.train()

    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-04, decay=0.1)
    vae_model = Model(inputs = Train.x, outputs = Train.x_ )
    vae_model.compile(optimizer = optimizer, loss = vae_loss, metrics = ['accuracy'] )

    train_size = x_train.shape[0] - x_train.shape[0]%batch_size
    valid_size = x_valid.shape[0] - x_valid.shape[0]%batch_size

    vae_model.fit(x = x_train[:train_size,:], y = x_train[:train_size,:], 
                  shuffle=True, epochs=epochs, 
                  batch_size = batch_size, 
                  validation_data = (x_test, x_test),
                  verbose = 1)
    
    score_train = vae_model.evaluate(x_valid[:valid_size,:], y = x_valid[:valid_size,:],
                               batch_size = batch_size,
                               verbose = 1)
    
    score_test = vae_model.evaluate(x_test, y = x_test,
                           batch_size = batch_size,
                           verbose = 1)
    
    scores.append(score(e,f,h,score_train[-1], score_test[-1])) #score_test[-1]))
    
    print("\n Train Acc: {}, Test Acc: {}".format(score_train[-1], 
                                                  score_test[-1])  )
    
scores = pd.DataFrame(scores)

scores.sort_values("test_score", ascending=False)

scores.to_pickle("dataset/vae_only_feature_extraction_scores.pkl")

