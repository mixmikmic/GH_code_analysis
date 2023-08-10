import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np 

from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler



(x_train, y_train), (x_test, y_test) = boston_housing.load_data()





(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

# Training Parameters
learning_rate = 0.1 
training_epochs = 200
batch_size = 100

# Network Parameters  ---> Layer params
n_hidden_1 = 3840 # 1st layer number of neurons
n_hidden_2 = 100 # 2nd layer number of neurons



# define base model
def first_deep_model():
# create model
    model = Sequential()
    model.add(Dense(n_hidden_1, input_dim=13, kernel_initializer='normal', activation='relu'))
    model.add(Dense(n_hidden_2,  input_dim=13, name = "Dense_2"))
    model.add(Activation('relu', name = "Relu2"))
    model.add(Dense(1, kernel_initializer='normal'))
# Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

#model.add(Dense(n_hidden_2, input_dim=13, kernel_initializer='normal', activation='relu'))



# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=first_deep_model, nb_epoch=100, batch_size=batch_size, verbose=0)

seed = 7
np.random.seed(seed)
(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()
scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_test = scale.fit_transform(X_test)
model = KerasRegressor(build_fn=first_deep_model, nb_epoch=training_epochs, batch_size=5,verbose=0)
tensorboard = TensorBoard(log_dir='./logs')

model.fit(X_train,Y_train, callbacks=[tensorboard])
res = model.predict(X_test)

score = model.score(X_test, Y_test)
score

from sklearn.metrics import mean_squared_error
score = mean_squared_error(Y_test, model.predict(X_test))
print(score)
from sklearn.metrics import mean_absolute_error
score = mean_absolute_error(Y_test, model.predict(X_test))
print(score)

#from keras.utils import plot_model
#plot_model(model, to_file='model.png')



