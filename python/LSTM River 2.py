import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
# Get some time series data
df = pd.read_csv("height2.csv")
df = df[['height', 'temp', 'rainfallh']]
df.dropna()

# This function reas the CSV and gets the necessary rows
def read_csv(path):
    df = pd.read_csv(path)
    df = df[['height', 'temp', 'rainfallh']]
    df.dropna()
    #X_test, actual = get_split(df)
    # Save it as a list
    return format_data(df)


def format_data(df):
    # According to the advice in the post located at 
    # http://stackoverflow.com/questions/39674713/neural-network-lstm-keras
    height2, predictors = get_split(df)
    df['single_input_vector'] = predictors.apply(tuple, axis=1).apply(list)
    # Double-encapsulate list so that you can sum it in the next step and keep time steps as separate elements
    df['single_input_vector'] = df.single_input_vector.apply(lambda x: [list(x)])
    df['cumulative_input_vectors'] = df.single_input_vector.cumsum()
    max_sequence_length = df.cumulative_input_vectors.apply(len).max()
    padded_sequences = pad_sequences(df.cumulative_input_vectors.tolist(), max_sequence_length).tolist()
    df['padded_input_vectors'] = pd.Series(padded_sequences).apply(np.asarray)
    print(len(df))
    X_train_init = np.asarray(df.padded_input_vectors)
    print(X_train_init.shape)
    s = np.hstack(X_train_init)
    fin = s.reshape(len(df),len(df),2)
    y_train = np.hstack(np.asarray(height2))
    return fin, y_train


def get_split(dataset):
    #print(dataset.drop('height',1))
    return dataset['height'], dataset.drop('height',1)

X_train, y_train = read_csv('height.csv')
#print(predictors[rainfallh].head())





#df['output_vector'].head()
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras import callbacks
import numpy as np
import random
import sys

def build_model(layers):
    print(layers)
    model = Sequential()

    model.add(LSTM(
        input_shape=(None,2),
        units=20,
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        50,
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        1))
    model.add(Activation("linear"))
    
    model.compile(loss="mse", optimizer="rmsprop")
    #print("> Compilation Time : ", time.time() - start)
    return model

callback =callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
model = build_model([2, 3535,3535,2])



print(model.summary())
print(y_train)
print(model.output_shape)

#load_data("height.csv", 20 ,True)
model = build_model([2, 3535, 3535, 2])
print(model.summary())
model.fit(
    X_train,
    y_train,
    batch_size=512,
    epochs=2,
    validation_split=0.05, 
    callbacks=[callback])

# Prediction function gotten from tutorial. Located at following url
#http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
from numpy import newaxis
def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

# Save the model
from keras.models import load_model
#model.save('my_model24.h5')
load_model('my_model.h5')

X_test, y =read_csv("height3.csv")
predict_sequences_multiple(model,X_test,10,100)

#predict_sequences_multiple(model, X_test,10,50)
model.predict(X_test)

# The actual values
print(y)

from keras.models import load_model 
model = load_model('model.h5')
from tensorflow.contrib.session_bundle import exporter
import tensorflow as tf
from keras import backend as K
sess = K.get_session()
K.set_learning_phase(0)
export_path = 'saved' # where to save the exported graph
export_version = 1 # version number (integer)

saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=model.input,
                                              scores_tensor=model.output)
model_exporter.init(sess.graph.as_graph_def(),
                    default_graph_signature=signature)
model_exporter.export(export_path, tf.constant(export_version), sess)





