import numpy as np
import os
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

def read_csv_files(path):
    dirpath, dirnames, filenames = list(os.walk(path))[0]    
    return [
        np.genfromtxt(dirpath + '/' + file, delimiter=',') for file in filenames 
        if os.path.splitext(file)[1] == '.csv'
    ]

DATA_FOLDER = '../data/trampoliny/'

positive_samples = read_csv_files(DATA_FOLDER + '42')
negative_samples = read_csv_files(DATA_FOLDER + 'ostatni')

print("Positive samples: %d" % len(positive_samples))
print("Negative samples: %d" % len(negative_samples))
plt.hist([len(x) for x in positive_samples])
plt.hist([len(x) for x in negative_samples])
plt.show()

plt.figure(figsize=(15,6))
plt.plot(positive_samples[0])
plt.show()

plt.figure(figsize=(15,6))
plt.plot(positive_samples[0][:,9:13])
plt.show()

plt.figure(figsize=(15,6))
plt.plot(positive_samples[0][:,0:9])
plt.show()

def normalize(*datasets):            
    all_samples = np.vstack([np.vstack(samples) for samples in datasets])    
    max_vals = np.max(all_samples, axis=0)
    min_vals = np.min(all_samples, axis=0)
    
    return [
        [(sample - min_vals) / (max_vals - min_vals) * 2 - 1 for sample in samples] 
        for samples in datasets
    ], min_vals, max_vals

(norm_positive_samples, norm_negative_samples), max_vals, min_vals = normalize(positive_samples, negative_samples)

from keras.preprocessing.sequence import pad_sequences

def pad(*datasets):
    max_length = max(sample.shape[0] for samples in datasets for sample in samples)
    return [pad_sequences(samples, maxlen=max_length, dtype=datasets[0][0].dtype) for samples in datasets]

norm_positive_samples, norm_negative_samples = pad(norm_positive_samples, norm_negative_samples)

plt.figure(figsize=(15,6))
plt.plot(norm_positive_samples[0][:,0:9])
plt.show()

training_X = np.vstack((norm_positive_samples[:,:,0:9], norm_negative_samples[:,:,0:9]))
training_Y = np.vstack((np.full((len(positive_samples), 1), 1.0), np.full((len(negative_samples), 1), 0.0)))

import random
training_set = list(zip(training_X, training_Y))
random.shuffle(training_set)

training_X, training_Y = zip(*training_set)
                             
training_X = np.array(training_X)
training_Y = np.array(training_Y)

from keras import Model
from keras.layers import LSTM, Input, Dense

inputs = Input(shape=(training_X.shape[1] * training_X.shape[2],))
x = Dense(256, activation='tanh')(inputs)
x = Dense(128, activation='tanh')(x)
x = Dense(64, activation='tanh')(x)
outputs = Dense(1, activation='sigmoid')(x)

ffn_model = Model(inputs, outputs)
ffn_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
ffn_model.summary()

ffn_training_X = training_X.reshape(training_X.shape[0], training_X.shape[1] * training_X.shape[2])

ffn_model.fit(ffn_training_X, training_Y, epochs=40, validation_split=0.1)

inputs = Input(shape=training_X.shape[1:])
x = LSTM(64, return_sequences=True, recurrent_activation='sigmoid')(inputs)
x = LSTM(64, recurrent_activation='sigmoid')(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs, outputs)
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(training_X, training_Y, epochs=30, validation_split=0.1)
model.save_weights('model_trampoline_9i.hdf')

from keras import Model
from keras.layers import LSTM, Input, Dense, Dropout

inputs = Input(shape=(None, training_X.shape[2]))
x = LSTM(64, return_sequences=True, recurrent_activation='sigmoid')(inputs)
x = LSTM(64, return_sequences=True, recurrent_activation='sigmoid')(x)
outputs = Dense(1, activation='sigmoid')(x)

cont_model = Model(inputs, outputs)
cont_model.summary()
cont_model.load_weights("model_trampoline_9i.hdf")

def show_prediction(test_case):
    print(test_case[1])

    c_prediction = cont_model.predict(np.expand_dims(test_case[0], axis=0))
    plt.figure(figsize=(15,6))
    plt.plot(test_case[0], 'silver')
    plt.plot(c_prediction[0], 'red' if test_case[1] == 0 else 'green')
    plt.show()

positive_test_sample = next(sample for sample in reversed(training_set) if sample[1] == 1)
negative_test_sample = next(sample for sample in reversed(training_set) if sample[1] == 0)

show_prediction(positive_test_sample)
show_prediction(negative_test_sample)

show_prediction((training_set[-5][0][45:], training_set[-5][1]))
show_prediction((training_set[-1][0][67:], training_set[-1][1]))

plt.figure(figsize=(25,6))
c_prediction = cont_model.predict(norm_positive_samples[:100,:,0:9])
plt.plot(np.squeeze(c_prediction).T, '#00800020')
c_prediction = cont_model.predict(norm_negative_samples[:100,:,0:9])
plt.plot(np.squeeze(c_prediction).T, '#80000020')
plt.show()

plt.figure(figsize=(25,6))
c_prediction = cont_model.predict(norm_positive_samples[:100,60:,0:9])
plt.plot(np.squeeze(c_prediction).T, '#00800020')
c_prediction = cont_model.predict(norm_negative_samples[:100,60:,0:9])
plt.plot(np.squeeze(c_prediction).T, '#80000020')
plt.show()

plt.figure(figsize=(25,6))
c_prediction = cont_model.predict(norm_positive_samples[:100,100:,0:9])
plt.plot(np.squeeze(c_prediction).T, '#00800020')
c_prediction = cont_model.predict(norm_negative_samples[:100,100:,0:9])
plt.plot(np.squeeze(c_prediction).T, '#80000020')
plt.show()

plt.figure(figsize=(25,6))
c_prediction = cont_model.predict(norm_positive_samples[:100,117:,0:9])
plt.plot(np.squeeze(c_prediction).T, '#00800020')
c_prediction = cont_model.predict(norm_negative_samples[:100,117:,0:9])
plt.plot(np.squeeze(c_prediction).T, '#80000020')
plt.show()

from sklearn.metrics import roc_curve, precision_recall_curve

threshold = 0.8

validation_X = training_X[:]
validation_Y = training_Y[:]

validation_prediction = cont_model.predict(validation_X)[:,-1]

fpr, tpr, thresholds = roc_curve(validation_Y, validation_prediction)

fig = plt.figure(figsize=(10, 8))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)

t_index = min(enumerate(thresholds), key=lambda x: abs(x[1] - threshold))[0]
s = plt.scatter(fpr[t_index], tpr[t_index])
s.axes.annotate(thresholds[t_index], (fpr[t_index] + 0.01, tpr[t_index] - 0.02))

    
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.show()

precision, recall, thresholds = precision_recall_curve(validation_Y, validation_prediction)

fig = plt.figure(figsize=(10, 8))
plt.plot(recall, precision)

t_index = min(enumerate(thresholds), key=lambda x: abs(x[1] - threshold))[0]
s = plt.scatter(recall[t_index], precision[t_index])
s.axes.annotate(thresholds[t_index], (recall[t_index] + 0.01, precision[t_index] + 0.002))

plt.xlabel('Recall')    
plt.ylabel('Precision')
plt.show()

from sklearn.metrics import confusion_matrix

confusion_matrix(validation_Y, [0 if x < threshold else 1 for x in validation_prediction])

