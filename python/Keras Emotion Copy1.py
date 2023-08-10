import glob
import librosa
import librosa.display
import numpy as np
import numpy
import _pickle as pickle
from sklearn import svm
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import NearestNeighbors

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.layers import Dropout
from keras.utils import plot_model
from IPython.display import SVG
from matplotlib import pyplot as plt

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


def parse_audio_files(path):
    features, labels = np.empty((0, 193)), np.empty(0)
    labels = []
    for fn in glob.glob(path):
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        except Exception as e:
            print("Error encountered while parsing file: ", fn)
            continue
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        labels = np.append(labels, fn.split("_")[3].split(".")[0])

    return np.array(features), np.array(labels)

tr_features, tr_labels = parse_audio_files('./train_80/*.wav')


tr_features = np.array(tr_features, dtype=pd.Series)
tr_labels = np.array(tr_labels, dtype=pd.Series)

X = tr_features.astype(int)
Y = tr_labels.astype(str)

X.shape, Y.shape

seed = 7
numpy.random.seed(seed)

encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

dummy_y = np_utils.to_categorical(encoded_Y)

dummy_y


def baseline_model():
    deep_model = Sequential()
    deep_model.add(Dense(100, input_dim=193, activation="relu", kernel_initializer="uniform"))
    deep_model.add(Dropout(0.5))
    deep_model.add(Dense(50, activation="relu", kernel_initializer="uniform"))
    deep_model.add(Dropout(0.5))
    deep_model.add(Dense(20, activation="relu", kernel_initializer="uniform"))
    deep_model.add(Dropout(0.5))
    deep_model.add(Dense(7, activation="softmax", kernel_initializer="uniform"))

    deep_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return deep_model


epoches = 100
batch_size = 25
verbose = 1

model = baseline_model()
result = model.fit(X, dummy_y, validation_split=0.1, batch_size=batch_size, epochs=epoches, verbose=verbose)

print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

print(result.history)

filename = 'keras_model.h5'

model.save(filename)

print('Model Saved..')

plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.title('keras model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('keras model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.summary()

# print("Baseline: %.2f%% (%.2f%%)" % (result.mean() * 100, result.std() * 100))

print(result.history)

filename = 'keras_model_v1.h5'

model.save(filename)

print('Model Saved..')

plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.title('keras model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

plt.plot(result.history['loss'])
plt.plot(result.history['val_loss'])
plt.title('keras model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

result.history.keys()

plt.plot(result.history['acc'])
plt.plot(result.history['val_acc'])
plt.legend(['train_acc', 'val_acc'], loc='lower right')
plt.show()









import glob
import librosa
import librosa.display
import numpy as np
import _pickle as pickle
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.models import Model

import glob
import librosa
import librosa.display
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.metrics import accuracy_score


def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
                                              sr=sample_rate).T, axis=0)
    return mfccs, chroma, mel, contrast, tonnetz


target_files = []


def parse_audio_files(path):
    labels = []
    features = np.empty((0, 193))
    for fn in glob.glob(path):
        try:
            mfccs, chroma, mel, contrast, tonnetz = extract_feature(fn)
        except Exception as e:
            print("Error encountered while parsing file: ", fn)
            continue
        ext_features = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
        features = np.vstack([features, ext_features])
        labels = np.append(labels, fn.split("_")[3].split(".")[0])
        target_files.append(fn)
    return np.array(features), np.array(labels)


ts_features, ts_labels = parse_audio_files('./test_20/*.wav')

ts_features = np.array(ts_features, dtype=pd.Series)
ts_labels = np.array(ts_labels, dtype=pd.Series)

test_true = ts_labels
test_class_label = ts_labels

encoder = LabelEncoder()
encoder.fit(ts_labels.astype(str))
encoded_Y = encoder.transform(ts_labels.astype(str))

ts_labels = np_utils.to_categorical(encoded_Y)

ts_labels.resize(ts_labels.shape[0], 7)

filename = 'keras_model.sav'

model = load_model('keras_model_v1.h5')

prediction = model.predict_classes(ts_features.astype(int))


test_predicted = []

labels_map = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

for i, val in enumerate(prediction):
    test_predicted.append(labels_map[val])

# print(test_predicted)
print("Accuracy Score:", accuracy_score(test_true, test_predicted))
print('Number of correct prediction:', accuracy_score(test_true, test_predicted, normalize=False), 'out of', len(ts_labels))

matrix = confusion_matrix(test_true, test_predicted)
classes = list(set(test_class_label))
classes.sort()
df = pd.DataFrame(matrix, columns=classes, index=classes)
plt.figure()
sn.heatmap(df, annot=True)
plt.show()

file = open("X_train_data_2.txt","w") 
file.write(X) 
file.close() 

file1 = open("X_train_data.txt","r") 
file1.read()
file1.close()



