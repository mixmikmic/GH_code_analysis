import pickle 
import numpy as np 

from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from keras.layers import Input, Dense, TimeDistributed
from keras.layers import Embedding, Activation
from keras.layers import GRU, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint, EarlyStopping


from keras.backend import tf

with open('conll.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['train']['X']
y = data['train']['y']
X_test = data['test']['X']
y_test = data['test']['y']
maxlen = data['stats']['maxlen']
word2ind = data['stats']['word2ind']
ind2word = data['stats']['ind2word']
label2ind = data['stats']['label2ind']
ind2label = data['stats']['ind2label']

print(ind2label)

def encode_one_hot(idx, dim):
    temp = [0]*dim
    temp[idx] = 1
    return temp

def encode_corpus(X, maxlen):
    X_enc = [[word2ind[word] for word in x] for x in X]
    return pad_sequences(X_enc, maxlen=maxlen, value=0)

def encode_labels(Y, maxlen, dim):
    Y_enc = [[label2ind[tag] for tag in y] for y in Y]
    Y_enc = pad_sequences(Y_enc, maxlen=maxlen, value=0)
    Y_enc = [[encode_one_hot(idx, dim) for idx in y] for y in Y_enc]
    return np.array(Y_enc)

dim = len(ind2label) + 1
print(dim)

X_enc = encode_corpus(X, maxlen)
y_enc = encode_labels(y, maxlen, dim)

X_test_enc = encode_corpus(X_test, maxlen)
y_test_enc = encode_labels(y_test, maxlen, dim)

validation_split = 0.1

indices = np.arange(X_enc.shape[0])
np.random.shuffle(indices)
X_enc = X_enc[indices]
y_enc = y_enc[indices]
num_validation_samples = int(validation_split * X_enc.shape[0])

X_train_enc = X_enc[:-num_validation_samples]
y_train_enc = y_enc[:-num_validation_samples]
X_val_enc = X_enc[-num_validation_samples:]
y_val_enc = y_enc[-num_validation_samples:]

print('Training and testing tensor shapes:')
print(X_train_enc.shape, X_val_enc.shape, X_test_enc.shape, y_train_enc.shape, y_val_enc.shape, y_test_enc.shape)

max_features = len(word2ind)+1
embedding_size = 128
hidden_size = 32
out_size = len(label2ind) + 1
batch_size = 32
epochs = 10

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=embedding_size,
                    input_length=maxlen, mask_zero=True))
model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
model.add(TimeDistributed(Dense(out_size)))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam')

filepath = "models/NER-Wikigold-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
callbacks_list = [checkpoint, earlystopping]

model.fit(X_train_enc, y_train_enc, batch_size=batch_size, epochs=epochs,
          validation_data=(X_val_enc, y_val_enc), callbacks=callbacks_list)

model.save('models/bidir_lstm.h5')

model = load_model('models/bidir_lstm.h5')

score = model.evaluate(X_test_enc, y_test_enc, batch_size=batch_size, verbose=1)
print('Raw test score:', score)

def unpad_sequences(yh, pr):
    coords = [np.where(yhh > 0)[0][0] for yhh in yh]
    yh = [yhh[co:] for yhh, co in zip(yh, coords)]
    ypr = [prr[co:] for prr, co in zip(pr, coords)]
    return yh, ypr

def score(yh, pr):
    yh, ypr = unpad_sequences(yh, pr)
    fyh = [c for row in yh for c in row]
    fpr = [c for row in ypr for c in row]
    return fyh, fpr

def compare_prediction_groumdtruth(model, X, y, verbose=True, indices=None):
    pr = model.predict(X)
    pr = pr.argmax(2)
    yh = y.argmax(2)
    fyh, fpr = score(yh, pr)
    print('Accuracy:', accuracy_score(fyh, fpr), end='\n\n')
    print('Confusion matrix:')
    print(confusion_matrix(fyh, fpr), end='\n\n')
    
    if verbose and indices != None:
        yh, ypr = unpad_sequences(yh, pr)
        for idx in indices:
            print('test sample', idx)
            print(yh[idx])
            print(ypr[idx], end='\n\n')

compare_prediction_groumdtruth(model, X_test_enc, y_test_enc, True, indices=[1,2,3,4,5,6])

