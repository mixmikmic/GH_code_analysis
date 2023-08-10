import numpy as np
import pickle

from sklearn.metrics import confusion_matrix, accuracy_score

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D

from keras.callbacks import ModelCheckpoint, EarlyStopping

### Load Data
with open('atis.pkl', 'rb') as f:
    train_set, valid_set, test_set, dicts = pickle.load(f)

w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']

# Create index to word/label dicts
idx2w  = {w2idx[k]:k for k in w2idx}
idx2ne = {ne2idx[k]:k for k in ne2idx}
idx2la = {labels2idx[k]:k for k in labels2idx}

vocab_len = len(w2idx)
# print(vocab_len)
# print(len(labels2idx))
# print(max(list(labels2idx.values())))

train_x, train_ne, train_label = train_set
val_x, val_ne, val_label = valid_set
test_x, test_ne, test_label = test_set

X_full = train_x + val_x + test_x
maxlen = max([len(x) for x in X_full])
print('Maximum sequence length:', maxlen)

def encode_one_hot(idx, dim):
    temp = [0]*dim
    temp[idx] = 1
    return temp

def encode_corpus(X, maxlen):
    #X_enc = [[w2idx[word] for word in x] for x in X]
    return pad_sequences(X, maxlen=maxlen, value=vocab_len)

def encode_labels(Y, maxlen, dim):
    #Y_enc = [[labels2idx[tag] for tag in y] for y in Y]
    Y_enc = pad_sequences(Y, maxlen=maxlen, value=dim-1)
    Y_enc = [[encode_one_hot(idx, dim) for idx in y] for y in Y_enc]
    return np.array(Y_enc)

dim = len(idx2la) + 1
print(dim)

X_enc = encode_corpus(train_x + val_x, maxlen)
y_enc = encode_labels(train_label + val_label, maxlen, dim)

X_test_enc = encode_corpus(test_x, maxlen)
y_test_enc = encode_labels(test_label, maxlen, dim)

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

# n_classes = len(idx2la)
# n_vocab = len(idx2w)

max_features = len(w2idx)+1
embedding_size = 100
hidden_size = 32
out_size = len(labels2idx) + 1
batch_size = 32
epochs = 10

# Define model
model = Sequential()
# model.add(Embedding(n_vocab,100))
model.add(Embedding(input_dim=max_features, output_dim=embedding_size,
                    input_length=maxlen, mask_zero=False))
model.add(Convolution1D(64,5,padding='same', activation='relu'))
model.add(Dropout(0.25))
model.add(GRU(100,return_sequences=True))
model.add(TimeDistributed(Dense(out_size, activation='softmax')))

model.summary()

model.compile('rmsprop', 'categorical_crossentropy')

filepath = "models/NER-ATIS-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
callbacks_list = [checkpoint, earlystopping]

model.fit(X_train_enc, y_train_enc, batch_size=batch_size, epochs=epochs,
          validation_data=(X_val_enc, y_val_enc), callbacks=callbacks_list)

model.save('models/conv_model.h5')

model = load_model('models/conv_model.h5')
model.summary()

score = model.evaluate(X_test_enc, y_test_enc, batch_size=batch_size, verbose=0)
print('Raw test score:', score)

def unpad_sequences(yh, pr):
    coords = [np.where(yhh < vocab_len)[0][0] for yhh in yh]
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

# def score(yh, pr):
#     coords = [np.where(yhh < vocab_len)[0][0] for yhh in yh]
#     yh = [yhh[co:] for yhh, co in zip(yh, coords)]
#     ypr = [prr[co:] for prr, co in zip(pr, coords)]
#     fyh = [c for row in yh for c in row]
#     fpr = [c for row in ypr for c in row]
#     return fyh, fpr

# # On the test set 
# pr = model.predict(X_test_enc)
# pr = pr.argmax(2)
# print(pr.shape)
# print(pr[0])
# print(pr[0][0])
# yh = y_test_enc.argmax(2)
# print(yh.shape)
# print(yh[0])

# fyh, fpr = score(yh, pr)
# print('Testing accuracy:', accuracy_score(fyh, fpr))
# print('Testing confusion matrix:')
# print(confusion_matrix(fyh, fpr))

