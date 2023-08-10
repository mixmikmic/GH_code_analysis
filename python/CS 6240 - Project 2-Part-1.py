from __future__ import print_function;
import sys;
import random;
from random import randint

import numpy as np;

from keras.models import Sequential;
from keras.layers import Dense, Activation;
from keras.layers import LSTM;
from keras.optimizers import RMSprop;
from keras.utils.data_utils import get_file;
from keras.models import load_model;

from sklearn.cross_validation import train_test_split;
from sklearn.metrics import *;
from sklearn.externals import joblib;

import matplotlib.pyplot as plt;
from IPython.display import clear_output
from keras.callbacks import ModelCheckpoint, Callback

import subprocess;
import h5py;

english_text = open('data/eng.txt').read().lower()
french_text = open('data/frn.txt').read().lower()

print('English corpus length:', len(english_text))
print('French corpus length:', len(french_text))

english_chars = sorted(list(set(english_text)))
french_chars = sorted(list(set(french_text)))

english_char_map = dict((c, i) for i, c in enumerate(english_chars))
french_char_map = dict((c, i) for i, c in enumerate(french_chars))

english_char_map_inverse = dict((i, c) for i, c in enumerate(english_chars))
french_char_map_inverse = dict((i, c) for i, c in enumerate(french_chars))

maxlen = 40
step = 3

english_sentences = []
english_next_chars = []
for i in range(0, len(english_text) - maxlen, step):
    english_sentences.append(english_text[i: i + maxlen])
    english_next_chars.append(english_text[i + maxlen])

french_sentences = []
french_next_chars = []
for i in range(0, len(french_text) - maxlen, step):
    french_sentences.append(french_text[i: i + maxlen])
    french_next_chars.append(french_text[i + maxlen])
    
print('nb English sequences:', len(english_sentences))
print('nb French sequences:', len(french_sentences))

print('Vectorization...')

char_len = max(len(english_chars), len(french_chars));

english_x = np.zeros((len(english_sentences), maxlen, char_len), dtype=np.bool)
english_y = np.zeros((len(english_sentences), char_len), dtype=np.bool)
for i, sentence in enumerate(english_sentences):
    for t, char in enumerate(sentence):
        english_x[i, t, english_char_map[char]] = 1
    english_y[i, english_char_map[english_next_chars[i]]] = 1
    
    
french_x = np.zeros((len(french_sentences), maxlen, char_len), dtype=np.bool)
french_y = np.zeros((len(french_sentences), char_len), dtype=np.bool)
for i, sentence in enumerate(french_sentences):
    for t, char in enumerate(sentence):
        french_x[i, t, french_char_map[char]] = 1
    french_y[i, french_char_map[french_next_chars[i]]] = 1

print("Finished!")

english_train_x, english_test_x, english_train_y, english_test_y = train_test_split(english_x, english_y, test_size=0.2, random_state=1024);
french_train_x, french_test_x, french_train_y, french_test_y = train_test_split(french_x, french_y, test_size=0.2, random_state=1024);

print('English Shapes');
print(english_train_x.shape);
print(english_train_y.shape);
print(english_test_x.shape);
print(english_test_y.shape);
print()
print('French Shapes');
print(french_train_x.shape);
print(french_train_y.shape);
print(french_test_x.shape);
print(french_test_y.shape);

def random_generate(test_x, key):
    labels = []
    feats = []
    if key == "english": 
        labels = [1 for i in range(100)]
    elif key == 'french': 
        labels = [0 for i in range(100)]
    else:
        return feats, labels;
    
    for i in range(100): 
        r1 = randint(0, len(test_x) - 1)
        ind = test_x[r1]
        
        r2 = randint(0, len(ind) - 5)
        
        sub_string = ind[r2:r2+5]
        
        feats.append(sub_string)
        
    return feats,labels
    
english_sample, english_labels = random_generate(english_test_x, 'english');
french_sample, french_labels = random_generate(french_test_x, 'french');

test_data = np.array(english_sample + french_sample);
test_labels = np.array(english_labels + french_labels);

def build_model(chars):
    print('Build model...')
    model = Sequential()
    model.add(LSTM(256, input_shape=(None, char_len)))
    model.add(Dense(char_len))
    model.add(Activation('softmax'))
    
    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy']);
    return model

def predict_on_sample(model, test_val):
    start = np.zeros((1, 1, char_len), dtype=bool);
    start_prob = model.predict(start);

    next_vec = start.copy()[0][0];
    probs = [];

    probs.append(start_prob[0,np.argwhere(test_val[0])[0][0]]);

    for idx, vec in enumerate(test_val):
        next_vec = np.append(next_vec, vec).reshape(1, idx+2, char_len)
        next_prob = model.predict(next_vec);

        probs.append(next_prob[0, np.argwhere(test_val[idx])[0][0]]);
        
    return np.sum(np.log(probs));

def predict_results(english_model, french_model):
    english_preds = np.array([predict_on_sample(english_model, x) for x in test_data]);
    french_preds = np.array([predict_on_sample(french_model, x) for x in test_data]);
    ratio_probs = english_preds - french_preds;
        
    fpr, tpr, _ = roc_curve(test_labels, ratio_probs);
    roc_auc = auc(fpr, tpr);
    
    print(roc_auc);

    return roc_auc, fpr, tpr;

def plot_roc_auc_curve(fpr, tpr, roc_auc, title): 
    plt.figure()
    lw = 2
    plt.semilogx(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.semilogx([np.min(fpr[fpr != 0]),1], [np.min(fpr[fpr != 0]), 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for ' + title)
    plt.legend(loc="lower right")
    plt.show()

def train_and_predict(total_epochs, step_size = 3, batch_size=2512):
    english_model = build_model(english_chars)
    french_model = build_model(french_chars);
    epochs_ran = 0;

    for i in range(0, total_epochs, step_size):
        history_english = english_model.fit(english_train_x, english_train_y,
                            batch_size=batch_size, epochs=step_size, shuffle=True);
        history_french = french_model.fit(french_train_x, french_train_y,
                            batch_size=batch_size, epochs=step_size, shuffle=True);

        
        epochs_ran = epochs_ran + step_size;
        
        roc = predict_results(english_model, french_model);
        
        if epochs_ran >= total_epochs:
            break;
            
    return [english_model, french_model, roc];

models_1 = train_and_predict(5, 5, 2512)

models_2 = train_and_predict(12, 3, 2512)

models_1[3][0].save('model_current_e.h5')
models_1[3][1].save('model_current_f.h5')

model_1 = load_model('model_current_e.h5')
model_2 = load_model('model_current_f.h5')

roc, fpr, tpr = predict_results(model_1, model_2);

plot_roc_auc_curve(fpr, tpr, roc, 'English/French Classification')

