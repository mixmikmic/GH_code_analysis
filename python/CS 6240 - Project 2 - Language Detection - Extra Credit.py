from __future__ import print_function;
import re;
import sys;
import random;
import unicodedata;
from random import randint;

import numpy as np;
import seaborn as sns;

from keras.models import Sequential;
from keras.layers import Dense, Activation;
from keras.layers import LSTM;
from keras.optimizers import RMSprop;
from keras.utils.data_utils import get_file;
from sklearn.cross_validation import train_test_split;
from sklearn.metrics import *;
from sklearn.externals import joblib;

import matplotlib.pyplot as plt;
from IPython.display import clear_output
from keras.callbacks import ModelCheckpoint, Callback

import subprocess;
import h5py;

def notify_slack(text):
    text = 'WebSearch: ' + text;
    subprocess.Popen('''curl -X POST --data-urlencode "payload={'channel' : '#random', 'username': 'webhookbot', 'text':'''+ '\'' + text + '\'' + '''}" https://hooks.slack.com/services/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''', shell=True)

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

languages = ['data/eng.txt',
             'data/frn.txt',
             'languages/dut.txt',
             'languages/ger.txt', 
             'languages/itn.txt', 
             'languages/ltn.txt', 
             'languages/por.txt', 
             'languages/romanized_jap.txt', 
             'languages/romanized_rus.txt', 
             'languages/spn.txt' ]

language_names = ['English',
                  'French',
                  'Dutch',
                  'German',
                  'Italian',
                  'Latin',
                  'Portugese',
                  'Japanese',
                  'Russian',
                  'Spanish']

all_text = []
for file in languages:
    text = open(file).read().lower()
    all_text.append(strip_accents(text))

for text in all_text:
    print(text[0:20])

all_chars = []
all_char_map = []
all_char_map_inverse = []
for idx, lang in enumerate(all_text):
    all_chars.append(sorted(list(set(lang))))
    all_char_map.append(dict((c, i) for i, c in enumerate(sorted(list(set(lang))))))
    all_char_map_inverse.append(dict((i, c) for i, c in enumerate(sorted(list(set(lang))))))
    
    print (language_names[idx])
    print ("\tCorpus length:", len(lang))
    print ("\tCharacter Count", len(all_chars[idx]))
    print ()

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3

all_sentences = []
all_next_chars = []

for idx, lang in enumerate(all_text):
    sentences = []
    next_chars = []
    for i in range(0, len(lang) - maxlen, step):
        sentences.append(lang[i: i + maxlen])
        next_chars.append(lang[i + maxlen])
    
    all_sentences.append(sentences)
    all_next_chars.append(next_chars)

    print (language_names[idx])
    print ("\tSequences:", len(sentences))
    print ()

print('Vectorization...')

char_len = max([len(x) for x in all_chars])

all_x = []
all_y = []

for idx, lang in enumerate(all_text):
    x = np.zeros((len(all_sentences[idx]), maxlen, char_len), dtype=np.bool)
    y = np.zeros((len(all_sentences[idx]), char_len), dtype=np.bool)
    
    for i, sentence in enumerate(all_sentences[idx]):
        for t, char in enumerate(sentence):
            x[i, t, all_char_map[idx][char]] = 1
        y[i, all_char_map[idx][all_next_chars[idx][i]]] = 1
    
    all_x.append(x)
    all_y.append(y)

print("Finished!")

all_train_x = []
all_test_x = []
all_train_y = []
all_test_y =[]

for idx, lang in enumerate(all_text):
    train_x, test_x, train_y, test_y = train_test_split(all_x[idx], all_y[idx], test_size=0.2, random_state=1024);
    all_train_x.append(train_x)
    all_test_x.append(test_x)
    all_train_y.append(train_y)
    all_test_y.append(test_y)

def random_generate(test_x_1, test_x_2, seed_1, seed_2):
    both_labels = []
    both_feats = []
    
    rands = [random.Random(), random.Random()]

    rands[0].seed(seed_1)
    rands[1].seed(sys.maxsize - seed_2)
    
    
    key = 1
    for test_x in [test_x_1, test_x_2]:
        labels = []
        feats = []
        for i in range(100): 
            r1 = rands[key].randint(0, len(test_x) - 1)
            
            ind = test_x[r1]
            r2 = rands[key].randint(0, len(ind) - 5)

            sub_string = ind[r2:r2+5]

            feats.append(sub_string)
            labels.append(key)
            
        both_labels.append(labels)
        both_feats.append(feats)
        key = key^1
        
    return both_feats, both_labels

all_samples_1 = []
all_labels_1 = []

all_samples_2 = []
all_labels_2 = []

for idx, test_x in enumerate(all_test_x):
    for idx2, test_x2 in enumerate(all_test_x):
        
        [[feats_1, feats_2], [labels_1, labels_2]] = random_generate(test_x, test_x2, idx, idx2)
        all_samples_1.append(feats_1)
        all_labels_1.append(labels_1)
        
        all_samples_2.append(feats_2)
        all_labels_2.append(labels_2)

# build the model: a single LSTM
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

def predict_results(model_1, model_2, lang_idx_1, lang_idx_2):
    
    test_data = np.array(all_samples_1[lang_idx_1] + all_samples_2[lang_idx_2]);
    test_labels = np.array(all_labels_1[lang_idx_1] + all_labels_2[lang_idx_2]);
    
    preds_1 = np.array([predict_on_sample(model_1, x) for x in test_data]);
    preds_2 = np.array([predict_on_sample(model_2, x) for x in test_data]);
    
    ratio_probs = preds_1 - preds_2;
        
    fpr, tpr, _ = roc_curve(test_labels, ratio_probs);
    roc_auc = roc_auc_score(test_labels, ratio_probs);

    return roc_auc, fpr, tpr;

def train_and_predict(lang_idx_1, lang_idx_2, total_epochs, batch_size=2048):
    model_1 = build_model(all_chars[lang_idx_1])
    model_2 = build_model(all_chars[lang_idx_2]);
    #notify_slack('--------------------------------------------------------------------------------------------------------------------------------------------');
    epochs_ran = 0;
    
    history_1 = model_1.fit(all_train_x[lang_idx_1], all_train_y[lang_idx_1],
                        batch_size=batch_size, epochs=total_epochs, shuffle=True, verbose=0);
    history_2 = model_2.fit(all_train_x[lang_idx_2], all_train_y[lang_idx_2],
                        batch_size=batch_size, epochs=total_epochs, shuffle=True, verbose=0);

    roc, _, _ = predict_results(model_1, model_2, lang_idx_1, lang_idx_2);
    #notify_slack(res);
           
    return roc, history_1, history_2, model_1, model_2;

all_models = []
for idx, lang in enumerate(all_text):
    model_results = []
    for idx_2, lang2 in enumerate(all_text):
        result = train_and_predict(lang_idx_1=idx, lang_idx_2=idx_2, total_epochs=8, batch_size=2512)
        model_results.append(result)
        print ("Finished: %s -> %s with ROC=%f" % (language_names[idx], language_names[idx_2], result[0]))
    all_models.append(model_results)
    notify_slack("Finished all models for " + language_names[idx])

all_roc = dict()
all_roc_list = []

all_fpr = dict()
all_tpr = dict()
all_loss = []

for idx_1 in range(len(all_models)):
    language_roc = []
    language_loss = []
    
    for idx_2 in range(len(all_models[idx_1])):
        roc, history_1, history_2, model_1, model_2 = all_models[idx_1][idx_2]
        roc_auc, fpr, tpr = predict_results(model_1, model_2, idx_1, idx_2)
        
        index = idx_1*len(all_models) + idx_2
        all_roc[index] = roc_auc
        all_fpr[index] = fpr
        all_tpr[index] = tpr
        
        language_roc.append(roc_auc)
        language_loss.append(history_2.history["loss"][-1])
        
        print ("%s - %s ROC: %f" % (language_names[idx_1], language_names[idx_2], roc_auc))
    print()
    
    all_roc_list.append(language_roc)
    all_loss.append(language_loss)

sns.set()
ax = sns.heatmap(all_roc_list, vmin=0, vmax=1, annot=True, yticklabels=language_names, xticklabels=language_names)
plt.xticks(rotation=45)
plt.title("ROC Heatmap")
sns.plt.show()

def plot_roc_auc(fpr_, tpr_, roc_, title, num_plot):
    plt.figure(figsize=(10,12))

    language_pairs = []
    for lang in language_names:
        for lang2 in language_names:
            language_pairs.append(lang[0:3]+"-"+lang2[0:3])
    
    lw=1.15
    for i in range(num_plot):
        plt.plot(fpr_[i], tpr_[i], lw=lw,
                 label='{0} (area = {1:0.2f})'
                 ''.format(language_pairs[i], roc_[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC AUC Curve for " + title)
    #plt.legend(loc=9, bbox_to_anchor=(0.5, -0.1))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.show()

plot_roc_auc(all_fpr, all_tpr, all_roc, "all Languages", len(all_fpr))

plot_roc_auc(all_fpr, all_tpr, all_roc, "English", 10)

