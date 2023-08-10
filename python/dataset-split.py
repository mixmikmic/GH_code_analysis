import numpy as np
import pandas as pd
import math
import random
import pickle
import json

# to make the experimens replicable
random.seed(123456)

df = pd.read_pickle('../data/atti-dirigenti-processed.pkl')

dataset = df[['OGGETTO', 'UFFICIO_DG', 'DATA_ATTO']]
dataset.shape

documents_per_office = dataset.groupby(['UFFICIO_DG']).count()
documents_per_office.describe()

value = 2000
sel_dataset = documents_per_office[documents_per_office.OGGETTO >= 2000]
sel_dataset.shape

sel_dataset['UFFICIO_DG'] = sel_dataset.index

sel_dataset.describe()

sel_dataset.head()

final_ds = dataset.merge(sel_dataset, how='inner', on=['UFFICIO_DG'])

final_ds.shape

final_ds = final_ds[['OGGETTO_x',"UFFICIO_DG"]]
final_ds.head()

len(set(final_ds['UFFICIO_DG']))

samples = []
labels = []

for text, label in final_ds.as_matrix():
    samples.append(text)
    labels.append(label)

samples[50000]

labels[50000]

samples = np.array(samples)
labels = np.array(labels)

with open('../data/dataset-dirigenti.pkl', 'wb') as o:
    pickle.dump((samples, labels), o)

dataset_path = '../data/dataset-dirigenti.pkl'

with open(dataset_path, 'rb') as f:
    samples, labels = pickle.load(f)

from sklearn.model_selection import StratifiedShuffleSplit

split_train_test = StratifiedShuffleSplit(1,test_size=0.2, random_state=123456)

for train, test in split_train_test.split(samples, labels):
    train_split_samples, test_split_samples = samples[train], samples[test]
    train_split_labels, test_split_labels = labels[train], labels[test]

index_label_dict = dict(enumerate(set(train_split_labels),0))
label_index_dict = {v:k for k,v in index_label_dict.items()}

label_index_dict

with open('data_dirigenti_label_index.json', 'w') as f: 
    json.dump(label_index_dict, f)

train_labels = np.array([label_index_dict[l] for l in train_split_labels])
test_labels = np.array([label_index_dict[l] for l in test_split_labels])

print(train_labels)
print(test_labels)

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('punkt')

punctuation = ['-', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}', '’', '”', '“', '``', "''"]
stop_words = set(stopwords.words('italian'))
stop_words.update(punctuation)

def tokenize_sample(samples, remove_stopwords=True, tokenizer=word_tokenize):
    for sample in samples:
        words = []
        sample = sample.replace('`', ' ')
        sample = sample.replace("'", " ")
        for w in tokenizer(sample):
            if remove_stopwords:
                if w not in stop_words:
                    words.append(w.lower())
            else:
                words.append(w.lower())
        yield words

train_samples_tokenized = tokenize_sample(train_split_samples, remove_stopwords=False, tokenizer=word_tokenize)

from collections import Counter

counter = Counter()

for words in train_samples_tokenized:
    counter.update(words)

counter.most_common()[:20]

index_word_dict = dict(enumerate([ k for k,v in counter.most_common()],3))
word_index_dict = {v:k for k,v in index_word_dict.items()}

with open('data_dirigenti_word_index.json', 'w') as f: 
    json.dump(word_index_dict, f)

with open('data_dirigenti_most_common.json', 'w') as f:
    json.dump(counter.most_common(), f)

pad_char = 0
start_char=1
oov_char=2

train_samples_tokenized = tokenize_sample(train_split_samples, remove_stopwords=False, tokenizer=word_tokenize)
test_samples_tokenized = tokenize_sample(test_split_samples, remove_stopwords=False, tokenizer=word_tokenize)

def samples_to_idx(tokenized_samples, word_index_dict):
    for sample in tokenized_samples:
        encoded_sample = []
        for w in sample:
            if w in word_index_dict:
                encoded_sample.append(word_index_dict[w])
            else:
                encoded_sample.append(oov_char)
        yield encoded_sample

train_sample = np.array(list(samples_to_idx(train_samples_tokenized, word_index_dict)))
test_data = np.array(list(samples_to_idx(test_samples_tokenized, word_index_dict)))

split_train_val = StratifiedShuffleSplit(1,test_size=0.1, random_state=123456)

for train, val in split_train_val.split(train_sample, train_labels):
    train_data, val_data = train_sample[train], train_sample[val]
    train_labels_, val_labels = train_labels[train], train_labels[val]

print('labels training {}'.format(train_labels_.shape))
print('labels validation {}'.format(val_labels.shape))
print('labels test {}'.format(test_labels.shape))

print('samples training {}'.format(train_data.shape))
print('samples validation {}'.format(val_data.shape))
print('samples test {}'.format(test_data.shape))

train_labels = train_labels_

np.savez_compressed('data_dirigenti.npz', 
                    x_train=train_data, y_train=train_labels, 
                    x_val=val_data, y_val=val_labels, 
                    x_test=test_data, y_test=test_labels)

loaded = np.load('data_dirigenti.npz')

loaded['x_train'].shape

