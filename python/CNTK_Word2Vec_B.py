from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import pickle
import random
import sys

import numpy as np
from cntk.initializer import uniform
from cntk.learner import learning_rate_schedule, sgd, UnitType
from cntk.ops import *
from cntk.trainer import Trainer
from cntk.utils import ProgressPrinter

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')


curr_epoch = 0
emb_size = 256
minibatch_size = 256
num_epochs = 2
skip_window = 1
vocab_size = 4096
words_per_epoch = 4096
words_seen = 0
words_to_train = 1024


data = list()
embeddings = None


dictpickle = 'w2v-dict.pkl'
datapickle = 'w2v-data.pkl'
embpickle = 'w2v-emb.pkl'

# Ensure we always get the same amount of randomness
np.random.seed(0)

def lrmodel(inp, out_dim):
    inp_dim = inp.shape[0]
    wt = parameter(shape=(inp_dim, out_dim), init=uniform(scale=1.0))
    b = parameter(shape=(out_dim), init=uniform(scale=1.0))
    out = times(inp, wt) + b
    return out

def train(emb_size, vocab_size):
    global embeddings, words_to_train

    inp = input_variable(shape=(vocab_size,))
    label = input_variable(shape=(vocab_size,))

    init_width = 0.5 / emb_size
    emb = parameter(shape=(vocab_size, emb_size), init=uniform(scale=init_width))
    embeddings = emb
    embinp = times(inp, emb)

    z = softmax(lrmodel(embinp, vocab_size))        # logistic regression model

    loss = - label * log(z) - ((1 - label) / (vocab_size - 1)) * log(1 - z)
    eval_error = classification_error(z, label)

    lr_per_sample = [0.003]*4 + [0.0015]*24 + [0.0003]
    lr_per_minibatch = [x * minibatch_size for x in lr_per_sample]
    lr_schedule = learning_rate_schedule(lr_per_minibatch, UnitType.minibatch)
    
    learner = sgd(z.parameters, lr=lr_schedule)
    trainer = Trainer(z, loss, eval_error, learner)

    return inp, label, trainer

def build_dataset():
    global data, num_epochs, words_per_epoch, words_to_train
    with open(datapickle, 'rb') as handle:
        data = pickle.load(handle)
    words_per_epoch = len(data)
    words_to_train = num_epochs * words_per_epoch

def generate_batch(batch_size, skip_window):
    """ Function to generate a training batch for the skip-gram model. """
    
    global data, curr_epoch, words_per_epoch, words_seen
    
    data_index = words_seen - curr_epoch * words_per_epoch
    num_skips = 2 * skip_window
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    
    batch = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    buffer = collections.deque(maxlen=span)
    
    for _ in range(span):
        buffer.append(data[data_index])
        words_seen += 1
        data_index += 1
        if data_index >= len(data):
            curr_epoch += 1
            data_index -= len(data)
    
    for i in range(batch_size // num_skips):
        target = skip_window    # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            
            targets_to_avoid.append(target)
            batch[i * num_skips + j, 0] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        
        buffer.append(data[data_index])
        words_seen += 1
        data_index += 1
        if data_index >= len(data):
            curr_epoch += 1
            data_index -= len(data)
    
    return batch, labels

def get_one_hot(origlabels):
    global minibatch_size, vocab_size
    labels = np.zeros(shape=(minibatch_size, vocab_size), dtype=np.float32)
    for t in range(len(origlabels)):
        if origlabels[t, 0] < vocab_size and origlabels[t, 0] >= 0:
            labels[t, origlabels[t, 0]] = 1.0
    return labels

build_dataset()

inp, label, trainer = train(emb_size, vocab_size)
print('Model Creation Done.')
pp = ProgressPrinter(50)
for _epoch in range(num_epochs):
    i = 0
    while curr_epoch == _epoch:
        features, labels = generate_batch(minibatch_size, skip_window)
        features = get_one_hot(features)
        labels = get_one_hot(labels)
        
        trainer.train_minibatch({inp: features, label: labels})
        pp.update_with_trainer(trainer)
        i += 1
        if i % 200 == 0:
            print('Saving Embeddings..')
            with open(embpickle, 'wb') as handle:
                pickle.dump(embeddings.value, handle)

    pp.epoch_summary()

test_features, test_labels = generate_batch(minibatch_size, skip_window)
test_features = get_one_hot(test_features)
test_labels = get_one_hot(test_labels)

avg_error = trainer.test_minibatch({inp: test_features, label: test_labels})
print('Avg. Error on Test Set: ', avg_error)

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
plot_only = 500
with open(embpickle, 'rb') as handle:
    final_embeddings = pickle.load(handle)

low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])

with open(dictpickle, 'rb') as handle:
    dictionary = pickle.load(handle)
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
labels = [reverse_dictionary[i] for i in range(plot_only)]
plot_with_labels(low_dim_embs, labels)



