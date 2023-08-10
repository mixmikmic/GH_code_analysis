get_ipython().system('python setup.py build_ext --inplace')

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

import time
import numpy as np
import mxnet as mx
from gensim.models import Word2Vec
from gensim.models.word2vec import Text8Corpus
from preprocessing import data_iterator_cython
import logging
import sys, random, time, math
from collections import namedtuple
from operator import itemgetter
from sklearn.preprocessing import normalize

get_ipython().system('wget http://mattmahoney.net/dc/text8.zip -O text8.gz && gzip -d text8.gz -f')

corpus = Text8Corpus("text8")
current_time = time.time()
model = Word2Vec(iter=1, sg=1)
model.build_vocab(corpus)
print "Building vocab took %s seconds" % (time.time() - current_time)

batch_data = []
batch_label = []
batch_label_weight = []

current_time = time.time()
job_batch, batch_size = [], 0
for sent_idx, sentence in enumerate(corpus):
    sentence_length = model._raw_word_count([sentence])

    # can we fit this sentence into the existing job batch?
    if batch_size + sentence_length <= model.batch_words:
        # yes => add it to the current job
        job_batch.append(sentence)
        batch_size += sentence_length
    else:
        sents = data_iterator_cython(model, job_batch, model.alpha)
        for sent in sents:
            batch_data.append(sent[0])
            batch_label.append(sent[1:])
        job_batch[:] = []
        batch_size = 0
print time.time() - current_time
print "Data prep took: ", time.time() - current_time

batch_data = mx.nd.array(batch_data)
batch_label = mx.nd.array(batch_label)

target_weight = mx.nd.zeros((batch_data.shape[0], model.negative+1))
target_weight[:,0] = 1

batch_data = mx.nd.expand_dims(batch_data, axis = 1)

batch_size = 512

nd_iter = mx.io.NDArrayIter(data = {"center_word" : batch_data, "target_words": batch_label},
                            label={ "labels":target_weight},
                            batch_size=batch_size, shuffle = True)

neg_dim = model.negative
vocab_size = len(model.wv.vocab)
dim = model.vector_size

def get_sym_makeloss(vocab_size, dim, batch_size):
    labels = mx.sym.Variable('labels') #1 positive and k "0" labels
    center_word = mx.sym.Variable('center_word')
    target_words = mx.sym.Variable('target_words') # 1 target + k negative samples
    center_vector = mx.sym.Embedding(data = center_word, input_dim = vocab_size,
                                  output_dim = dim, name = 'syn0_embedding')
    target_vectors = mx.sym.Embedding(data = target_words, input_dim = vocab_size,
                                   output_dim = dim, name = 'syn1_embedding')
    pred = mx.sym.batch_dot(target_vectors, center_vector, transpose_b=True)
    sigmoid = mx.sym.sigmoid(mx.sym.flatten(pred))
    loss = mx.sym.sum(labels * mx.sym.log(sigmoid) + (1 - labels) * mx.sym.log(1 - sigmoid), axis=1)
    loss *= -1.0
    loss_layer = mx.sym.MakeLoss(loss, normalization="batch")
    return loss_layer

def mean_loss(label, pred):
    return np.mean(pred)

nd_iter.reset()
sym = get_sym_makeloss(vocab_size, dim, batch_size)
network = mx.mod.Module(sym, data_names=("center_word", "target_words",), label_names=("labels",),context=mx.gpu())
network.bind(data_shapes=nd_iter.provide_data, label_shapes=nd_iter.provide_label)
current_time = time.time()
network.fit(nd_iter, num_epoch=1,optimizer='adam',
            eval_metric=mx.metric.CustomMetric(mean_loss),
            optimizer_params={'learning_rate': .001},
            batch_end_callback=mx.callback.Speedometer(batch_size, 1000),
            initializer=mx.initializer.Uniform(scale=.01))
print time.time() - current_time

all_vecs = network.get_params()[0]["syn0_embedding_weight"].asnumpy()
all_vecs = normalize(all_vecs, copy=False)

model.wv.syn0 = all_vecs
model.wv.syn0norm = all_vecs

model.most_similar("car")



