from __future__ import print_function
import os
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import numpy as np

import chainer

train, val, test = chainer.datasets.get_ptb_words()

print('train type: ', type(train), train.shape, train)
print('val   type: ', type(val), val.shape, val)
print('test  type: ', type(test), test.shape, test)

ptb_dict = chainer.datasets.get_ptb_words_vocabulary()
print('Number of vocabulary', len(ptb_dict))
print('ptb_dict', ptb_dict)

ptb_word_id_dict = ptb_dict
ptb_id_word_dict = dict((v,k) for k,v in ptb_word_id_dict.iteritems())

# Same with https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt
print([ptb_id_word_dict[i] for i in train[:300]])

# Same with https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt
print([ptb_id_word_dict[i] for i in val[:300]])

# Same with https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt
print([ptb_id_word_dict[i] for i in test[:300]])



