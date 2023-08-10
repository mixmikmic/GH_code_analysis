from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import itertools as it
from functools import partial
import nltk
from string import punctuation
import pickle
from collections import Counter

PATH_DATA = '../data/amazon/food/reviews_df.msg'
stop = nltk.corpus.stopwords.words('english')

stemmer = nltk.stem.SnowballStemmer('english')

df = pd.read_msgpack(PATH_DATA)#.head(10000)

get_ipython().run_cell_magic('time', '', 'purge_words = stop + list(punctuation) + [\'"\', \'\\\'\', \'\\\\\']\ndata_words = df.Text\\\n    .apply(lambda row: [stemmer.stem(word.lower())\n                        for word in nltk.word_tokenize(row) \n                        if word.lower() not in purge_words])')

get_ipython().run_cell_magic('time', '', '# Find most common words\nc = Counter()\n_ = data_words.map(lambda words: c.update(words))\nthresh = 2**15-1  # should be 2**16 but I want less\nfreq_words, _ = zip(*c.most_common(thresh))\nfreq_words = set(freq_words)')

get_ipython().run_cell_magic('time', '', '# Go through again and filter out uncommon words\ndata_words = data_words\\\n    .map(lambda words: [word for word in words if word in freq_words])')

vocab = set(it.chain(*data_words))
vocab_size = len(vocab) + 1  # +1 for UNK

max_len = max(map(len, data_words))

vocab_size

get_ipython().run_cell_magic('time', '', "le = LabelEncoder()\nle.fit(list(vocab))\nle.classes_ = np.insert(le.classes_, 0, '')  # UNK tok @ 0")

enc_d = dict(zip(le.classes_, range(len(le.classes_))))

get_ipython().run_cell_magic('time', '', 'data_words_enc = data_words.map(lambda l: list(map(enc_d.get, l)))')

data_words_enc.to_msgpack('../data/amazon/food/reviews_txt_enc_s.msg', compress='blosc')

pickle.dump(list(le.classes_), open('../data/amazon/food/vocab.p', 'wb'))

