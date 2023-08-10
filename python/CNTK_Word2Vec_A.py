# Import the relevant modules to be used later
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import os
import pickle
import random
import sys
import zipfile

from six.moves import urllib
from six.moves import xrange

# Initializing globals

vocab_size = 4096
data = list()
dictpickle = 'w2v-dict.pkl'
datapickle = 'w2v-data.pkl'

def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    url = 'http://mattmahoney.net/dc/'
    if not os.path.exists(filename):
        print('Downloading Sample Data..')
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

def read_data(filename):
    """Read the file as a list of words"""
    data = list()
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            data += line.split()
    return data


def read_data_zip(filename):
    """Extract the first file enclosed in a zip file as a list of words"""
    with zipfile.ZipFile(filename) as f:
        bdata = f.read(f.namelist()[0]).split()
    data = [x.decode() for x in bdata]
    return data

def build_dataset(words):
    global data, vocab_size
    
    print('Building Dataset..')
    
    print('Finding the N most common words in the dataset..')
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocab_size - 1))
    print('Done')
    
    dictionary = dict()
    
    for word, _ in count:
        dictionary[word] = len(dictionary)
    
    print('Integerizing the data..')
    data = list()
    unk_count = 0
    
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    
    print('Done')
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    
    print('Saving Vocabulary..')
    with open(dictpickle, 'wb') as handle:
        pickle.dump(dictionary, handle)
    print('Done')
    
    print('Saving the processed dataset..')
    with open(datapickle, 'wb') as handle:
        pickle.dump(data, handle)
    print('Done')
    
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

def process_text(filename):

    if filename == 'runexample':
        print('Running on the example data..')
        filename = maybe_download('text8.zip', 31344016)
        words = read_data_zip(filename)
    else:
        print('Running on the user specified data')
        words = read_data(filename)
    
    build_dataset(words)
    
# Running on Example Data (i.e. Text8 Corpus)
process_text('runexample')

