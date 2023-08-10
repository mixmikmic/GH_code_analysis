import collections
from packages.vocab import Vocab
import os
input_dir = '/home/irteam/users/data/CNN_DailyMail/cnn/2.stories_tokenized_100/'

vocab = Vocab(50000)
batch_count=0
file_list = os.listdir(path=input_dir)

for file_name in file_list:
    with open(os.path.join(input_dir,file_name)) as f:
        text = f.read()
        text = vocab.preprocess_string(text,[(":==:"," "),('\n\n', " ")])
        word_list = vocab.tokenize(text)
        vocab.feed_to_counter(word_list)
    batch_count+=1
    print("%d batch(s) covered!" %batch_count)
    counter = vocab.counter
    vocab.counter_to_vocab(counter)

with open('counter_cnn.pckl','wb') as f:
    pickle.dump(counter,f)

vocab.counter_to_vocab(counter)

vocab.max_size

import pickle

import numpy as np
np.save('word2idx.npy',vocab.w2i)
np.save('idx2word.npy',vocab.i2w)

w2i = np.load('word2idx.npy').item()
i2w = np.load('idx2word.npy').item()

w2i['prune']

i2w[11217]



