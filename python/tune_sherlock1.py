from wordvector import WordVector
from windowmodel import WindowModel
from plot_util import plot_results
import docload

import numpy as np
import sklearn.utils

files = ['../data/adventures_of_sherlock_holmes.txt',
        '../data/hound_of_the_baskervilles.txt',
        '../data/sign_of_the_four.txt']
word_array, dictionary, num_lines, num_words = docload.build_word_array(
    files, vocab_size=50000, gutenberg=True)
print('Document loaded and processed: {} lines, {} words.'
      .format(num_lines, num_words))

x, y = WindowModel.build_training_set(word_array)

# shuffle and split 10% validation data
x_shuf, y_shuf = sklearn.utils.shuffle(x, y, random_state=0)
split = round(x_shuf.shape[0]*0.9)
x_val, y_val = (x_shuf[split:, :], y_shuf[split:, :])
x_train, y_train = (x[:split, :], y[:split, :])
results_list = []

count = 0
for embed_size in [32, 64, 128]:
    for hid_size in [32, 64, 128]:
        print('{}) embed size = {}, hid_size = {}'.format(count,embed_size, hid_size))
        count += 1
        graph_params = {'batch_size': 32,
                        'vocab_size': np.max(x)+1,
                        'embed_size': embed_size,
                        'hid_size': hid_size,
                        'neg_samples': 64,
                        'learn_rate': 0.003,
                        'embed_noise': 0.1,
                        'hid_noise': 0.1,                        
                        'optimizer': 'RMSProp'}
        model = WindowModel(graph_params)
        results = model.train(x_train, y_train, x_val, y_val, epochs=80, verbose=False)
        results_list.append((graph_params, results))

plot_results(results_list)

results_list2 = []

count = 0
for learn_rate in [0.0003, 0.001, 0.003, 0.01, 0.03]:
    for he_size in [64, 128]:
        print('{}) learn_rate = {}, hid_size = embed_size = {}'
              .format(count, learn_rate, he_size))
        count += 1
        graph_params = {'batch_size': 32,
                        'vocab_size': np.max(x)+1,
                        'embed_size': he_size,
                        'hid_size': he_size,
                        'neg_samples': 64,
                        'learn_rate': learn_rate,
                        'embed_noise': 0.1,
                        'hid_noise': 0.1,                        
                        'optimizer': 'RMSProp'} 
        model = WindowModel(graph_params)
        results = model.train(x_train, y_train, x_val, y_val, epochs=120, verbose=False)
        results_list2.append((graph_params, results))

plot_results(results_list2)

