from wordvector import WordVector
from windowmodel import WindowModel
import docload
from plot_util import plot_results

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
for embed_noise in [0.01, 0.1, 1]:
    for dummy in range(2):  # run each sim twice
        print('{}) embed noise = {}, run #{}'.format(count, embed_noise, dummy))
        count += 1
        graph_params = {'batch_size': 32,
                        'vocab_size': np.max(x)+1,
                        'embed_size': 128,
                        'hid_size': 128,
                        'neg_samples': 64,
                        'learn_rate': 0.003,
                        'embed_noise': embed_noise,
                        'optimizer': 'RMSProp'}
        model = WindowModel(graph_params)
        results = model.train(x_train, y_train, x_val, y_val, epochs=80, verbose=False)
        results_list.append((graph_params, results))

plot_results(results_list)

results_list2 = []
count = 0
for trunc_norm in [True, False]:
    for dummy in range(2):  # run each sim twice
        print('{}) truncated normal? {}, run #{}'.format(count, trunc_norm, dummy))
        count += 1
        graph_params = {'batch_size': 32,
                        'vocab_size': np.max(x)+1,
                        'embed_size': 128,
                        'hid_size': 128,
                        'neg_samples': 64,
                        'learn_rate': 0.003,
                        'embed_noise': 0.1,
                        'hid_noise': 0.1,
                        'trunc_norm': trunc_norm,
                        'optimizer': 'RMSProp'}  
        model = WindowModel(graph_params)
        results = model.train(x_train, y_train, x_val, y_val, epochs=80, verbose=False)
        results_list2.append((graph_params, results))

plot_results(results_list2)

results_list3 = []
count = 0
for hid_noise in [0.1, 1, 10]:
    for dummy in range(2):  # run each sim twice
        print('{}) hidden layer sigma {}, run #{}'.format(count, hid_noise, dummy))
        count += 1
        graph_params = {'batch_size': 32,
                        'vocab_size': np.max(x)+1,
                        'embed_size': 128,
                        'hid_size': 128,
                        'neg_samples': 64,
                        'learn_rate': 0.003,
                        'embed_noise': 0.1,
                        'hid_noise': hid_noise,
                        'optimizer': 'RMSProp'}
        model = WindowModel(graph_params)
        results = model.train(x_train, y_train, x_val, y_val, epochs=80, verbose=False)
        results_list3.append((graph_params, results))

plot_results(results_list3)

