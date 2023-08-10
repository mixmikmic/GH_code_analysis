from wordvector import WordVector
from windowmodel import WindowModel
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

print('Building training set ...')
x, y = WindowModel.build_training_set(word_array)

# shuffle and split 10% validation data
x_shuf, y_shuf = sklearn.utils.shuffle(x, y, random_state=0)
split = round(x_shuf.shape[0]*0.9)
x_val, y_val = (x_shuf[split:, :], y_shuf[split:, :])
x_train, y_train = (x[:split, :], y[:split, :])

print('Training set built.')
graph_params = {'batch_size': 32,
                'vocab_size': np.max(x)+1,
                'embed_size': 64,
                'hid_size': 64,
                'neg_samples': 64,
                'learn_rate': 0.01,
                'momentum': 0.9,
                'embed_noise': 0.1,
                'hid_noise': 0.3,
                'optimizer': 'Momentum'}
model = WindowModel(graph_params)
print('Model built. Vocab size = {}. Document length = {} words.'
      .format(np.max(x)+1, len(word_array)))

print('Training ...')
results = model.train(x_train, y_train, x_val, y_val, epochs=120, verbose=False)

word_vector_embed = WordVector(results['embed_weights'], dictionary)
word_vector_nce = WordVector(results['nce_weights'], dictionary)

print(word_vector_embed.most_common(100))

word = "seven"
print('Embedding layer: 8 closest words to:', "'" + word + "'")
print(word_vector_embed.n_closest(word=word, num_closest=8, metric='cosine'), '\n')
print('Hidden-to-output layer: 8 closest words to:', "'" + word + "'")
print(word_vector_nce.n_closest(word=word, num_closest=8, metric='cosine'))

word = "laughing"
print('8 closest words to:', "'" + word + "'")
print(word_vector_nce.n_closest(word=word, num_closest=8, metric='cosine'))

word = "mr"
print('8 closest words to:', "'" + word + "'")
print(word_vector_nce.n_closest(word=word, num_closest=8, metric='cosine'))

print(word_vector_nce.analogy('had', 'has', 'was', 5))

print(word_vector_nce.analogy('boot', 'boots', 'arm', 5))

# grab 100 word passage from book
reverse_dict = word_vector_nce.get_reverse_dict()
passage = [x for x in map(lambda x: reverse_dict[x], word_array[12200:12300])]

# print passage with some crude formatting (e.g. space after comma)
readable = ''
for word in passage:
    if word == '"':
        readable += word
    elif word in ['?', '!', '.', ',']:
        readable += word + ' '
    else: 
        readable += ' ' + word
print(readable)

# use model to replace words in original passage with predicted words
# need to grab 2 words before and after passage
x, y = WindowModel.build_training_set(word_array[(12200-2):(12300+2)])
y_hat = model.predict(x, 120)
passage_predict = [x for x in map(lambda x: reverse_dict[x], y_hat[0])]

# print predicted passage
readable = ''
for word in passage_predict:
    if word == '"':
        readable += word
    elif word in ['?', '!', '.', ',']:
        readable += word + ' '
    else: 
        readable += ' ' + word
print(readable)

