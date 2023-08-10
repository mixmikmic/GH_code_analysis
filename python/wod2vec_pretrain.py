import os
import re
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile
from tensorflow.python.framework import ops
import collections
from tqdm import tqdm

batch_size=32
vocabulary_size=7
embedding_size=300

# Create a text cleaning function
def clean_text(text_string):
    text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)
    text_string = " ".join(text_string.split())
    text_string = text_string.lower()
    return(text_string)

def tokenizer(text):
    text = [document.lower().replace('\n', '').split() for document in text]
    return text

# sentences = ' '.join(text_data_train)
# words = sentences.split()
words=['man','king','mango','roads','dinner','food','morning']
word = tokenizer(words)
print('Data size', len(words))
 

# get unique words and map to glove set
print('Unique word count', len(set(words))) 

word

# drop rare words
vocabulary_size = 8


def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count = unk_count + 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)

data

count

dictionary

reverse_dictionary

GLOVE_DIR=""
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.42B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# prepare embedding matrix
num_words = vocabulary_size
embedding_matrix = np.zeros((num_words, embedding_size))
for word, i in dictionary.items():
    if i >= vocabulary_size:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        print("{} Words:{} Vectors:{}".format(i,word,embedding_vector))

embedding_matrix.shape

i=1
vocab=[]
for x in count[1:]:
    vocab.append(count[i][0])
    i=i+1

vocab

# Change texts into numeric vectors
vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(1)
#fit the vocab from glove
pretrain = vocab_processor.fit(vocab)
#transform inputs
text_processed = np.array(list(vocab_processor.transform(words)))

text_processed

words

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(nb_words=7)
tokenizer.fit_on_texts(words)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

word_index

graph = tf.Graph()
with graph.as_default(), tf.device('/cpu:0'):

    # Input data.
    test_dataset = tf.constant(data, dtype=tf.int32)
    # Variables.
    embeddings = tf.Variable(
        tf.random_uniform([num_words, embedding_size], -1.0, 1.0))
    embedding_placeholder = tf.placeholder(tf.float32, [num_words, embedding_size])
    embedding_init = embeddings.assign(embedding_placeholder)
    
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, test_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding_matrix})
    
    print('Initialized')
    sim = similarity.eval()
    for i in range(6):
        valid_word = dictionary[words[i]]
        top_k = 1 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % reverse_dictionary.get(valid_word)
        for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log = '%s %s,' % (log, close_word)
        print(log)



