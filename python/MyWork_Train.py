import pickle
import re
import numpy as np
import tensorflow as tf
from collections import Counter

title_file = open('data/Wechat_title.parsed.txt', encoding='utf8')
content_file = open('data/Wechat_content.parsed.txt', encoding='utf8')

title_line = title_file.readlines()
content_line = content_file.readlines()

pkl_file = open('data.pkl', 'wb')
data  = {
    'title':title_line,
    'content':content_line,
    'keyword':None
}

pickle.dump(data, pkl_file)
pkl_file.close()

pkl_file = open('data.pkl', 'rb')

data = pickle.load(pkl_file)

pkl_file.close()

title = data['title']
content = data['content']
#title_words = [t.split() for t in title]
#content_words = [c.split() for c in content]

from itertools import chain

def generate_vocab(lst):
    word_counter = Counter([w for curr_line in lst for w in curr_line.split() ])
    return word_counter

vocabcounter = generate_vocab(title+content)
vocabcounter = sorted(vocabcounter.items(), key=lambda d:d[1], reverse=True)
#print(sorted(vocabcounter.items(), key=lambda d:d[1], reverse=True))
#print(vocabsort)
vocab = [w[0] for w in vocabcounter ]
# print(vocab)

# create look up table for vocab: word2idx and idx2word
word2idx = {w:i for i, w in enumerate(vocab)}
idx2word = {i:w for i, w in enumerate(vocab)}
print(idx2word[0])
print(word2idx['可以'])

from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')

plt.plot([w[1] for w in vocabcounter])
plt.gca().set_xscale("log", nonposx="clip")
plt.gca().set_yscale("log", nonposy="clip")
plt.title("Word distribution in title and content.")
plt.xlabel("words in the title and content")
plt.ylabel("word frequency")

# Code here 

def load_wordembedding(file_dir):
    
    file = open(file_dir, encoding='utf8')
    line = file.readline()
    words = []
    vectors = []
    while line:
        temp_tokens = re.split(",", line) # line.split(",")
        if len(temp_tokens) != 201:
            print("error length: %d\n"%len(temp_tokens), temp_tokens[0:3])
            line = file.readline()
            continue
        words.append(temp_tokens[0])
        vectors.append([float(x) for x in temp_tokens[1:201]])
        line = file.readline()
    return np.array(words), np.array(vectors)

def test_wordembedding():
    words, vectors = load_wordembedding("data/word_vector.csv")
    print(words.shape, vectors.shape)
    assert words.shape== (9846,)
    assert vectors.shape == (9846,200)

#code here: test the data we have so far.  
test_wordembedding()
#print(title_words[0])
#print(content_words[0])
words, vectors = load_wordembedding("data/word_vector.csv")

#embedding_words2idx = {w:i for i, w in enumerate(words)}
#print(words[:10])
matched_words = []
idx_vocab2idx_embedding={}
for i in range(len(words)):
    w = words[i]
    if w in vocab:
        matched_words.append(w)
        idx_vocab2idx_embedding[word2idx[w]] = i

print(idx_vocab2idx_embedding[0], idx2word[0], words[189])

import pickle
FN = 'vocabulary-embedding'
with open('data/%s.pkl'%FN,'wb') as fp:
    pickle.dump((matched_words, vectors, idx2word, word2idx, idx_vocab2idx_embedding),fp,-1)
    

X = [ c.split() for c in content]
Y = [ t.split() for t in title]

with open('data/%s.data.pkl'%FN,'wb') as fp:
    pickle.dump((X, Y), fp, -1)

maxlend=25 # 0 - if we dont want to use description at all
maxlenh=25
maxlen = maxlend + maxlenh
rnn_size = 512 # must be same as 160330-word-gen
rnn_layers = 3  # match FN1
batch_norm=False

activation_rnn_size = 40 if maxlend else 0

# training parameters
seed=42
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
optimizer = 'adam'
LR = 1e-4
batch_size=64
nflips=10

nb_train_samples = 200 #30000
nb_val_samples = 100 #3000

nb_unknown_words = 10 # we use the last 10 words to replace unknown words.  Actually, we can just use one for it, e.g., (的/了)

# load data from pickle file 
import pickle
FN = 'vocabulary-embedding'
with open('data/%s.pkl'%FN, 'rb') as fp:
    matched_words_, vectors, idx2word, word2idx, idx_vocab2idx_embedding = pickle.load(fp)
    
vocab_size, embedding_size = vectors.shape

with open('data/%s.data.pkl'%FN, 'rb') as fp:
    X, Y = pickle.load(fp)
tf.global_variables_initializer
print(vocab_size, embedding_size, len(X), len(Y))

print ('number of examples:',len(X),len(Y))
print ('dimension of embedding space for words:',embedding_size)
print ('vocabulary size', vocab_size, 'the last %d words can be used as place holders for unknown/oov words'%nb_unknown_words)
print ('total number of different words',len(idx2word), len(word2idx))
print ('number of words outside vocabulary which we can substitue using glove similarity', len(idx_vocab2idx_embedding))
print ('number of words that will be regarded as unknonw(unk)/out-of-vocabulary(oov)',len(idx2word)-vocab_size-len(idx_vocab2idx_embedding))

from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)
len(X_train), len(Y_train), len(X_test), len(Y_test)
#print(X_train[:3])
#print(Y_train[:3])





x = tf.placeholder(tf.float32, shape=(batch_size, embedding_size))









