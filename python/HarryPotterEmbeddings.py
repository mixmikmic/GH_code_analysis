# Importing Packages
import codecs
import os
import re
import time
import gensim
import pandas as pd
import glob
import nltk
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('pylab inline')

# Books present
books = sorted(glob.glob("data/harrypotter/*.txt"))

print "Available Books: \n"
for i in books:
    print i.split("/")[2].split("_")[0]

# Read data from all books to single corpus variable
temp = ""
t = ""
chars = []
for book in books:
    print 
    print "Reading " + str(book).split("/")[2].split("_")[0]
    with codecs.open(book, "rb", "utf-8") as infile:
        temp += infile.read()
        chars.append(len(temp))
        print "Characters read so far " + str(len(temp))

lens = []
lens.append(chars[0])
for i in xrange(1, len(chars)):
    lens.append(chars[i] - chars[i-1])
lens

y = lens
N = len(y)
x = [i+1 for i in range(N)]
width = 1/1.5

pylab.xlabel("Book")
pylab.ylabel("Length")
plt.bar(x, y, width, color="red", align='center')

# Split into sentences
sentences = nltk.tokenize.sent_tokenize(temp)
print "Total Sentences are " + str(len(sentences))

# sentences to list of words
sent_words = []
total_tokens = 0
for raw_sent in sentences:
    clean = nltk.word_tokenize(re.sub("[^a-zA-Z]"," ", raw_sent.strip().lower()))
    tokens = [i for i in clean if len(i) > 1]
    total_tokens += len(tokens)
    sent_words.append(tokens)

print "Total tokens are " + str(total_tokens)

# capture collocations
bigram = gensim.models.Phrases(sent_words)
final = []
for s in sent_words:
    processed_sent = bigram[s]
    final.append(processed_sent)

# Sample first two sentences
final[:2]

num_features = 300
min_word_count = 3
num_workers = 3
context_size = 7
seed = 1

model = gensim.models.Word2Vec(sent_words, window=context_size,                                min_count=min_word_count, workers=num_workers,                                seed=seed, size=num_features
                              )

model.train(sent_words)

print 'Vocabulary ' + str(len(model.wv.vocab))

if not os.path.exists("model"):
    os.makedirs("model")
model.save(os.path.join("model", "harry2vec.w2v"))

# words 
print 'Similar kind of words for AZKABAN: '
print [i[0] for i in model.wv.most_similar('azkaban')]
print '\n'
print 'Similar kind of words for SNAPE: '
print [i[0] for i in model.wv.most_similar('snape')]

start = time.time()
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix = model.wv.syn0
all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)
stop = time.time() - start
print 'Time taken is ' + str(stop)

points = pd.DataFrame(
    [
        (word, coords[0], coords[1])
        for word, coords in [
            (word, all_word_vectors_matrix_2d[model.vocab[word].index])
            for word in model.vocab
        ]
    ],
    columns=["word", "x", "y"]
)
points.head(20)

sns.set_context("poster")
points.plot.scatter("x", "y", s=10, figsize=(20, 12))



