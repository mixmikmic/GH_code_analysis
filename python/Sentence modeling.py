import re
import numpy as np

from gensim.models import Word2Vec
from nltk.corpus import gutenberg
from multiprocessing import Pool
from scipy import spatial
from sklearn.decomposition import PCA

sentences = list(gutenberg.sents('shakespeare-hamlet.txt'))   # import the corpus and convert into a list

print('Type of corpus: ', type(sentences))
print('Length of corpus: ', len(sentences))

print(sentences[0])    # title, author, and year
print(sentences[1])
print(sentences[10])

for i in range(len(sentences)):
    sentences[i] = [word.lower() for word in sentences[i] if re.match('^[a-zA-Z]+', word)]

print(sentences[0])    # title, author, and year
print(sentences[1])
print(sentences[10])

# set threshold to consider only sentences longer than certain integer
threshold = 5

for i in range(len(sentences)):
    if len(sentences[i]) < 5:
        sentences[i] = None

sentences = [sentence for sentence in sentences if sentence is not None] 

print('Length of corpus: ', len(sentences))

model = Word2Vec(sentences = sentences, size = 100, sg = 1, window = 3, min_count = 1, iter = 10, workers = Pool()._processes)

model.init_sims(replace = True)

# converting each word into its vector representation
for i in range(len(sentences)):
    sentences[i] = [model[word] for word in sentences[i]]

print(sentences[0])    # vector representation of first sentence

# define function to compute weighted vector representation of sentence
# parameter 'n' means number of words to be accounted when computing weighted average
def sent_PCA(sentence, n = 2):
    pca = PCA(n_components = n)
    pca.fit(np.array(sentence).transpose())
    variance = np.array(pca.explained_variance_ratio_)
    words = []
    for _ in range(n):
        idx = np.argmax(variance)
        words.append(np.amax(variance) * sentence[idx])
        variance[idx] = 0
    return np.sum(words, axis = 0)

sent_vectorized = []

# computing vector representation of each sentence
for sentence in sentences:
    sent_vectorized.append(sent_PCA(sentence))

# vector representation of first sentence
list(sent_PCA(sentences[0])) == list(sent_vectorized[0])

# define a function that computes cosine similarity between two words
def cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)

# similarity between 11th and 101th sentence in the corpus
print(cosine_similarity(sent_vectorized[10], sent_vectorized[100]))

