from __future__ import unicode_literals
from __future__ import division

a = 'Refrigerador Brastemp CFR45 20L frostfree'
b = 'Geladeira Brastemp CFR45 20L com desgelo automático'

# Value for similar tokens:
tokensA = a.split()
tokensB = b.split()
set(tokensA).intersection(tokensB)

similar = len(set(tokensA).intersection(tokensB))
total = len(set(tokensA).union(tokensB))
print '{} similars from {} tokens: {:0.2f}% of similarity'.format(similar, total, similar/total*100)

# several other metrics. See jellyfish, fuzzywuzzy, metaphone, etc
import jellyfish
import fuzzywuzzy
import metaphone

print metaphone.doublemetaphone('caza')
print metaphone.doublemetaphone('casa')

# The Jaro–Winkler distance metric is designed and best suited for short strings such as person names. 
jellyfish.jaro_distance(a,b)

# read the corpus
import codecs

# this could be done in a iterate way for performance in huge corpus
with codecs.open('corpus.txt', encoding='utf8') as fp:
    corpus = fp.read()

# sent and word tokenize with ntlk
# It may take a while to process
from nltk import sent_tokenize, word_tokenize
sentences = [[w.lower() for w in word_tokenize(sentence, language='portuguese')] for sentence in sent_tokenize(corpus, language='portuguese')]

# It may take a while to train
from gensim.models import Word2Vec
model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=8)
model.init_sims(replace=True)

model.most_similar('geladeira')

tokensA = [t.lower() for t in tokensA]
vectorsA = sum([model[token] for token in tokensA if token in model.vocab])

tokensB = [t.lower() for t in tokensB]
vectorsB = sum([model[token] for token in tokensB if token in model.vocab])

from nltk.cluster.util import cosine_distance
print 'Similarity: {}'.format(abs(1 - cosine_distance(vectorsA, vectorsB)))



