import spacy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import codecs
import math

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from gensim.models import KeyedVectors

# Display plots in this notebook, instead of externally. 
from pylab import rcParams
rcParams['figure.figsize'] = 16, 8
get_ipython().magic('matplotlib inline')
nlp = spacy.load('es')

doc = nlp('Las naranjas y las manzanas se parecen')
pd.DataFrame([[word.text, word.tag_, word.pos_] for word in doc], columns=['Token', 'TAG', 'POS'])

pd.DataFrame([[word.text, math.exp(word.prob), word.prob] for word in doc], columns=['Token', 'Prob', 'Log Prob'])

doc2 = nlp(u"La premier alemana Angela Merkel visitó Buenos Aires esta semana")
for ent in doc2.ents:
    print('{} \t {}'.format(ent, ent.label_))

orange = doc[1]
apple = doc[4]
orange.similarity(apple)

doc.similarity(doc2)

from bs4 import BeautifulSoup
import urllib.request
html = urllib.request.urlopen('http://api-editoriales.clarin.com/api/instant_articles?limit=100').read()
soup = BeautifulSoup(html, 'html.parser')
news = [
    BeautifulSoup(c.text, 'html.parser').text.split('function', 1)[0] 
    for item in soup.findAll('item') 
        for c in item.children if c.name == 'content:encoded'
]
news[0][0:100]

corpus = nlp('\n'.join(news))

visited = {}
nouns = []
for word in corpus:
    if word.pos_.startswith('N') and len(word.string) < 15:
        token = word.string.strip().lower()
        if token in visited:
            visited[token] += 1
            continue
        else:
            visited[token] = 1
            nouns.append(word)
nouns = sorted(nouns, key=lambda w: -visited[w.string.strip().lower()])[:150]
pd.DataFrame([[w.text, visited[w.string.strip().lower()]] for w in nouns], columns=['Noun', 'Freq'])

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y, s=1.0)
        plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
    plt.show()
#     plt.savefig(filename)

# Creating the tsne plot [Warning: will take time]
tsne = TSNE(perplexity=30.0, n_components=2, init='pca', n_iter=5000)

low_dim_embedding = tsne.fit_transform(np.array([word.vector for word in nouns]))

# Finally plotting and saving the fig 
plot_with_labels(low_dim_embedding, [word.text for word in nouns])



