import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

from gensim.models.word2vec import Word2Vec

from collections import OrderedDict

models = OrderedDict([
    (year, Word2Vec.load('models/bpo/{}.bin'.format(year)))
    for year in range(1720, 1960, 20)
])

def cosine_series(anchor, query):
    
    series = OrderedDict()
    
    for year, model in models.items():
        
        series[year] = (
            model.similarity(anchor, query)
            if query in model else 0
        )

    return series

import numpy as np
import statsmodels.api as sm

def lin_reg(series):

    x = np.array(list(series.keys()))
    y = np.array(list(series.values()))

    x = sm.add_constant(x)

    return sm.OLS(y, x).fit()

def plot_cosine_series(anchor, query, w=5, h=4):
    
    series = cosine_series(anchor, query)
    
    fit = lin_reg(series)

    x1 = list(series.keys())[0]
    x2 = list(series.keys())[-1]

    y1 = fit.predict()[0]
    y2 = fit.predict()[-1]
    
    print(query)
    
    plt.figure(figsize=(w, h))
    plt.ylim(0, 1)
    plt.title(query)
    plt.xlabel('Year')
    plt.ylabel('Similarity')
    plt.plot(list(series.keys()), list(series.values()))
    plt.plot([x1, x2], [y1, y2], color='gray', linewidth=0.5)
    plt.show()

plot_cosine_series('literature', 'poetry')
plot_cosine_series('literature', 'fiction')
plot_cosine_series('literature', 'polite')

import enchant

dictionary = enchant.Dict('en_US')

def union_neighbor_vocab(anchor, topn=200):
    
    vocab = set()
    
    for year, model in models.items():
        similar = model.most_similar(anchor, topn=topn)
        vocab.update([s[0] for s in similar if dictionary.check(s[0])])
        
    return vocab

union_vocab = union_neighbor_vocab('literature')

data = []
for token in union_vocab:
    
    series = cosine_series('literature', token)
    fit = lin_reg(series)
    
    data.append((token, fit.params[1], fit.pvalues[1]))

import pandas as pd

df1 = pd.DataFrame(data, columns=('token', 'slope', 'p'))

pd.set_option('display.max_rows', 1000)

df1.sort_values('slope', ascending=False).head(50)

for i, row in df1.sort_values('slope', ascending=False).head(20).iterrows():
    plot_cosine_series('literature', row['token'], 3, 2)

df1.sort_values('slope', ascending=True).head(50)

for i, row in df1.sort_values('slope', ascending=True).head(20).iterrows():
    plot_cosine_series('literature', row['token'], 3, 2)

def intersect_neighbor_vocab(anchor, topn=1000):
    
    vocabs = []
    
    for year, model in models.items():
        similar = model.most_similar(anchor, topn=topn)
        vocabs.append(set([s[0] for s in similar if dictionary.check(s[0])]))
        
    return set.intersection(*vocabs)

intersect_vocab = intersect_neighbor_vocab('literature')

data = []
for token in intersect_vocab:
    
    series = cosine_series('literature', token)
    fit = lin_reg(series)
    
    data.append((token, fit.params[1], fit.pvalues[1]))

import pandas as pd

df2 = pd.DataFrame(data, columns=('token', 'slope', 'p'))

df2.sort_values('slope', ascending=False)

for i, row in df2.sort_values('slope', ascending=False).iterrows():
    plot_cosine_series('literature', row['token'], 3, 2)

