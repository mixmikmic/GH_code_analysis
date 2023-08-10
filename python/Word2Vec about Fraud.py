import nltk
import pandas as pd
# %pylab inline
from nltk.corpus import stopwords
import gensim
from gensim import corpora, models, similarities
import re
import random
import time

oag_data = pd.read_csv("../data/OAG Complaints-Online_Final.csv")

# oag_data.info()

oag_doc = list(oag_data['COMPLAINT_DESCRIPTION'])

# drop duplicate
oag_doc = list(set(oag_doc))
oag_doc = oag_doc[1:]
len(oag_doc)

oag_doc[0]

# data preprocessing
texts = []
for doc in oag_doc:
    try:
        re.split('\.|\,|\n| ',doc)
        single_doc = []
        for word in re.split('\.|\,|\n| ',doc):
            if word.lower() not in stopwords.words('english') and 'xx' not in word.lower() and len(word)>3 and '$' not in word and '--' not in word:
                single_doc.append(word.lower())
        texts.append(single_doc)
    except:
        pass
len(texts)

print texts[0]

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1] for text in texts]
len(texts)

# Phrases Detection
bigram = gensim.models.Phrases(texts, threshold=50.0)
phrases_texts = bigram[texts]

print phrases_texts[0]

# transform into dictionary
dictionary = corpora.Dictionary(phrases_texts)

# store the dictionary, for future reference
# dictionary.save('phrases_texts_oag.dict')

print (dictionary)

# gensim.models.word2vec.Word2Vec(sentences=None, 
# size=100, alpha=0.025, window=5, min_count=5, max_vocab_size=None, 
# sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, 
# negative=5, cbow_mean=1, hashfxn=<built-in function hash>, iter=5, 
# null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000)
# train word2vec
model = gensim.models.word2vec.Word2Vec(phrases_texts, size=60, sg=1, window=5, workers=2)

# save the model
# model.save('word2vec_model_oag')

model.most_similar(positive=['fraud'], topn=20)

model.most_similar(positive=['google'], topn=20)

model.most_similar(positive=["attorney_general"], topn=20)

company = ['uber','chase','citibank','google','facebook','yelp','apple','instagram','starbucks','amazon','airbnb','sony','dell','tinder','linkedin']
sim = [(i,model.similarity('fraud', i)) for i in company]
sorted(sim, key=lambda item: -item[1])



