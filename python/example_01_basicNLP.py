import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining and the weather is sweet'])

count_1g = CountVectorizer()
count_2g = CountVectorizer(ngram_range=(2,2))
bag_1g = count_1g.fit_transform(docs)
bag_2g = count_2g.fit_transform(docs)

print '1-gram: ', count_1g.vocabulary_

print '2-gram: ', count_2g.vocabulary_

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()

np.set_printoptions(precision=2)
print tfidf.fit_transform(count_1g.fit_transform(docs)).toarray()

from nltk.stem.porter import PorterStemmer #pip install nltk

text='runners like running and thus they run'
print text

porter = PorterStemmer()
text_stem = []
for word in text.split():
    stem = porter.stem(word)
    text_stem.append(stem)
    print '%s -> %s'%( word, stem )
print text_stem

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')
text_stem_rm = []
for w in text_stem:
    if w not in stop:
        text_stem_rm.append(w)
print text_stem_rm



