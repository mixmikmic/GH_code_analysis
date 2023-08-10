from __future__ import unicode_literals 

import gzip
import json

# load json ecommerce dump
corpus = list()
with gzip.open('ecommerce.json.gz') as fp:
    for line in fp:
        entry = line.decode('utf8')
        corpus.append(json.loads(entry))

from pprint import pprint
pprint(corpus[0])

print corpus[0]['descr']

import gensim
print gensim.summarization.summarize(corpus[0]['descr'])

len(corpus)

# let's build a classifier for product categories
# for speed up the example lets only consider the first 10k products
dataset = list()
for entry in corpus[:10000]:
    if 'cat' in entry:
        dataset.append( (entry['name'], entry['cat'].lower().strip()) )

len(dataset)

pprint(dataset[:10])

# how many distinc categories do we have and how many items per category?
from collections import Counter
counter = Counter([cat for prod, cat in dataset])

pprint(counter.most_common())

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import nltk
stopwords = nltk.corpus.stopwords.words('portuguese')

classifier = Pipeline([('vect', TfidfVectorizer()), ('clf', SVC(kernel='linear', probability=True))])
encoder = LabelEncoder()
# Please check on http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html

data = [prod for prod, cat in dataset]
labels = [cat for prod, cat in dataset]
len(data)

target = encoder.fit_transform(labels)

encoder.classes_.item(1)

classifier.fit(data, target)

classifier.predict(["Refrigerador Brastemp com função frostfree"])

print encoder.classes_[9]

probs = classifier.predict_proba(["Ventilador"])

guess = [( class_, probs.item(n)) for n, class_ in enumerate(encoder.classes_)]
pprint(guess)

from operator import itemgetter
for cat, proba in sorted(guess, key=itemgetter(1), reverse=True):
    print '{}: {:.4f}'.format(cat,proba)



