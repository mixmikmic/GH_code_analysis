from __future__ import unicode_literals

from pattern.web import Twitter

twitter = Twitter(language='pt')

dataset = list()
for tweet in twitter.search('sensacional', cached=False, count=1000):
    dataset.append( (tweet.text, 'pos') )

for tweet in twitter.search('odiei', cached=False, count=1000):
    dataset.append( (tweet.text, 'neg') )

len(dataset)

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

classifier = Pipeline([('vect', TfidfVectorizer()), ('clf', SVC(kernel='linear', probability=True))])
encoder = LabelEncoder()

data = [prod for prod, cat in dataset]
labels = [cat for prod, cat in dataset]
len(data)

target = encoder.fit_transform(labels)
classifier.fit(data, target)

encoder.classes_.item( classifier.predict(['incrível produto, muito bom!'])[0])



