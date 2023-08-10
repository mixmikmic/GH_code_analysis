categories = ['rec.sport.baseball', 'rec.sport.hockey',
               'comp.graphics', 'sci.med']

from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train',
                                  categories=categories, shuffle=True, random_state=42)

twenty_train.target_names

len(twenty_train.data)

len(twenty_train.target)

# the newsgroup header.
print("\n".join(twenty_train.data[0].split("\n")[:3]))

print(twenty_train.target_names[twenty_train.target[0]])

twenty_train.target[:10]

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

count_vect.vocabulary_.get(u'algorithm')

count_vect.vocabulary_.get(u'ergonomic')

# and something that does not exist returns null
count_vect.vocabulary_.get(u'deodato')

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_counts, twenty_train.target)

docs_new = [
    'home runs are exciting',
    'OpenGL on the GPU is fast',
    'diagnosis of a viral respiratory infection',
    'blue line power play goals are rock-em-sock-em'
]

X_new_counts = count_vect.transform(docs_new)

predicted = clf.predict(X_new_counts)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

from sklearn.pipeline import Pipeline

import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
    categories=categories, shuffle=True, random_state=42)

docs_test = twenty_test.data

text_clf_basic = Pipeline([('vect', CountVectorizer()),
                     ('clf', MultinomialNB()),
])
text_clf_basic.fit(twenty_train.data, twenty_train.target)
predicted = text_clf_basic.predict(docs_test)
np.mean(predicted == twenty_test.target)  

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()), # ~91% without
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
])
text_clf.fit(twenty_train.data, twenty_train.target)  
predicted2 = text_clf.predict(docs_test)
np.mean(predicted2 == twenty_test.target)

from sklearn import metrics
print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))

metrics.confusion_matrix(twenty_test.target, predicted)

