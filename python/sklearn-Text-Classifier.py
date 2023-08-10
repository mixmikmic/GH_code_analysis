#!/Users/jessica/anaconda/lib/python3.5

import sklearn

from sklearn.datasets import fetch_20newsgroups

categories= ['alt.atheism', 'soc.religion.christian',
             'comp.graphics', 'sci.med']

# Load 20 Newsgroups training dataset
twenty_train = fetch_20newsgroups(subset='train', 
                                 categories=categories, shuffle=True,
                                random_state=42)

twenty_train

twenty_train.target_names

twenty_train.data[0]

len(twenty_train.filenames)

print(twenty_train.filenames[0],"\n", twenty_train.filenames[1])

# Print first three lines of first loaded file
print("\n".join(twenty_train.data[0].split("\n")[:3]))

# Print category label of first loaded file
print(twenty_train.target_names[twenty_train.target[0]])

twenty_train.target[:10]

for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts.shape

count_vect.vocabulary_.get(u'algorithm')

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)

# Predict the outcome on a new document
docs_new = ['God is love', 'OpenGL on the GPU is fast', 'There is no God']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))

## Building a Pipeline

from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())
                    ])

text_clf = text_clf.fit(twenty_train.data, twenty_train.target)

## Evaluation of the performance on the test set
import numpy as np
twenty_test = fetch_20newsgroups(subset='test',
                                categories=categories, shuffle=True,
                                random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == twenty_test.target)

