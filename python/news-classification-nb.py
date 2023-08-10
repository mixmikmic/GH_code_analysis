import pandas as pd
news = pd.read_csv('uci-news-aggregator.csv').sample(frac=0.1)

len(news)

news.head()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

X = news['TITLE']
y = encoder.fit_transform(news['CATEGORY'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)

len(X_train)

len(X_test)

type(X_train)

X_train.head()

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=3)

train_vectors = vectorizer.fit_transform(X_train)
test_vectors = vectorizer.transform(X_test)

train_vectors

X_train.iloc[1]

train_vectors[1]

type(train_vectors)

# one-hot vector
train_vectors[1].toarray()

from sklearn.metrics import accuracy_score

train_vectors.toarray()

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(train_vectors.toarray(), y_train)

pred = clf.predict(test_vectors.toarray())
accuracy_score(y_test, pred, )

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(train_vectors, y_train)

pred = clf.predict(test_vectors)
accuracy_score(y_test, pred, )

from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(train_vectors, y_train)

pred = clf.predict(test_vectors.toarray())
accuracy_score(y_test, pred, )

