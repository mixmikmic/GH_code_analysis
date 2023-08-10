import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

dataset_url = "https://gist.githubusercontent.com/avannaldas/9b80417c3ecaab4336e15ff98344e1cd/raw/b72939e627dde07535ceb6fda421dc17258c81d1/dataset-txtclf-spamorham"
data = pd.read_csv(dataset_url, sep='\t', names=['Status', 'Message'])

data.describe()

data['Status'].value_counts()

data.loc[data['Status']=='spam', 'Status']=1
data.loc[data['Status']=='ham', 'Status']=0

X=data["Message"]
y=data["Status"]

countVec = CountVectorizer()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

y_train = y_train.astype('int')
y_test = y_test.astype('int')

X_train_vec = countVec.fit_transform(X_train)
X_test_vec = countVec.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

predicted = clf.predict(X_test_vec)

accuracy_score(y_test, predicted)

clf.predict(X_test_vec[3])

countVec.inverse_transform(X_test_vec[3])



