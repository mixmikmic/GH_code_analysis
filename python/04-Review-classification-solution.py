import pandas as pd

reviews = pd.read_csv('../data/en_reviews.csv', sep='\t', header=None, names =['rating', 'text'])
reviews[35:45]

target = reviews['rating']
data = reviews['text']
names = ['Class 1', 'Class 2', 'Class 3','Class 4', 'Class 5']

#reduce number of classes
#target = list(map(lambda t: 1 if t==4 or t==5 else 0, target))
#names = ["Negative", "Positive"]

print(data[:5])
print(target[:5])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
print('Train size: {}'.format(len(X_train)))
print('Test size: {}'.format(len(X_test)))

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from nltk.tokenize.casual import casual_tokenize

from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

baselineModel = Pipeline([('vec', CountVectorizer(tokenizer=lambda x: casual_tokenize(x))),
                          ('clf', DummyClassifier(strategy='stratified'))
                         ])
                          
clfModel = Pipeline([('vec', CountVectorizer(tokenizer=lambda x: casual_tokenize(x))),
                     ('tfidf', TfidfTransformer()),
                     #('clf', MultinomialNB())
                     #('clf', LogisticRegression())
                     #('clf', GradientBoostingClassifier(n_estimators=300))
                     ('clf', SVC(kernel='linear'))
                    ])

baselineModel.fit(X_train, y_train)
clfModel.fit(X_train, y_train)

y_baseline = baselineModel.predict(X_test)
y_pred = clfModel.predict(X_test)

from sklearn import metrics

print("BASELINE REPORT")
print("Accuracy: {}".format(metrics.accuracy_score(y_test, y_baseline)))
print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, y_baseline))
print(metrics.classification_report(y_test, y_baseline,
                                            target_names=names))
print()
print("ML MODEL REPORT")
print("Accuracy: {}".format(metrics.accuracy_score(y_test, y_pred)))
print("Confusion matrix:")
print(metrics.confusion_matrix(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred,
                                            target_names=names))

