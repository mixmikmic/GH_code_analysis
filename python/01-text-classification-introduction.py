import pandas as pd
df = pd.read_csv('../data/spam.csv', sep='\t')
df.head()

data = df['Text']
target = df['Target'].replace('ham', 1).replace('spam', 0)
names = ['spam', 'ham']
print(data[:5])
print(target[:5])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
print('Train size: {}'.format(len(X_train)))
print('Test size: {}'.format(len(X_test)))

from nltk.tokenize.casual import casual_tokenize

sms = data[4]
print(sms)

tokenizer = lambda text: casual_tokenize(text, preserve_case=False)

print(tokenizer(sms))

from nltk.corpus import stopwords
print(stopwords.words('english'))
stopword_tokenizer = lambda text: [w for w in tokenizer(text) if not w in set(stopwords.words('english'))]

stopword_tokenizer(sms)

from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

stem_tokenizer = lambda text: [stemmer.stem(w) for w in stopword_tokenizer(text)]

print (stem_tokenizer(sms))

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(tokenizer=stem_tokenizer)
vectorizer.fit(X_train)
print (vectorizer.transform([sms]))

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(vectorizer.transform(X_train))
print(tfidf_transformer.transform(vectorizer.transform([sms])))

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier

clf_pipeline = Pipeline([('vec', vectorizer),
                         ('tfidf', tfidf_transformer),
                         #('lr', LogisticRegression())
                         #('gbc', GradientBoostingClassifier(n_estimators=100, max_depth=4))
                         ('svm', svm.SVC(kernel='linear'))
                        ])
clf_pipeline.fit(X_train, y_train)

from sklearn import metrics
from sklearn.metrics import accuracy_score

y_pred = clf_pipeline.predict(X_test)

print ("Test accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
print ()
print(metrics.classification_report(y_test, y_pred, digits=4))

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

y_pred = clf_pipeline.predict(X_train)

print ("Train accuracy: {:.2f}".format(accuracy_score(y_train, y_pred)))
print ()
print(metrics.classification_report(y_train, y_pred, digits=4))

