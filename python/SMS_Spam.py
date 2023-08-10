# Install NLTK package if not already installed. Uncomment the last line
# Anaconda prompt should be opened with admin previliges (Run as adminstrator)
#Download NLTK if not already downloaded

#!conda install nltk      #Uncomment if required
#import nltk              #Uncomment if required
#nltk.download()          #Uncomment if required

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score, confusion_matrix

df= pd.read_csv("sms_spam.csv")

df.head()

stopwords.words('english')

#TFIDF Vectorizer
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

vectorizer

df.type.replace('spam', 1, inplace=True)

df.type.replace('ham', 0, inplace=True)

df.head()

df.shape

##Our dependent variable will be 'spam' or 'ham' 
y = df.type

#Convert df.txt from text to features
X = vectorizer.fit_transform(df.text)

print (y.shape)
print (X.shape)

##Split the test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

##Train Naive Bayes Classifier
## Fast (One pass)
## Not affected by sparse data, so most of the 8605 words dont occur in a single observation
clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)

y_test

df[df.type.isnull()]

clf.predict_proba(X_test)[:,1]

##Check model's accuracy
print("ROC =",roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]))
confusion_matrix(y_test,clf.predict(X_test))

