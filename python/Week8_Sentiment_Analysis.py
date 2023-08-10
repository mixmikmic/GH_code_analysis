import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split
from sklearn import naive_bayes
from sklearn.metrics import roc_auc_score

df= pd.read_csv("../data/UMICH_SI650_Sentiment_Classification.txt", sep='\t', names=['liked', 'txt'])

df.head()

#TFIDF Vectorizer, just like before
stopset = set(stopwords.words('english'))
vectorizer = TfidfVectorizer(use_idf=True, lowercase=True, strip_accents='ascii', stop_words=stopset)

#in this case our dependent variable will be liked as 0 (didn't like the movie) or 1 (liked the movie)
y = df.liked

#convert df.txt from text to features
X= vectorizer.fit_transform(df.txt)

#6918 observations x 2022 unique words.
print (y.shape)
print (X.shape)

#Test Train Split as usual
X_train, X_test,y_train, y_test = train_test_split(X, y, random_state=42)

#we will train a naive_bayes classifier
clf = naive_bayes.MultinomialNB()
clf.fit(X_train, y_train)

#We can test our model's accuracy like this:

roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])

movie_reviews_array=np.array(["Jupiter Ascending was a disapointing and terrible movie"])

movie_review_vector = vectorizer.transform(movie_reviews_array)

print (clf.predict(movie_review_vector))



