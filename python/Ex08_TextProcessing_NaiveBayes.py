import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
df = pd.read_csv('../datasets/emails.csv', sep=',', names=["label", "text"]) 
df.shape

df.head()

## lets see how many "good" and "bad" emails we have
df.groupby(["label"]).count()

# simple example of CountVectorizer. Try to understand whats going on here.
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(["Hello good day", "Good day to to you"])
print (cv.vocabulary_)
pd.DataFrame(X.todense(), columns=['day', 'good', 'hello', 'to', 'you'])

cv.transform(["day day good"]).toarray()

## lets train a model that predicts if an email is good or bad
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB

vectorizer = CountVectorizer()
text_vectorized = vectorizer.fit_transform(df.text)
text_vectorized_array = text_vectorized.toarray()
gnb = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(text_vectorized_array, df.label)
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Number of mislabeled email out of a total %d test emails : %d"
      % (X_test.shape[0],(y_test != y_pred).sum()))

## Predicitng a single email 
email_text = ["This is a new email text I wonder if the model is going to classify it as good or bad"]

## Notice that we only use transform, not fit_transform! We must convert the text to the same features
## that we used to fit our model. Also note that new words in email_text (that didn't appear in email.csv)
## would be ignored. 
email_text_vectorized = vectorizer.transform(email_text) ## creates a single row with 11351 features
gnb.predict(email_text_vectorized.toarray())

## In our corpus there are 11351 words. email_text_vectorized is a list with 11351 elements, each element 
## holds a number which represents the number of times that word appeared in email_text. We expact to have max 20 1's, because 
## thats the length of email_text. If a word appears in email_text but not in our corpus it is ignored
sum (email_text_vectorized.toarray()[0])

from sklearn.feature_extraction.text import TfidfVectorizer

tfidVectorizer = TfidfVectorizer()
text_vectorized = tfidVectorizer.fit_transform(df.text)
text_vectorized_array = text_vectorized.toarray()

gnb = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(text_vectorized_array, df.label)

gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)

print("Number of mislabeled email out of a total %d test emails : %d"
      % (X_test.shape[0],(y_test != y_pred).sum()))



