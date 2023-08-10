# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #quoting = 3 --ignoring double quotes

dataset.head(n=5)

dataset.shape

dataset['Review'][0]

import re
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
review

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) ## .sub remove punctations and numbers
    review = review.lower() # lowering the capital letter
    review = review.split() # tokenizing like word_tokenize
    ps = PorterStemmer() # used for stemming(removing words having no mean e.g the,a,if etc. )
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # remove stemming part
    review = ' '.join(review)
    corpus.append(review)

corpus[0:5]

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

Accuracy = (55+91)/200
print ("Accuracy is",Accuracy)

