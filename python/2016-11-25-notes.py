import nltk

nltk.__version__

import sklearn

sklearn.__version__

nltk.download('movie_reviews')

from nltk.corpus import movie_reviews

fileids = movie_reviews.fileids()

fileids[0]

len(fileids)

print(movie_reviews.categories(fileids[0]))
print(movie_reviews.categories(fileids[999]))
print(movie_reviews.categories(fileids[1000]))
print(movie_reviews.categories(fileids[1999]))

review_1000 = movie_reviews.raw(fileids[1000])
review_1000[:200]

nltk.download('stopwords')

from nltk.corpus import stopwords

stop = stopwords.words('english')

print(stop)

len(stop)

from string import punctuation

punctuation

import re

re.sub('[aeiouAEIOU]','X','Pen pineapple apple pen')

re.sub('[' + punctuation + ']','7',"What?! What?! I don't know what.")

no_punc_review_1000 = re.sub('[' + punctuation + ']',' ',review_1000)
clean_review_1000 = [word.lower() for word in no_punc_review_1000.split() if word.lower() not in stop]

print(clean_review_1000[:20])

def clean_review(review):
    no_punc = re.sub('[' + punctuation + ']',' ',review)
    clean_review = [word.lower() for word in no_punc.split() if word.lower() not in stop]
    return clean_review

clean_review("Worst! Movie! Ever! But not as bad as 'Titanic'; I really didn't like that movie... But I saw it twice?!")

len(movie_reviews.words())

movie_words = [word.lower() for word in movie_reviews.words()
               if word.lower() not in stop and word.lower() not in punctuation]

from collections import Counter

top_2000 = [item[0] for item in Counter(movie_words).most_common(2000)]

Counter(['a','a','b','c','c','c'])

print(top_2000[:20])

import numpy as np

def word2vector(fileid):
    vec = np.zeros(2000)
    review = re.sub("[^a-zA-Z]"," ", movie_reviews.raw(fileid)).split()
    for i in range(0,2000):
        if top_2000[i] in review:
            vec[i] = 1
        else:
            vec[i] = 0
    return vec

movie_reviews.raw(fileids[0])[:100]

word2vector(fileids[0])

n_files = len(fileids)
X = np.zeros((n_files,2000))
for i in range(0,n_files):
    X[i,:] = word2vector(fileids[i])

y = [0 if movie_reviews.categories(fileid) == ['neg'] else 1 for fileid in fileids]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()

clf.fit(X_train,y_train)

clf.score(X_test,y_test)

