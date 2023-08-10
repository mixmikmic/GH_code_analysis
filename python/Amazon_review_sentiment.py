import pandas as pd
import ast
import time
import re
import scipy

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn import svm

get_ipython().run_line_magic('matplotlib', 'inline')

# # read entire file
# with open('reviews_Toys_and_Games_5.json', 'rb') as f:
#     data = f.readlines()
    
# # remove the trailing "\n" from each element
# data = map(lambda x: x.rstrip(), data)
# print 'number reviews:', len(data)

# # for now, let's restrict to the first 30k obs
# # (if I want to run with the full dataset, will probably need to use ec2)
# data = data[:50000]

# # convert list to a dataframe
# t1 = time.time()
# df = pd.DataFrame()
# count = 0
# for r in data:
#     r = ast.literal_eval(r)
#     s  = pd.Series(r,index=r.keys())
#     df = df.append(s,ignore_index=True)
#     if count % 1000 ==0:
#         print count
#     count+=1
# t_process = time.time() - t1
# print 'process time (seconds):', t_process  #takes 8s for 1000, so should take 8*167/60=22min for 167k
# del data

# # the above step is slow, so let's write this to csv so we don't have to do it again
# df.to_csv('Toys_and_Games.csv', index=False)

df = pd.read_csv('Toys_and_Games.csv')
print df.shape
print df.head(3)

df['overall'].value_counts().plot(kind='bar', color='cornflowerblue')

print len(df)
df = df[df['reviewText'].notnull()]
print len(df)
df = df[df['overall'].notnull()]
print len(df)

X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], 
                                                   df['overall'],
                                                   test_size=.2, random_state=1)

# instantiate the vectorizer
vect = CountVectorizer()

# tokenize train and test text data
X_train_dtm = vect.fit_transform(X_train)
print "number words in training corpus:", len(vect.get_feature_names())
X_test_dtm = vect.transform(X_test)

nb = MultinomialNB()
get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')

# make class predictions
y_pred = nb.predict(X_test_dtm)

# calculate accuracy, precision, recall, and F-measure of class predictions
def eval_predictions(y_test, y_pred):
    print 'accuracy:', metrics.accuracy_score(y_test, y_pred)
    print 'precision:', metrics.precision_score(y_test, y_pred, average='weighted')
    print 'recall:', metrics.recall_score(y_test, y_pred, average='weighted')
    print 'F-measure:', metrics.f1_score(y_test, y_pred, average='weighted')
eval_predictions(y_test, y_pred)

# print message text for the first 3 false positives
print 'False positives:'
print
for x in X_test[y_test < y_pred][:2]:
    print x
    print

# print message text for the first 3 false negatives
print 'False negatives:'
print
for x in X_test[y_test > y_pred][:2]:
    print x[:500]
    print

import string
import nltk
from nltk.stem import WordNetLemmatizer

def no_punctuation_unicode(text):
    '''.translate only takes str. Therefore, to use .translate in the 
    tokenizer in TfidfVectorizer I need to write a function that converts 
    unicode -> string, applies .translate, and then converts it back'''
    str_text = str(text)
    no_punctuation = str_text.translate(None, string.punctuation)
    unicode_text = no_punctuation.decode('utf-8')
    return unicode_text

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

stoplist = [word.decode('utf-8') for word in nltk.corpus.stopwords.words('english')] 

wnl = WordNetLemmatizer()

def prep_review(review):
    lower_case = review.lower()
    no_punct = no_punctuation_unicode(lower_case)
    tokens = nltk.word_tokenize(no_punct)    # weird to tokenize within the vectorizer, 
    # but not sure how else to apply functions to each token
    has_letters = [t for t in tokens if re.search('[a-zA-Z]',t)]
    drop_numbers  = [t for t in has_letters if not hasNumbers(t)]
    drop_stops = [t for t in drop_numbers if not t in stoplist] 
    lemmed = [wnl.lemmatize(word) for word in drop_stops]
    return lemmed

# tokenize train and test text data
vect = CountVectorizer(tokenizer=prep_review)
X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)

# nltk.download()

# instantiate and train model
nb = MultinomialNB()
get_ipython().run_line_magic('time', 'nb.fit(X_train_dtm, y_train)')

# evaluate model
y_pred = nb.predict(X_test_dtm)
eval_predictions(y_test, y_pred)

tfidf_vectorizer_1 = TfidfVectorizer(min_df=5, max_df=0.8)
tfidf_train_1 = tfidf_vectorizer_1.fit_transform(X_train)
tfidf_test_1 = tfidf_vectorizer_1.transform(X_test)

# instantiate and train model, kernel=rbf 
svm_rbf = svm.SVC(random_state=12345)
get_ipython().run_line_magic('time', 'svm_rbf.fit(tfidf_train_1, y_train)')

# evaulate model
y_pred_1 = svm_rbf.predict(tfidf_test_1)
eval_predictions(y_test, y_pred_1)

# instantiate and train model, kernel=linear
svm_rbf = svm.SVC(kernel='linear', random_state=12345)
get_ipython().run_line_magic('time', 'svm_rbf.fit(tfidf_train_1, y_train)')

# evaulate model
y_pred_1 = svm_rbf.predict(tfidf_test_1)
eval_predictions(y_test, y_pred_1)

tfidf_vectorizer_2 = TfidfVectorizer(tokenizer=prep_review, min_df=5, max_df=0.8)
tfidf_train_2 = tfidf_vectorizer_2.fit_transform(X_train)
tfidf_test_2 = tfidf_vectorizer_2.transform(X_test)

# kernel=rbf
print 'kernel=rbf'
svm_rbf = svm.SVC(random_state=1)
get_ipython().run_line_magic('time', 'svm_rbf.fit(tfidf_train_2, y_train)')
y_pred_2 = svm_rbf.predict(tfidf_test_2)
eval_predictions(y_test, y_pred_2)
print 

print 'kernel=linear'
svm_rbf = svm.SVC(kernel='linear', random_state=1)
get_ipython().run_line_magic('time', 'svm_rbf.fit(tfidf_train_2, y_train)')
y_pred_2 = svm_rbf.predict(tfidf_test_2)
eval_predictions(y_test, y_pred_2)

compare_tokens = pd.DataFrame(
    {'unprocessed': tfidf_vectorizer_1.get_feature_names()[:10],
     'preprocessed': tfidf_vectorizer_2.get_feature_names()[:10],
    })
compare_tokens

# randomized search
param_dist = {'C': scipy.stats.expon(scale=100), 
              'gamma': scipy.stats.expon(scale=.1)}
n_iter_search = 1
rand_search = RandomizedSearchCV(svm.SVC(random_state=1), param_dist, cv=5, n_iter=n_iter_search, n_jobs=8)
rand_search.fit(tfidf_train_1, y_train)
bp = rand_search.best_params_
print 'Best parameters:', bp

# fit and evaluate model with parameters from grid search
model = svm.SVC(C = bp['C'], gamma = bp['gamma'], random_state=1)
model.fit(tfidf_train_1, y_train)
y_pred = model.predict(tfidf_test_1)
eval_predictions(y_test, y_pred)

