from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import string
import re

np.random.seed(777)

df = pd.read_csv('../data/sms_data_uci.csv', encoding='latin')
df = df[['v1', 'v2']]
df.columns = ['Label', 'Message']

df.shape

df.head()

cv = CountVectorizer(stop_words='english', max_features=500)

df_train, df_test = train_test_split(df, test_size=0.2)

train_X = cv.fit_transform(df_train['Message']).toarray()

test_X = cv.transform(df_test['Message']).toarray()

train_X.shape, test_X.shape

tf_spam = dict()
tf_ham = dict()

spam_word_count = 0
ham_word_count = 0

spam_count = 0
ham_count = 0

for word_id in range(500):
    tf_spam[word_id] = 0
    tf_ham[word_id] = 0

for d_id, row  in enumerate(zip(train_X, df_train['Label'])):
    label = row[1]
    if label == 'spam':
        spam_count += 1
    else:
        ham_count += 1
    for word_id, count in enumerate(row[0]):
        if count:
            if label == 'spam':
                tf_spam[word_id] = tf_spam.get(word_id, 0) + 1
                spam_word_count += 1
            else:
                tf_ham[word_id] = tf_ham.get(word_id, 0) + 1
                ham_word_count += 1

prob_spam = np.log(spam_count) - np.log(spam_count + ham_count)
prob_ham = np.log(ham_count) - np.log(spam_count + ham_count)

prob_spam, prob_ham

tf_spam_prob = dict()
for word_id in tf_spam:
    tf_spam_prob[word_id] = np.log(tf_spam[word_id] + 1) - np.log(spam_count + 2) 

tf_ham_prob = dict()
for word_id in tf_ham:
    tf_ham_prob[word_id] = np.log(tf_ham[word_id] + 1) - np.log(ham_count + 2)

def predict(messages):
    """
    source: https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html
    """
    result = []
    for msg in messages:
        spam_prob = 0
        ham_prob = 0
        for word_id, count in enumerate(msg):
            if count:
                spam_prob += tf_spam_prob[word_id]
                ham_prob += tf_ham_prob[word_id]
            else:
                spam_prob += np.log(1 - np.exp(tf_spam_prob[word_id]))
                ham_prob += np.log(1 - np.exp(tf_ham_prob[word_id]))
        spam_prob += prob_spam
        ham_prob += prob_ham
        if spam_prob > ham_prob:
            result.append(1)
        else:
            result.append(0)
    return result

res_2 = predict(test_X)

accuracy_score(res_2, df_test['Label'].map({'ham': 0, 'spam': 1}))

from sklearn.naive_bayes import BernoulliNB

clf = BernoulliNB()

clf.fit(train_X, df_train['Label'])

accuracy_score(clf.predict(test_X), df_test['Label'])

clf.class_log_prior_

prob_ham, prob_spam

for i in range(10):
    print(tf_ham_prob[i])

clf.feature_log_prob_[0,:][0:10]

for i in range(10):
    print(tf_spam_prob[i])

clf.feature_log_prob_[1,:][0:10]



