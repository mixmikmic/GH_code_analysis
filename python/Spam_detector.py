import pandas as pd

df = pd.read_csv('SMSSpamCollection/SMSSpamCollection', header=None, sep='\t', names=['label', 'text'])
df.head()

df.label = df.label.map({'ham':0, 'spam':1})
print df.shape
df.head()

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
for i in documents:
    lower_case_documents.append(i.lower())
print(lower_case_documents)

sans_punctuation_documents = []
import string

for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(None, string.punctuation))
print(sans_punctuation_documents)

preprocessed_documents = []
for i in sans_punctuation_documents:
    preprocessed_documents.append(i.split(' '))
print(preprocessed_documents)

from collections import Counter

frequency_list = []
for i in preprocessed_documents:
    frequency_counts = Counter(i)
    frequency_list.append(frequency_counts)

print frequency_list, '\n'

import pprint
pprint.pprint(frequency_list)

documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

from sklearn.feature_extraction.text import CountVectorizer

count_vec = CountVectorizer()

count_vec

count_vec.fit(documents)

count_vec.get_feature_names()

print count_vec.transform(documents)

doc_array = count_vec.transform(documents).toarray()
print doc_array

frequency_matrix = pd.DataFrame(doc_array, 
                                columns = count_vec.get_feature_names())
frequency_matrix

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.text, df.label, random_state=1)

print X_train.shape, X_test.shape, y_train.shape, y_test.shape

count_vec = CountVectorizer()
train = count_vec.fit_transform(X_train)
test = count_vec.transform(X_test)

# probability of a person having Diabetes
P_D = 0.01

# probability of getting a positive result on a test done for detecting diabetes, given that you have diabetes
P_Pos_D = 0.9

# probability of getting a negative result on a test done for detecting diabetes, given that you do not have diabetes
P_Neg_nD = 0.9

# what is P_D_Pos?

# P_Pos = P_D * P_Pos_D + P_nD * P_Pos_nD

P_Pos = 0.01 * 0.9 + (1 - 0.01) * (1 - 0.9)

P_Pos

# P_D_Pos = P_D * P_Pos_D / P_Pos

P_D_Pos = 0.01 * 0.9 / P_Pos

# the probability of an individual having diabetes, given that, that individual got a positive test result
P_D_Pos

# P_nD_Pos = P_nD * P_Pos_nD / P_Pos

P_nD_Pos = (1 - 0.01) * (1 - 0.9) / P_Pos

# Probability of an individual not having diabetes, given that that individual got a positive test result
P_nD_Pos

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(train, y_train)

pred = model.predict(test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print confusion_matrix(y_test, pred)

print accuracy_score(y_test, pred)
print precision_score(y_test, pred)
print recall_score(y_test, pred)
print f1_score(y_test, pred)

y_test.head()

pred

