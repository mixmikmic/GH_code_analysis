import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd

get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# noob documents for training :P
spam = [
    "you have won a lottery",
    "congratulations! you have a bonus",
    "this is bomb",
    "to use the credit, please click the link",
    "thank you for subscription. please click the link",
    "bomb"
]
Y_spam = [1 for i in range(len(spam)) ]

non_spam = [
    "i am awesome",
    "i have a meeting tomorrow",
    "you are smart",
    "get me out of here",
    "call me later"
]
Y_non_spam = [0 for i in range(len(non_spam)) ]

# feature extraction
count_vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(spam + non_spam)
X_train_vectorized = count_vectorizer.transform(spam + non_spam)

# Naive Bayes Model
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, Y_spam + Y_non_spam)

documents = [
    "call you",
    "you have won"
]
predictions = model.predict(count_vectorizer.transform(documents))
print(predictions)

# convert to pandas dataframe for seamless training
spam_df = pd.DataFrame(spam, columns=['text'])
spam_df['target'] = 1
non_spam_df = pd.DataFrame(non_spam, columns=['text'])
non_spam_df['target'] = 0

# final data
data = pd.concat([spam_df, non_spam_df], ignore_index=True)
data

# feature extraction
count_vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(data['text'])
X_train_vectorized = count_vectorizer.transform(data['text'])
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, Y_spam + Y_non_spam)
documents = [
    "call you",
    "you have won"
]
predictions = model.predict(count_vectorizer.transform(documents))
print(predictions)

# load training set
data = pd.read_csv('data/spam.csv')

data['target'] = np.where(data['target']=='spam',1, 0)
print(len(data))
data.head(10)

X_train, X_test, Y_train, Y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)
X_train.shape, X_test.shape

# extract features
vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)
X_train_vectorized.toarray().shape

# create Naive Bayes model
model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, Y_train)

# Calculate True Positive Rate vs False Positive Rate
predictions = model.predict(vectorizer.transform(X_test))
score = roc_auc_score(Y_test, predictions)
score

# create custom confusion matrix for evaluation
result = {}
cm = np.zeros((2, 2))
precisions = np.zeros(2)
recalls = np.zeros(2)
for t, p in zip(Y_test, predictions):
    cm[t][p] += 1

tp = np.diag(cm)
fn = np.sum(cm, axis=1) - tp
fp = np.sum(cm, axis=0) - tp

for i in range(2):
    p_denom = tp[i] + fp[i]
    r_denom = tp[i] + fn[i]
    precisions[i] = 0 if p_denom == 0 else tp[i]/p_denom
    recalls[i] = 0 if r_denom == 0 else tp[i]/r_denom
    
# calculate from sklearn
fpr_, tpr_, _ = roc_curve(Y_test,  predictions)
#tpr = cm[1][1] / (cm[1][1] + cm[1][0])
tpr = recalls[1]
fpr = cm[0][1] / (cm[0][1] + cm[0][0])

precision = np.average(precisions)
recall = np.average(recalls)
f1 = 2 * precision * recall / (precision + recall)

print("Confusion Matrix:\n{}".format(cm))
print("Recall for Spam:\n{}".format(recall))
print("Precision for Spam:\n{}".format(precision))

plt.plot(fpr_, tpr_, label="spam, auc="+str(score))
plt.legend(loc=4)
plt.show()

test_docs = [
    "you have won a lottery",
    "click the link",
    "i have a meeting"
]
predictions = model.predict(vectorizer.transform(test_docs))
predictions



