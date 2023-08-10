import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer        

text_pos = []
labels_pos = []
with open("./pos_tweets.txt") as f:
    for i in f: 
        text_pos.append(i) 
        labels_pos.append('pos')

text_neg = []
labels_neg = []
with open("./neg_tweets.txt") as f:
    for i in f: 
        text_neg.append(i)
        labels_neg.append('neg')

training_text = text_pos[:int((.8)*len(text_pos))] + text_neg[:int((.8)*len(text_neg))]
training_labels = labels_pos[:int((.8)*len(labels_pos))] + labels_neg[:int((.8)*len(labels_neg))]

test_text = text_pos[int((.8)*len(text_pos)):] + text_neg[int((.8)*len(text_neg)):]
test_labels = labels_pos[int((.8)*len(labels_pos)):] + labels_neg[int((.8)*len(labels_neg)):]

vectorizer = CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    max_features = 85
)

features = vectorizer.fit_transform(
    training_text + test_text)

features_nd = features.toarray() # for easy use

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test  = train_test_split(
        features_nd[0:len(training_text)], 
        training_labels,
        train_size=0.80, 
        random_state=1234)

from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression()

log_model = log_model.fit(X=X_train, y=y_train)

y_pred = log_model.predict(X_test)

import random
spl = random.sample(range(len(test_pred)), 10)
for text, sentiment in zip(test_text, test_pred[spl]):
    print (sentiment, text)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

import nltk

def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})

print(format_sentence("The cat is very cute"))

pos = []
with open("./pos_tweets.txt") as f:
    for i in f: 
        pos.append([format_sentence(i), 'pos'])

neg = []
with open("./neg_tweets.txt") as f:
    for i in f: 
        neg.append([format_sentence(i), 'neg'])

training = pos[:int((.8)*len(pos))] + neg[:int((.8)*len(neg))]
test = pos[int((.8)*len(pos)):] + neg[int((.8)*len(neg)):]

from nltk.classify import NaiveBayesClassifier

classifier = NaiveBayesClassifier.train(training)

classifier.show_most_informative_features()

example1 = "Twilio is an awesome company!"

print(classifier.classify(format_sentence(example1)))

example2 = "I'm sad that Twilio doesn't have even more blog posts!"

print(classifier.classify(format_sentence(example2)))

example3 = "I have no headache!"

print(classifier.classify(format_sentence(example3)))

from nltk.classify.util import accuracy
print(accuracy(classifier, test))



