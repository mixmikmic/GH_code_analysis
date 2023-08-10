import nltk
import random
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import pickle

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

stop_words = stopwords.words("english")
documents = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)
            ]
random.shuffle(documents)

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
all_words.most_common(20)

feature_words = list(all_words.keys())[:5000]
def find_features(document):
    words = set(document)
    feature = {}
    for w in feature_words:
        feature[w] = (w in words)
    return feature

feature_sets = [(find_features(rev), category) for (rev, category) in documents]

training_set = feature_sets[:1900]
testing_set = feature_sets[1900:]

## TO-DO: To build own naive bayes algorithm
# classifier = nltk.NaiveBayesClassifier.train(training_set)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

## saving it in a pickle
MNB_pickle = open("MNB_pickle.pickle", "wb")
pickle.dump(MNB_classifier, MNB_pickle)
MNB_pickle.close()
print("Multinomial classifier accuracy : ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

## BernoulliNB 

BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)

BNB_pickle = open("BNB_pickle.pickle", "wb")
pickle.dump(BNB_classifier, BNB_pickle)
BNB_pickle.close()

print("Bernoulli classifier accuracy : ", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)

LogisticRegression_pickle = open("LogisticRegression.pickle", "wb")
pickle.dump(LogisticRegression_classifier, LogisticRegression_pickle)
LogisticRegression_pickle.close()

print("LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)

SGDClassifier_pickle = open("SGDClassifier.pickle", "wb")
pickle.dump(SGDClassifier_classifier, SGDClassifier_pickle)
SGDClassifier_pickle.close()

print("SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)

SVC_classifier_pickle = open("SVC_classifier.pickle", "wb")
pickle.dump(SVC_classifier, SVC_classifier_pickle)
SVC_classifier_pickle.close()

print("SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

LinearSVC_pickle = open("LinearSVC.pickle", "wb")
pickle.dump(LinearSVC_classifier, LinearSVC_pickle)
LinearSVC_pickle.close()

print("LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

NuSVC_pickle = open("LinearSVC.pickle", "wb")
pickle.dump(NuSVC_classifier, NuSVC_pickle)
NuSVC_pickle.close()

print("NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

### using the old naive_bayes classifier
naive_bayes_pickle = open("naivebayes.pickle", "rb")
naive_bayes_classifier = pickle.load(naive_bayes_pickle)
naive_bayes_pickle.close()

print("Naive bayes classifier accuracy percent:", (nltk.classify.accuracy(naive_bayes_classifier, testing_set))*100)

