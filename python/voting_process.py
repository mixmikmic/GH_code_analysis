import nltk
import random
from nltk.corpus import movie_reviews
import pickle

from nltk.classify import ClassifierI
from statistics import mode

## defing the voteclassifier class
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

# pickle_obj = open("documents.pickle", "wb")
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
# pickle.dump(documents, pickle_obj)
# pickle_obj.close()

# pickle_obj = open("documents.pickle", "rb")
# documents = pickle.load(pickle_obj)
# pickle_obj.close()

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]
        
training_set = featuresets[:1900]
testing_set =  featuresets[1900:]

original_nb = open("naive_bayes.pickle", "rb")
naive_bayes_classifier = pickle.load(original_nb)
original_nb.close()

pickle_file = open("MNB_pickle.pickle", "rb")
MNB_classifier = pickle.load(pickle_file)
pickle_file.close()

pickle_file = open("BNB_pickle.pickle", "rb")
BernoulliNB_classifier = pickle.load(pickle_file)
pickle_file.close()

pickle_file = open("LogisticRegression.pickle", "rb")
LogisticRegression_classifier = pickle.load(pickle_file)
pickle_file.close()

pickle_file = open("SGDClassifier.pickle", "rb")
SGDClassifier_classifier = pickle.load(pickle_file)
pickle_file.close()


pickle_file = open("LinearSVC.pickle", "rb")
LinearSVC_classifier = pickle.load(pickle_file)
pickle_file.close()

pickle_file = open("NuSVC_classifier.pickle", "rb")
NuSVC_classifier = pickle.load(pickle_file)
pickle_file.close()

print("naive bayes: ", (nltk.classify.accuracy(naive_bayes_classifier, testing_set))*100)
print("MNB_classifier: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)
print("BernoulliNB_classifier: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)
print("LogisticRegression_classifier: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
print("SGDClassifier_classifier: ", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100)
print("LinearSVC_classifier: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
print("NuSVC_classifier: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

voted_classifier = VoteClassifier(
    naive_bayes_classifier,
    MNB_classifier,
    BernoulliNB_classifier,
    LogisticRegression_classifier,
    SGDClassifier_classifier,
    LinearSVC_classifier,
    NuSVC_classifier
)
print("Voted classifier accuracy : ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print("Classification:", voted_classifier.classify(testing_set[0][0]), "Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:", voted_classifier.classify(testing_set[1][0]), "Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:", voted_classifier.classify(testing_set[2][0]), "Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:", voted_classifier.classify(testing_set[3][0]), "Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:", voted_classifier.classify(testing_set[4][0]), "Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:", voted_classifier.classify(testing_set[5][0]), "Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)

