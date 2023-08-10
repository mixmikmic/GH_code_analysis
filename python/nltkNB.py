import nltk
import random
from nltk.corpus import movie_reviews
import pprint
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
import pickle

movie_reviews.categories()

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

all_words["hate"]  ## counting the occurences of a single word

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

## TO-DO: To build own naive bais algorithm
classifier = nltk.NaiveBayesClassifier.train(training_set)

## Testing it's accuracy
print("Naive bayes classifier accuracy percentage : ", (nltk.classify.accuracy(classifier, testing_set))*100)

classifier.show_most_informative_features(20)

save_classifier = open("naivebayes.pickle", "wb") ## 'wb' tells to write it using bytes
pickle.dump(classifier, save_classifier)
save_classifier.close()

