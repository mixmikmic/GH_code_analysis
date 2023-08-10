import numpy
from scipy import sparse
from sklearn.metrics import roc_auc_score

import codecs
# python kung fu. No need to understand this code
with codecs.open('./data/1-restaurant-train.csv') as f:
    labels, reviews = zip(*[line.split('\t') for line in f.readlines()])
with codecs.open('./data/1-restaurant-test.csv') as f:
    kaggle_test_reviews = f.readlines()

import pandas
from IPython.display import FileLink
# supplementary function. Again, just use it
def create_solution(predictions, filename='1-restaurant-predictions.csv'):
    result = pandas.DataFrame({'Id': numpy.arange(len(predictions)), 'Solution': predictions})
    result.to_csv('data/{}'.format(filename), index=False)
    return FileLink('data/{}'.format(filename))

answers = numpy.array(labels, dtype=int) >= 4

from sklearn.feature_extraction.text import CountVectorizer
# take 100000 most frequent words
vectorizer = CountVectorizer(max_features=100000)
vectorizer.fit(reviews)

counts = vectorizer.transform(reviews) #.toarray()
kaggle_test_counts = vectorizer.transform(kaggle_test_reviews) #.toarray()

counts

# take first 100 lines, convert to numpy
counts[:100].toarray() 
# but doing above operation with whole dataset is a very bad idea: ~10 ** 10 elements

n_elements = counts.shape[0] * counts.shape[1]
n_nonzero_elements = counts.getnnz()

# real amount of elements is ~1000 times less
print n_elements / float(n_nonzero_elements)

counts

counts.T

# we can see that transposing is instant, because it simply changed rows and columns.
get_ipython().magic('timeit counts.T')

counts.sum()

counts.sum(axis=0)

# converted to numpy.array, then converted from matrix to a single row
numpy.array(counts.sum(axis=0))[0]

counts.max(axis=1) # max times one word was repeated in the review

def add_new_features(sparse_counts):
    # added two new features to the dataset: number of words in the sentence and 'max repetitions'
    # horizontal stacking of matrices. Important - second and third are also matrices, but with one column
    return sparse.hstack([sparse_counts, sparse_counts.sum(axis=1), sparse_counts.max(axis=1)])

new_counts = add_new_features(counts)
# the same operations should be done on kaggle test (to use the same model)
new_kaggle_counts = add_new_features(kaggle_test_counts)

def add_character_count(counts, reviews, characters):
    new_features = numpy.zeros(shape=[len(reviews), len(characters)])
    for i, character in enumerate(characters):
        for j in range(len(reviews)):
            new_features[j, i] = reviews[j].lower().count(character)
    print new_features
    return sparse.hstack([counts, new_features])

characters = [')', '(', '!', 'like', ' ']
new_counts = add_character_count(new_counts, reviews, characters=characters)
new_kaggle_counts = add_character_count(new_kaggle_counts, kaggle_test_reviews, characters=characters)

from sklearn.cross_validation import train_test_split

train_counts, test_counts, train_answers, test_answers =     train_test_split(new_counts, answers, train_size=10000, random_state=42)

train_counts # works correctly with sparse types too!

# logistic regression was explained in the lectures.
from sklearn.linear_model import LogisticRegression

# C is an inverse regularization, smaller C -> stronger regularization
logreg_clf = LogisticRegression(C=1.)
logreg_clf.fit(train_counts, train_answers)

# important: logistic regression is a CLASSIFIER, so we use .predict_proba!
print 'train: ', roc_auc_score(train_answers, logreg_clf.predict_proba(train_counts)[:, 1])
print 'test:  ', roc_auc_score(test_answers, logreg_clf.predict_proba(test_counts)[:, 1])

create_solution(logreg_clf.predict_proba(new_kaggle_counts)[:, 1], filename='1-restaurant-predictions-logreg.csv')

# elastic net is a linear regression with both L1 and L2 regularizations, it's more general compared to Ridge
# sidenote: logistic regression also has L1 regularization
from sklearn.linear_model import ElasticNet, Ridge

# ridge_reg = ElasticNet(l1_ratio=0.5, alpha=0.1, max_iter=5)
ridge_reg = Ridge()
ridge_reg.fit(train_counts, train_answers)

# Ridge (and ElasticNet) is a regression method, so we use .predict method
print 'train: ', roc_auc_score(train_answers, ridge_reg.predict(train_counts))
print 'test:  ', roc_auc_score(test_answers, ridge_reg.predict(test_counts))

create_solution(ridge_reg.predict(new_kaggle_counts), filename='1-restaurant-predictions-ridge.csv')

