# Define our vectorizer

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer='word', lowercase=True)

# Sample sentence to tokenize
my_text = 'This module introduced many concepts in text analysis.'

cv1 = CountVectorizer(lowercase=True)
cv2 = CountVectorizer(stop_words = 'english', lowercase=True)

tk_func1 = cv1.build_analyzer()
tk_func2 = cv2.build_analyzer()

import pprint
pp = pprint.PrettyPrinter(indent=2, depth=1, width=80, compact=True)

print('Tokenization:')
pp.pprint(tk_func1(my_text))

print()

print('Tokenization (with Stop words):')
pp.pprint(tk_func2(my_text))

import nltk
mvr = nltk.corpus.movie_reviews

from sklearn.datasets import load_files

data_dir = '/usr/share/nltk_data/corpora/movie_reviews'

mvr = load_files(data_dir, shuffle = False)
print('Number of Reviews: {0}'.format(len(mvr.data)))

print(len(mvr))

print(mvr.data[:2])

print(mvr.target[:500])

from sklearn.model_selection import train_test_split

mvr_train, mvr_test, y_train, y_test = train_test_split(
    mvr.data, mvr.target, test_size=0.25, random_state=23)

print('Number of Reviews in train: {0}'.format(len(mvr_train)))
print('Number of Reviews in test: {0}'.format(len(mvr_test)))
print(y_train)

#print(mvr_test)

import string
import nltk
from nltk.stem.porter import PorterStemmer

example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
stemmer = PorterStemmer()

for w in example_words:
    print(stemmer.stem(w))

new_text = "It is important to be very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."

tokens = nltk.word_tokenize(new_text)

print(tokens)

print("===============")

tokens = [token for token in tokens if token not in string.punctuation]

for w in tokens:
    print(stemmer.stem(w))

# Split into training and testing
from sklearn.datasets import fetch_20newsgroups

train = fetch_20newsgroups(data_home='/dsa/data/DSA-8630/newsgroups/', subset='train',                            shuffle=True, random_state=23)

test = fetch_20newsgroups(data_home='/dsa/data/DSA-8630/newsgroups/', subset='test',                           shuffle=True, random_state=23)

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

cv = CountVectorizer()

train_counts = cv.fit_transform(train['data']) # returns a Document-Term Matrix
train_counts

print(train['target'])

print(train_counts[:2])

test_data = cv.transform(test['data'])
test_data

nb = MultinomialNB()

clf = nb.fit(train_counts, train['target'])
predicted = clf.predict(test_data)
print(predicted)

print("NB prediction accuracy = {0:5.1f}%".format(100.0 * clf.score(test_data, test['target'])))

from sklearn.pipeline import Pipeline

tools = [('cv', CountVectorizer()), ('nb', MultinomialNB())]
clf = Pipeline(tools)

clf = clf.fit(train['data'], train['target'])
predicted = clf.predict(test['data'])

print("NB prediction accuracy = {0:5.1f}%".format(100.0 * clf.score(test['data'], test['target'])))

from sklearn.feature_extraction.text import TfidfVectorizer

tools = [('tf', TfidfVectorizer()), ('nb', MultinomialNB())]
clf = Pipeline(tools)

# set_params() of TfidfVectorizer below, sets the parameters of the estimator. 
# The method works on simple estimators as 
# well as on nested objects (such as pipelines). 
# The pipelines have parameters of the form <component>__<parameter> 
# so that it’s possible to update each component of a nested object.
clf.set_params(tf__stop_words = 'english')

clf = clf.fit(train['data'], train['target'])
predicted = clf.predict(test['data'])

print("NB (TF-IDF with Stop Words) prediction accuracy = {0:5.1f}%"      .format(100.0 * clf.score(test['data'], test['target'])))

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer

clf = Pipeline([('vect', CountVectorizer(stop_words = 'english')),
                ('tfidf', TfidfTransformer()),
                ('lr', LogisticRegression())])


clf = clf.fit(train['data'], train['target'])
predicted = clf.predict(test['data'])

print("LR (TF-IDF with Stop Words) prediction accuracy = {0:5.1f}%".      format(100.0 * clf.score(test['data'], test['target'])))

