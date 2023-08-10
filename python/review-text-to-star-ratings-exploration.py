# This cell just sets up some data to use.
import pickle

with open('abilify.p', 'rb') as f:
    data = pickle.load(f)
    
reviews = [datum['comment'] for datum in data]
satisfaction_ratings = [datum['satisfaction'] for datum in data]

print(data[0])

reviews[:10]

satisfaction_ratings[:10]

len(reviews)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english', min_df = 2, lowercase=True, ngram_range=(1,2))
X_train_counts = count_vect.fit_transform(reviews)
#min_df - a word has to occur in (x) documents to be considered a feature

len(count_vect.vocabulary_)

X_train_counts[0]

from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer()
X_train_tfidf = transformer.fit_transform(X_train_counts)

print(X_train_tfidf)
#prints the location in the sparse matrix and the tfidf score

# Classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Regression
from sklearn.svm import SVR

# Model selection
from sklearn.model_selection import cross_val_score, KFold, train_test_split

X_train_tfidf_dense = X_train_tfidf.toarray()

X_train, X_test, y_train, y_test = train_test_split(
    X_train_tfidf_dense, satisfaction_ratings, test_size=0.33, random_state=8)

regression = SVR(kernel='linear', C=1e3)
regression.fit(X_train, y_train)
reg_results = regression.predict(X_test)

print(len(reg_results))
print(reg_results[:10])
print(y_test[:10])

