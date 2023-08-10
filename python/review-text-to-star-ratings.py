# This cell just sets up some data to use.
import pickle

with open('abilify.p', 'rb') as f:
    data = pickle.load(f)
    
reviews = [datum['comment'] for datum in data]
satisfaction_ratings = [datum['satisfaction'] for datum in data]

print(data[0])

reviews[:10]

satisfaction_ratings[:10]

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

#TODO Refactor to use model_selection module
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB



X_train_tfidf_dense = X_train_tfidf.toarray()

tree = DecisionTreeClassifier()
cross_val_score(tree, X_train_tfidf_dense, satisfaction_ratings, cv=5)

forest = RandomForestClassifier(n_estimators = 10)
cross_val_score(tree, X_train_tfidf_dense, satisfaction_ratings, cv=5)

logreg = LogisticRegression()
cross_val_score(tree, X_train_tfidf_dense, satisfaction_ratings, cv=5)

nb = MultinomialNB()
cross_val_score(tree, X_train_tfidf_dense, satisfaction_ratings, cv=5)

