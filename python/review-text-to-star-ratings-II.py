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
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Regression
from sklearn.svm import SVR

X_train_tfidf_dense = X_train_tfidf.toarray()
print(len(X_train_tfidf_dense))

models = []
models.append(('DTree', DecisionTreeClassifier()))
models.append(('RandForest', RandomForestClassifier(n_estimators = 10)))
models.append(('LogReg', LogisticRegression()))
models.append(('NaiveBayes', MultinomialNB()))

results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, random_state = 8)
    cv_results = cross_val_score(model, X_train_tfidf_dense, satisfaction_ratings, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

results

reg_models = []
reg_models.append(('rbf', SVR(kernel='rbf', C=1e3, gamma=0.1)))
reg_models.append(('linear', SVR(kernel='linear', C=1e3)))
reg_models.append(('quadratic', SVR(kernel='poly', C=1e3, degree=2)))

reg_results = []
reg_names = []

for name, model in reg_models:
    kfold = KFold(n_splits=10, random_state = 8)
    cv_results = cross_val_score(model, X_train_tfidf_dense, satisfaction_ratings, cv=kfold, scoring='neg_mean_absolute_error')
    reg_results.append(cv_results)
    reg_names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

