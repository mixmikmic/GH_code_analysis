from sklearn.datasets import fetch_20newsgroups
categories =['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

print twenty_train.data[0]

twenty_train.target[:10]

twenty_train.target_names[:10]

[twenty_train.target_names[t] for t in twenty_train.target[:10]]

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(stop_words='english', min_df = 3, lowercase=True, ngram_range=(1,2))
X_train_counts = count_vect.fit_transform(twenty_train.data)
#min_df - a word has to occur in (x) documents to be considered a feature

count_vect.vocabulary_.items()[:10]
#this is a dictionary so it has .items()

len(count_vect.vocabulary_)

X_train_counts[0]

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer

transformer = TfidfTransformer()
#model... like clf
X_train_tfidf = transformer.fit_transform(X_train_counts)

print X_train_tfidf
#prints the location in the sparse matrix and the tfidf score
reversed_vocab = dict()
reversed_vocab = {v:k for (k,v) in count_vect.vocabulary_.items()}
"""
for key in count_vect.vocabulary_:
    reversed_vocab[count_vect.vocabulary_[key]] = key    
"""

tfidfvect = TfidfVectorizer()

from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

"""
statistically you would want to normalize word counts 
between 0 and 1 but in practice TFIDF is a useful because gives
different weight to rare terms
"""

X_train_tfidf_dense = X_train_tfidf.toarray()
tree = DecisionTreeClassifier()
print cross_val_score(tree, X_train_tfidf_dense, twenty_train.target, cv=3)

forest = RandomForestClassifier(n_estimators = 10)
cross_val_score(forest, X_train_tfidf_dense, twenty_train.target, cv=3)

logreg = LogisticRegression()
cross_val_score(logreg,X_train_tfidf, twenty_train.target, cv = 5)

nb = MultinomialNB()
cross_val_score(nb,X_train_tfidf, twenty_train.target, cv = 5)

forest.fit(X_train_tfidf_dense, twenty_train.target)

x= forest.feature_importances_

x[x>0.]



