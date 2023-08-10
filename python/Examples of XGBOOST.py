import pandas as pd
df = pd.read_csv('/Users/user/Jupyter/ml_notes/x_gboost_examples/data/new_labeled_corpus.csv', sep = '|').dropna()
X = df['content'].values
y = df['labels'].values
df.head()

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X = count_vect.fit_transform(X)
X.toarray()
print (type(X))

print (type(X))
print (type(y))

num_columnas_del_df = df['labels'].shape[0]
print '\n count col\n',num_columnas_del_df

#Hacemos validacion cruzada
from sklearn.cross_validation import KFold
kf = KFold(n=num_columnas_del_df, n_folds=10, shuffle=True, random_state=False)
print '\n\n........Cross validating........\n\n'
for train_index, test_index in kf:
    print "\nEntrenó con las opiniones que tienen como indice:\n", train_index,     "\nProbó con las opiniones que tiene como indice:\n", test_index
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

y_=[ ]
prediction_=[ ]

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, y_train)
predictions = gbm.predict(X)
predictions

from sklearn import cross_validation
acc_scores = cross_validation.cross_val_score(gbm, X_train, y_train, cv=10)

print (acc_scores)

print("\nAccuracy: %0.2f (+/- %0.2f)" % (acc_scores.mean(), acc_scores.std() * 2))



