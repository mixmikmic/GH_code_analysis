import pandas as pd
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# print first n number of rows
train.head(5)

# print last n number of rows
train.tail(5)

# print first n number of rows
test.head(5)

# print last n number of rows
test.tail(3)

# returns non-NA/null observations 
train.count()

# returns non-NA/null observations 
test.count()

train.groupby('Survived').count()['PassengerId']

# Use following method to extract a sub-set of columns from original DataFrame
numerical_features = train[['Age', 'Fare', 'Pclass']]

#returns non-NA/null observations
numerical_features.count()

numerical_features_without_na = numerical_features.dropna()
mean = numerical_features_without_na.mean()
print mean

# Now you can impute all missing values with the mean vector we just calculated
imputed_features_training = numerical_features.fillna(mean)

imputed_features_training.count()

# Now we can convert our DataFrame to numpy arrays as shown below
X_train = imputed_features_training.values
y_train = train['Survived'].values

numerical_features_testing = test[['Age', 'Fare', 'Pclass']]

numerical_features_testing.count()

imputed_features_testing = numerical_features_testing.fillna(mean)

imputed_features_testing.count()

get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.cross_validation import KFold\nfrom sklearn.preprocessing import scale\n\nfolds = KFold(y_train.shape[0], n_folds=5, shuffle=True)\ncv_accuracies = []\nfor trining_idx, testing_idx in folds:\n    X_train_cv = X_train[trining_idx]\n    y_train_cv = y_train[trining_idx]\n    \n    X_test_cv = X_train[testing_idx]\n    y_test_cv = y_train[testing_idx]\n    \n    logistic_regression = LogisticRegression()\n    logistic_regression.fit(scale(X_train_cv), y_train_cv)\n    y_predict_cv = logistic_regression.predict(scale(X_test_cv))\n    current_accuracy = accuracy_score(y_test_cv, y_predict_cv)\n    cv_accuracies.append(current_accuracy)\n    print 'cross validation accuracy: %f' %(current_accuracy)\n    \nprint '---------------------------------------'\nprint 'average corss validation accuracy: %f' %(sum(cv_accuracies)/len(cv_accuracies))    \nprint '---------------------------------------'")

from sklearn.linear_model import LogisticRegression

get_ipython().magic('pinfo LogisticRegression')

# First train Logistic Regression using the full training dataset
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train, y_train)
X_test = imputed_features_testing.values
y_test = logistic_regression.predict(X_test)

# save data
test_result = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':y_test})
test_result.to_csv('../data/submission.csv', index=False)

train[['Sex']].head()

# categorical feature should be encoded before feeding to scikit-learn algorithms
pd.get_dummies(train['Sex'], prefix='Sex').head()

more_features_training = pd.concat([train[['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']], 
                                  pd.get_dummies(train[['Sex']]),
                                  pd.get_dummies(train[['Embarked']])], axis=1)

more_features_training.head(5)

more_features_training.count()

mean = more_features_training.dropna().mean()
more_features_training_without_nan = more_features_training.fillna(mean)

more_features_training_without_nan.count()

# Now we can convert our DataFrame to numpy arrays as shown below
X_train = imputed_features_training.values
y_train = train['Survived'].values

get_ipython().run_cell_magic('time', '', "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import accuracy_score\nfrom sklearn.cross_validation import KFold\nfrom sklearn.preprocessing import scale\n\nX_train = more_features_training_without_nan.values\ny_train = train['Survived'].values\n\nfolds = KFold(y_train.shape[0], n_folds=5, shuffle=True)\ncv_accuracies = []\nfor trining_idx, testing_idx in folds:\n    X_train_cv = X_train[trining_idx]\n    y_train_cv = y_train[trining_idx]\n    \n    X_test_cv = X_train[testing_idx]\n    y_test_cv = y_train[testing_idx]\n    \n    logistic_regression = LogisticRegression()\n    logistic_regression.fit(scale(X_train_cv), y_train_cv)\n    y_predict_cv = logistic_regression.predict(scale(X_test_cv))\n    current_accuracy = accuracy_score(y_test_cv, y_predict_cv)\n    cv_accuracies.append(current_accuracy)\n    print 'cross validation accuracy: %f' %(current_accuracy)\n    \nprint '---------------------------------------'\nprint 'average corss validation accuracy: %f' %(sum(cv_accuracies)/len(cv_accuracies))  ")

get_ipython().run_cell_magic('time', '', "from sklearn.ensemble import RandomForestClassifier\n\nX_train = more_features_training_without_nan.values\ny_train = train['Survived'].values\n\nfolds = KFold(y_train.shape[0], n_folds=5, shuffle=True)\ncv_accuracies = []\nfor trining_idx, testing_idx in folds:\n    X_train_cv = X_train[trining_idx]\n    y_train_cv = y_train[trining_idx]\n    \n    X_test_cv = X_train[testing_idx]\n    y_test_cv = y_train[testing_idx]\n    \n    random_forest = RandomForestClassifier(n_estimators = 100)\n    random_forest.fit(scale(X_train_cv), y_train_cv)\n    y_predict_cv = random_forest.predict(scale(X_test_cv))\n    current_accuracy = accuracy_score(y_test_cv, y_predict_cv)\n    cv_accuracies.append(current_accuracy)\n    print 'cross validation accuracy: %f' %(current_accuracy)\n\n    \nprint '---------------------------------------'\nprint 'average corss validation accuracy: %f' %(sum(cv_accuracies)/len(cv_accuracies)) \nprint '---------------------------------------\\n'")

get_ipython().run_cell_magic('time', '', "from sklearn.svm import SVC\n\nX_train = more_features_training_without_nan.values\ny_train = train['Survived'].values\n\nfolds = KFold(y_train.shape[0], n_folds=5, shuffle=True)\ncv_accuracies = []\nfor trining_idx, testing_idx in folds:\n    X_train_cv = X_train[trining_idx]\n    y_train_cv = y_train[trining_idx]\n    \n    X_test_cv = X_train[testing_idx]\n    y_test_cv = y_train[testing_idx]\n    \n    svc = SVC(C = 1.4)\n    svc.fit(scale(X_train_cv), y_train_cv)\n    y_predict_cv = svc.predict(scale(X_test_cv))\n    current_accuracy = accuracy_score(y_test_cv, y_predict_cv)\n    cv_accuracies.append(current_accuracy)\n    print 'cross validation accuracy: %f' %(current_accuracy)\n\n    \nprint '---------------------------------------'\nprint 'average corss validation accuracy: %f' %(sum(cv_accuracies)/len(cv_accuracies)) ")

get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.grid_search import GridSearchCV\n\nX_train = more_features_training_without_nan.values\ny_train = train[\'Survived\'].values\n\ncls = RandomForestClassifier()\nparameters = {\n    \'n_estimators\' : [10, 20, 40, 80, 160, 320, 640],\n    \'max_depth\' : [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40],\n    \'criterion\' : [\'gini\', \'entropy\']\n}\ngs = GridSearchCV(cls, parameters, cv=5, n_jobs=8, scoring="accuracy")\ngs.fit(X_train, y_train)\n\nprint gs.best_score_')

