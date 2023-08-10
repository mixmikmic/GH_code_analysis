import pandas as pd

df = pd.read_csv('singapore-roadnames-final-classified.csv')

df

# let's pick a random 10% to train with

import random
random.seed(1965)
train_test_set = df.loc[random.sample(df.index, int(len(df) / 10))]

X = train_test_set['road_name']
y = train_test_set['classification']

zip(X,y)[::10]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_true = train_test_split(X, y)

df.classification.value_counts()

from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(ngram_range=(1,4), analyzer='char')

# fit_transform for the training data
X_train_feats = vect.fit_transform(X_train)
# transform for the test data
# because we need to match the ngrams that were found in the training set 
X_test_feats  = vect.transform(X_test) 

print type(X_train_feats)
print X_train_feats.shape
print X_test_feats.shape

from sklearn.svm import LinearSVC
clf = LinearSVC()

model = clf.fit(X_train_feats, y_train)

y_predicted = model.predict(X_test_feats)

y_predicted

from sklearn.metrics import accuracy_score

accuracy_score(y_true, y_predicted)

def classify(X, y):
    # do the train-test split
    X_train, X_test, y_train, y_true = train_test_split(X, y)

    # get our features
    X_train_feats = vect.fit_transform(X_train)
    X_test_feats  = vect.transform(X_test) 

    # train our model
    model = clf.fit(X_train_feats, y_train)
    
    # predict labels on the test set
    y_predicted = model.predict(X_test_feats)
    
    # return the accuracy score obtained
    return accuracy_score(y_true, y_predicted)

scores = list()
num_expts = 100
for i in range(num_expts):
    score = classify(X,y)
    scores.append(score)
    
print sum(scores) / num_expts

