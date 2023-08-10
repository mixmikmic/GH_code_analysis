#Create Dataframe
import pandas as pd

df = pd.read_csv('/Users/forsythd/Desktop/article_contents.csv')

#Change tag to Binary 1/0
df['tag'] = df['tag'].map({'Disasters': 0, 'Conflict and violence': 1}) 

#Re Order Columns
df = df[['country', 'url', 'title', 'meta_description', 'content','tag']]

df.head()

#Value Counts on tag
df['tag'].value_counts()

#replace nan with space in meta description and content
df['meta_description'].fillna('', inplace=True)
df['content'].fillna('', inplace=True)
df['title'].fillna('', inplace=True)

#Split Data into a CV Set 70% and an Evaluation Set 30%
from sklearn.model_selection import train_test_split
X_CV, X_Eval, y_CV, y_Eval = train_test_split(df, df['tag'], test_size=0.3,
                                                    random_state=0,stratify=df['tag'])

#Print CV Set Stats
print('Length',len(X_CV))
print()
print('Value Counts')
print(X_CV['tag'].value_counts())

#Print Eval Set Stats
print('Length',len(X_Eval))
print()
print('Value Counts')
print(X_Eval['tag'].value_counts())

#Pull Appropriate Features
combined_text = X_CV['title'] +' '+ X_CV['meta_description'] +' '+  X_CV['content']

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

count_vect = CountVectorizer(ngram_range=(1,4),stop_words='english')
X_CV_counts = count_vect.fit_transform(combined_text)


tfidf_transformer = TfidfTransformer()
X_CV_tfidf = tfidf_transformer.fit_transform(X_CV_counts)



X_CV_tfidf

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 
                     'gamma': [.01, .03, 0.1, 0.3, 1.0, 3.0],
                     'class_weight':[{0:1,1:1},{0:1, 1:4}, {0:1, 1:5}, {0:1, 1:10}],
                     'C': [1/x for x in [0.1, 0.3, 1.0, 3.0, 10.0]]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_CV_tfidf, y_CV)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

#Best model from GS for recall
clf.best_params_

combined_text_eval = X_Eval['title'] + ' ' + X_Eval['meta_description'] + ' ' +  X_Eval['content']

X_Eval_counts = count_vect.transform(combined_text_eval)

X_Eval_tfidf = tfidf_transformer.transform(X_Eval_counts)

from sklearn import svm
#Train model on all of CV training set
clf = svm.SVC(kernel='rbf', C=10.0,gamma=.03,class_weight={0: 1, 1: 4}).fit(X_CV_tfidf, y_CV)
#Score on Evaluation Set
clf.score(X_Eval_tfidf, y_Eval)

from sklearn.metrics import classification_report
y_pred = clf.predict(X_Eval_tfidf)
target_names = ['Disasters', 'Conflict and violence']
print(classification_report(y_Eval, y_pred, target_names=target_names))

#Confusion Matrix
from sklearn.metrics import confusion_matrix

confusion_matrix(y_Eval, y_pred)

#Create DF of eval text, actual y, and predicted y
pd.options.display.max_colwidth = 150

pred_df = pd.DataFrame({ 'text' : combined_text_eval,
                         'actual' : y_Eval,
                         'pred' : y_pred})

pred_df = pred_df[['text', 'actual','pred']]
pred_df.head()

#Correctly predicted as Disaster
pred_df[(pred_df.actual ==0) & (pred_df.pred ==0)]

#Correctly predicted as Conflict and Violence
pred_df[(pred_df.actual ==1) & (pred_df.pred ==1)]

#Incorrectly predicted as Disaster
pred_df[(pred_df.actual ==1) & (pred_df.pred ==0)]

#Incorrectly predicted as Conflict and Violence
pred_df[(pred_df.actual ==0) & (pred_df.pred ==1)]

