import pandas as pd
from sklearn import svm
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, cross_val_score
from nltk.corpus import stopwords

corpus_feat = pd.read_csv('corpus_liwc_mtx.csv.gz', compression='gzip')
corpus = pd.read_csv('corpus.csv.gz', compression='gzip')
stopwords = stopwords.words("portuguese")

# filter corpus
corpus = corpus[corpus['qual_a_melhor_classificao_para_esse_texto:confidence'] == 1]
corpus = corpus.reset_index()

# fix labels to binary
def classFit(x):
    if x['qual_a_melhor_classificao_para_esse_texto'] == "diario":
        return 1
    else:
        return -1
    
corpus['class'] = corpus.apply(classFit,axis=1)
target = corpus['class'].values

print(corpus['qual_a_melhor_classificao_para_esse_texto'].values[:2])
print(corpus['class'][:2])

from scipy.sparse import hstack

corpus_feat.drop('Unnamed: 0', axis=1,inplace=True)
corpus_feat.drop('confidence', axis=1,inplace=True)
corpus_feat.drop('wc', axis=1,inplace=True)
liwc_data = corpus_feat.drop('class', 1).values

tfidf_data = TfidfVectorizer(max_features=900, strip_accents='unicode', stop_words=stopwords).fit_transform(corpus.content)

data = hstack((tfidf_data, liwc_data))
data.shape

from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()

precision = cross_val_score(model, data, target, cv=10, scoring='precision').mean()
acc = cross_val_score(model, data, target, cv=10, scoring='accuracy').mean()
recall = cross_val_score(model, data, target, cv=10, scoring='recall').mean()
print(precision)
print(acc)
print(recall)

model = svm.LinearSVC(C=50)

precision = cross_val_score(model, data, target, cv=10, scoring='precision').mean()
acc = cross_val_score(model, data, target, cv=10, scoring='accuracy').mean()
recall = cross_val_score(model, data, target, cv=10, scoring='recall').mean()
print(precision)
print(acc)
print(recall)

from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np
import oll

## manual 10-fold cross-validation
kf = KFold(n_splits=2, random_state=None, shuffle=False)

methods = ["P" ,"AP" ,"PA" ,"PA1","PA2" ,"PAK","CW" ,"AL"]

for m in methods:
    model = oll.oll(m, C=1)

    accuracy = []
    precision = []
    recall = []
    
    for train_index, test_index in kf.split(data):

        X_train, X_test = data.toarray()[train_index], data.toarray()[test_index]
        y_train, y_test = target[train_index], target[test_index]
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)

        accuracy.append(accuracy_score(y_test, predicted))
        precision.append(precision_score(y_test, predicted))
        recall.append(recall_score(y_test, predicted))

    print(m + ': acc(' + str(np.mean(accuracy)) 
          + '), prec(' + str(np.mean(precision))
          + '), rec(' + str(np.mean(recall)) + ')'
         )

from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier()

precision = cross_val_score(model, data.toarray(), target, cv=10, scoring='precision').mean()
acc = cross_val_score(model, data.toarray(), target, cv=10, scoring='accuracy').mean()
recall = cross_val_score(model, data.toarray(), target, cv=10, scoring='recall').mean()
fscore = cross_val_score(model, data.toarray(), target, cv=10, scoring='f1').mean()
print(precision)
print(fscore)



