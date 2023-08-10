#Download the libraries
import nltk
import re
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import sys
import time
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn import tree

print os.getcwd();

os.chdir("/Users/imacair/Desktop/Products3/")

data= pd.read_csv('Final_Manual_3006.csv',
                    encoding='latin-1',delimiter=',')

#Converts text into ASCII

#data.text = data.text.str.encode('ascii','replace')

data.message = data.message.str.encode('ascii','replace')

len(data)

#sample_data= data.sample(n=64000)



sample_data= data

#data_t=sample_data["text"]

data_t=sample_data["message"]

len(data_t)



#lowercase
data_t = data_t.str.lower()

data_s=sample_data["sentiment"]

np.unique(data_s)

#data_s= data_s[~np.isnan(data_s)]

final = data
res5= pd.DataFrame( index=range(0,len(data_t)),columns = {'new_sent'} )
res5[(final.sentiment==u'2')] = '-1'
res5[(final.sentiment==u'1')] = '-1'
res5[(final['sentiment']==u'3')] = '1'
res5[(final['sentiment']==u'4')] = '1'
res5[(final['sentiment']==u'N')] = '0'
res5[(final['sentiment']==u"n")] = '0'
final=pd.concat([final, res5], axis=2)

#final['sentiment'] = final['sentiment'][~pd.isnull(final['sentiment'])]

#final['sentiment']=np.nan_to_num(final['sentiment'])
#final['sentiment'] = final[~np.isnan('sentiment')]
#y = 
#final['sentiment'][~np.isfinite(final['sentiment'])]

np.unique(final.new_sent)

with open('abbrev.csv', mode='r') as infile:
    reader = csv.reader(infile)
    replacement = {rows[0].lower():rows[1].lower() for rows in reader              
                  }

#replacement

#replacement = {
##'r':'are',
#'y':'why',
#'u':'you'}

##How in works
s1 = 'y r u l8'

s2 = ' '.join([replacement.get(w, w) for w in s1.split()])
s2

result = pd.DataFrame()
result = final

for i in range(len(result)):
    data_t.values[i]=' '.join([replacement.get(w, w) for w in data_t.values[i].split()])


text = data_t.to_string()
text = nltk.word_tokenize(text)
fdist = nltk.FreqDist(text)
s2 = set([w for w in set(text) if len(w) > 2 and fdist[w] > 2])
for i in range(len(result)):
    data_t.values[i]=' '.join(filter(lambda w: w in s2,data_t.values[i].split()))



from nltk.corpus import stopwords
s=set(stopwords.words('english'))
for i in range(len(result)):
    data_t.values[i]=' '.join(filter(lambda w: not w in s,data_t.values[i].split()))

data_t

data_t.head(10)

#lowercase
data_t = data_t.str.lower()
#Remove urls
data_t= data_t.str.replace(r'(http.*) |(http.*)$|\n', "",)
#Remove twitter handles
data_t = data_t.str.replace(r"@\\w+", "")
#remove htmls
data_t = data_t.str.replace(r'<.*?>', "")
#Remove citations
data_t = data_t.str.replace(r'@[a-zA-Z0-9]*', "")

#remove _
#data_t = data_t.str.replace(r'\_+',"")

data_t.head(10)

from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.model_selection import KFold

data_train, data_test, label_train, label_test = train_test_split(data_t, final.new_sent, test_size=0.3, random_state=2340)

#data_train, data_test, label_train, label_test = train_test_split(data_t, data_s, test_size=0.3, random_state=2340)

t0 = time.time()
vectorizer = TfidfVectorizer(    sublinear_tf=True,
                                 use_idf=True,stop_words = 'english')
train_vectors = vectorizer.fit_transform(data_train)
test_vectors = vectorizer.transform(data_test)
t1 = time.time()
time_vec = t1-t0

print(time_vec)

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(C=0.6, kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, label_train)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(C=0.6, kernel='linear')
t0 = time.time()
classifier_linear.fit(train_vectors, label_train)
t1 = time.time()
prediction_linear = classifier_linear.predict(test_vectors)
t2 = time.time()
time_linear_train = t1-t0
time_linear_predict = t2-t1

print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(label_test, prediction_linear))
confusion_matrix(label_test, prediction_linear)

# Perform classification with SVM, kernel=linear
classifier_liblinear = svm.LinearSVC()
t0 = time.time()
classifier_liblinear.fit(train_vectors, label_train)
t1 = time.time()
prediction_liblinear = classifier_liblinear.predict(test_vectors)
t2 = time.time()
time_liblinear_train = t1-t0
time_liblinear_predict = t2-t1

from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Perform classification with random forest
classifier_rf = RandomForestClassifier()
t0 = time.time()
classifier_rf.fit(train_vectors, label_train)
t1 = time.time()
prediction_rf = classifier_rf.predict(test_vectors)
t2 = time.time()
time_rf_train = t1-t0
time_rf_predict = t2-t1

# Perform classification with Multinomial Naïve Bayes.
classifier_nb = MultinomialNB()
t0 = time.time()
classifier_nb.fit(train_vectors, label_train)
t1 = time.time()
prediction_nb = classifier_nb.predict(test_vectors)
t2 = time.time()
time_nb_train = t1-t0
time_nb_predict = t2-t1


print("Results for SVC(kernel=linear)")
print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
print(classification_report(label_test, prediction_linear))
print("Results for LinearSVC()")
print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
print(classification_report(label_test, prediction_liblinear))
print("Results for MultinomialNB()")
print("Training time: %fs; Prediction time: %fs" % (time_nb_train, time_nb_predict))
print(classification_report(label_test, prediction_nb))

from sklearn.ensemble import VotingClassifier
from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10, random_state=8)

estimators = []
#estimators.append(('rbf_svm', classifier_rbf ))
estimators.append(('linear_svm',classifier_liblinear))
estimators.append(('linear2_svm',classifier_linear))
estimators.append(('random_forest',classifier_rf))
estimators.append(('multi_naive',classifier_nb))
#estimators.append(('lstm',model))

ensemble = VotingClassifier(estimators)


results = model_selection.cross_val_score(ensemble, train_vectors, label_train, cv=kfold)
print(results.mean())

label_tests = np.asarray(label_test)

label_tests

df =[]
df.append('linear_svm')
df.append('linear2_svm')
df.append('random_forest')
df.append('multi_naive')
df.append('label_tests')

prediction_linear

df

df=[prediction_linear, prediction_liblinear, prediction_nb,label_tests]

df = pd.DataFrame(df)
df = df.transpose()
df.columns = ['prediction_linear', 'prediction_liblinear', 'prediction_nb','label_tests']
#df

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
import numpy as np


lr = RandomForestClassifier()
sclf = StackingClassifier(classifiers=[classifier_liblinear,classifier_linear,classifier_nb], 
                          meta_classifier=lr)

print('3-fold cross validation:\n')

for clf, label in zip([classifier_liblinear,classifier_linear, classifier_nb,sclf], 
                      ['linear_svm', 
                       'linear2_svm',
                       'multi_naive',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, train_vectors, label_train, 
                                              cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
import numpy as np


lr = RandomForestClassifier()
sclf = StackingClassifier(classifiers=[classifier_liblinear,classifier_linear, classifier_nb], 
                          meta_classifier=lr)

print('3-fold cross validation:\n')

for clf, label in zip([classifier_liblinear,classifier_linear,classifier_nb,sclf], 
                      ['linear_svm', 
                       'linear2_svm',
                       'multi_naive',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, train_vectors, label_train, 
                                              cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
import numpy as np


lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[classifier_liblinear,classifier_linear, classifier_nb], 
                          meta_classifier=lr)

print('3-fold cross validation:\n')

for clf, label in zip([classifier_liblinear,classifier_linear,classifier_nb,sclf], 
                      ['linear_svm', 
                       'linear2_svm',
                       'multi_naive',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, train_vectors, label_train, 
                                              cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))

from sklearn import model_selection
from mlxtend.classifier import StackingClassifier
import numpy as np


lr = classifier_linear
sclf = StackingClassifier(classifiers=[classifier_liblinear, classifier_nb], 
                          meta_classifier=lr)

print('3-fold cross validation:\n')

for clf, label in zip([classifier_liblinear,classifier_nb,sclf], 
                      ['linear_svm', 
                       'multi_naive',
                       'StackingClassifier']):

    scores = model_selection.cross_val_score(clf, train_vectors, label_train, 
                                              cv=10, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
import numpy as np


lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[classifier_liblinear,classifier_linear, classifier_nb], 
                          meta_classifier=lr)


sclf.fit(train_vectors, label_train)

prediction_sclf = sclf.predict(test_vectors)
confusion_matrix(label_test, prediction_sclf)


prediction_sclf = sclf.predict(test_vectors)

#Convert to np arrays
label_tests = np.asarray(label_test)


#Create a Data Frame
df=[ prediction_linear, prediction_liblinear,prediction_nb,prediction_sclf]
df = pd.DataFrame(df)
df = df.transpose()
df.columns = ['prediction_linear', 'prediction_liblinear','prediction_nb','staking']
df

#Convert to np arrays
label_tests = np.asarray(label_test)


#Create a Data Frame
df=[ prediction_linear, prediction_liblinear,prediction_nb]
df = pd.DataFrame(df)
df = df.transpose()
df.columns = ['prediction_linear', 'prediction_liblinear','prediction_nb']
df

from collections import Counter
df2= pd.DataFrame( index=range(0,len(data_test)),columns = {'final'} )
for i in range(len(data_test)):
    d=Counter(df.ix[i,:])
    dfin=d.most_common(1)[0][0]
    df2.values[i]= dfin
df["final"] = df2

df





