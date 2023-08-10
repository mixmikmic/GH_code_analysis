# for Python 2: use print only as a function
# from __future__ import print_function

import ast

import pandas as pd
import numpy as np
from collections import defaultdict
import collections
import gensim

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.preprocessing import MultiLabelBinarizer 
from sklearn.multiclass import OneVsRestClassifier 
from sklearn.multioutput import MultiOutputClassifier                  # included from scikit-learn version 0.18.1 and onwards
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

path = 'C:\Users\Pushpendra\Desktop\qa\datascience_datasets.csv'

d_set = pd.read_csv(path,header=None,names=['tag','title'],dtype=object)

d_set.head()

y_labels = []                                                                  # contains values of d_set['tag']
for i in d_set['tag']:  
    y_labels.append(ast.literal_eval(i))                                       # to remove unicodeed string 
y_labels = [j for i in y_labels for j in i] 
y_labels = list(set(y_labels))

# for i in range(len(y_labels)):
#     if (y_labels[i].find('-') > -1):
#         y_labels[i] = y_labels[i].replace('-','')

import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

class lemmatokenizer(object):
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.token_pattern = r"(?u)\b\w\w+\b"       
#         self.wnl = WordNetLemmatizer()
    def __call__(self,doc):                                                     # here, doc is one string sentence
        token_pattern = re.compile(self.token_pattern)
        return [self.stemmer.stem(t) for t in token_pattern.findall(doc)]       # return lambda doc: token_pattern.findall(doc) 
#         return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

vect_title = CountVectorizer(max_df=0.5,min_df=5,stop_words='english',tokenizer=lemmatokenizer(),ngram_range=(1,3))

tfidf_vect_title = TfidfVectorizer(smooth_idf=False,max_df=0.5,min_df=5,stop_words='english',tokenizer=lemmatokenizer(),ngram_range=(1,3))

le = preprocessing.LabelEncoder()  
le.fit(y_labels) 
d_set['label_num'] = pd.Series([le.transform(ast.literal_eval(i)) for i in d_set['tag']])
d_set.head()

new_y_labels = d_set['label_num'].values.tolist()

# print (new_y_labels)

mlb = MultiLabelBinarizer() 
mlb.fit(new_y_labels)

# mlb.transform(new_y_labels).shape

y_tag_dtm = mlb.transform(new_y_labels) 

# print (type(y_tag_dtm))
# y_tag_dtm = pd.Series(y_tag_dtm) 

y_tag_dtm.shape

X_labels = d_set['title'].values.tolist()

# print (X_labels)

vect_title.fit(X_labels)
X_title_dtm = vect_title.transform(X_labels)

# vect_title.get_feature_names()
# vect_title.get_params
# print (X_title_dtm.toarray())

X_title_dtm



from sklearn.decomposition import PCA

pca = PCA(n_components=100).fit(X_title_dtm.toarray())
pca_samples = pca.transform(X_title_dtm.toarray())

pca_df = pd.DataFrame(np.round(pca_samples,4))

print (pca_df.head())



new_df = pd.DataFrame(X_title_dtm.toarray(),columns=vect_title.get_feature_names())

# new_df.head()
# new_df.ix[1].to_dict().values()

new_df.shape

d = collections.Counter(vect_title.get_feature_names())
# print (d['ai']) 

new_df['target_list'] = [i for i in y_tag_dtm] 

# new_df.columns[:100]  
# new_df.ix[0]

# new_df.head()  

tfidf_vect_title.fit(X_labels)
X_title_dtm_tfidf = tfidf_vect_title.transform(X_labels)

X_title_dtm_tfidf

new_df_of_tfidf = pd.DataFrame(X_title_dtm_tfidf.toarray(),columns=tfidf_vect_title.get_feature_names()) 

new_df_of_tfidf['target_list'] = [i for i in y_tag_dtm] 

y = new_df_of_tfidf['target_list'] 
X = new_df_of_tfidf.drop('target_list',axis=1)  

# y = new_df['target_list']
# X = new_df.drop('target_list',axis=1)

# # X = X.ix[:]                                           # it will return each feature row wise to X
# # X = X.values

# print (type(X))



X = np.array(X.values.tolist())                           # it will convert list to numpy ndarray
y = np.array(y.values.tolist())

# print (X[0]) 

pca_X = PCA(n_components=200).fit_transform(X)  
pca_X = np.round(pca_X,4)

pca_y = PCA(n_components=50).fit_transform(y)  
pca_y = np.round(pca_y,4)

print (pca_y) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)   

# X_train, X_test, y_train, y_test = train_test_split(pca_X, pca_y, test_size=0.2, random_state=1)   



# clf = Pipeline([('classifier',OneVsRestClassifier(SVC(probability=True,random_state=0)))])  # just to for Pipeline example

knn_clf = KNeighborsClassifier(n_neighbors=5)
# mnb_clf = MultinomialNB()                                                                   # not working for MultiLabelinput

# time_pass_y = np.random.randint(2,size=(2838,1))                                            # produce ndarray of size 2838 X 1

knn_clf.fit(X_train, y_train)
# mnb_clf.fit(X_train, y_train) 

knn_pred = knn_clf.predict(X_test)  
# mnb_pred = mnb_clf.predict(X_test)

svc_clf = OneVsRestClassifier(SVC(probability=True,random_state=0))
svc_clf.fit(X_train, y_train)
svc_pred = svc_clf.predict(X_test)  

knn_clf.score(X_test, y_test) 

# """ it will give 0.0, since it's matching with the exact no. of target labels and not giving credit for 
#     partial correct prediction of labels. """

# svc_clf.score(X_test, y_test)        

knn_report = metrics.classification_report(y_test[:], knn_pred[:]) 
knn_f1_score = metrics.f1_score(y_test[:], knn_pred[:], average='samples') 
knn_precision_recall_fscore = metrics.precision_recall_fscore_support(y_test, knn_pred, average='samples')  # on full data-set
knn_avg_precision_score = metrics.average_precision_score(y_test, knn_pred, average='samples')
knn_roc_auc_score = metrics.roc_auc_score(y_test, knn_pred, average='samples')

# mnb_report = metrics.classification_report(y_test[:100], mnb_pred[:100])  #throwing error mnb_clf can't work on multilabel O/P

svc_report = metrics.classification_report(y_test[:], svc_pred[:]) 
svc_f1_score = metrics.f1_score(y_test[:], svc_pred[:], average='samples') 
svc_precision_recall_fscore = metrics.precision_recall_fscore_support(y_test, svc_pred, average='samples')  # on full data-set
svc_avg_precision_score = metrics.average_precision_score(y_test, svc_pred, average='samples')
svc_roc_auc_score = metrics.roc_auc_score(y_test, svc_pred, average='samples')

# """  beacuse it will also give 0.0 for the same reason discussed above. """

# metrics.accuracy_score(y_true=y_test[:], y_pred=svc_pred[:])          # I think it's same as calculating hamming_score

# print "For svc_clf (LinearSVC) : "
# print "precision, recall, fbeta_score, support : ",svc_precision_recall_fscore
# print "f1_score : ",svc_f1_score
print "avg. precision_score : ",svc_avg_precision_score 
print "roc_auc_score : ",svc_roc_auc_score

metrics.accuracy_score(y_true=y_test[:], y_pred=knn_pred[:])          # I think it's same as calculating hamming_score

# print (knn_report)                                   # its type is str

print "For knn_clf (KNearestNeighbours) : "
print "precision, recall, fbeta_score, support : ",knn_precision_recall_fscore
print "f1_score : ",knn_f1_score
print "avg. precision_score : ",knn_avg_precision_score 
print "roc_auc_score : ",knn_roc_auc_score

# def does_test_tag_match(d, list_of_tags):      # no need for this function

test = ["why is overfitting bad in machine learning ?"]
# test = ["what is lstm ?"] 

# test_dtm = vect_title.transform(test)                                           # without tfidf
test_dtm = tfidf_vect_title.transform(test)                                       # with tfidf

# print (test_dtm.toarray()[0])
status = False
for i in test_dtm.toarray()[0]:
    if (i!=0):
        status = True
        break

ans = knn_clf.predict(test_dtm.toarray())
ans = mlb.inverse_transform(ans)

if (len(ans[0])==0 or status==False):
    print ("sorry, we can't predict your category!!!")
else:
    ans = le.inverse_transform(ans)
    print (ans)
    
    
# mlb.transform([[224,0,100]]) 
# ans 
# test_dtm.toarray()

forest = RandomForestClassifier(n_estimators=10, random_state=0)
rf_clf = MultiOutputClassifier(forest, n_jobs=-1)
rf_clf.fit(X_train, y_train)
rf_pred = rf_clf.predict(X_test)

rf_clf 

metrics.accuracy_score(y_true=y_test[:100], y_pred=rf_pred[:100])          # I think it's same as calculating hamming_score

rf_clf.score(X_test, y_test)

# rf_clf.predict_log_proba(X_test)                                               # no such function in MultiOutputClassifier()

# print (knn_clf.predict_proba(X_train))                                         # running smooth

# print (rf_clf.predict_proba(X_train[:5]))                                      # showing error

rf_report = metrics.classification_report(y_test[:], rf_pred[:])
rf_f1_score = metrics.f1_score(y_test, rf_pred, average='samples')  
rf_precision_recall_fscore = metrics.precision_recall_fscore_support(y_test, rf_pred, average='samples')  # on full data-set
rf_avg_precision_score = metrics.average_precision_score(y_test, rf_pred, average='samples')
rf_roc_auc_score = metrics.roc_auc_score(y_test, rf_pred, average='samples') 

# print (rf_report) 

print "For rf_clf (RandomForest) : "
print "precision, recall, fbeta_score, support : ",rf_precision_recall_fscore
print "f1_score : ",rf_f1_score  
print "avg. precision_score : ",rf_avg_precision_score 
print "roc_auc_score : ",rf_roc_auc_score

test = ["what is lstm ?"] 

# test_dtm = vect_title.transform(test)                                            # without tfidf
test_dtm = tfidf_vect_title.transform(test)                                        # with tfidf

status = False
for i in test_dtm.toarray()[0]:
    if (i!=0):
        status = True
        break

ans = rf_clf.predict(test_dtm.toarray())
ans = mlb.inverse_transform(ans)
if (len(ans[0])==0 or status==False):
    print ("sorry, we can't predict your category!!!")
else:
    ans = le.inverse_transform(ans)
    print (ans)
    
# mlb.transform([[224,0,100]]) 
# ans 
# test_dtm.toarray()

from sklearn.externals import joblib
joblib.dump(rf_clf, 'datascience_classifier.pkl')
# new_clf = joblib.load('classifier.pkl')

new_pkl_clf = joblib.load('datascience_classifier.pkl')

new_pkl_clf

test = ["why is overfitting bad in machine learning ?"] 

# test_dtm = vect_title.transform(test)                                           # without tfidf
test_dtm = tfidf_vect_title.transform(test)                                       # with tfidf       

status = False
for i in test_dtm.toarray()[0]:
    if (i!=0):
        status = True
        break
        
ans = new_pkl_clf.predict(test_dtm.toarray())
ans = mlb.inverse_transform(ans)
if (len(ans[0])==0 or status==False):
    print ([["sorry, we can't predict your category!!!"]]) 
else:
    ans = le.inverse_transform(ans)
    print (ans)





