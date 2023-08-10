import pandas as pd
df_dbp = pd.read_csv('dbp_types_en.csv')
df_dbp.head()

types = ['person', 'organisation', 'location', 'other']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_dbp['ocr'], df_dbp[types].as_matrix(), random_state=0)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

from sklearn.feature_extraction.text import TfidfVectorizer

count_vect = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(2,5), analyzer='char_wb', max_features=10000)
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

len(count_vect.vocabulary_.keys())

from sklearn.externals import joblib
joblib.dump(count_vect, 'dbp_types_en_vct.pkl') 

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

clf = OneVsRestClassifier(SVC(probability=True, kernel='linear', class_weight='balanced', C=1.0, verbose=True))
clf.fit(X_train_counts, y_train)

joblib.dump(clf, 'dbp_types_en_clf.pkl') 

pred = clf.predict(X_test_counts)
pred

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, pred)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, pred)

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

print('average precision', precision_score(y_test, pred, average='macro'))
print('average recall', recall_score(y_test, pred, average='macro'))
print('average f1', f1_score(y_test, pred, average='macro'))

scores = {}
scores['precision'] = precision_score(y_test, pred, average=None)
scores['recall'] = recall_score(y_test, pred, average=None)
scores['f1'] = f1_score(y_test, pred, average=None)

pd.DataFrame(data=scores, index=types)



