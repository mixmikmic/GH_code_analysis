import pandas as pd

df_orig = pd.read_csv('news_topics.csv')
df_orig.head()

df_orig.shape

df_sru = pd.read_csv('news_topics_sru.csv')
df_sru.head()

df_sru.shape

df = pd.concat([df_orig, df_sru])
df.shape

df = df.drop(['url', 'topic'], axis=1)
df.head()

topics = ['politics', 'business', 'culture', 'science', 'sports', 'crime', 'disasters', 'environment', 'health',
          'education', 'religion', 'lifestyle', 'other']

df['sum_0'] = (df[topics] == 0).sum(axis=1)
df['sum_1'] = (df[topics] == 1).sum(axis=1)
df['sum_2'] = (df[topics] == 2).sum(axis=1)

df.head(10)

non_unique = df.loc[df['sum_1'] != 1]
non_unique.shape

df.shape[0]

counts = [df.shape[0] - df[t].value_counts()[0] for t in topics]
primary_counts = [df.shape[0] - (df[t].value_counts()[0] + df[t].value_counts()[2]) for t in topics]
data = {'topic': topics, 'any': counts, 'primary': primary_counts}
counts = pd.DataFrame(data=data)
counts

import matplotlib.pyplot as plt

ax = counts[['any','primary']].plot(kind='bar', title ="Topic distribution",
                                    figsize=(15, 10), legend=True, fontsize=12)
plt.show()

df[topics].as_matrix()

df[topics] = df[topics].replace(to_replace=2, value=1)
df[topics].as_matrix()

from sklearn.model_selection import train_test_split

topics = ['politics', 'business', 'culture', 'science', 'sports']

X_train, X_test, y_train, y_test = train_test_split(df['ocr'], df[topics].as_matrix(), random_state=0)

X_train.shape, y_train.shape

X_test.shape, y_test.shape

from sklearn.feature_extraction.text import TfidfVectorizer

count_vect = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(2,5), analyzer='char_wb', max_features=10000)
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

len(count_vect.vocabulary_.keys())

from sklearn.externals import joblib
joblib.dump(count_vect, 'news_topics_nl_vct.pkl') 

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

clf = OneVsRestClassifier(SVC(probability=True, kernel='linear', class_weight='balanced', C=1.0, verbose=True))
clf.fit(X_train_counts, y_train)

clf.classes_

clf.multilabel_

joblib.dump(clf, 'news_topics_nl_clf.pkl') 

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

pd.DataFrame(data=scores, index=topics)

article = '''Minister Romme en partijgenoot Lieftinck van het CDA verlieten gisteren het Binnenhof
        om met de priester in de kerk te gaan praten over de zin en onzin van religieuze geboortebeperking.
        Computer op komst.'''

article_counts = count_vect.transform([article])
clf.predict_proba(article_counts)

df_dbp = pd.read_csv('dbp_topics_en.csv')
df_dbp.head()

df_dbp.shape

X_test_counts_dbp = count_vect.transform(df_dbp['ocr'])

y_test_dbp = df_dbp[topics].as_matrix()

pred_dbp = clf.predict(X_test_counts_dbp)
pred_dbp

scores = {}
scores['precision'] = precision_score(y_test_dbp, pred_dbp, average=None)
scores['recall'] = recall_score(y_test_dbp, pred_dbp, average=None)
scores['f1'] = f1_score(y_test_dbp, pred_dbp, average=None)

pd.DataFrame(data=scores, index=topics)

X_train, X_test, y_train, y_test = train_test_split(df_dbp['ocr'], df_dbp[topics].as_matrix(), random_state=0)

count_vect = TfidfVectorizer(min_df=5, max_df=0.9, ngram_range=(2,5), analyzer='char_wb', max_features=10000)
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

joblib.dump(count_vect, 'dbp_topics_en_vct.pkl')

clf = OneVsRestClassifier(SVC(probability=True, kernel='linear', class_weight='balanced', C=1.0, verbose=True))
clf.fit(X_train_counts, y_train)

clf.multilabel_

joblib.dump(clf, 'dbp_topics_en_clf.pkl')

pred = clf.predict(X_test_counts)
pred

roc_auc_score(y_test, pred)

print('average precision', precision_score(y_test, pred, average='macro'))
print('average recall', recall_score(y_test, pred, average='macro'))
print('average f1', f1_score(y_test, pred, average='macro'))

scores = {}
scores['precision'] = precision_score(y_test, pred, average=None)
scores['recall'] = recall_score(y_test, pred, average=None)
scores['f1'] = f1_score(y_test, pred, average=None)

pd.DataFrame(data=scores, index=topics)

df_dbp = df_dbp.drop(['url', 'topic'], axis=1)
df = df.drop(['sum_0', 'sum_1', 'sum_2'], axis=1)
df_comb = pd.concat([df, df_dbp])
df_comb.head()

X_train, X_test, y_train, y_test = train_test_split(df_comb['ocr'], df_comb[topics].as_matrix(), random_state=0)

X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)

clf = OneVsRestClassifier(SVC(probability=True, kernel='linear', class_weight='balanced', C=1.0))
clf.fit(X_train_counts, y_train)

pred = clf.predict(X_test_counts)
pred

roc_auc_score(y_test, pred)

print('average precision', precision_score(y_test, pred, average='macro'))
print('average recall', recall_score(y_test, pred, average='macro'))
print('average f1', f1_score(y_test, pred, average='macro'))

scores = {}
scores['precision'] = precision_score(y_test, pred, average=None)
scores['recall'] = recall_score(y_test, pred, average=None)
scores['f1'] = f1_score(y_test, pred, average=None)

pd.DataFrame(data=scores, index=topics)



