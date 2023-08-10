
import pandas as pd

gtdb_path = "../data/csv/GTDB"
csv_path =  gtdb_path + '.csv'
encoding = ['latin1', 'iso8859-1', 'utf-8'][1]
gtdb_df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
gtdb_df.columns


import re

attack_regex = re.compile(r"attack")
attack_column_list = [column for column in gtdb_df.columns for m in [attack_regex.search(column)] if m]
attack_column_list


gtdb_df['attacktype1_txt'].unique()


attack_types_path = "../data/csv/AttackTypes"
csv_path =  attack_types_path + '.csv'
attack_types_df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
attack_types_df


ucdp_path = "../data/csv/UCDP"
csv_path =  ucdp_path + '.csv'
ucdp_df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
ucdp_df.columns


scad_path = "../data/csv/SCAD"
csv_path =  scad_path + '.csv'
scad_df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
scad_df.columns


rand_path = "../data/csv/RAND"
csv_path =  rand_path + '.csv'
rand_df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
rand_df.columns


acled_path = "../data/csv/ACLED"
csv_path =  acled_path + '.csv'
acled_df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
acled_df.columns


def concat_independendent_variables(df):
    X = pd.Series([])
    for row_index, row_series in df.iterrows():
        row_concat = row_series.astype('str').str.cat(sep=' ').strip()
        row_concat = sq_regex.sub(r'', row_concat)
        row_concat = nonalpha_regex.sub(r' ', row_concat)
        X = X.append(pd.Series([row_concat]), ignore_index=True)
    
    return X

nonalpha_regex = re.compile(r"[^a-zA-Z]+")
sq_regex = re.compile(r"'")


import time

t0 = time.time()
X = pd.Series([])
y = pd.Series([])
for csv_file in ['acled', 'rand', 'scad', 'ucdp', 'GTDB']:
    if csv_file == "GTDB":
        gtdb_path = "../data/csv/GTDB"
        csv_path =  gtdb_path + '.csv'
    else:
        relabeled_path = "../data/csv/mike_"
        csv_path =  relabeled_path + csv_file + '.csv'
    df = pd.read_csv(csv_path, encoding=encoding, low_memory=False)
    df.fillna(value="", inplace=True)
    if csv_file == "GTDB":
        important_columns = [column for column in df.columns if (column not in attack_column_list)]
    else:
        important_columns = df.columns.tolist()[:-1]
    X = X.append(concat_independendent_variables(df[important_columns]), ignore_index=True)
    if csv_file == "GTDB":
        y = y.append(df['attacktype1'].map(lambda x: int(x)-1), ignore_index=True)
    else:
        y = y.append(df[df.columns.tolist()[-1]].map(lambda x: attack_types_df['Attack Type'].tolist().index(x)), 
                     ignore_index=True)
t1 = time.time()
print(t1-t0, time.ctime(t1))


from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
import numpy as np

class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=0)
csv_path = "../data/csv/"
csv_files = [join(csv_path, f) for f in listdir(csv_path) if isfile(join(csv_path, f))]
gtdb_train = Bunch(filenames=np.asarray(csv_files),
                   target_names=attack_types_df['Attack Type'].tolist(),
                   DESCR=None,
                   target=np.asarray(y_train.tolist()),
                   data=X_train.tolist(),
                   description="The GTDB dataset concatoned into one column (minus the target columns)")
gtdb_test = Bunch(filenames=np.asarray(csv_files),
                   target_names=attack_types_df['Attack Type'].tolist(),
                   DESCR=None,
                   target=np.asarray(y_test.tolist()),
                   data=X_test.tolist(),
                   description="The GTDB dataset concatoned into one column (minus the target columns)")
gtdb_all = Bunch(filenames=np.asarray(csv_files),
                   target_names=attack_types_df['Attack Type'].tolist(),
                   DESCR=None,
                   target=np.asarray(y.tolist()),
                   data=X.tolist(),
                   description="The GTDB dataset concatoned into one column (minus the target columns)")


gtdb_train.data[:3]


for t in gtdb_train.target[:3]:
    print(gtdb_train.target_names[t])


from sklearn.feature_extraction.text import CountVectorizer

t0 = time.time()
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(gtdb_train.data)
t1 = time.time()
print(t1-t0, time.ctime(t1))
X_train_counts.shape


count_vect.vocabulary_.get(u'ayacucho')


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape


from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf, gtdb_train.target)

docs_new = gtdb_test.data[:3]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, gtdb_train.target_names[category]))


from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])
text_clf = text_clf.fit(gtdb_train.data, gtdb_train.target)
predicted = text_clf.predict(gtdb_test.data)
np.mean(predicted == gtdb_test.target) 


from sklearn.linear_model import SGDClassifier

t0 = time.time()
text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                     ('clf', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
])
text_clf = text_clf.fit(gtdb_train.data, gtdb_train.target)
predicted = text_clf.predict(gtdb_test.data)
t1 = time.time()
print(t1-t0, time.ctime(t1))
np.mean(predicted == gtdb_test.target)


from sklearn import metrics

print(metrics.classification_report(gtdb_test.target, predicted, target_names=gtdb_test.target_names))


metrics.confusion_matrix(gtdb_test.target, predicted)


from sklearn.model_selection import GridSearchCV

t0 = time.time()

parameters = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3, 1e-4),
              'clf__loss': ('log', 'modified_huber'),
}
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(penalty='l2', n_iter=5, random_state=42)),
])
gs_all_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_all_clf = gs_all_clf.fit(gtdb_all.data, gtdb_all.target)

t1 = time.time()
print(t1-t0, time.ctime(t1))

# 0.7536404915378021
gs_all_clf.best_score_


t0 = time.time()
predicted = gs_all_clf.predict(gtdb_test.data)
t1 = time.time()
print(t1-t0, time.ctime(t1))
np.mean(predicted == gtdb_test.target)


t0 = time.time()

for df in [acled_df, rand_df, scad_df, ucdp_df, gtdb_df]:
    data = concat_independendent_variables(df).tolist()
    df['predicted_id'] = gs_all_clf.predict(data)
    df['predicted_type'] = df['predicted_id'].map(lambda x: gtdb_all.target_names[x])
    df['probabilities'] = pd.Series(list(gs_all_clf.predict_proba(data)))
    df['probability'] = df.apply(lambda row: "{0:.1f}%".format(row['probabilities'][row['predicted_id']]*100), axis=1)
    df.drop(['predicted_id','probabilities'], axis=1, inplace=True)

t1 = time.time()
print(t1-t0, time.ctime(t1))


csv_folder = "../data/csv/"
gtdb_df.to_csv(csv_folder+"gtdb_df.csv", sep=',', encoding=encoding, index=False)
acled_df.to_csv(csv_folder+"acled_df.csv", sep=',', encoding=encoding, index=False)
rand_df.to_csv(csv_folder+"rand_df.csv", sep=',', encoding=encoding, index=False)
scad_df.to_csv(csv_folder+"scad_df.csv", sep=',', encoding=encoding, index=False)
ucdp_df.to_csv(csv_folder+"ucdp_df.csv", sep=',', encoding=encoding, index=False)



