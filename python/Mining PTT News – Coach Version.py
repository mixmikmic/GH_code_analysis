# init for plotting

get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.style.use('ggplot')

# init the utils

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

# doc: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
vectorizer = CountVectorizer(
    # the default regex select tokens of 2+ alphanumeric characters
    # it filters 催 停 到 from 催 太猛 停車 停 到 衝進 藥局,
    # but they are significant in Chinese, we rewrite a regex here
    token_pattern=r'(?u)\b\w+\b',
    # unigram and bigram
    # unigram: 催 / 太猛 / 衝進 -> 催 / 太猛 / 衝進
    # bigram: 催 / 太猛 / 衝進 -> 催 太猛 / 太猛 衝進
    ngram_range=(1, 2)
)

# smat: sparse matrix
input_term_count_smat = vectorizer.fit_transform([
    '新聞 中午 吃 什麼',
    '新聞 急症 沒錢醫',
])

print(vectorizer.get_feature_names())
# rows are inputs
# columns are counts of a feature
print(input_term_count_smat.toarray())

with open('corpus.txt') as f:
    print(next(f), end='')
    print(next(f), end='')
    print(next(f), end='')

with open('corpus.txt') as f:
    input_term_count_smat = vectorizer.fit_transform(f)

# mat: matrix
term_count_mat = input_term_count_smat.sum(axis=0)
term_count_arr = np.asarray(term_count_mat).reshape(-1)
sorted_term_count_arr = np.sort(term_count_arr)

# doc: http://pandas.pydata.org/pandas-docs/stable/visualization.html
sorted_term_count_s = pd.Series(sorted_term_count_arr)
sorted_term_count_s.plot()

sorted_term_count_s.describe()

vectorizer.stop_words_

feature_names = vectorizer.get_feature_names()
sorted_term_idx_arr = np.argsort(term_count_arr)

n = 10

print('Top', n, 'terms and counts:')
print()
for term_idx in sorted_term_idx_arr[:-n:-1]:
    print(feature_names[term_idx], term_count_arr[term_idx])

vectorizer_2 = CountVectorizer(
    min_df=1, max_df=0.5,
    token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1, 2)
)

with open('corpus.txt') as f:
    input_term_count_smat_2 = vectorizer_2.fit_transform(f)
    
term_count_mat_2 = input_term_count_smat_2.sum(axis=0)
term_count_arr_2 = np.asarray(term_count_mat_2).reshape(-1)
sorted_term_count_arr_2 = np.sort(term_count_arr_2)

sorted_term_count_s_2 = pd.Series(sorted_term_count_arr_2)
sorted_term_count_s_2.plot()

sorted_term_count_s_2.describe()

vectorizer_2.stop_words_

feature_names = vectorizer_2.get_feature_names()
sorted_term_idx_arr = np.argsort(term_count_arr_2)

n = 10

print('Top', n, 'terms and counts:')
print()
for term_idx in sorted_term_idx_arr[:-n:-1]:
    print(feature_names[term_idx], term_count_arr_2[term_idx])

with open('target.txt') as f:
    print(next(f), end='')
    print(next(f), end='')
    print(next(f), end='')

with open('target.txt') as f:
    push_score_sums = [int(line) for line in f]

push_score_sum_s = pd.Series(push_score_sums)
push_score_sum_s.plot.hist(bins=100)

push_score_sum_s.describe()

from sklearn.feature_extraction.text import TfidfVectorizer

# doc: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
vectorizer = TfidfVectorizer(
    min_df=1, max_df=0.5,
    token_pattern=r'(?u)\b\w+\b',
    ngram_range=(1, 2)
)

with open('corpus.txt') as f:
    X = vectorizer.fit_transform(f)
    
with open('target.txt') as f:
    # let's use -1 to represent 被噓, and 0 for other cases
    y = [-1 if int(line) <= -1 else 0 for line in f]

from sklearn.model_selection import train_test_split

# doc: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn import svm
from sklearn.metrics import classification_report

# doc: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
svc = svm.SVC(kernel='linear', C=1, class_weight='balanced')

print('SVC')
print()

get_ipython().magic('time svc.fit(X_train, y_train)')
print()

# doc: http://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score
print('Traning Set Accuracy:', svc.score(X_train, y_train))
print('Validation Accuracy:', svc.score(X_test, y_test))
print()

print(classification_report(y_test, svc.predict(X_test)))

class_i = 0
n = 10

feature_names = vectorizer.get_feature_names()
sorted_feature_idx_arr = np.argsort(svc.coef_.toarray())[class_i]

print('Top', n, 'positive features:')
print()

for fidx in sorted_feature_idx_arr[:-n:-1]:
    print(feature_names[fidx], svc.coef_[class_i, fidx])
    
print()
    
print('Top', n, 'negative features:')
print()

for fidx in sorted_feature_idx_arr[:n]:
    print(feature_names[fidx], svc.coef_[class_i, fidx])

# doc: http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
lsvc = svm.LinearSVC(C=1, penalty='l1', dual=False)

print('Linear SVC')
print()

get_ipython().magic('time lsvc.fit(X_train, y_train)')
print()

print('Training Set Shape:', X_train.shape)
print('Traning Set Accuracy:', lsvc.score(X_train, y_train))
print('Validation Accuracy:', lsvc.score(X_test, y_test))
print()

print(classification_report(y_test, lsvc.predict(X_test)))

from sklearn.feature_selection import SelectFromModel

# doc: http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
sfm = SelectFromModel(lsvc, prefit=True)
X_train_2 = sfm.transform(X_train)
X_test_2 = sfm.transform(X_test)

lsvc_2 = svm.LinearSVC(C=1, penalty='l1', dual=False)

print('Linear SVC #2')
print()

get_ipython().magic('time lsvc_2.fit(X_train_2, y_train)')
print()

print('Training Set Shape:', X_train_2.shape)
print('Traning Set Accuracy:', lsvc_2.score(X_train_2, y_train))
print('Validation Accuracy:', lsvc_2.score(X_test_2, y_test))
print()

print(classification_report(y_test, lsvc_2.predict(X_test_2)))

from sklearn.model_selection import GridSearchCV

# CV: Cross-Validation, http://scikit-learn.org/stable/modules/cross_validation.html
# Grid Search: http://scikit-learn.org/stable/modules/grid_search.html
# doc: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
gscv = GridSearchCV(
    svm.LinearSVC(),
    {'C': [1, 10, 100, 1000]},
    cv=5
)

print('Grid Search CV on LinearSVC')
print()

get_ipython().magic('time gscv.fit(X_train, y_train)')
print()

print('Best Parameters:', gscv.best_params_)
print('Best Traning Set Accuracy:', gscv.best_score_)

best_lsvc = svm.LinearSVC(**gscv.best_params_)

print('Best Linear SVC')
print()

get_ipython().magic('time best_lsvc.fit(X_train, y_train)')
print()

print('Traning Set Accuracy:', best_lsvc.score(X_train, y_train))
print('Validation Accuracy:', best_lsvc.score(X_test, y_test))
print()

print(classification_report(y_test, best_lsvc.predict(X_test)))

from ptt_corpus_tokenizer import preprocess_and_tokenizie
best_lsvc.predict(vectorizer.transform([
    preprocess_and_tokenizie('[新聞] 爭風吃醋！國中生臉書約談判　持西瓜刀'),
    preprocess_and_tokenizie('[新聞] 「炫妻狂魔」黑人狂PO范范美照　引古詞讚'),
    preprocess_and_tokenizie('[新聞] 分手後驚覺懷孕！前女友轉生氣　怒告男：'),
]))

