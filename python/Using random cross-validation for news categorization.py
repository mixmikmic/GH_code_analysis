import News_Categorization_MNB as nc
nc.import_data()
cont = nc.count_data(nc.labels, nc.categories)

X = nc.titles
y = nc.categories

from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.pipeline import Pipeline 
from sklearn import metrics 
import numpy as np

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
Ntimes = 10

def k_fold(shuffle=False,stratified=False):
    f1s = [0] * Ntimes
    accus = [0] * Ntimes
    nlabels = len(nc.labels)
    
    if stratified:
        kf = StratifiedKFold(n_splits=Ntimes, shuffle=shuffle)
    else:
        kf = KFold(n_splits=Ntimes, shuffle=shuffle)

    k = 0
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        text_clf = Pipeline([('vect', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', MultinomialNB()),
                                 ])
        text_clf = text_clf.fit(X_train, y_train)
        predicted = text_clf.predict(X_test)

        accus[k] = metrics.accuracy_score(y_test, predicted)


        f1s[k] = metrics.f1_score(y_test, predicted, labels=nc.labels,
                                   average=None)        
        k+=1
    
    return (accus,f1s)

(accus1,f1s1) = k_fold(shuffle=False,stratified=False)
(accus2,f1s2) = k_fold(shuffle=False,stratified=True)
(accus3,f1s3) = k_fold(shuffle=True,stratified=False)
(accus4,f1s4) = k_fold(shuffle=True,stratified=True)

get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt

x = list(range(len(accus1)))
y0 = accus1
y1 = accus2
y2 = accus3
y3 = accus4

plt.plot(x,y0,color='cyan',marker='o')
plt.plot(x,y1,color='green',marker='s')
plt.plot(x,y2,color='blue',marker='*')
plt.plot(x,y3,color='red',marker='d')
plt.legend(["K-Fold NoShuffle", "Strat.K-Fold NoShuffle", "K-Fold Shuffle", 
            "Strat.K-Fold Shuffle"], loc='best')
plt.title("Accuracy by method")
plt.ylabel("Accuracy")
plt.xlabel("Repetition")
plt.grid(True)
plt.show()

import pandas as pd

lista = [nc.conf_interv1("Accuracy ",accus) for accus in [accus1,accus2,accus3,accus4]]
mean,half = zip(*lista)
df = pd.DataFrame([list(mean),list(half)]).transpose()
df.columns = ["mean accuracy","half range"]
df.index = ["K-Fold NoShuffle", "Strat.K-Fold NoShuffle", "K-Fold Shuffle", "Strat.K-Fold Shuffle"]
df

# F1-score
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(nrows=2, ncols=2, sharex='col', sharey='row')

def multiplot(ax,f1s,tit):
    matf1 = np.matrix(f1s)
    x = list(range(len(f1s)))
    y0 = list(matf1[:, 0].flat)
    y1 = list(matf1[:, 1].flat)
    y2 = list(matf1[:, 2].flat)
    y3 = list(matf1[:, 3].flat)

    fig.suptitle("F1-score results", fontsize=14)

    l1 = ax.plot(x,y0)
    l2 = ax.plot(x,y1)
    l3 = ax.plot(x,y2)
    l4 = ax.plot(x,y3)
    ax.legend(['b', 'e', 'm', 't'], loc='best')
    ax.set_title(tit)
    ax.grid(True)

multiplot(ax1,f1s1, "K-Fold NoShuffle")
multiplot(ax2,f1s2, "Strat.K-Fold NoShuffle")
multiplot(ax3,f1s3, "K-Fold Shuffle")
multiplot(ax4,f1s4, "Strat.K-Fold Shuffle")

plt.show()

lista = [np.mean(f1s, axis=0) for f1s in [f1s1,f1s2,f1s3,f1s4]]
mat = np.matrix(lista)
meanxcol = mat.mean(0)
mat2 = np.append(mat,meanxcol, axis=0)
meanxrow = mat2.mean(1)
mat3 = np.append(mat2,meanxrow, axis=1)
df = pd.DataFrame(mat3) 
df.columns = ['business', 'entertainment', 'health', 'technology','mean']
df.index = ["K-Fold NoShuffle", "Strat.K-Fold NoShuffle", "K-Fold Shuffle", 
            "Strat.K-Fold Shuffle",'mean']
df

