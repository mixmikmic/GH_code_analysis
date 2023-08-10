import pandas as pd
import numpy as np

with open("tweets_AAPL.csv",'rb') as dfile:
    dstr = str(dfile.read())[1:5]

get_ipython().run_cell_magic('time', '', 'with open("tweets_AAPL.csv",\'rb\') as dfile:\n    dstr = str(dfile.read())[2:]\ndarr = dstr.split(\'\\\\r\')\ndmat = [rs.split(",") for rs in darr]\ndmat = [r[:1] + [",".join(r[1:-5])] + r[-5:] for r in dmat]\ndmat = [r for r in dmat if len(r) == 7]\ndf = pd.DataFrame(dmat)\ndf.columns = ["datestr","twt","open","high","low","close","adjclose"]\ndf["twtarr"] = df["twt"].apply(lambda text: text.lower().split())')

from datetime import datetime

df = df[df["datestr"] != ""]
get_ipython().run_line_magic('time', 'df["date"] = df["datestr"].apply(     lambda date: datetime.strptime(date,"%m/%d/%y"))')
datedf = df[["date","open"]].groupby("date").aggregate(lambda gp: tuple(set(gp))[0])
pval = datedf.values.astype(float)
up = ((pval[1:] - pval[:-1]) > 0).astype(int)
datedf["openup"] = list(up.T[0]) + [0]

get_ipython().run_line_magic('time', 'df["openup"] = df["date"].apply(lambda date: price["openup"].loc[date])')

import gensim
get_ipython().run_line_magic('time', 'w2vM = gensim.models.Word2Vec(df["twtarr"])')

# number of samples to aggregate
Ns = int(1e6)

get_ipython().run_cell_magic('time', '', '# MEAN AGGREGATION\n# tvecs = np.array([np.array([w2vM[t] if t in w2vM\n#                                 else np.zeros((100,))\n#                             for t in twt]).mean(axis=0)\n#                  for twt in df["twtarr"][:Ns]])\n# tvecs = np.array([np.array([w2vM[t]\n#                             for t in twt\n#                             if t in w2vM]).mean(axis=0)\n#                  for twt in df["twt"][:Ns]])\n# SUM AGGREGATION\n# tvecs = np.array([np.array([w2vM[t] if t in w2vM\n#                                 else np.zeros((100,))\n#                             for t in twt]).sum(axis=0)\n#                  for twt in sentDf["SentimentText"][:Ns]])\n# MEAN AGGREGATION\n%time df["twtvecs"] = df["twtarr"][:Ns].apply(\\\n    lambda twt: np.array([w2vM[t] \\\n                         if t in w2vM else np.zeros((100,)) \\\n                         for t in twt]).mean(axis=0))')

print(df.shape)
get_ipython().run_line_magic('time', 'veclens = df["twtvecs"].apply(lambda vec: int(np.prod(vec.shape)))')
df_empty = df[veclens != 100]
get_ipython().run_line_magic('time', 'df = df[veclens == 100]')
print(df.shape)

dategps = df[["date","twtvecs"]].groupby("date")
datevecs = [np.mean(gp.values[:,1], axis=0) for k,gp in dategps]
datedf["datevecs"] = datevecs

import sklearn
from sklearn import ensemble,svm,neural_network,discriminant_analysis
from sklearn.metrics import roc_curve,auc,precision_recall_curve

from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

def roc_auc(clf,X,y):
    probs = clf.predict_proba(X)
    fpr, tpr, thresholds = roc_curve(y, probs[:, 1])
    roc_auc = auc(fpr, tpr)
    roc_str = 'ROC (AUC Gain = %0.2f)' % (roc_auc - 0.5)
    plt.plot(fpr, tpr, lw=1,label=roc_str)
    plt.plot([0,1],[0,1],label="RAN CLF")
    plt.title(roc_str)
    plt.show()

def prrc_auc(clf,X,y):
    probs = clf.predict_proba(X)
    pr, rc, thresholds = precision_recall_curve(y, probs[:, 1])
    roc_auc = auc(rc, pr)
    roc_str = 'Prec vs Recall (AUC Gain = %0.2f)' % (roc_auc - np.mean(y))
    plt.plot(rc,pr, lw=1,label=roc_str)
    plt.plot([0,1],[np.mean(y),np.mean(y)],label="RAN CLF")
    plt.axis([0,1,0,1])
    plt.title(roc_str)
    plt.show()

def evaluate(clf,X,y):
    yhat = clf.predict(X)
    accu = np.mean(yhat == y)
    prec = np.mean(y[yhat == 1])
    recl = np.mean(yhat[y == 1])
    f1 = 2 * prec * recl / (prec + recl)
    print("Accuracy",accu,"Precision",prec,"Recall",recl,"F1",f1)

# dvecs = np.array([vec.T for vec in avgvecs["vecs"].values])
X = np.array([vec.T for vec in datedf["datevecs"].values])
y = datedf["openup"].values
# inverse classifier
# y = 1 - y

# generate test/train split
ratio = 0.8
tidx = np.random.rand(X.shape[0]) < ratio
pidx = ~tidx

rf = sklearn.ensemble.RandomForestClassifier()
rf.max_depth = 5
clf = rf
get_ipython().run_line_magic('time', 'clf.fit(X[tidx],y[tidx])')

print("TEST")
get_ipython().run_line_magic('time', 'evaluate(clf,X[pidx],y[pidx])')

print("TRAIN")
get_ipython().run_line_magic('time', 'evaluate(clf,X[tidx],y[tidx])')

print("ROC AUC")
get_ipython().run_line_magic('time', 'roc_auc(clf,X[pidx],y[pidx])')

print("PRECISION/RECALL AUC")
get_ipython().run_line_magic('time', 'prrc_auc(clf,X[pidx],y[pidx])')

mlp = sklearn.neural_network.MLPClassifier()
clf = mlp
get_ipython().run_line_magic('time', 'clf.fit(X[tidx],y[tidx])')

print("TEST")
get_ipython().run_line_magic('time', 'evaluate(clf,X[pidx],y[pidx])')

print("TRAIN")
get_ipython().run_line_magic('time', 'evaluate(clf,X[tidx],y[tidx])')

print("ROC AUC")
get_ipython().run_line_magic('time', 'roc_auc(clf,X[pidx],y[pidx])')

print("PRECISION/RECALL AUC")
get_ipython().run_line_magic('time', 'prrc_auc(clf,X[pidx],y[pidx])')

qda = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
clf = qda
get_ipython().run_line_magic('time', 'clf.fit(X[tidx],y[tidx])')

print("TEST")
get_ipython().run_line_magic('time', 'evaluate(clf,X[pidx],y[pidx])')

print("TRAIN")
get_ipython().run_line_magic('time', 'evaluate(clf,X[tidx],y[tidx])')

print("ROC AUC")
get_ipython().run_line_magic('time', 'roc_auc(clf,X[pidx],y[pidx])')

print("PRECISION/RECALL AUC")
get_ipython().run_line_magic('time', 'prrc_auc(clf,X[pidx],y[pidx])')

# number of samples to train on
N = int(1e6)
X = np.array([x.T for x in df["twtvecs"][:N].values])
y = df["openup"][:N].values
# inverse classifier
# y = 1 - y

# generate test/train split
ratio = 0.8
tidx = np.random.rand(min(X.shape[0],N)) < ratio
pidx = ~tidx

rf = sklearn.ensemble.RandomForestClassifier()
rf.max_depth = 5
clf = rf
get_ipython().run_line_magic('time', 'clf.fit(X[tidx],y[tidx])')

print("TEST")
get_ipython().run_line_magic('time', 'evaluate(clf,X[pidx],y[pidx])')

print("TRAIN")
get_ipython().run_line_magic('time', 'evaluate(clf,X[tidx],y[tidx])')

print("ROC AUC")
get_ipython().run_line_magic('time', 'roc_auc(clf,X[pidx],y[pidx])')

print("PRECISION/RECALL AUC")
get_ipython().run_line_magic('time', 'prrc_auc(clf,X[pidx],y[pidx])')

mlp = sklearn.neural_network.MLPClassifier()
clf = mlp
get_ipython().run_line_magic('time', 'clf.fit(X[tidx],y[tidx])')

print("TEST")
get_ipython().run_line_magic('time', 'evaluate(clf,X[pidx],y[pidx])')

print("TRAIN")
get_ipython().run_line_magic('time', 'evaluate(clf,X[tidx],y[tidx])')

print("ROC AUC")
get_ipython().run_line_magic('time', 'roc_auc(clf,X[pidx],y[pidx])')

print("PRECISION/RECALL AUC")
get_ipython().run_line_magic('time', 'prrc_auc(clf,X[pidx],y[pidx])')

qda = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()
clf = qda
get_ipython().run_line_magic('time', 'clf.fit(X[tidx],y[tidx])')

print("TEST")
get_ipython().run_line_magic('time', 'evaluate(clf,X[pidx],y[pidx])')

print("TRAIN")
get_ipython().run_line_magic('time', 'evaluate(clf,X[tidx],y[tidx])')

print("ROC AUC")
get_ipython().run_line_magic('time', 'roc_auc(clf,X[pidx],y[pidx])')

print("PRECISION/RECALL AUC")
get_ipython().run_line_magic('time', 'prrc_auc(clf,X[pidx],y[pidx])')

