import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE

from sklearn.linear_model import RandomizedLasso

get_ipython().run_line_magic('matplotlib', 'inline')
cmap_bold = ListedColormap(['#00FF00','#FF0000'])

sys.path.append('../utils')

import DataAggregation as da
import AlgoUtils as au

algos_dd = {
    "LogisticRegression": {"C": 1e9},
    "LogisticRegressionB": {"C": 1e9, "class_weight":'balanced'},
    "KNeighborsClassifier": {"n_neighbors": 7},
    "LinearDiscriminantAnalysis": {},
    "QuadraticDiscriminantAnalysis": {},
    "SVC": {}
}

fcols = ["d_mean:d_std:d_max:l_range",
         "d_mean:d_std:l_range",
         "d_std:l_range",
         "l_range",
         "d_std",
         "d_max"]
algos_str = ["LogisticRegression", 
             "LogisticRegressionB", 
             "KNeighborsClassifier",
             "LinearDiscriminantAnalysis",
             "QuadraticDiscriminantAnalysis"]

a2 = da.GetFrames("../data/device_failure.csv", "a2")
a7 = da.GetFrames("../data/device_failure.csv", "a7")
a4 = da.GetFrames("../data/device_failure.csv", "a4", ldays=-30, lday_strict=False)
tdf = a2.df_sfeature.drop("failure", axis=1).join(a7.df_sfeature.drop("failure", axis=1)).join(a4.df_sfeature)

tdf.head()

algo_str = "QuadraticDiscriminantAnalysis"
scols = ["a2d_std", "a7d_std","a7l_range","a7d_mean", "a7d_max"]
analysisdf = au.do_clf_validate_new(tdf, algo_str,algos_dd[algo_str], scols, "failure")

algo_str = "QuadraticDiscriminantAnalysis"
scols = tdf.columns[:-1]
analysisdf = au.do_clf_validate_new(tdf, algo_str,algos_dd[algo_str], scols, "failure")



X = tdf[tdf.columns[:-1]]
y = tdf["failure"]
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_new = sel.fit_transform(X)
sel.get_support()
cnt = 0
allcols = tdf.columns[:-1]
for decision in sel.get_support():
    if decision == True:
        print allcols[cnt]
    cnt = cnt + 1

#Adding a2l_range, a4d_max and a4l_range based on above analysis
scols = ["a2l_range", "a2d_std", "a7d_std","a7l_range","a4d_max", "a4l_range"]
algo_str = "QuadraticDiscriminantAnalysis"
scols = ["a2l_range", "a2d_std", "a7d_std","a7l_range","a4d_max", "a4l_range"]
analysisdf = au.do_clf_validate_new(tdf, algo_str,algos_dd[algo_str], scols, "failure")



fcols = tdf.columns[:-1] 
X = tdf[fcols]
Y = tdf["failure"]
rlasso = RandomizedLasso(alpha=0.025)
rlasso.fit(X, Y)
 
print "Features sorted by their score:"
print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), 
                 fcols), reverse=True)



fcols = [x for x in tdf.columns[:-1]]
X = tdf[fcols]
Y = tdf["failure"]
clf = LogisticRegression()
rfe = RFE(clf, n_features_to_select=2)
rfe.fit(X,Y)
 
print "Features sorted by their rank:"
print sorted(zip(rfe.ranking_, fcols))

#a4d_mean: Need to add this
#a2l_range, a4l_range: Already added this based on Variance Threshold analysis
#a7l_range, a7d_max, a7d_std, a7d_mean: This is already in baseline
algo_str = "QuadraticDiscriminantAnalysis"
scols = ["a2l_range", "a2d_std", "a7d_std","a7l_range","a4d_max", "a4l_range", "a4d_mean"]
analysisdf = au.do_clf_validate_new(tdf, algo_str,algos_dd[algo_str], scols, "failure")



fcdf = pd.DataFrame(tdf.corr()["failure"])
fcdf.loc[:,"failure_abs"] = fcdf.failure.map(lambda x: abs(x))
fcdf.sort_values(by="failure_abs", ascending=False, inplace=True)
fcdf.drop(["failure_abs"], axis=1, inplace=True)
fcdf



tdf[tdf["failure"] == 0].corr()

tdf[tdf["failure"] == 1][fcols].corr()

#Based on: a2d_std Vs a2d_mean: +ve corr for good. -ve corr for bad
#Possibly no improvement because of lack of signal in this selection.
#May need to cross validate
algo_str = "QuadraticDiscriminantAnalysis"
scols = ["a2l_range", "a2d_std", "a2d_mean", "a7d_std","a7l_range","a4d_max", "a4l_range", "a4d_mean"]
analysisdf = au.do_clf_validate_new(tdf, algo_str,algos_dd[algo_str], scols, "failure")

algo_str = "QuadraticDiscriminantAnalysis"
scols = tdf.columns[:-1]
analysisdf = au.do_clf_validate_new(tdf, algo_str,algos_dd[algo_str], scols, "failure")



