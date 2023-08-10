get_ipython().magic('matplotlib notebook')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn_pandas import DataFrameMapper

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# features_imputed_set1.csv
features = pd.read_csv('background_imputed_1.csv', low_memory=False, index_col='challengeID')
# features.drop('idnum', 1, inplace=True) # remove idnum column
print("Features shape: {}".format(features.shape))
features.sort_index(kind='mergesort', inplace=True) # mergesort is the only stable sort
features.head()

# some columns are not numeric; encode them
from sklearn.preprocessing import LabelEncoder
str_col_features = features.select_dtypes(include=['object'])
label_encoders={} # save the label encoders to allow for inverse transforms
for col_name in str_col_features.columns:
    le = LabelEncoder()
    dmy = le.fit(features[col_name])
    features[col_name] = le.transform(features[col_name])
    label_encoders[col_name] = le
features.head()

# scale data to unit variance and 0 mean
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
indices_f = features.index
colnames_f = features.columns
features = ss.fit_transform(features)
features = pd.DataFrame(data=features, index=indices_f, columns=colnames_f)
features.head()

from sklearn.model_selection import train_test_split

full_y = pd.read_csv('train.csv', index_col='challengeID')
train_y = full_y['gpa'] # we are specifically interested in GPA
print("Target shape: {}".format(train_y.shape))
train_y.dropna(how='any', inplace=True) # drop those with no reported GPA
print("Target shape (no na): {}".format(train_y.shape))

# get the rows for which we can predict
y_indices = train_y.index.values.tolist()
full_x = features
train_x = features.loc[y_indices]
print("Features shape (final): {}".format(train_x.shape))

train_x, test_x, train_y, test_y = train_test_split(
    train_x, train_y, test_size=0.2, random_state=0)

train_x.shape
train_x.head()
train_y.shape
train_y.head()
test_x.shape
test_x.head()
test_y.shape
test_y.head()

from sklearn.feature_selection import VarianceThreshold
# remove features with 0 variance

# mapper from pandas to numpy
mapper = DataFrameMapper([(train_x.columns, None)])
indices = train_x.index
colnames = train_x.columns

vt = VarianceThreshold()
train_vt_x = vt.fit_transform(mapper.fit_transform(train_x))
train_x = pd.DataFrame(data=train_vt_x, index=indices, columns=colnames[vt.get_support(True)])
train_x.head()

# Test set needs variance threshold treatment
# mapper from pandas to numpy
mapper_test = DataFrameMapper([(test_x.columns, None)])
indices = test_x.index
colnames = test_x.columns

test_vt_x = vt.transform(mapper.fit_transform(test_x))
test_x = pd.DataFrame(data=test_vt_x, index=indices, columns=colnames[vt.get_support(True)])
test_x.head()

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# mapper from pandas to numpy
mapper = DataFrameMapper([(train_x.columns, None)])
indices = train_x.index
colnames = train_x.columns

# Use Lasso Regression
fs_model_lasso = LassoCV().fit(mapper.fit_transform(train_x), train_y.as_matrix())
fs_lasso = SelectFromModel(fs_model_lasso, prefit=True, threshold=0.0005)
train_fs_lasso_x = fs_lasso.transform(train_x)
train_fs_lasso_x = pd.DataFrame(data=train_fs_lasso_x, index=indices, columns=colnames[fs_lasso.get_support(True)])
train_fs_lasso_x.head()

train_fs_x = fs_lasso.transform(train_x)
test_fs_x = fs_lasso.transform(test_x)

from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from sklearn.model_selection import cross_val_score

## DEFAULT COMMENTED OUT B/C THIS TAKES FOREVER TO RUN

# # adapted from http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_fa_model_selection.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-fa-model-selection-py
# n_components = [200, 300, 400]

# def compute_scores(X, y):
#     pca = PCA(svd_solver='full')
# #     fa = FactorAnalysis()

#     pca_scores, fa_scores = [], []
#     for n in n_components:
#         pca.n_components = n
# #         fa.n_components = n
#         pca_scores.append(np.mean(cross_val_score(pca, X, y)))
# #         fa_scores.append(np.mean(cross_val_score(fa, X)))
#         print("Finished for %d" % n)

#     return pca_scores #, fa_scores

# pca_scores = compute_scores(train_x, train_y) #, pca_scores
# n_components_pca = n_components[np.argmax(pca_scores)]
# # n_components_fa = n_components[np.argmax(fa_scores)]

# # pca = PCA(svd_solver='full', n_components='mle')
# # pca.fit(train_x)
# # n_components_pca_mle = pca.n_components_

# print("best n_components by PCA CV = %d" % n_components_pca)
# # print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
# # print("best n_components by PCA MLE = %d" % n_components_pca_mle)

# best number of factor components is 100 according to cv (versus 100,500,1000,5000)
# best number of PCA components is 400 according to cv (versus 10,25,50,100,200, 300, 400)
pca = PCA(svd_solver='full', n_components=100)
pca.fit(train_x, train_y)

pca_comps = np.array(pca.components_)
pca_comps.shape
variance_ratios = np.array(pca.explained_variance_ratio_) # seems like first 6 components have the most
variance_ratios
pca_comps_high_var = pca_comps[variance_ratios > 0.01] # get components with high variance
pca_comps_high_var.shape

for i in range(1,len(pca_comps_high_var)):
    comp = pca_comps_high_var[i]
    print("Investigating component {} with variance {}".format(i, variance_ratios[i]) )
    relevant_features = np.array(features.columns.values[((comp > 0.01) == True)]) # indices of top 10 features that lend more than 0.01 to the component's variance
    relevant_features.shape
    relevant_features

# transform data sets, looking for features with high variance ratios
train_reduced_x = pca.transform(train_x) #[:,variance_ratios > 0.002]
test_reduced_x = pca.transform(test_x) #[:,variance_ratios > 0.002]
train_reduced_x.shape

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error

# Test on
# Lasso - Linear Regression
# Lasso
# Random Forest/ Decision Tree
# SVR
# Kernel Ridge Regression?
# 

#clf - the classifier, params - params to CV, transform - to do any feature selection
classifiers = [
    {
        'clf': DecisionTreeRegressor(),
        'params': {
            "max_depth": [1000, None],
            "max_features": [None],
            "criterion": ["mse", "mae"]
        }
    },
    {
        'clf': RandomForestRegressor(),
        'params': {"max_depth": [1000, None],
                  "max_features": [None],
                  "criterion": ["mse", "mae"]}
    },
    {
        'clf': Lasso(),
        'params': {
            "alpha": [0.5, 1, 2, 4, 8],
            "fit_intercept": [True, False],
            "normalize": [True, False]
        }
    },
    {
        'clf': LinearRegression(),
        'params': {
            "fit_intercept": [True, False],
            "normalize": [True, False]
        }
        
    },
    {
        'clf': ElasticNet(),
        'params': {
            "alpha": [0.5, 1, 2, 4, 8],
            "l1_ratio": [0.1, 0.2, 0.4, 0.6, 0.8],
            "fit_intercept": [True, False],
            "normalize": [True, False],
            "positive": [True, False]
        }
    },
#     {
#         'clf': KernelRidge(),
#         'params': {
            
#         }
        
#     },
    {
        'clf': LinearSVR(),
        'params': {
            "C": [1, 2, 4]
        }
        
    }
]

# run grid search without feature selection
# keeping track of best classifier
best_mse = 100
best_clf = 0
for clf in classifiers: 
    model = clf['clf']
    print("~~~ Fitting and Testing {} ~~~".format(type(model).__name__))
    grid_search = GridSearchCV(model, param_grid=clf['params'])

    dmy=grid_search.fit(train_reduced_x, train_y)
    grid_search.get_params()

    clf = grid_search.best_estimator_
    dmy=clf.fit(train_reduced_x, train_y)
    pred_y = clf.predict(test_reduced_x)

    mse = mean_squared_error(test_y, pred_y)
    "----MSE: {}----".format(mse)
    if (mse < best_mse):
        best_mse = mse
        best_clf = clf

