import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

import os, sys
import numpy as np
import pandas as pd

# ---------------------------
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import GaussianNB
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.decomposition import FactorAnalysis
from scipy.stats import pearsonr

np.set_printoptions(suppress=True) # suppress scientific notation
np.random.seed(18937)

datasource = "datasets/HR_analytics.csv"
print(os.path.exists(datasource))

dataset = pd.read_csv(datasource).sample(frac = 1).reset_index(drop = True)
dataset.columns

del dataset["Unnamed: 0"]

dataset.head().transpose()

dataset["salary"].head()

np.expand_dims(dataset["salary"], 1)

encoder = LabelBinarizer()

salary_features = encoder.fit_transform(np.expand_dims(dataset["salary"], 1))

salary_features

for j, _class in enumerate(encoder.classes_):
    print(j, _class)    
    print('salary_{}'.format(_class.replace('\x20', '_'))) # this is building out the column names
    # whitespaces ( ) are replaces with underscores (_)
    print("===================")

for j, _class in enumerate(encoder.classes_):
    dataset.loc[:, 'salary_{}'.format(_class.replace('\x20', '_'))] = salary_features[:, j]
    # JACKY: loop through each enumeration of the classes (low, medium, high)
    # and set the corresponding salary column to either 0 or 1.
    
dataset.head().transpose()

dataset["sales"].head()

encoder = LabelBinarizer()
sales_features = encoder.fit_transform(np.expand_dims(dataset["sales"], 1))

for j, _class in enumerate(encoder.classes_):
    dataset.loc[:, "sales_{}".format(_class.replace("\x20", "_"))] = sales_features[:, j]
    
dataset.info()

dataset.head().transpose()

columns_to_keep = [0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
df = dataset.copy() # copy for later
df = df.iloc[:, columns_to_keep]
X = np.array(dataset.iloc[:, columns_to_keep])
y = np.array(dataset["left"])

list(df.columns)

print(X.shape)

print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

def mutual_info_session(X_train, y_train, k, df):
    selector = SelectKBest(mutual_info_classif, k)
    selector.fit(X_train, y_train)    
    print("Selected feature indices:", selector.get_support(True))
    print("")
    print("Selected feature column names: \n", np.array(df.columns[selector.get_support(True)]))
    model = GaussianNB()
    model.fit(selector.transform(X_train), y_train)
    chi2_sklearn, pvalue_sklearn = chi2(selector.transform(X_train), y_train)
    print("")
    print("Selected Model Score: ", model.score(selector.transform(X_test), y_test))
    print("")
    print("p-values:", pvalue_sklearn)
    
print("MUTUAL INFO METHOD: \n")
for i in range(1, 16):
    print("k =", i)
    mutual_info_session(X_train, y_train, i, df)
    print("*******************************************")

def chi_feature_session(X_train, y_train, k, df):
    selector = SelectKBest(chi2, k)
    selector.fit(X_train, y_train)
    print("Selected feature indices:", selector.get_support(True))
    print("")
    print("Selected feature column names: \n", np.array(df.columns[selector.get_support(True)]))
    model = GaussianNB()
    model.fit(selector.transform(X_train), y_train)
    chi2_sklearn, pvalue_sklearn = chi2(selector.transform(X_train), y_train)
    print("")
    print("Selected Model Score: ", model.score(selector.transform(X_test), y_test))
    print("")
    print("p-values:", pvalue_sklearn)

print("CHI SQUARE METHOD: \n")
for i in range(1, 16):
    print("k =", i)
    chi_feature_session(X_train, y_train, i, df)
    print("*******************************************")

class ForwardSelector(object):
    def __init__(self, estimator):
        self.estimator = estimator
        
    def fit(self, X, y, k):
        selected = np.zeros(X.shape[1]).astype(bool) # holds indicators of whether each feature is selected
        
        score = lambda X_features: clone(self.estimator).fit(X_features, y).score(X_features, y)
        # fit and score model based on some subset of features
        
        selected_indices = lambda: list(np.flatnonzero(selected))
        
        while np.sum(selected) < k: # keep looping until k features are selected
            rest_indices = list(np.flatnonzero(~selected)) # indices to unselected columns
            scores = list()
            
            for i in rest_indices:
                feature_subset = selected_indices() + [i]
                s = score(X[:, feature_subset])
                scores.append(s)
            idx_to_add = rest_indices[np.argmax(scores)]
            selected[idx_to_add] = True
        self.selected = selected.copy()
        return self
    
    def transform(self, X):
        return X[:, self.selected]
    
    def get_support(self, indices = False):
        return np.flatnonzero(self.selected) if indices else self.selected
    

def forward_selection_session(X_train, y_train, k, df):
    model = GaussianNB()
    selector = ForwardSelector(model)
    selector.fit(X_train, y_train, k)
    print("Selected feature indices:", selector.get_support(True))
    print("")
    print("Selected feature column names: \n", np.array(df.columns[selector.get_support(True)])) 
    model = GaussianNB()
    model.fit(selector.transform(X_train), y_train)
    chi2_sklearn, pvalue_sklearn = chi2(selector.transform(X_train), y_train)
    print("")
    print("Selected Model Score: ", model.score(selector.transform(X_test), y_test))
    print("")
    print("p-values:", pvalue_sklearn)

print("FORWARD SELECTION METHOD: \n")
for i in range(1, 16):
    print("k =", i)
    forward_selection_session(X_train, y_train, i, df)
    print("*******************************************")

# Add code below this comment
# ---------------------------
def Do_PCA(X, y, k):
    pca = PCA(n_components = k)
    pca.fit(X)
    print("PCA Variance Ratio:")
    print(pca.explained_variance_ratio_)
    print("")
    print("PCA Correlation Coefficient:")
    X_PCA = pca.transform(X)
    corr = np.array([pearsonr(X_PCA[:,i], y)[0] for i in range(X_PCA.shape[1])])
    print(corr)
    print("")
    X_train, X_test, y_train, y_test = train_test_split(X_PCA, y, test_size = 0.20)
    print("X Shape:", X.shape)
    print("X Train Shape:", X_train.shape)
    model = GaussianNB()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print("")
    print("PCA Model Score:", score)
    return pca

def Do_FA(X, y, k):
    fa = FactorAnalysis(n_components = k)
    fa.fit(X)
    fa.explained_variance_ = np.flip(np.sort(np.sum(fa.components_**2, axis=1)), axis=0)
    total_variance = np.sum(fa.explained_variance_) + np.sum(fa.noise_variance_)
    fa.explained_variance_ratio_ = fa.explained_variance_ / total_variance
    print("FA Variance Ratio:")
    print(fa.explained_variance_ratio_)
    print("")
    X_FA = fa.transform(X)
    print("FA Correlation Coefficient:")
    corr = np.array([pearsonr(X_FA[:, i], y)[0] for i in range(X_FA.shape[1])])
    print(corr)
    print("")
    X_train, X_test, y_train, y_test = train_test_split(X_FA, y, test_size = 0.20)
    print("X Shape:", X.shape)
    print("X Train Shape:", X_train.shape)
    model = GaussianNB()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)   
    print("")
    print("FA Model Score:", score)
    return fa

for i in range(1, 6):
    print("# of Components:", i)
    print("")
    Do_PCA(X, y, i)
    print("*******************************")

for i in range(1, 6):
    print("# of Components:", i)
    print("")
    Do_FA(X, y, i)
    print("*******************************")

# Add code below this comment
# ---------------------------

def screePCA(k):
    pca_scree = Do_PCA(X, y, k) 
    x_ticks = np.arange(len(pca_scree.components_)) + 1
    plt.xticks(x_ticks)
    plt.plot(x_ticks, pca_scree.explained_variance_)
    plt.show()
    return pca_scree
    
def screeFA(k):
    fa_scree = Do_FA(X, y, k)
    x_ticks = np.arange(len(fa_scree.components_)) + 1
    plt.xticks(x_ticks)
    plt.plot(x_ticks, fa_scree.explained_variance_)
    plt.show()
    return fa_scree
    
def screeBoth(k):
    pca_k = screePCA(k)
    print("==================================================")
    fa_k = screeFA(k)
    print("==================================================")
    x_ticks = np.arange(len(pca_k.components_)) + 1
    plt.xticks(x_ticks)
    plt.plot(x_ticks, np.log(pca_k.explained_variance_), "b") # PCA
    plt.plot(x_ticks, np.log(fa_k.explained_variance_), "r") # FA
    plt.show()
    print("Blue line is PCA. Red line is FA.")

screeBoth(5)



