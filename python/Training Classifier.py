import os
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
get_ipython().run_line_magic('matplotlib', 'inline')

all_train = pd.read_csv('train_test.csv', names=['img_name','1st_result','1st_score','2nd_result','2nd_score',
                                           '3rd_result','3rd_score','4th_result','4th_score',
                                           '5th_result','5th_score','segment','cls'])
train = pd.read_csv('training.csv', names=['img_name','1st_result','1st_score','2nd_result','2nd_score',
                                           '3rd_result','3rd_score','4th_result','4th_score',
                                           '5th_result','5th_score','segment','cls'])
test = pd.read_csv('testing.csv', names=['img_name','1st_result','1st_score','2nd_result','2nd_score',
                                         '3rd_result','3rd_score','4th_result','4th_score',
                                         '5th_result','5th_score','segment','cls'])
all_train.drop('img_name', axis=1, inplace=True)
test.drop('img_name', axis=1, inplace=True)
train.drop('img_name', axis=1, inplace=True)
train.head()

def preprocess(features): 
    features = features.drop('1st_score', axis=1)    
    features = features.drop('2nd_score', axis=1)
    features = features.drop('3rd_score', axis=1)    
    features = features.drop('4th_score', axis=1)
    features = features.drop('5th_score', axis=1)
    
    for c in features.columns:
        features[c]=features[c].fillna(-1)
        if features[c].dtype == 'object':
            lbl = LabelEncoder()
            lbl.fit(list(features[c].values))
            features[c] = lbl.transform(list(features[c].values))
    return features

all_train = preprocess(all_train)
train = preprocess(train)
test = preprocess(test)
train.head()

x_train = train.drop(['cls'], axis=1)
y_train = train['cls'].values
print('train:',x_train.shape, y_train.shape)

x_valid = test.drop(['cls'], axis=1)
y_valid = test['cls'].values
print('test:',x_valid.shape, y_valid.shape)

mean_all = np.mean(all_train['cls'].values)
mean_train = np.mean(y_train)
mean_valid = np.mean(y_valid)
print('correct:', mean_all,',',mean_train,',',mean_valid)

def xgb_run(params, x_train, x_valid, y_train, y_valid):
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
  
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds=100, verbose_eval=10)

    d_test = xgb.DMatrix(x_valid)
    y_xgb = clf.predict(d_test)

    sns.set(font_scale = 1.5)
    xgb.plot_importance(clf)
    
    predictions = [round(value) for value in y_xgb]
    mean_valid = np.mean(predictions)
    print('mean:', mean_valid)
    
    return clf

# reg:linear, reg:logistic, binary:logistic, rank:pairwise,reg:gamma,reg:tweedie,
# multi:softprob, multi:softmax, count:poisson, binary:logitraw
params = {
    'eta': 0.01,
    'objective': 'reg:linear',
    'eval_metric': 'error',
    'max_depth': 4,
    'silent': 1,
    'lambda':0.8,
    'subsample': 0.8,
    'alpha': 0.4,
    'gamma': 0,
}
clf = xgb_run(params, x_train, x_valid, y_train, y_valid)

def pca_fit(features):
    pca = PCA(n_components=2)
    return pca.fit_transform(features)

def tsne_fit(features):
    pca = PCA(n_components=4)
    tsne = TSNE(n_components=2)
    features = tsne.fit_transform(pca.fit_transform(features))
    return features

def plot(features, cls):
    plt.figure(figsize=(30,25))
    cmap = cm.rainbow(np.linspace(0.0, 1.0, 2))
    colors = cmap[cls]
        
    x = features[:, 0]
    y = features[:, 1]
    plt.scatter(x, y, c=colors)
    plt.show()

features = pca_fit(x_train)
plot(features, y_train)

del all_train, train, test, x_train, x_valid, y_train, y_valid, mean_train, mean_valid

train = pd.read_csv('train_test.csv', names=['img_name','1st_result','1st_score','2nd_result','2nd_score',
                                           '3rd_result','3rd_score','4th_result','4th_score',
                                           '5th_result','5th_score','segment','cls'])
train.drop('img_name', axis=1, inplace=True)

breeding_sites = [ 
                'pot, flowerpot', 
                'stupa, tope', 
                'water jug', 
                'water bottle', 
                'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin', 
                'greenhouse, nursery, glasshouse', 
                'milk can', 
                'barrel, cask', 
                'canoe', 
                'rain barrel', 
                'lakeside, lakeshore', 
                'Dutch oven' 
                ]

def select_rows(df, breeding_sites, threshold=0.1):
    df = df[df['1st_score'] >= threshold]
    print('after threshold:',len(df))
    temp = df.loc[df['1st_result'].isin(breeding_sites)]
    return temp

print('all_train:',train.shape)
train_selected = select_rows(train, breeding_sites, threshold=0.15)             
train_selected = preprocess(train_selected)

x_train = train_selected.drop(['cls'], axis=1)
y_train = train_selected['cls'].values
print('train:', x_train.shape, y_train.shape,'\n')

mean_train = np.mean(y_train)
print('inception correctness:', mean_train,'\n')

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=7)
print('valid:', x_valid.shape, y_valid.shape)

mean_train = np.mean(y_train)
mean_valid = np.mean(y_valid)
print('mean_train, mean_valid:', mean_train,',',mean_valid)

features = pca_fit(x_train)
plot(features, y_train)

# reg:linear, reg:logistic, binary:logistic, rank:pairwise,reg:gamma,reg:tweedie,
# multi:softprob, multi:softmax, count:poisson, binary:logitraw
params = {
    'eta': 0.01,
    'objective': 'reg:linear',
    'eval_metric': 'error',
    'max_depth': 6,
    'silent': 1,
    'lambda':0.8,
    'subsample': 0.8,
    'alpha': 0.4,
    'gamma': 0,
}
clf = xgb_run(params, x_train, x_valid, y_train, y_valid)

clf.save_model('xgb.model')

bst = xgb.Booster() 
bst.load_model('xgb.model')
bst.predict(xgb.DMatrix(x_train))

from sklearn.svm import SVC
from collections import Counter
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import time

start = time.time()

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft')

params = {'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200]}

grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
grid = grid.fit(x_train, y_train)
print('accuracy:',grid.score(x_valid, y_valid))
print('time:',time.time()-start,'seconds')

start = time.time()

dt = DecisionTreeClassifier(max_depth=6)
knn = KNeighborsClassifier(n_neighbors=7)
svm = SVC(kernel='rbf', probability=True)
eclf_2 = VotingClassifier(estimators=[('dt', dt), ('knn', knn), ('svc', svm)], voting='soft', weights=[2,1,2])

dt = dt.fit(x_train, y_train)
knn = knn.fit(x_train, y_train)
svm = svm.fit(x_train, y_train)
eclf_2 = eclf_2.fit(x_train, y_train)

print('accuracy:',eclf_2.score(x_valid, y_valid))
print('time:',time.time()-start,'seconds')

start = time.time()

rnd = RandomForestClassifier(random_state=1)
svm = SVC(kernel='linear')
dt = DecisionTreeClassifier(max_depth=6)
knn = KNeighborsClassifier(n_neighbors=7)

rnd.fit(x_train, y_train)
svm.fit(x_train, y_train)
dt.fit(x_train, y_train)
knn.fit(x_train, y_train)

print('Random Forest\'s accuracy:',rnd.score(x_valid, y_valid))
print('SVM\'s accuracy:',svm.score(x_valid, y_valid))
print('Decision Tree\'s accuracy:',dt.score(x_valid, y_valid))
print('Knn\'s accuracy:',dt.score(x_valid, y_valid))
print('time:',time.time()-start,'seconds')

