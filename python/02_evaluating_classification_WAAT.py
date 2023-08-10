import io
import requests
import time # for timestamps
import pickle # save models

import numpy as np
import pandas as pd
from ast import literal_eval # parsing hp after tuner

from cls_tuning import * # my helper functions

from sklearn.metrics import recall_score, classification_report, confusion_matrix

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

# fix random seed for reproducibility
seed = 2302
np.random.seed(seed)

path = 'https://raw.githubusercontent.com/laufergall/ML_Speaker_Characteristics/master/data/generated_data/'

url = path + "feats_ratings_scores_train.csv"
s = requests.get(url).content
feats_ratings_scores_train = pd.read_csv(io.StringIO(s.decode('utf-8')))

url = path + "feats_ratings_scores_test.csv"
s = requests.get(url).content
feats_ratings_scores_test = pd.read_csv(io.StringIO(s.decode('utf-8')))

with open(r'..\data\generated_data\feats_names.txt') as f:
    feats_names = f.readlines()
feats_names = [x.strip().strip('\'') for x in feats_names] 

with open(r'..\data\generated_data\items_names.txt') as f:
    items_names = f.readlines()
items_names = [x.strip().strip('\'') for x in items_names] 

with open(r'..\data\generated_data\traits_names.txt') as f:
    traits_names = f.readlines()
traits_names = [x.strip().strip('\'') for x in traits_names] 

# extracting scores

feats_ratings_scores_all = feats_ratings_scores_train.append(feats_ratings_scores_test)

scores = feats_ratings_scores_all.groupby(['spkID','speaker_gender']).mean()[traits_names]
scores.reset_index(inplace=True)

# pairplot of the 5 traits

myfig = sns.pairplot(scores.drop('spkID', axis=1), hue='speaker_gender', hue_order=['male','female'])
filename = r'\pairplot_traits_allspeakers.png'
# myfig.savefig(r'.\figures' + filename, bbox_inches = 'tight')  


# # same pairplot with numbers instead of labels for traits and larger label
# scores2 = scores.rename(index=str, columns={"warmth": "1", "attractiveness": "2", "compliance": "4", "confidence": "3", "maturity": "5"})
# cols = scores2.columns.tolist()
# cols = cols[0:4] + [cols[5]] + [cols[4]] + [cols[6]]
# scores2 = scores2[cols]

# plt.rcParams["axes.labelsize"] = 30
# plt.rcParams["legend.fontsize"] = 30

# myfig = sns.pairplot(scores2.drop('spkID', axis=1), hue='speaker_gender', hue_order=['male','female'])
 
# filename = r'\pairplot_traits_allspeakers2.png'
# # myfig.savefig(r'.\figures' + filename, bbox_inches = 'tight')  

# scatter plot

sns.lmplot('warmth', 'attractiveness', data = scores, hue="speaker_gender") 

# histogram, kernel density estimation
sns.jointplot('warmth', 'attractiveness', data = scores, kind="kde").set_axis_labels("warmth", "attractiveness")

# applying k-means

n_clusters=3

kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(scores[[traits_names[0],traits_names[1]]])

scores['class'] = pd.Categorical(kmeans.labels_).rename_categories(['high','low','mid'])

print(scores['class'].value_counts())   

myfig = sns.lmplot(traits_names[0],traits_names[1], data = scores, hue="class", hue_order = ['low','mid','high'], fit_reg=False, aspect=1)
filename = r'\kmeans3_WAAT_allspeakers.png'
# myfig.savefig(r'.\figures' + filename, bbox_inches = 'tight')  

# remove speakers in the mid class

scores_class = scores.loc[ scores['class'] != 'mid', ['spkID','speaker_gender','class']]

scores_class['class'] = pd.Categorical(scores['class'], categories=['low','high'])

print(scores_class.head())

print(scores_class.groupby(['speaker_gender','class']).count())

# get stratified random partition for train and test

scores_class['genderclass'] = scores_class[['speaker_gender', 'class']].apply(lambda x: ''.join(x), axis=1)

indexes = np.arange(0,len(scores))
classes = scores_class['class']
train_i, test_i, train_y, test_y = train_test_split(indexes, 
                                                    classes, 
                                                    test_size=0.25, 
                                                    stratify = scores_class['genderclass'], 
                                                    random_state=2302)

scores_class_train = scores_class.iloc[train_i,:] 
scores_class_test = scores_class.iloc[test_i,:] 

print(scores_class_train['genderclass'].value_counts())
print(scores_class_test['genderclass'].value_counts())

# save these data for other evaluations
# scores_class.iloc[:,0:3].to_csv(r'..\data\generated_data\speakerIDs_cls_WAAT_all.csv', index=False)
# scores_class_train.iloc[:,0:3].to_csv(r'..\data\generated_data\speakerIDs_cls_WAAT_train.csv', index=False)
# scores_class_test.iloc[:,0:3].to_csv(r'..\data\generated_data\speakerIDs_cls_WAAT_test.csv', index=False)

# Number of speakers in Train: 137
# Number of speakers in Test: 46
# Number of w-high speakers in Train: 43
# Number of m-high speakers in Train: 34
# Number of w-low speakers in Train: 34
# Number of m-low speakers in Train: 26
# Number of w-high speakers in Test: 15
# Number of m-high speakers in Test: 12
# Number of w-low speakers in Test: 11
# Number of m-low speakers in Test: 8

# merge features and scores train/test

feats_class_train = feats_ratings_scores_all.merge(scores_class_train) 
feats_class_test = feats_ratings_scores_all.merge(scores_class_test) 

# Standardize speech features  

dropcolumns = ['name','spkID','speaker_gender','class','genderclass'] + items_names + traits_names

# learn transformation on training data
scaler = StandardScaler()
scaler.fit(feats_class_train.drop(dropcolumns, axis=1))

# numpy n_instances x n_feats
feats_s_train = scaler.transform(feats_class_train.drop(dropcolumns, axis=1))
feats_s_test = scaler.transform(feats_class_test.drop(dropcolumns, axis=1)) 

# training data. Features and labels
X = feats_s_train
y = feats_class_train['class'].cat.codes

# test data. Features and labels
Xt = feats_s_test
yt = feats_class_test['class'].cat.codes

# split train data into 80% and 20% subsets - with balance in trait and gender
# give subset A to the inner hyperparameter tuner
# and hold out subset B for meta-evaluation
AX, BX, Ay, By = train_test_split(X, y, test_size=0.20, stratify = feats_class_train['genderclass'], random_state=2302)

print('Number of instances in A (hyperparameter tuning):',AX.shape[0])
print('Number of instances in B (meta-evaluation):',BX.shape[0])
    

# dataframe with results from hp tuner to be appended
tuning_all = pd.DataFrame()

# list with tuned classifiers trained on training data, to be appended
trained_all = []

# label: str to keep track of the different runs in the filename
label=''

# save splits

# original features and class
feats_class_train.to_csv(r'.\data_while_tuning\feats_class_train.csv', index=False)
feats_class_test.to_csv(r'.\data_while_tuning\feats_class_test.csv', index=False)

# train/test partitions, features and labels
np.save(r'.\data_while_tuning\X.npy', X)
np.save(r'.\data_while_tuning\y.npy', y)
np.save(r'.\data_while_tuning\Xt.npy', Xt)
np.save(r'.\data_while_tuning\yt.npy', yt)

# # A/B splits, features and labels
np.save(r'.\data_while_tuning\AX.npy', AX)
np.save(r'.\data_while_tuning\BX.npy', BX)
np.save(r'.\data_while_tuning\Ay.npy', Ay)
np.save(r'.\data_while_tuning\By.npy', By)


# original features and class
feats_class_train = pd.read_csv(r'.\data_while_tuning\feats_class_train.csv')
feats_class_test = pd.read_csv(r'.\data_while_tuning\feats_class_test.csv')
feats_names = pd.read_csv(r'.\data_while_tuning\feats_names.csv', header=None)
feats_names = feats_names.values.tolist()

# train/test partitions, features and labels
X = np.load(r'.\data_while_tuning\X.npy')
y = np.load(r'.\data_while_tuning\y.npy')
Xt = np.load(r'.\data_while_tuning\Xt.npy')
yt = np.load(r'.\data_while_tuning\yt.npy')

# A/B splits, features and labels
AX = np.load(r'.\data_while_tuning\AX.npy')
BX = np.load(r'.\data_while_tuning\BX.npy')
Ay = np.load(r'.\data_while_tuning\Ay.npy')
By = np.load(r'.\data_while_tuning\By.npy')

# label: str to keep track of the different runs in the filename
label=''

# Loading outpus of hp tuning from disk
tuning_all, trained_all = load_tuning(label)

# save tuning_all (.csv) and trained_all (nameclassifier.sav)
save_tuning(tuning_all, trained_all, label)

from sklearn.naive_bayes import GaussianNB

"""
Naive Bayes Classifier
"""
def get_GaussianNB2tune():

    model = GaussianNB()
    hp = dict()
    return 'GaussianNB', model, hp

# Hyperparameter tuning with this model
tuning, trained = hp_tuner(AX, BX, Ay, By, [get_GaussianNB2tune], feats_names)

# update lists of tuning info and trained classifiers
tuning_all = tuning_all.append(tuning, ignore_index=True)
trained_all.append(trained)

# open generated file with results of fitting search
 
sgrid = pd.read_csv(r'.\data_while_tuning\GaussianNB_tuning.csv')
print(sgrid['params'].head())

# params to dataframe
params_dict = sgrid['params'].apply(lambda x: literal_eval(x) ).to_dict()
params_df = pd.DataFrame(data = params_dict).transpose()

# plot acc vs. k
sns.pointplot(x='selecter__k', y='mean_acc_A', data=sgrid.join(params_df)) 

from sklearn.linear_model import LogisticRegression

"""
Logistic Regression
"""
def get_LogisticRegression2tune():

    model = LogisticRegression()
    hp = dict(
        #classifier__penalty = ['l1','l2'],
        classifier__C = np.logspace(-3,3,num=7)
    )
    return 'LogisticRegression', model, hp

# Hyperparameter tuning with this model
tuning, trained = hp_tuner(AX, BX, Ay, By, [get_LogisticRegression2tune], feats_names)

# update lists of tuning info and trained classifiers
tuning_all = tuning_all.append(tuning, ignore_index=True)
trained_all.append(trained)

# open generated file with results of fitting search
 
sgrid = pd.read_csv(r'.\data_while_tuning\LogisticRegression_tuning.csv')

# params to dataframe
params_dict = sgrid['params'].apply(lambda x: literal_eval(x) ).to_dict()
params_df = pd.DataFrame(data = params_dict).transpose()

# plot acc vs. params
sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__C', data=sgrid.join(params_df)) 

from sklearn.neighbors import KNeighborsClassifier

"""
K Nearest Neighbors
"""
def get_KNeighborsClassifier2tune():

    model = KNeighborsClassifier()
    hp = dict(
        classifier__n_neighbors = list(range(1,40))
    )
    return 'KNeighborsClassifier', model, hp

# Hyperparameter tuning with this model
tuning, trained = hp_tuner(AX, BX, Ay, By, [get_KNeighborsClassifier2tune], feats_names)

# update lists of tuning info and trained classifiers
tuning_all = tuning_all.append(tuning, ignore_index=True)
trained_all.append(trained)

# open generated file with results of fitting search
 
sgrid = pd.read_csv(r'.\data_while_tuning\KNeighborsClassifier_tuning.csv')

# params to dataframe
params_dict = sgrid['params'].apply(lambda x: literal_eval(x) ).to_dict()
params_df = pd.DataFrame(data = params_dict).transpose()

# plot acc vs. params
params_df = params_df.loc[params_df['classifier__n_neighbors']<10,:] # selecting only lower k
sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__n_neighbors', data=sgrid.join(params_df)) 

from sklearn.svm import SVC

"""
Support Vector Machines
"""
def get_SVClinear2tune():
    
    model = SVC()
    hp = dict(
        classifier__C = np.logspace(-5,3,num=9),
        classifier__kernel = ['linear']
    )
    return 'SVClinear', model, hp

def get_SVCpoly2tune():
    
    model = SVC()
    hp = dict(
        classifier__C = np.logspace(-4,2,num=7),
        classifier__kernel = ['poly'],
        classifier__degree = [2,3,4,5], 
        classifier__gamma = ['auto']#,np.logspace(-5,2,num=8)]
    )
    return 'SVCpoly', model, hp

def get_SVCrbf2tune():
    
    model = SVC()
    hp = dict(
        classifier__C = np.logspace(-5,5,num=20),
        classifier__kernel = ['rbf'],
        classifier__gamma = np.logspace(-5,5,num=20)
    )
    return 'SVCrbf', model, hp

def get_SVCsigmoid2tune():
    
    model = SVC()
    hp = dict(
        classifier__C = np.logspace(-5,3,num=9),
        classifier__kernel = ['sigmoid'],
        classifier__gamma = np.logspace(-5,3,num=9)
    )
    return 'SVCsigmoid', model, hp

# Hyperparameter tuning with SVM with different kernels
tuning, trained = hp_tuner(AX, BX, Ay, By, [get_SVClinear2tune], feats_names)

# update lists of tuning info and trained classifiers
tuning_all = tuning_all.append(tuning, ignore_index=True)
trained_all.append(trained)

# tune with poly kernel
tuning, trained = hp_tuner(AX, BX, Ay, By, [get_SVCpoly2tune], feats_names)

# update lists of tuning info and trained classifiers
tuning_all = tuning_all.append(tuning, ignore_index=True)
trained_all.append(trained)

# save tuning_all (.csv) and trained_all (nameclassifier.sav)
save_tuning(tuning_all, trained_all)

# tune with rbf kernel
tuning, trained = hp_tuner(AX, BX, Ay, By, [get_SVCrbf2tune], feats_names)

# update lists of tuning info and trained classifiers
tuning_all = tuning_all.append(tuning, ignore_index=True)
trained_all.append(trained)

# save tuning_all (.csv) and trained_all (nameclassifier.sav)
save_tuning(tuning_all, trained_all)

# tune with sigmoid kernel
tuning, trained = hp_tuner(AX, BX, Ay, By, [get_SVCsigmoid2tune], feats_names)

# update lists of tuning info and trained classifiers
tuning_all = tuning_all.append(tuning, ignore_index=True)
trained_all.append(trained)

# save tuning_all (.csv) and trained_all (nameclassifier.sav)
save_tuning(tuning_all, trained_all)

# Linear kernel: open generated file with results of fitting search
 
sgrid = pd.read_csv(r'.\data_while_tuning\SVClinear_tuning.csv')

# params to dataframe
params_dict = sgrid['params'].apply(lambda x: literal_eval(x) ).to_dict()
params_df = pd.DataFrame(data = params_dict).transpose()

# plot acc vs. params
sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__C', data=sgrid.join(params_df)) 

# Poly kernel: open generated file with results of fitting search
 
sgrid = pd.read_csv(r'.\data_while_tuning\SVCpoly_tuning.csv')

# params to dataframe
params_dict = sgrid['params'].apply(lambda x: literal_eval(x) ).to_dict()
params_df = pd.DataFrame(data = params_dict).transpose()

# plot acc vs. params
sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__C', data=sgrid.join(params_df))

# plot acc vs. params
sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__degree', data=sgrid.join(params_df))

# rbf kernel: open generated file with results of fitting search
 
sgrid = pd.read_csv(r'.\data_while_tuning\SVCrbf_tuning.csv')

# params to dataframe
params_dict = sgrid['params'].apply(lambda x: literal_eval(x) ).to_dict()
params_df = pd.DataFrame(data = params_dict).transpose()

# plot acc vs. params
sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__C', data=sgrid.join(params_df))


# plot acc vs. params
sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__gamma', data=sgrid.join(params_df))

# sigmoid kernel: open generated file with results of fitting GridSearchCV
 
sgrid = pd.read_csv(r'.\data_while_tuning\SVCsigmoid_tuning.csv')

# params to dataframe
params_dict = sgrid['params'].apply(lambda x: literal_eval(x) ).to_dict()
params_df = pd.DataFrame(data = params_dict).transpose()

# plot acc vs. params
sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__C', data=sgrid.join(params_df))

from sklearn.svm import SVC

def get_SVCrbf2finetune():
    
    model = SVC()
    hp = dict(
        classifier__C = np.arange(1,100,5),
        classifier__kernel = ['rbf'],
        classifier__gamma = np.logspace(-5,-3,num=3)
    )
    return 'SVCrbf', model, hp

k_gridsearch = np.arange(50, 88, 1)


tuning, trained = hp_tuner(AX, BX, Ay, By, 
                               [get_SVCrbf2finetune], 
                               feats_names, 
                               k_gridsearch,
                               'random',
                               n_iter=100
                              )

# update lists of tuning info and trained classifiers
tuning_all = tuning_all.append(tuning, ignore_index=True)
trained_all.append(trained)

# save tuning_all (.csv) and trained_all (nameclassifier.sav)
save_tuning(tuning_all, trained_all)

from sklearn.tree import DecisionTreeClassifier

"""
Decision Trees
"""
def get_DecisionTreeClassifier2tune():
    
    model = DecisionTreeClassifier(random_state=2302)
    hp = dict(
        classifier__max_depth = np.arange(30,50),
        classifier__min_samples_leaf=np.arange(1,4)
    )
    return 'DecisionTreeClassifier', model, hp

k_gridsearch = np.arange(2, 89, 5)

# Hyperparameter tuning with this model
tuning, trained = hp_tuner(AX, BX, Ay, By, 
                               [get_DecisionTreeClassifier2tune], 
                               feats_names, 
                               k_gridsearch,
                               'grid',
                               n_iter=10
                              )

# update lists of tuning info and trained classifiers
tuning_all = tuning_all.append(tuning, ignore_index=True)
trained_all.append(trained)

# open generated file with results of fitting GridSearchCV
 
sgrid = pd.read_csv(r'.\data_while_tuning\DecisionTreeClassifier_tuning.csv')

# params to dataframe
params_dict = sgrid['params'].apply(lambda x: literal_eval(x) ).to_dict()
params_df = pd.DataFrame(data = params_dict).transpose()

# plot acc vs. params
sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__min_samples_leaf', data=sgrid.join(params_df))

sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__max_depth', data=sgrid.join(params_df))

from sklearn.ensemble import RandomForestClassifier

"""
Random Forest
"""
def get_RandomForestClassifier2tune():
    
    model = RandomForestClassifier(random_state=2302, max_features = None)
    hp = dict(
        classifier__n_estimators = np.arange(2,10)
    )
    return 'RandomForestClassifier', model, hp

k_gridsearch = np.arange(2, 89)

# Hyperparameter tuning with this model
tuning, trained = hp_tuner(AX, BX, Ay, By, 
                               [get_RandomForestClassifier2tune], 
                               feats_names, 
                               k_gridsearch,
                               'grid',
                               n_iter=10
                              )

# open generated file with results of fitting GridSearchCV
 
sgrid = pd.read_csv(r'.\data_while_tuning\RandomForestClassifier_tuning.csv')

# params to dataframe
params_dict = sgrid['params'].apply(lambda x: literal_eval(x) ).to_dict()
params_df = pd.DataFrame(data = params_dict).transpose()

# plot acc vs. params
fig, ax = plt.subplots(figsize=(15,15))
sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__n_estimators', data=sgrid.join(params_df))

from sklearn.ensemble import RandomForestClassifier

"""
Random Forest
"""
def get_RandomForestClassifier2tune():
    
    model = RandomForestClassifier(random_state=2302, max_features = None)
    hp = dict(
        classifier__n_estimators = np.arange(8,20)
    )
    return 'RandomForestClassifier', model, hp

k_gridsearch = np.arange(30, 89)

# Hyperparameter tuning with this model
tuning, trained = hp_tuner(AX, BX, Ay, By, 
                               [get_RandomForestClassifier2tune], 
                               feats_names, 
                               k_gridsearch,
                               'grid',
                               n_iter=10
                              )

# open generated file with results of fitting GridSearchCV
 
sgrid = pd.read_csv(r'.\data_while_tuning\RandomForestClassifier_tuning.csv')

# params to dataframe
params_dict = sgrid['params'].apply(lambda x: literal_eval(x) ).to_dict()
params_df = pd.DataFrame(data = params_dict).transpose()

# plot acc vs. params
fig, ax = plt.subplots(figsize=(15,15))
sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__n_estimators', data=sgrid.join(params_df))

from sklearn.ensemble import RandomForestClassifier

"""
Random Forest
"""
def get_RandomForestClassifier2tune():
    
    model = RandomForestClassifier(random_state=2302, max_features = None)
    hp = dict(
        classifier__n_estimators = np.arange(27,35)
    )
    return 'RandomForestClassifier', model, hp

k_gridsearch = np.arange(70, 89)

# Hyperparameter tuning with this model
tuning, trained = hp_tuner(AX, BX, Ay, By, 
                               [get_RandomForestClassifier2tune], 
                               feats_names, 
                               k_gridsearch,
                               'random',
                               n_iter=50
                              )

# open generated file with results of fitting GridSearchCV
 
sgrid = pd.read_csv(r'.\data_while_tuning\RandomForestClassifier_tuning.csv')

# params to dataframe
params_dict = sgrid['params'].apply(lambda x: literal_eval(x) ).to_dict()
params_df = pd.DataFrame(data = params_dict).transpose()

# plot acc vs. params
fig, ax = plt.subplots(figsize=(15,15))
sns.pointplot(x='selecter__k', y='mean_acc_A',hue='classifier__n_estimators', data=sgrid.join(params_df))

from sklearn.dummy import DummyClassifier

model = DummyClassifier(strategy='uniform')
model.fit(AX, Ay)
By_pred = model.predict(BX)
score_on_B = recall_score(By, By_pred, average='macro')
d = {
    'classifiers_names': ['DummyClassifier'],
    'best_accs': score_on_B,
    'best_hps': '',
    'sel_feats': '',
    'sel_feats_i': ''
    }

tuning = pd.DataFrame(data = d)
trained = model.fit(X, y)

# update lists of tuning info and trained regressors
tuning_all = tuning_all.append(tuning, ignore_index=True)
trained_all.append([trained])

# save tuning_all (.csv) and trained_all (nameregressor.sav)
save_tuning(tuning_all, trained_all, label)

# original features and class
feats_class_train = pd.read_csv(r'.\data_while_tuning\feats_class_train.csv')
feats_class_test = pd.read_csv(r'.\data_while_tuning\feats_class_test.csv')
feats_names = pd.read_csv(r'.\data_while_tuning\feats_names.csv', header=None)
feats_names = feats_names.values.tolist()

# train/test partitions, features and labels
X = np.load(r'.\data_while_tuning\X.npy')
y = np.load(r'.\data_while_tuning\y.npy')
Xt = np.load(r'.\data_while_tuning\Xt.npy')
yt = np.load(r'.\data_while_tuning\yt.npy')

# Loading outpus of hp tuning from disk
label=''
tuning_all, trained_all = load_tuning(label)
tuning_all

# select the classifier that gave the maximum acc on B set
best_accs = tuning_all['best_accs']
i_best = best_accs.idxmax()

print('Selected classifier based on the best performance on B: %r (accB = %0.2f)' % (tuning_all.loc[i_best,'classifiers_names'], round(best_accs[i_best],2)))

def mj_preds(yt, yt_pred):
    """
    Majority voting of class predictions of instances, grouped by speakers

    Input:
    - yt: true class for all instances
    - yt_pred: predicted class for all instances

    Output:
    - yt: true class per speaker
    - yt_pred_spk: predicted class per speaker
    - yt_pred_spk_conf: conf measure for the prediction per speaker, ranging from 0 to 1
    """
    # new df with true and predicted classes
    test_scores = pd.DataFrame(data = feats_class_test[['spkID','class']])
    test_scores['yt'] = yt
    test_scores['yt_pred'] = yt_pred

    # group by speakers and compute prediction confidence based on the distance from 0.5
    test_scores_spk = test_scores.groupby('spkID').mean()
    conf = abs(0.5-test_scores_spk)/0.5
    yt_pred_spk_conf = conf['yt_pred']

    # compute scores after 'majority voting'
    yt_spk = test_scores_spk.round().astype(int)['yt']
    yt_pred_spk = test_scores_spk.round().astype(int)['yt_pred']

    return yt_spk, yt_pred_spk, yt_pred_spk_conf



def plot_WAAT_preds(yt_spk, yt_pred_spk, yt_pred_spk_conf, cls, avg_pc_acc):
    """
    Given the true and predicted class per speaker and confidence, plot the WAAT space
    color coded by correct/incorrect predictions, with point shape indicating the confidence
    of the prediction.

    Plot saved to disk.

    Input:
    - yt: true class per speaker
    - yt_pred_spk: predicted class per speaker
    - yt_pred_spk_conf: conf measure for the prediction per speaker, ranging from 0 to 1
    - cls: name of classifier
    - avg_pc_acc: average per-class accuracy to show on figure title
    """

    is_correct = yt_pred_spk==yt_spk

    d = {
        'yt_spk': yt_spk,
        'yt_pred_spk': yt_pred_spk,
        'yt_pred_spk_conf': yt_pred_spk_conf,
        'is_correct': is_correct
        }

    df = pd.DataFrame(data = d)
    df.reset_index(inplace=True)

    # scores of speakers of the test set
    df = df.merge(scores[['spkID','warmth','attractiveness']])

    plt.style.use(['default'])
    alpha = 0.5

    plt.figure()

    plt.scatter(df.loc[df['is_correct']==True,'warmth'], df.loc[df['is_correct']==True,'attractiveness'],
                marker="o", alpha=alpha, s=10+100*df.loc[df['is_correct']==True,'yt_pred_spk_conf'], label='correct')

    plt.scatter(df.loc[df['is_correct']==False,'warmth'], df.loc[df['is_correct']==False,'attractiveness'],
                marker="o", alpha=1, s=10+100*df.loc[df['is_correct']==False,'yt_pred_spk_conf'], label='not correct', color="red")

    plt.xlabel('warmth')
    plt.ylabel('attractiveness')
    plt.title('Binary WAAT classification with '+ cls + ', average per-class accuracy=' + str(round(avg_pc_acc,2)))
    plt.legend()

    # save plot
    filename = r'\cls_WAAT_binary_predictions_'+cls+'.png'
    plt.savefig(r'.\figures' + filename, bbox_inches = 'tight')

# go through performace on dev and on test for all classifiers


# removing duplicates from tuning_all (same classifier tuned twice with different searchers)
indexes = tuning_all['classifiers_names'].drop_duplicates(keep='last').index.values

# dataframe for summary of performances
performances = pd.DataFrame(tuning_all.loc[indexes,['classifiers_names','best_accs']])


for i in indexes:

    yt_pred = trained_all[i][0].predict(Xt)

    # score per instance
    score_on_test = recall_score(yt, yt_pred, average='macro')

    # score per speaker
    yt_spk, yt_pred_spk, yt_pred_spk_conf = mj_preds(yt, yt_pred)
    score_on_test_spk = recall_score(yt_spk, yt_pred_spk, average='macro')

    print("%r -> (per instance) Average per-class accuracy on B: %.2f" % (tuning_all.loc[i,'classifiers_names'], tuning_all.loc[i,'best_accs'])) 
    print("%r -> (per instance) Average per-class accuracy on test: %.2f" % (tuning_all.loc[i,'classifiers_names'], score_on_test)) 
    print("%r -> (per speaker) Average per-class accuracy on test: %.2f" % (tuning_all.loc[i,'classifiers_names'], score_on_test_spk)) 

    performances.loc[i,'score_on_test']=score_on_test
    performances.loc[i,'score_on_test_spk']=score_on_test_spk   
                                           
    cm = confusion_matrix(yt_spk, yt_pred_spk)
    print(cm)
    print("")
    
    # generate plot of the WAAT space with the confidence of the predictions
    plot_WAAT_preds(yt_spk, yt_pred_spk, yt_pred_spk_conf, tuning_all.loc[i,'classifiers_names'], score_on_test_spk)

# pointplot performance

performances = performances.rename(index=str, columns={
    'classifiers_names':'classifiers',
    'best_accs':'on B set (per instance)',
    'score_on_test':'on test set (per instance)',
    'score_on_test_spk':'on test set (per speaker)'
})

# sort by 'on test set (per speaker)' column
performances_sorted = performances.sort_values(['on test set (per speaker)']).reset_index()

performances_melt = pd.melt(performances_sorted, id_vars=['classifiers'], value_vars=['on B set (per instance)',
                                                                               'on test set (per instance)',
                                                                               'on test set (per speaker)'])
performances_melt.rename(columns={
    'value':'average per-class accuracy',
    'variable': 'evaluation'
}, inplace=True)

fig, ax = plt.subplots(figsize=[10, 6])
myfig=sns.pointplot(x='classifiers', y='average per-class accuracy', hue='evaluation', 
                    data=performances_melt[performances_melt['evaluation'].isin(['on B set (per instance)',
                                                                               'on test set (per instance)'])]);
        
# rotate x axis and limit y axis
for item in myfig.get_xticklabels():
    item.set_rotation(90)
    
# save plot
filename = r'\cls_WAAT_binary_performance.png'
plt.savefig(r'.\figures' + filename, bbox_inches = 'tight')

