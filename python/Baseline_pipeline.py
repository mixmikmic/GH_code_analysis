get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 7.0)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from Bio import SeqIO
from nltk import bigrams
from nltk import trigrams


from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn import preprocessing
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import dask.dataframe as dd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from scipy import interp
from itertools import cycle

"""
Writing a method to build tri-grams
input: fasta_file
output: a csv type file of tri-gram values
"""


def build_tri_grams(input_file, output_file):
    tri_dic = defaultdict(int)

    out_handle = open(output_file, "w")
    for rec in SeqIO.parse(input_file, "fasta"):
        tok = rec.description.split("|")[-1]
        #print tok
        #print tok[0], tok[-1]
        #if tok == 'toxin':
        tri_tokens = trigrams(rec.seq)

        # tri_tokens is a generator - you can only go through it once
        for item in ((tri_tokens)):
            if '-' in item:
                continue
            tri_str = item[0] + item[1] + item[2]
            #print bi_str
            tri_dic[tri_str] += 1

        for index, item in enumerate(sorted(all_tri_grams)):
            if index > 0:
                out_handle.write(',')
            out_handle.write("%s" % tri_dic[item])
        out_handle.write("\n")
        tri_dic.update({}.fromkeys(tri_dic, 0)) # setting all key values to be zero again


    out_handle.close()

build_tri_grams('less_than_30_pos_neg_bacteriocin.fa', 'less_than_30_pos_neg_bacteriocin_trigrams')

final_bac_data = pd.read_csv("less_than_30_pos_neg_bacteriocin_trigrams", names=sorted(all_tri_grams))

# taking the values from panda into numpy array
final_bac_array = final_bac_data.values
print final_bac_array.shape

y = np.vstack((np.ones((346, 1)), np.zeros((346,1))))

print final_bac_array.shape
print y.shape

"""
Compressing the test data
"""
x_test_truncated_compressed = tsvd.transform(x_test)
print x_test_truncated_compressed.shape

# display a 2D plot of the digit classes in the latent space
# plt.cm.get_cmap("brg", 3)
#import matplotlib
#x_test_encoded = encoder.predict(x_test)
#print x_test_encoded.shape
#colors = ['red','green','blue','purple']
X_reduced = TruncatedSVD(n_components=200, random_state=0).fit_transform(final_bac_array)
X_embedded = TSNE(n_components=2, perplexity=30, verbose=2).fit_transform(X_reduced)
vis_x = X_embedded[:, 0]
vis_y = X_embedded[:, 1]

print tsvd.explained_variance_ratio_.sum()

print X_reduced.shape

"""
Manual nested cross validation and ROC curve from that

10-fold-------------------------------------------


"""

# ROC curve stuff
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
colors = cycle(['cyan', 'crimson', 'seagreen', 'yellow', 'blue', 'darkorange', 'darkviolet', 'fuchsia',
               'deepskyblue', 'lightcoral'])
lw = 2
i = 0



# Dividing data set for outer cv
cv_outer = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 43)

precision_scores = []
recall_scores = []
f1_scores = []


# Outer cv
for (train, test), color in zip(cv_outer.split(X_reduced, y), colors):
    pipe_svc = Pipeline([('clf', SVC(random_state=1, probability = True))])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']}, {'clf__C': param_range, 
             'clf__gamma': param_range, 'clf__kernel': ['rbf']}]

    
    #Inner cv
    cv_inner = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='f1', cv= cv_inner)
    
    gs.fit(X_reduced[train], y[train])
    
    print gs.best_params_
    
    scores = gs.best_estimator_.predict(X_reduced[test])
    precision_s = precision_score(y[test], scores)
    recall_s = recall_score(y[test], scores)
    f1_s = f1_score(y[test], scores)
    #print scores
    
    precision_scores.append(precision_s)
    recall_scores.append(recall_s)
    f1_scores.append(f1_s)
    print "Precision", precision_s
    print "Recall:", recall_s
    print "F1:", f1_s
    
    probas_ = gs.best_estimator_.predict_proba(X_reduced[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    i += 1
    #break
    
print "Precision mean and std:", np.mean(precision_scores), np.std(precision_scores) 
print "Recall mean and std:", np.mean(recall_scores), np.std(recall_scores)
print "F1 meand and std:", np.mean(f1_scores), np.std(f1_scores)

# Draw ROC curves
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= cv_outer.get_n_splits(X_reduced, y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

"""
10 fold Manual nested cross validation 50 times
"""
colors = cycle(['cyan', 'crimson', 'seagreen', 'yellow', 'blue', 'darkorange', 'darkviolet', 'fuchsia',
               'deepskyblue', 'lightcoral'])

outer_random_seed_list = [3,13,23,33,43,53,63,73,83,93, 103,113,123,133,143,153,163,173,183,193,
                         203,213,223,233,243,253,263,273,283,293, 303,313,323,333,343,353,363,373,383,393,
                          403,413,423,433,443,453,463,473,483,493]
random_seed_list = [2,12,22,32,42,52,62,72,82,92, 102,112,122,132,142,152,162,172,182,192,
                   202,212,222,232,242,252,262,272,282,292, 302,312,322,332,342,352,362,372,382,392,
                   402,412,422,432,442,452,462,472,482,492]

precision_scores_mean_list = []
recall_scores_mean_list = []
f1_scores_mean_list = []

precision_scores_std_list = []
recall_scores_std_list = []
f1_scores_std_list = []

# Outer cv
for index, rand_seed_i in enumerate(random_seed_list):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    cv_outer = StratifiedKFold(n_splits = 10, shuffle = True, random_state = outer_random_seed_list[index])
    for (train, test), color in zip(cv_outer.split(X_reduced, y), colors):
        pipe_svc = Pipeline([('clf', SVC(random_state=1, probability = True))])
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [{'clf__C': param_range, 'clf__kernel': ['linear']}, {'clf__C': param_range, 
             'clf__gamma': param_range, 'clf__kernel': ['rbf']}]


        #Inner cv
        cv_inner = StratifiedKFold(n_splits = 5, shuffle = True, random_state = rand_seed_i)
        gs = GridSearchCV(estimator=pipe_svc, param_grid=param_grid, scoring='f1', cv= cv_inner)

        gs.fit(X_reduced[train], y[train])

        scores = gs.best_estimator_.predict(X_reduced[test])
        precision_s = precision_score(y[test], scores)
        recall_s = recall_score(y[test], scores)
        f1_s = f1_score(y[test], scores)
        #print scores

        precision_scores.append(precision_s)
        recall_scores.append(recall_s)
        f1_scores.append(f1_s)
        
    print 'Loop:', index
    print 'Precision:', np.mean(precision_scores), np.std(precision_scores) 
    print 'Recall:', np.mean(recall_scores), np.std(recall_scores)
    print 'F1:', np.mean(f1_scores), np.std(f1_scores)
    
    precision_scores_mean_list.append(np.mean(precision_scores))
    recall_scores_mean_list.append(np.mean(recall_scores))
    f1_scores_mean_list.append(np.mean(f1_scores))
    
    precision_scores_std_list.append(np.std(precision_scores))
    recall_scores_std_list.append(np.std(recall_scores))
    f1_scores_std_list.append(np.std(f1_scores))
    
print 'Precision for 10 times:', np.mean(precision_scores_mean_list), np.std(precision_scores_std_list)
print 'Recall for 10 times:', np.mean(recall_scores_mean_list), np.std(recall_scores_std_list)
print 'F1 for 10 times:', np.mean(f1_scores_mean_list), np.std(f1_scores_std_list)

import numpy as np
from scipy import stats
precision_scores_mean_list = []
recall_scores_mean_list = []
f1_scores_mean_list = []
with open('temp.txt', 'r') as in_handle:
    for line in in_handle:
        if line.split()[0] == 'Precision:':
            precision_scores_mean_list.append(float(line.split()[1]))
        if line.split()[0] == 'Recall:':
            recall_scores_mean_list.append(float(line.split()[1]))
        if line.split()[0] == 'F1:':
            f1_scores_mean_list.append(float(line.split()[1]))
            
print 'Precision mean for 50 times:', np.mean(precision_scores_mean_list), 'Std. error:', stats.sem(precision_scores_mean_list)
print 'Recall mean for 50 times:', np.mean(recall_scores_mean_list), 'Std. error:', stats.sem(recall_scores_mean_list)
print 'F1 mean for 50 times:', np.mean(f1_scores_mean_list), 'Std. error:', stats.sem(f1_scores_mean_list)
            

"""
Manual nested cross validation and ROC curve from that

10-fold-------------------------------------------


"""

# ROC curve stuff
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
colors = cycle(['cyan', 'crimson', 'seagreen', 'yellow', 'blue', 'darkorange', 'darkviolet', 'fuchsia',
               'deepskyblue', 'lightcoral'])
lw = 2
i = 0



# Dividing data set for outer cv
cv_outer = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 43)

precision_scores = []
recall_scores = []
f1_scores = []


# Outer cv
for (train, test), color in zip(cv_outer.split(X_reduced, y), colors):
    pipe_logis = Pipeline([('logis', LogisticRegression(random_state = 1))])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'logis__C': param_range, 'logis__penalty': ['l1']}, 
             {'logis__C': param_range, 'logis__penalty': ['l2']}]

    
    #Inner cv
    cv_inner = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    gs = GridSearchCV(estimator=pipe_logis, param_grid=param_grid, scoring='f1', cv= cv_inner)
    
    gs.fit(X_reduced[train], y[train])
    
    print gs.best_params_
    
    scores = gs.best_estimator_.predict(X_reduced[test])
    precision_s = precision_score(y[test], scores)
    recall_s = recall_score(y[test], scores)
    f1_s = f1_score(y[test], scores)
    #print scores
    
    precision_scores.append(precision_s)
    recall_scores.append(recall_s)
    f1_scores.append(f1_s)
    print "Precision", precision_s
    print "Recall:", recall_s
    print "F1:", f1_s
    
    probas_ = gs.best_estimator_.predict_proba(X_reduced[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    i += 1
    #break
    
print "Precision mean and std:", np.mean(precision_scores), np.std(precision_scores) 
print "Recall mean and std:", np.mean(recall_scores), np.std(recall_scores)
print "F1 mean and std:", np.mean(f1_scores), np.std(f1_scores)

# Draw ROC curves
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= cv_outer.get_n_splits(X_reduced, y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

"""
10 fold Manual nested cross validation 50 times
"""
colors = cycle(['cyan', 'crimson', 'seagreen', 'yellow', 'blue', 'darkorange', 'darkviolet', 'fuchsia',
               'deepskyblue', 'lightcoral'])

outer_random_seed_list = [3,13,23,33,43,53,63,73,83,93, 103,113,123,133,143,153,163,173,183,193,
                         203,213,223,233,243,253,263,273,283,293, 303,313,323,333,343,353,363,373,383,393,
                          403,413,423,433,443,453,463,473,483,493]
random_seed_list = [2,12,22,32,42,52,62,72,82,92, 102,112,122,132,142,152,162,172,182,192,
                   202,212,222,232,242,252,262,272,282,292, 302,312,322,332,342,352,362,372,382,392,
                   402,412,422,432,442,452,462,472,482,492]

precision_scores_mean_list = []
recall_scores_mean_list = []
f1_scores_mean_list = []

precision_scores_std_list = []
recall_scores_std_list = []
f1_scores_std_list = []

# Outer cv
for index, rand_seed_i in enumerate(random_seed_list):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    cv_outer = StratifiedKFold(n_splits = 10, shuffle = True, random_state = outer_random_seed_list[index])
    for (train, test), color in zip(cv_outer.split(X_reduced, y), colors):
        pipe_logis = Pipeline([('logis', LogisticRegression(random_state = 1))])
        param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [{'logis__C': param_range, 'logis__penalty': ['l1']}, 
                 {'logis__C': param_range, 'logis__penalty': ['l2']}]


        #Inner cv
        cv_inner = StratifiedKFold(n_splits = 5, shuffle = True, random_state = rand_seed_i)
        gs = GridSearchCV(estimator=pipe_logis, param_grid=param_grid, scoring='f1', cv= cv_inner)

        gs.fit(X_reduced[train], y[train])

        scores = gs.best_estimator_.predict(X_reduced[test])
        precision_s = precision_score(y[test], scores)
        recall_s = recall_score(y[test], scores)
        f1_s = f1_score(y[test], scores)
        #print scores

        precision_scores.append(precision_s)
        recall_scores.append(recall_s)
        f1_scores.append(f1_s)
        
    print 'Loop:', index
    print 'Precision:', np.mean(precision_scores), np.std(precision_scores) 
    print 'Recall:', np.mean(recall_scores), np.std(recall_scores)
    print 'F1:', np.mean(f1_scores), np.std(f1_scores)
    
    precision_scores_mean_list.append(np.mean(precision_scores))
    recall_scores_mean_list.append(np.mean(recall_scores))
    f1_scores_mean_list.append(np.mean(f1_scores))
    
    precision_scores_std_list.append(np.std(precision_scores))
    recall_scores_std_list.append(np.std(recall_scores))
    f1_scores_std_list.append(np.std(f1_scores))
    
print 'Precision for 10 times:', np.mean(precision_scores_mean_list), np.std(precision_scores_std_list)
print 'Recall for 10 times:', np.mean(recall_scores_mean_list), np.std(recall_scores_std_list)
print 'F1 for 10 times:', np.mean(f1_scores_mean_list), np.std(f1_scores_std_list)

precision_scores_mean_list = [0.86773643527273292, 0.87253825063405155, 0.85983623651329177, 0.85579045180207414, 0.86452404896590951, 0.86996682755303445, 0.88730606055202821, 0.87425662974131568, 0.85787800237397016, 0.84561718966993971, 0.84924435034555668, 0.87031691974902292, 0.84515918269806944, 0.88228603777797332, 0.88498762698359479, 0.84695279845667026, 0.89360630826000842, 0.8810832791829496, 0.85621310878603618, 0.86779130475999544, 0.88767902971347967, 0.83866637893570195, 0.86778397022299458, 0.90411463032381612, 0.87971981313893077, 0.86321676326111807, 0.85421198687375155, 0.88802156250812436, 0.83971810130360058, 0.86679882962141031, 0.86480925707474943, 0.87621274873236121, 0.86198912809240069, 0.86803579868934444, 0.85367604402864961, 0.87575865226727301, 0.86842176901671009, 0.84446162418648196, 0.8333475295090963, 0.83943483142244746, 0.86798939631800209, 0.8615392040361215, 0.85043582291747322, 0.85935204435204438, 0.88725504628374652, 0.83464424054802255, 0.85044361448618599, 0.85429330614594523, 0.85453704032651401, 0.87790658942081268]
recall_scores_mean_list = [0.84134453781512608, 0.82084033613445373, 0.84689075630252086, 0.82126050420168073, 0.83151260504201685, 0.84411764705882353, 0.82991596638655474, 0.84630252100840342, 0.84705882352941175, 0.84168067226890764, 0.86100840336134454, 0.82966386554621852, 0.83243697478991607, 0.81999999999999995, 0.81798319327731106, 0.84689075630252098, 0.81756302521008395, 0.82890756302521018, 0.84621848739495797, 0.81554621848739506, 0.81504201680672261, 0.83218487394957974, 0.82294117647058818, 0.82386554621848751, 0.81252100840336128, 0.82084033613445373, 0.85596638655462187, 0.8322689075630253, 0.85521008403361343, 0.84705882352941164, 0.83210084033613452, 0.82974789915966396, 0.84647058823529409, 0.8373949579831933, 0.83243697478991607, 0.80605042016806716, 0.83815126050420174, 0.87025210084033622, 0.83453781512605052, 0.87831932773109234, 0.84647058823529409, 0.83420168067226896, 0.84689075630252098, 0.83285714285714296, 0.82672268907563018, 0.85823529411764699, 0.84378151260504208, 0.8360504201680673, 0.85798319327731087, 0.85848739495798321]
f1_scores_mean_list = [0.8495970029571065, 0.84299825565560926, 0.85091209408664648, 0.83469013704845363, 0.84251250521592769, 0.85365976152300116, 0.85392253109037775, 0.85350210500699342, 0.85041584981295648, 0.83863221819701561, 0.85327297708459471, 0.84634931848437545, 0.83378137472115754, 0.84558133623704923, 0.84598157228247128, 0.84453030790129091, 0.85254869828239388, 0.85114196201051961, 0.84633337755335525, 0.83908501571724481, 0.84883240618054534, 0.83241502016364122, 0.84103352406333765, 0.85768870512224082, 0.84183672759346673, 0.83549543991065889, 0.85142197826863908, 0.85592180205506074, 0.84305864321556379, 0.85548884336194675, 0.84593356465483982, 0.8480154431473389, 0.85049869307158799, 0.84759454286591596, 0.83941824420105315, 0.83807928126957987, 0.85087583453571458, 0.85383033574440059, 0.82891928856674046, 0.85489772504238337, 0.85250737443058466, 0.84430822102460323, 0.84354726132585367, 0.84298744873147824, 0.8512042200541885, 0.84357297646098428, 0.84337859040762608, 0.83788192082147861, 0.85230022251850457, 0.8647281372444573]

print 'Precision mean for 50 times:', np.mean(precision_scores_mean_list), 'Std. error:', stats.sem(precision_scores_mean_list)
print 'Recall mean for 50 times:', np.mean(recall_scores_mean_list), 'Std. error:', stats.sem(recall_scores_mean_list)
print 'F1 mean for 50 times:', np.mean(f1_scores_mean_list), 'Std. error:', stats.sem(f1_scores_mean_list)

print precision_scores_mean_list
print recall_scores_mean_list
print f1_scores_mean_list

"""
Manual nested cross validation and ROC curve from that

10-fold-------------------------------------------


"""

# ROC curve stuff
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
colors = cycle(['cyan', 'crimson', 'seagreen', 'yellow', 'blue', 'darkorange', 'darkviolet', 'fuchsia',
               'deepskyblue', 'lightcoral'])
lw = 2
i = 0



# Dividing data set for outer cv
cv_outer = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 43)

precision_scores = []
recall_scores = []
f1_scores = []


# Outer cv
for (train, test), color in zip(cv_outer.split(X_reduced, y), colors):
    pipe_dect = Pipeline([('dect', DecisionTreeClassifier(random_state=1))])
    #param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'dect__criterion': ['gini', 'entropy'], 'dect__max_depth': [3,4,5,6,7,8,9,10,None]},
             {'dect__criterion': ['gini', 'entropy'], 'dect__max_depth': [3,4,5,6,7,8,9,10,None],
             'dect__class_weight': ['balanced']}]

    
    #Inner cv
    cv_inner = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    gs = GridSearchCV(estimator=pipe_dect, param_grid=param_grid, scoring='f1', cv= cv_inner)
    
    gs.fit(X_reduced[train], y[train])
    
    print gs.best_params_
    
    scores = gs.best_estimator_.predict(X_reduced[test])
    precision_s = precision_score(y[test], scores)
    recall_s = recall_score(y[test], scores)
    f1_s = f1_score(y[test], scores)
    #print scores
    
    precision_scores.append(precision_s)
    recall_scores.append(recall_s)
    f1_scores.append(f1_s)
    print "Precision", precision_s
    print "Recall:", recall_s
    print "F1:", f1_s
    
    probas_ = gs.best_estimator_.predict_proba(X_reduced[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    i += 1
    #break
    
print "Precision mean and std:", np.mean(precision_scores), np.std(precision_scores) 
print "Recall mean and std:", np.mean(recall_scores), np.std(recall_scores)
print "F1 mean and std:", np.mean(f1_scores), np.std(f1_scores)

# Draw ROC curves
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= cv_outer.get_n_splits(X_reduced, y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

"""
10 fold Manual nested cross validation 50 times
"""
colors = cycle(['cyan', 'crimson', 'seagreen', 'yellow', 'blue', 'darkorange', 'darkviolet', 'fuchsia',
               'deepskyblue', 'lightcoral'])

outer_random_seed_list = [3,13,23,33,43,53,63,73,83,93, 103,113,123,133,143,153,163,173,183,193,
                         203,213,223,233,243,253,263,273,283,293, 303,313,323,333,343,353,363,373,383,393,
                          403,413,423,433,443,453,463,473,483,493]
random_seed_list = [2,12,22,32,42,52,62,72,82,92, 102,112,122,132,142,152,162,172,182,192,
                   202,212,222,232,242,252,262,272,282,292, 302,312,322,332,342,352,362,372,382,392,
                   402,412,422,432,442,452,462,472,482,492]

precision_scores_mean_list = []
recall_scores_mean_list = []
f1_scores_mean_list = []

precision_scores_std_list = []
recall_scores_std_list = []
f1_scores_std_list = []

# Outer cv
for index, rand_seed_i in enumerate(random_seed_list):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    cv_outer = StratifiedKFold(n_splits = 10, shuffle = True, random_state = outer_random_seed_list[index])
    for (train, test), color in zip(cv_outer.split(X_reduced, y), colors):
        pipe_dect = Pipeline([('dect', DecisionTreeClassifier(random_state=1))])
        #param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [{'dect__criterion': ['gini', 'entropy'], 'dect__max_depth': [3,4,5,6,7,8,9,10,None]},
             {'dect__criterion': ['gini', 'entropy'], 'dect__max_depth': [3,4,5,6,7,8,9,10,None],
             'dect__class_weight': ['balanced']}]


        #Inner cv
        cv_inner = StratifiedKFold(n_splits = 5, shuffle = True, random_state = rand_seed_i)
        gs = GridSearchCV(estimator=pipe_dect, param_grid=param_grid, scoring='f1', cv= cv_inner)

        gs.fit(X_reduced[train], y[train])

        scores = gs.best_estimator_.predict(X_reduced[test])
        precision_s = precision_score(y[test], scores)
        recall_s = recall_score(y[test], scores)
        f1_s = f1_score(y[test], scores)
        #print scores

        precision_scores.append(precision_s)
        recall_scores.append(recall_s)
        f1_scores.append(f1_s)
        
    print 'Loop:', index
    print 'Precision:', np.mean(precision_scores), np.std(precision_scores) 
    print 'Recall:', np.mean(recall_scores), np.std(recall_scores)
    print 'F1:', np.mean(f1_scores), np.std(f1_scores)
    
    precision_scores_mean_list.append(np.mean(precision_scores))
    recall_scores_mean_list.append(np.mean(recall_scores))
    f1_scores_mean_list.append(np.mean(f1_scores))
    
    precision_scores_std_list.append(np.std(precision_scores))
    recall_scores_std_list.append(np.std(recall_scores))
    f1_scores_std_list.append(np.std(f1_scores))
    
print 'Precision for 50 times:', np.mean(precision_scores_mean_list), np.std(precision_scores_std_list)
print 'Recall for 50 times:', np.mean(recall_scores_mean_list), np.std(recall_scores_std_list)
print 'F1 for 50 times:', np.mean(f1_scores_mean_list), np.std(f1_scores_std_list)

precision_scores_mean_list = []
recall_scores_mean_list = []
f1_scores_mean_list = []
with open('dec_tree_temp.txt', 'r') as in_handle:
    for line in in_handle:
        if line.split()[0] == 'Precision:':
            precision_scores_mean_list.append(float(line.split()[1]))
        if line.split()[0] == 'Recall:':
            recall_scores_mean_list.append(float(line.split()[1]))
        if line.split()[0] == 'F1:':
            f1_scores_mean_list.append(float(line.split()[1]))
            
print 'Precision mean for 50 times:', np.mean(precision_scores_mean_list), 'Std. error:', stats.sem(precision_scores_mean_list)
print 'Recall mean for 50 times:', np.mean(recall_scores_mean_list), 'Std. error:', stats.sem(recall_scores_mean_list)
print 'F1 mean for 50 times:', np.mean(f1_scores_mean_list), 'Std. error:', stats.sem(f1_scores_mean_list)
     

"""
Manual nested cross validation and ROC curve from that

10-fold-------------------------------------------


"""

# ROC curve stuff
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
colors = cycle(['cyan', 'crimson', 'seagreen', 'yellow', 'blue', 'darkorange', 'darkviolet', 'fuchsia',
               'deepskyblue', 'lightcoral'])
lw = 2
i = 0



# Dividing data set for outer cv
cv_outer = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 43)

precision_scores = []
recall_scores = []
f1_scores = []


# Outer cv
for (train, test), color in zip(cv_outer.split(X_reduced, y), colors):
    pipe_randf = Pipeline([('randf', RandomForestClassifier(random_state=1))])
    #param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [{'randf__n_estimators': [10, 13, 15, 17, 20], 'randf__criterion': ['gini', 'entropy'], 
               'randf__max_depth': [3,4,5,6,7,8,9,10,None]},
             {'randf__n_estimators': [10, 13, 15, 17, 20], 'randf__criterion': ['gini', 'entropy'], 
              'randf__max_depth': [3,4,5,6,7,8,9,10,None], 'randf__class_weight': ['balanced']}]

    
    #Inner cv
    cv_inner = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
    gs = GridSearchCV(estimator=pipe_randf, param_grid=param_grid, scoring='f1', cv= cv_inner)
    
    gs.fit(X_reduced[train], y[train])
    
    print gs.best_params_
    
    scores = gs.best_estimator_.predict(X_reduced[test])
    precision_s = precision_score(y[test], scores)
    recall_s = recall_score(y[test], scores)
    f1_s = f1_score(y[test], scores)
    #print scores
    
    precision_scores.append(precision_s)
    recall_scores.append(recall_s)
    f1_scores.append(f1_s)
    print "Precision", precision_s
    print "Recall:", recall_s
    print "F1:", f1_s
    
    probas_ = gs.best_estimator_.predict_proba(X_reduced[test])
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=lw, color=color,
             label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
    i += 1
    #break
    
print "Precision mean and std:", np.mean(precision_scores), np.std(precision_scores) 
print "Recall mean and std:", np.mean(recall_scores), np.std(recall_scores)
print "F1 mean and std:", np.mean(f1_scores), np.std(f1_scores)

# Draw ROC curves
plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
         label='Luck')

mean_tpr /= cv_outer.get_n_splits(X_reduced, y)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
         label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

"""
10 fold Manual nested cross validation 50 times
"""
colors = cycle(['cyan', 'crimson', 'seagreen', 'yellow', 'blue', 'darkorange', 'darkviolet', 'fuchsia',
               'deepskyblue', 'lightcoral'])

outer_random_seed_list = [3,13,23,33,43,53,63,73,83,93, 103,113,123,133,143,153,163,173,183,193,
                         203,213,223,233,243,253,263,273,283,293, 303,313,323,333,343,353,363,373,383,393,
                          403,413,423,433,443,453,463,473,483,493]
random_seed_list = [2,12,22,32,42,52,62,72,82,92, 102,112,122,132,142,152,162,172,182,192,
                   202,212,222,232,242,252,262,272,282,292, 302,312,322,332,342,352,362,372,382,392,
                   402,412,422,432,442,452,462,472,482,492]

precision_scores_mean_list = []
recall_scores_mean_list = []
f1_scores_mean_list = []

precision_scores_std_list = []
recall_scores_std_list = []
f1_scores_std_list = []

# Outer cv
for index, rand_seed_i in enumerate(random_seed_list):
    precision_scores = []
    recall_scores = []
    f1_scores = []
    cv_outer = StratifiedKFold(n_splits = 10, shuffle = True, random_state = outer_random_seed_list[index])
    for (train, test), color in zip(cv_outer.split(X_reduced, y), colors):
        pipe_randf = Pipeline([('randf', RandomForestClassifier(random_state=1))])
        #param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        param_grid = [{'randf__n_estimators': [10, 13, 15, 17, 20], 'randf__criterion': ['gini', 'entropy'], 
               'randf__max_depth': [3,4,5,6,7,8,9,10,None]},
             {'randf__n_estimators': [10, 13, 15, 17, 20], 'randf__criterion': ['gini', 'entropy'], 
              'randf__max_depth': [3,4,5,6,7,8,9,10,None], 'randf__class_weight': ['balanced']}]


        #Inner cv
        cv_inner = StratifiedKFold(n_splits = 5, shuffle = True, random_state = rand_seed_i)
        gs = GridSearchCV(estimator=pipe_randf, param_grid=param_grid, scoring='f1', cv= cv_inner)

        gs.fit(X_reduced[train], y[train])

        scores = gs.best_estimator_.predict(X_reduced[test])
        precision_s = precision_score(y[test], scores)
        recall_s = recall_score(y[test], scores)
        f1_s = f1_score(y[test], scores)
        #print scores

        precision_scores.append(precision_s)
        recall_scores.append(recall_s)
        f1_scores.append(f1_s)
        
    print 'Loop:', index
    print 'Precision:', np.mean(precision_scores), np.std(precision_scores) 
    print 'Recall:', np.mean(recall_scores), np.std(recall_scores)
    print 'F1:', np.mean(f1_scores), np.std(f1_scores)
    
    precision_scores_mean_list.append(np.mean(precision_scores))
    recall_scores_mean_list.append(np.mean(recall_scores))
    f1_scores_mean_list.append(np.mean(f1_scores))
    
    precision_scores_std_list.append(np.std(precision_scores))
    recall_scores_std_list.append(np.std(recall_scores))
    f1_scores_std_list.append(np.std(f1_scores))
    
print 'Precision for 50 times:', np.mean(precision_scores_mean_list), np.std(precision_scores_std_list)
print 'Recall for 50 times:', np.mean(recall_scores_mean_list), np.std(recall_scores_std_list)
print 'F1 for 50 times:', np.mean(f1_scores_mean_list), np.std(f1_scores_std_list)

precision_scores_mean_list = []
recall_scores_mean_list = []
f1_scores_mean_list = []
with open('rand_forest_temp.txt', 'r') as in_handle:
    for line in in_handle:
        if line.split()[0] == 'Precision:':
            precision_scores_mean_list.append(float(line.split()[1]))
        if line.split()[0] == 'Recall:':
            recall_scores_mean_list.append(float(line.split()[1]))
        if line.split()[0] == 'F1:':
            f1_scores_mean_list.append(float(line.split()[1]))
            
print 'Precision mean for 50 times:', np.mean(precision_scores_mean_list), 'Std. error:', stats.sem(precision_scores_mean_list)
print 'Recall mean for 50 times:', np.mean(recall_scores_mean_list), 'Std. error:', stats.sem(recall_scores_mean_list)
print 'F1 mean for 50 times:', np.mean(f1_scores_mean_list), 'Std. error:', stats.sem(f1_scores_mean_list)
     

