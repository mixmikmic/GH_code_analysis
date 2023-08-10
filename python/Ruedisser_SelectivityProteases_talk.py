import pandas as pd
import numpy as np
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, auc, roc_curve, roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV

from rdkit.Chem import rdMolDescriptors as rdmd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Draw import SimilarityMaps
import SelectivityMaps   # SelectivityMaps

get_ipython().magic('matplotlib inline')

#set cell width to max
display(HTML("<style>.container { width:100% !important; }</style>")) 

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

SimilarityMaps.__file__

from rdkit import rdBase
print("RDkit version: %s, pandas version: %s, numpy version: %s, sklearn version: %s" %(rdBase.rdkitVersion, pd.__version__, np.__version__, sklearn.__version__))

panel = pd.read_csv(r"chembl_F10THRTRY.csv")
panel.head(1)

# prior probabilities for active for each target

for target in ['F10', 'trypsin', 'thrombin']:
    print("ratio active/(active + inactive) for target %s:\t %f") %(target, sum(list(panel[target]))/float(len(list(panel[target]))))

# 2**3 selectivity classes

def selclass(label1, label2, label3):
    return label3 + 2 * label2 + 4 *label1

panel['selclass'] = pd.DataFrame(map(selclass, panel['F10'], panel['trypsin'], panel['thrombin']))
grouped = panel.groupby('selclass').count()

params = {'legend.fontsize': 14,
          'figure.figsize': (8, 8),
         'axes.labelsize': 14,
         'axes.titlesize': 14,
         'xtick.labelsize': 14,
         'ytick.labelsize': 14}

plt.rcParams.update(params)

plt.bar(grouped.index, grouped['reference'], align = 'center')
plt.ylabel('Counts')
plt.title('Counts for each class')
plt.xlim(-0.6, 7.6)

#show class labels
df = panel.groupby("selclass", group_keys = "false").sum()
df[df != 0] = 1
df

def compute_morgan2(Smiles):
    mol = Chem.MolFromSmiles(Smiles)
    if mol is None:
        return None
    return rdmd.GetMorganFingerprintAsBitVect(mol, 2, 2048)

panel['morgan2'] = panel['smiles'].map(compute_morgan2)

def maptonumpy(x):
    arr = np.zeros(len(x))
    DataStructs.ConvertToNumpyArray(x, arr)
    return arr

panel['np_morgan2'] = panel['morgan2'].map(maptonumpy)
panel.dropna(subset=['morgan2'], inplace=True)

top_n = 20 # select the top 20 publications (ranking according to molecules count per publication)

references = pd.DataFrame(panel.groupby(['reference'])['reference'].count())
del references.index.name
print "===================================================================="
print "The first %s publications provide data for %s molecules out of %s." %(top_n, references.sort_values(by='reference', ascending=False).head(top_n)['reference'].sum(), panel.shape[0])
print "===================================================================="

ref = references.sort_values(by='reference', ascending=False)['reference']
ref_n = references.sort_values(by='reference', ascending=False).head(top_n)['reference']

plt.figure(figsize=(10, 8))
sns.barplot(x=ref_n.values, y=ref_n.index, orient="h")

train =  panel[panel['reference'].isin(ref_n.index.tolist())==True]   #DataFrame.isin returns a boolean dataframe with True for a match
test = panel[panel['reference'].isin(ref_n.index.tolist())==False]

X_train = train['np_morgan2'].tolist()
X_test = test['np_morgan2'].tolist()

#lists for single-output (so) classifier
targets = panel.columns[2:5].tolist()

Y_train=[]
Y_test=[]

for i in range(len(targets)):
    Y_train.append(train[targets[i]].tolist())
    Y_test.append(test[targets[i]].tolist())

#tuples for multi-output (mo) classifier
Y_train_mo = zip(Y_train[0], Y_train[1], Y_train[2])
Y_test_mo = zip(Y_test[0], Y_test[1], Y_test[2])

X =  panel['np_morgan2'].tolist()
  
Y_3D = panel[targets]

Y_list=[]

for i in range(len(targets)):
    Y_list.append(panel[targets[i]].tolist())
        
skf_list = []
for i in range(len(targets)):
    skf_list.append(StratifiedKFold(Y_list[i], 2, shuffle=True))

#train random forest classifier on multi-output (mo), perform parameter search for each fold and target

params = {'max_depth':[10,20],'min_samples_split':[2,8],'min_samples_leaf':[2,5],'n_estimators':[20,50,100]}

precision_mo_list = []; recall_mo_list = []
fpr_mo_list = []; tpr_mo_list = []
roc_mo_auc_list = []; proba_mo_list = []

for i in range(len(targets)):
    #switch training and test set
    for j in [0, 1]:
        if j == 0:
            x_train, y_train = np.array(X_train), np.array(Y_train_mo)
            x_test, y_test = np.array(X_test), np.array(Y_test_mo)[:, i]
        else:
            x_train, y_train = np.array(X_test), np.array(Y_test_mo)
            x_test, y_test = np.array(X_train), np.array(Y_train_mo)[:, i]

        cv = StratifiedKFold(y_train[:, i],n_folds=10,shuffle=True)
        gs = GridSearchCV(RandomForestClassifier(), params, cv=cv,verbose=0,refit=True)
        gs.fit(x_train, y_train)
        clf_rf = gs.best_estimator_
        # print gs.best_params_

        proba_mo = clf_rf.predict_proba(x_test)[i][:, -1]
        precision, recall, thresholds = precision_recall_curve(y_test, proba_mo)
        fpr, tpr, threshold = roc_curve(y_test, proba_mo)
        roc_auc = roc_auc_score(y_test, proba_mo)

        precision_mo_list.append(precision); recall_mo_list.append(recall)
        fpr_mo_list.append(fpr); tpr_mo_list.append(tpr)
        roc_mo_auc_list.append(roc_auc); proba_mo_list.append(proba_mo)

#train random forest classifier on single-output (so)
precision_list = []; recall_list = []
fpr_list = []; tpr_list = []
roc_auc_list = []; prob_list = []


for i in range(len(targets)):
    #switch training and test set
    for j in [0, 1]:
        if j == 0:
            x_train, y_train = np.array(X_train), np.array(Y_train[i])
            x_test, y_test = np.array(X_test), np.array(Y_test[i])
        else:
            x_train, y_train = np.array(X_test),  np.array(Y_test[i])
            x_test, y_test = np.array(X_train), np.array(Y_train[i])

        cv = StratifiedKFold(y_train,n_folds=10,shuffle=True)
        gs = GridSearchCV(RandomForestClassifier(), params, cv=cv,verbose=0,refit=True)
        gs.fit(x_train, y_train)
        clf_rf = gs.best_estimator_
        # print gs.best_params_

        proba_1D = [p[1] for p in clf_rf.predict_proba(x_test).tolist()]
        precision, recall, thresholds = precision_recall_curve(y_test, proba_1D)
        fpr, tpr, threshold = roc_curve(y_test, proba_1D)
        roc_auc = roc_auc_score(y_test, proba_1D)

        precision_list.append(precision); recall_list.append(recall)
        fpr_list.append(fpr); tpr_list.append(tpr)
        prob_list.append(proba_1D); roc_auc_list.append(roc_auc)

# helper functions to plot precission versus recall and ROC curves

def plot_prec(precision, recall, targets, label, colors =['r', 'g', 'b']):
    """plot precission and recall for target in targets 
    and for each fold"""
    for i in range(len(targets)):  
        fold_1 = plt.plot(recall[2*i], precision[2*i], label='fold 1: %s' % targets[i])
        fold_2 = plt.plot(recall[2*i+1], precision[2*i+1], '--', label='fold 2: %s' % targets[i])   
        plt.setp(fold_1, color=colors[i])
        plt.setp(fold_2, color=colors[i])
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(label)
    plt.legend(loc="lower left")
    plt.plot()
    
def plot_proc(fpr, tpr, roc, targets, label, colors = ['r', 'g', 'b']):
    """plot ROC for target in targets 
    and for each fold"""
    for i in range(len(targets)):    
        fold_1 = plt.plot(fpr[2*i], tpr[2*i], label='fold 1: %s area = %0.4f' % (targets[i], roc[2*i]))
        fold_2 = plt.plot(fpr[2*i+1], tpr[2*i+1], '--', label='fold 2: %s area = %0.4f' % (targets[i], roc[2*i+1]))   
        plt.setp(fold_1, color=colors[i])
        plt.setp(fold_2, color=colors[i])
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.title(label)
    plt.legend(loc="lower right")

plt.figure(figsize=(20,10))

plt.subplot(2, 2, 1)
fig_lc_pr_so = plot_prec(precision_list, recall_list, targets, 'Precision and recall for single-output model')
plt.subplot(2, 2, 2)
fig_lc_pr_mo = plot_prec(precision_mo_list, recall_mo_list, targets, 'Precision and recall for multi-output model')

plt.subplot(2, 2, 3)
fig_lc_roc_so = plot_proc(fpr_list, tpr_list, roc_auc_list, targets, 'ROC single-output model')
plt.subplot(2, 2, 4)
fig_lc_roc_mo = plot_proc(fpr_mo_list, tpr_mo_list, roc_mo_auc_list, targets, 'ROC multi-output model')

plt.tight_layout()

#SO classifier on training/test for each protease

precision_list = []; recall_list = []
fpr_list = []; tpr_list = []
roc_auc_list = []; prob_list = []

for i in range(len(targets)):
    for train, test in skf_list[i]:
        x_train= np.array(X)[train]; y_train = np.array(Y_list[i])[train]
        x_test= np.array(X)[test]; y_test = np.array(Y_list[i])[test]
        
        cv = StratifiedKFold(y_train,n_folds=10,shuffle=True)
        gs = GridSearchCV(RandomForestClassifier(), params, cv=cv,verbose=0,refit=True)
        gs.fit(x_train, y_train)
        clf_rf = gs.best_estimator_
        # print gs.best_params_
        
        proba_1D = [p[1] for p in clf_rf.predict_proba(x_test).tolist()]
        precision, recall, thresholds = precision_recall_curve(y_test, proba_1D)
        fpr, tpr, threshold = roc_curve(y_test, proba_1D)
        roc_auc = roc_auc_score(y_test, proba_1D)
        
        precision_list.append(precision); recall_list.append(recall)
        fpr_list.append(fpr); tpr_list.append(tpr)
        prob_list.append(proba_1D); roc_auc_list.append(roc_auc)

#MO classifier on training/test for each protease

precision_mo_list = []; recall_mo_list = []
fpr_mo_list = []; tpr_mo_list = []
roc_mo_auc_list = []; proba_mo_list = []


for i in range(len(targets)):
    for train, test in skf_list[0]:
        x_train= np.array(X)[train]; y_train = np.array(Y_3D)[:,0:3][train]
        x_test= np.array(X)[test]; y_test = np.array(Y_3D)[:,0:3][test]
        
        #clf_rf.fit(x_train, y_train)
        #cv = KFold(n=len(x_train),n_folds=10,shuffle=True)
        cv = StratifiedKFold(y_train[:, i],n_folds=10,shuffle=True)
        gs = GridSearchCV(RandomForestClassifier(), params, cv=cv,verbose=0,refit=True)
        gs.fit(x_train, y_train)
        clf_rf = gs.best_estimator_
        # print gs.best_params_
        
        proba_mo = clf_rf.predict_proba(x_test)[i][:,-1]
        precision, recall, thresholds = precision_recall_curve(y_test[:,i], proba_mo)
        fpr, tpr, threshold = roc_curve(y_test[:,i], proba_mo)
        roc_auc = roc_auc_score(y_test[:,i], proba_mo)
        
        precision_mo_list.append(precision); recall_mo_list.append(recall)
        fpr_mo_list.append(fpr); tpr_mo_list.append(tpr)
        roc_mo_auc_list.append(roc_auc); proba_mo_list.append(proba_mo)

plt.figure(figsize=(20,10))
plt.subplot(2, 2, 1)
plot1 = plot_prec(precision_list, recall_list, targets, 'Precision and recall for single-output model')
plt.subplot(2, 2, 2)
plot2 = plot_prec(precision_mo_list, recall_mo_list, targets, 'Precision and recall for multi-output model')

plt.subplot(2, 2, 3)
plot_proc(fpr_list, tpr_list, roc_auc_list, targets, 'ROC single-output model')
plt.subplot(2, 2, 4)
plot_proc(fpr_mo_list, tpr_mo_list, roc_mo_auc_list, targets, 'ROC multi-output model')

plt.tight_layout()

#encode image
from io import BytesIO
from base64 import b64encode

def img_func(x):
   b = BytesIO()
   x.savefig(b, format='png', dpi=70, bbox_inches='tight')
   return '<img alt="2d array" src="data:image/png;base64,' + b64encode(b.getvalue()) + '"/>'

#show similarity maps for one molecules
tomol = lambda x: Chem.MolFromSmiles(x)
mols_subset = map(tomol, panel['smiles'][0:1])
chemblID_subset = panel['parent__cmpd__chemblid'][0:1]
references_subset =  panel['reference'][0:1]

#nested lists for encoded images for SimilarityMaps
#clone (deepcopy) of estimator does not preserve fitting parameters: therefore classifier needs to be refitted for each training set

imgs = [[], [], []]

x_train = [np.array(X)[train], np.array(X_train), np.array(X_test)]
y_train = [np.array(Y_3D)[:,0:3][train], np.array(Y_train_mo), np.array(Y_test_mo)]

target_nr = 0 # show maps for F10

for i in range(len(x_train)):
    cv = StratifiedKFold(y_train[i][:, target_nr],n_folds=10,shuffle=True)
    gs = GridSearchCV(RandomForestClassifier(), params, cv=cv,verbose=0,refit=True)
    gs.fit(x_train[i], y_train[i])
    clf_rf = gs.best_estimator_
    
    for m in mols_subset:
        fp = lambda m,i: SimilarityMaps.GetMorganFingerprint(m, i, nBits=2048)
        fig, maxweight =  SelectivityMaps.GetSimilarityMapForModel(m, fp, lambda x: SelectivityMaps.getProbaprod(x, clf_rf.predict_log_proba, target_nr=target_nr), size=(180, 180), weightsScaling=False)
        imgs[i].append(img_func(fig))
        plt.close(fig)

s = """<table width=1200>
<tr>
   <th>trained on stratified 2-fold</th> 
   <th>trained on top 20 publications</th> 
   <th>trained on remaining publications</th>
   <th>chembl_id</th>
   <th>reference</th>
</tr>"""

for i in range(len(imgs[0])):
    s = s + """
    <tr>
       <th>%s</th>
       <th>%s</th>
       <th>%s</th>
       <th>%s</th>
       <th>%s</th>
    </tr>"""%(imgs[0][i], imgs[1][i], imgs[2][i], chemblID_subset[i], references_subset[i])
s = s + """
</table>"""

t=HTML(s)
t

