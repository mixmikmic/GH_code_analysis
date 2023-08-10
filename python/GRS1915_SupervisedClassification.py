get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import pandas as pd
import numpy as np
import sklearn
import pickle
import feature_extraction
import generaltools as gt

datadir = "/scratch/daniela/data/grs1915/"

with open(datadir+"grs1915_greedysearch_res.dat" ,'r') as f:
    data = pickle.load(f)

scores = data["scores"]
ranking = data["ranking"]

np.array(ranking)+1

max_scores = []
for s in scores:
    max_scores.append(np.max(s))

sns.set_style("whitegrid")
plt.rc("font", size=24, family="serif", serif="Computer Sans")
plt.rc("axes", titlesize=20, labelsize=20) 
plt.rc("text", usetex=True)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

fig, ax = plt.subplots(1,1,figsize=(9,6))

ax.scatter(np.arange(1,len(max_scores)+1,1), max_scores, marker="o", c=sns.color_palette()[0], s=40)
ax.set_xlabel("Number of features")
ax.set_ylabel("Validation $F_1$ score")
ax.set_xlim(0,len(max_scores)+1.5)
ax.set_ylim(0.5,1)
plt.savefig(datadir+"grs1915_feature_accuracy.pdf", format="pdf")

np.argmax(max_scores)

from paper_figures import load_data

features, labels, lc, hr, tstart, nseg,    features_lb, labels_lb, lc_lb, hr_lb, nseg_lb,     fscaled, fscaled_lb, fscaled_full, labels_all =         load_data(datadir, tseg=1024.0, log_features=None)


labels_all_labelled = labels_all[labels_all != "None"]

labels_all_labelled = pd.Series(labels_all_labelled)

labels_all_labelled.value_counts()

classified_label_fractions = labels_all_labelled.value_counts()/np.sum(labels_all_labelled.value_counts())
classified_label_fractions

features_train_full = features["train"]
features_val_full = features["val"]
features_test_full = features["test"]

ftrain_labelled = features_lb["train"]
fval_labelled = features_lb["val"]
ftest_labelled = features_lb["test"]

labels_train = labels_lb["train"]
labels_val =  labels_lb["val"]
labels_test = labels_lb["test"]

print(len(np.unique(labels_train)))
print(len(np.unique(labels_val)))
print(len(np.unique(labels_test)))

print(features_train_full.shape)
print(features_val_full.shape)
print(features_test_full.shape)

print(len(labels_train))
print(len(labels_val))
print(len(labels_test))

np.sum([len(labels_train), len(labels_val), len(labels_test)])

fscaled_train = fscaled_lb["train"]
fscaled_val = fscaled_lb["val"]
fscaled_test = fscaled_lb["test"]

fscaled_train_full = fscaled["train"]
fscaled_val_full = fscaled["val"]
fscaled_test_full = fscaled["test"]

def scatter(f1_full, f2_full, labels, log1=False, log2=False, alpha=0.5, palette="Set3"):
    if log1:
        f1 = np.log(f1_full)
    else:
        f1 = f1_full
        
    if log2:
        f2 = np.log(f2_full)
    else:
        f2 = f2_full
        
    #f1_labelled = f1[labels != "None"]
    #f2_labelled = f2[labels != "None"]
    
    unique_labels = np.unique(labels)
    unique_labels = np.delete(unique_labels, np.where(unique_labels == "None")[0])
    #print("unique labels : " + str(unique_labels))
    
    # make a Figure object
    fig, ax = plt.subplots(1,1,figsize=(12,9))
    
    # first plot the unclassified examples:
    ax.scatter(f1[labels == "None"], f2[labels == "None"], color="grey", alpha=alpha)

    # now make a color palette:
    current_palette = sns.color_palette(palette, len(unique_labels))
    
    for l, c in zip(unique_labels, current_palette):
        ax.scatter(f1[labels == l], f2[labels == l], color=c, alpha=alpha, label=l)
        
    plt.legend()
    
    return fig, ax
    
    

fig, ax = scatter(fscaled_full[:,0], fscaled_full[:,1], 
                  labels_all, log1=False, log2=False, alpha=0.7)

from sklearn.decomposition import PCA

fscaled_full = np.vstack([fscaled["train"], fscaled["val"], fscaled["test"]])

labels_all = np.hstack([labels["train"], labels["val"], labels["test"]])

pc = PCA(n_components=2)
fscaled_pca = pc.fit(fscaled_full).transform(fscaled_full)

alpha=0.8
palette = "Set3"

sns.set_style("whitegrid")
plt.rc("font", size=24, family="serif", serif="Computer Sans")
plt.rc("axes", titlesize=20, labelsize=20) 
plt.rc("text", usetex=True)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

unique_labels = np.unique(labels_all)
unique_labels = np.delete(unique_labels, np.where(unique_labels == "None")[0])
#print("unique labels : " + str(unique_labels))

# make a Figure object
fig, ax = plt.subplots(1,1,figsize=(15,9))

# first plot the unclassified examples:
ax.scatter(fscaled_pca[labels_all == "None",0], fscaled_pca[labels_all == "None",1], color="grey", alpha=alpha)

# now make a color palette:
current_palette = sns.color_palette(palette, len(unique_labels))

for l, c in zip(unique_labels, current_palette):
    ax.scatter(fscaled_pca[labels_all == l,0], fscaled_pca[labels_all == l,1], s=40,
               color=c, alpha=alpha, label=l)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.xlim(-6.2, 7.0)
plt.ylim(-7.0, 8.0)
plt.legend()

labels_train = labels["train"]
labels_val = labels["val"]
labels_test = labels["test"]

chaotic = ["beta", "lambda", "kappa", "mu"]
deterministic = ["theta", "rho", "alpha", "nu", "delta"]
stochastic = ["phi", "gamma", "chi"]

labels_train_phys, labels_val_phys, labels_test_phys = [], [], []
for l in labels_train:
    if l in chaotic:
        labels_train_phys.append("chaotic")
    elif l in deterministic:
        labels_train_phys.append("deterministic")
    elif l in stochastic:
        labels_train_phys.append("stochastic")
    else:
        labels_train_phys.append(l)
        
for l in labels_test:
    if l in chaotic:
        labels_test_phys.append("chaotic")
    elif l in deterministic:
        labels_test_phys.append("deterministic")
    elif l in stochastic:
        labels_test_phys.append("stochastic")
    else:
        labels_test_phys.append(l)

for l in labels_val:
    if l in chaotic:
        labels_val_phys.append("chaotic")
    elif l in deterministic:
        labels_val_phys.append("deterministic")
    elif l in stochastic:
        labels_val_phys.append("stochastic")
    else:
        labels_val_phys.append(l)

 

labels_all_phys = np.hstack([labels_train_phys, labels_val_phys, labels_test_phys])
labels_unique_phys = np.unique(labels_all_phys)
labels_unique_phys = np.delete(labels_unique_phys, 0)
print("unique labels: " + str(labels_unique_phys))


labels_phys = {"train":labels_train_phys,
              "test": labels_test_phys,
              "val": labels_val_phys}

alpha=0.8
palette = "Set3"

sns.set_style("whitegrid")
plt.rc("font", size=24, family="serif", serif="Computer Sans")
plt.rc("axes", titlesize=20, labelsize=20) 
plt.rc("text", usetex=True)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

# make a Figure object
fig, ax = plt.subplots(1,1,figsize=(15,9))

# first plot the unclassified examples:
ax.scatter(fscaled_pca[labels_all_phys == "None",0], 
           fscaled_pca[labels_all_phys == "None",1], 
           color="grey", alpha=alpha)

# now make a color palette:
current_palette = sns.color_palette(palette, len(labels_unique_phys))

for l, c in zip(labels_unique_phys, current_palette):
    ax.scatter(fscaled_pca[labels_all_phys == l,0], 
               fscaled_pca[labels_all_phys == l,1], s=40,
               color=c, alpha=alpha, label=l)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.xlim(-6.2, 7.0)
plt.ylim(-7.0, 8.0)
plt.legend()


import plotting

sns.set_style("whitegrid")
fig, ax = plt.subplots(1,1,figsize=(9,6))
ax = plotting.scatter(fscaled_pca, labels_all_phys, ax=ax)
plt.xlim(-7, 7)
plt.ylim(-8, 8)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
ax1 = plotting.scatter(fscaled_pca, labels_all, ax=ax1)
ax2 = plotting.scatter(fscaled_pca, labels_all_phys, ax=ax2)
ax1.set_xlim(-7, 9)
ax1.set_ylim(-8, 8)
ax2.set_xlim(-7, 9)
ax2.set_ylim(-8, 8)

import paper_figures

ax1, ax2 = paper_figures.features_pca(fscaled_full, labels, axes=None,
                 alpha=0.8, palette="Set3")

ax1.set_ylim(-8, 10)
ax2.set_ylim(-8, 10)

ax2.set_ylabel("")
plt.tight_layout()
plt.savefig(datadir+"grs1915_features_pca.pdf", format="pdf")

from sklearn.lda import LDA

lda = LDA(solver='svd', shrinkage=None, priors=None, n_components=2, 
          store_covariance=False, tol=0.0001)

fscaled_lda = lda.fit_transform(fscaled_full[:,:11], labels_all)

alpha=0.8
palette = "Set3"

sns.set_style("whitegrid")
plt.rc("font", size=24, family="serif", serif="Computer Sans")
plt.rc("axes", titlesize=20, labelsize=20) 
plt.rc("text", usetex=True)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

# make a Figure object
fig, ax = plt.subplots(1,1,figsize=(15,9))

# first plot the unclassified examples:
ax.scatter(fscaled_lda[labels_all_phys == "None",0], 
           fscaled_lda[labels_all_phys == "None",1], 
           color="grey", alpha=alpha)

# now make a color palette:
current_palette = sns.color_palette(palette, len(labels_unique_phys))

for l, c in zip(labels_unique_phys, current_palette):
    ax.scatter(fscaled_lda[labels_all_phys == l,0], 
               fscaled_lda[labels_all_phys == l,1], s=40,
               color=c, alpha=alpha, label=l)

plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.xlim(-6.2, 7.0)
plt.ylim(-7.0, 8.0)
plt.legend()



from sklearn.manifold import TSNE
fscaled_tsne = TSNE(n_components=2).fit_transform(fscaled_full[:,:11])

ax1, ax2 = paper_figures.features_pca(fscaled_full[:,:11], labels, 
                                      axes=None, algorithm="tsne",
                                      alpha=0.8, palette="Set3")


ax2.set_ylabel("")
plt.tight_layout()
plt.savefig(datadir+"grs1915_features_tsne.pdf", format="pdf")

fscaled_train = fscaled_lb["train"]
fscaled_test = fscaled_lb["test"]
fscaled_val = fscaled_lb["val"]

labels_train = labels_lb["train"]
labels_test = labels_lb["test"]
labels_val = labels_lb["val"]

fscaled_train.shape

nfeatures = 10

from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

params = {'n_neighbors': [1, 3, 5, 10, 15, 20, 25, 30, 50, 60, 80, 100, 120, 150]}#, 'max_features': }
grid = GridSearchCV(KNeighborsClassifier(), param_grid=params, verbose=1, 
                    n_jobs=10, scoring="f1_macro")
grid.fit(fscaled_train[:,:nfeatures], labels_train)

print(grid.best_params_)
print(grid.score(fscaled_train[:,:nfeatures], labels_train))
print(grid.score(fscaled_val[:,:nfeatures], labels_val))
print(grid.score(fscaled_test[:,:nfeatures], labels_test))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

params_c =  [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2, 5, 10.0, 20, 50, 100.0, 1000.0]

scores = np.zeros_like(params_c)
for i,c in enumerate(params_c):
    lr = LogisticRegression(penalty="l2", class_weight="balanced", multi_class="multinomial",
                            C=c, solver="lbfgs")

    lr.fit(fscaled_train[:,:nfeatures], labels_train)
    scores[i] = lr.score(fscaled_val[:,:nfeatures], labels_val)

max_score = np.max(scores)
print(max_score)
max_ind = np.argmax(scores)
print(max_ind)
lr_max_c = params_c[max_ind]
print(lr_max_c)

lr_best = LogisticRegression(penalty="l2", class_weight="balanced", multi_class="multinomial",
                            C=lr_max_c, solver="lbfgs")

lr_best.fit(fscaled_train[:,:nfeatures], labels_train)


labels_lr = lr_best.predict(fscaled_val[:,:nfeatures])
labels_lr_test = lr_best.predict(fscaled_test[:,:nfeatures])

labels_lr_all = lr_best.predict(fscaled_full[:,:nfeatures])
print("Test data set: " + str(lr_best.score(fscaled_test[:,:nfeatures], labels_test)))

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
ax1 = plotting.scatter(fscaled_pca, labels_all, ax=ax1)
ax2 = plotting.scatter(fscaled_pca, labels_lr_all, ax=ax2)
ax1.set_xlim(-7, 12)
ax1.set_ylim(-10, 10)
ax2.set_xlim(-7, 12)
ax2.set_ylim(-10, 10)

from sklearn.svm import LinearSVC

svm = sklearn.svm.LinearSVC(penalty="l2", loss="squared_hinge", dual=False,
                            class_weight="balanced", multi_class="ovr")

params_c =  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2, 5, 10.0, 20, 50, 100.0, 1000.0]

scores = np.zeros_like(params_c)
for i,c in enumerate(params_c):
    svm = sklearn.svm.LinearSVC(penalty="l2", loss="squared_hinge", dual=False,
                            class_weight="balanced", multi_class="ovr", C=c)

    svm.fit(fscaled_train[:,:nfeatures], labels_train)
    scores[i] = svm.score(fscaled_val[:,:nfeatures], labels_val)

max_score = np.max(scores)
print(max_score)
max_ind = np.where(scores == max_score)[0][0]
print(max_ind)
svm_max_c = params_c[max_ind]
print(svm_max_c)

svm_best = sklearn.svm.LinearSVC(penalty="l2", loss="squared_hinge", dual=False,
                        class_weight="balanced", multi_class="ovr", C=svm_max_c)

svm_best.fit(fscaled_train[:,:nfeatures], labels_train)

labels_svm = svm_best.predict(fscaled_val[:,:nfeatures])
labels_svm_test = svm_best.predict(fscaled_test[:,:nfeatures])

labels_svm_all = svm_best.predict(fscaled_full[:,:nfeatures])
print("Test data set: " + str(svm_best.score(fscaled_test[:,:nfeatures], labels_test)))
print("Linear model versus SVM: " + str(svm_best.score(fscaled_full[:,:nfeatures], labels_lr_all)))

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6))
ax1 = plotting.scatter(fscaled_pca, labels_all, ax=ax1)
ax2 = plotting.scatter(fscaled_pca, labels_svm_all, ax=ax2)

from sklearn.ensemble import RandomForestClassifier

params_max_depth = [7, 10, 20,40, 50, 100, 200, 500]
params_max_features = [2,4,6,8,10]

scores = np.zeros((len(params_max_depth), len(params_max_features)))

for i, md in enumerate(params_max_depth):
    for j, mf in enumerate(params_max_features):
        
        rfc = RandomForestClassifier(n_estimators=500, 
                                     max_features=mf, 
                                     max_depth=md)
        
        rfc.fit(fscaled_train[:,:nfeatures], labels_train)

        scores[i,j] = rfc.score(fscaled_val[:,:nfeatures], labels_val)
        

max_score = np.max(scores)
print(max_score)
max_ind = np.where(scores == max_score)

print(max_ind)

print(max_ind)
rfc_best =  RandomForestClassifier(n_estimators=500, 
                              max_depth=params_max_depth[max_ind[0][0]], 
                              max_features=params_max_features[max_ind[1][0]])

rfc_best.fit(fscaled_train[:,:nfeatures], labels_train)

labels_rfc = rfc_best.predict(fscaled_val[:,:nfeatures])
labels_rfc_test = rfc_best.predict(fscaled_test[:,:nfeatures])

print("Test data set: " + str(rfc_best.score(fscaled_test[:,:nfeatures], labels_test)))

feature_names = np.array(["LC mean", "LC median", "LC variance", "LC skew", "LC kurtosis",
                         r"$\nu_{\mathrm{max}}$", "PSD A", "PSD B", "PSD C", "PSD D", "PC 1", "PC 2",
                         "HR1 mean", "HR2 mean", "HR1 variance", "HR covariance", "HR2 variance",
                         "HR1 skew", "HR2 skew", "HR1 kurtosis", "HR2 kurtosis",
                         "LM 1", "LM 2", "LM 3", "LM 4", "LM 5", "LM 6", "LM 7", "LM 8", "LM 9", "LM 10",
                         "PCA 1", "PCA 2", "PCA 3", "PCA 4", "PCA 5", "PCA 6", "PCA 7", "PCA 8", "PCA 9", "PCA 10"])
print(feature_names.shape)
feature_names_ranked = feature_names[ranking]

feature_names_ranked

importances = rfc_best.feature_importances_
std = np.std([rfc_best.feature_importances_ for tree in rfc_best.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


# Print the feature ranking
print("Feature ranking:")

for f in range(fscaled_train[:,:nfeatures].shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(16,6))
plt.title("Feature importances")
plt.bar(range(fscaled_train[:,:nfeatures].shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(fscaled_train.shape[1]), feature_names_ranked[:nfeatures][indices], rotation=90)
plt.xlim([-1, nfeatures])
plt.show()

from sklearn.feature_selection import SelectFromModel

lr = LogisticRegression(C=lr_max_c, penalty="l1", class_weight=None, 
                        solver="liblinear")
lr.fit(fscaled_train, labels_train)

model = SelectFromModel(lr, prefit=True)

fscaled_new = model.transform(fscaled_full)

fscaled_new.shape

svm = sklearn.svm.LinearSVC(penalty="l1", loss="squared_hinge",
                            class_weight=None, dual=False,
                           C=svm_max_c)

svm.fit(fscaled_train, labels_train)

model = SelectFromModel(svm, prefit=True)

fscaled_new = model.transform(fscaled_full)
fscaled_new_train = model.transform(fscaled_train)
fscaled_new_val = model.transform(fscaled_val)
fscaled_new_test = model.transform(fscaled_test)

fscaled_new.shape

svm.fit(fscaled_new_train, labels_train)

svm.score(fscaled_new_test, labels_test)

fscaled_new_test.shape

labels_val_svm = svm.predict(fscaled_new_val)

features = {"train":features["train"][:,:nfeatures],
           "val": features["val"][:,:nfeatures],
            "test": features["test"][:,:nfeatures]}

features_lb = {"train": features_lb["train"][:,:nfeatures],
              "val": features_lb["val"][:,:nfeatures],
              "test": features_lb["test"][:,:nfeatures]}

features_all_full = np.vstack([features["train"], features["val"], features["test"]])
features_all_lb = np.vstack([features_lb["train"], features_lb["val"], features_lb["test"]])

import feature_engineering

fscaled, fscaled_lb = feature_engineering.scale_features(features, features_lb)

fscaled_train = fscaled_lb["train"]
fscaled_val = fscaled_lb["val"]
fscaled_test = fscaled_lb["test"]

fscaled_full = np.concatenate([fscaled["train"], fscaled["val"], fscaled["test"]])
fscaled_lb_full = np.concatenate([fscaled_train, fscaled_val, fscaled_test])

labels_train = labels_lb["train"]
labels_val = labels_lb["val"]
labels_test = labels_lb["test"]

lr = LogisticRegression(C=lr_max_c, penalty="l2", class_weight="balanced", multi_class="multinomial",
                        solver="lbfgs")
lr.fit(fscaled_train[:,:nfeatures], labels_train)

print(lr.score(fscaled_train[:,:nfeatures], labels_train))
print(lr.score(fscaled_val[:,:nfeatures], labels_val))

labels_lr_val = lr.predict(fscaled_val[:,:nfeatures])
labels_lr_test = lr.predict(fscaled_test[:,:nfeatures])

labels_lr_all = lr.predict(fscaled_full[:,:nfeatures])
print("Test data set: " + str(lr.score(fscaled_test[:,:nfeatures], 
                                       labels_test)))

proba_lr_val = lr.predict_proba(fscaled_val[:,:nfeatures])
proba_lr_test = lr.predict_proba(fscaled_test[:,:nfeatures])

lr.classes_

proba_lr_test.shape

from sklearn.metrics import confusion_matrix
import matplotlib.cm as cmap
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix', fig=None, ax=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    if ax is None and fig is None:
        fig, ax = plt.subplots(1,1, figsize=(10, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j], fontdict={"size":16},
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')

    return fig, ax

len(np.unique(labels_lr_test))

sns.set_style("white") 
plt.rc("font", size=24, family="serif", serif="Computer Sans")
plt.rc("axes", titlesize=20, labelsize=20) 
plt.rc("text", usetex=True)


labels_for_plotting = [r"$\%s$"%l for l in lr.classes_]

cm = confusion_matrix(np.hstack([labels_val, labels_test]),
                      np.hstack([labels_lr_val, labels_lr_test]), labels=lr.classes_)
#cm = confusion_matrix(labels_val, labels_lr_val, labels=lr.classes_)



fig, ax  = plt.subplots(1,1, figsize=(12,10))
fig, ax = plot_confusion_matrix(cm, labels_for_plotting, fig=fig, ax=ax)


reload(plotting)

import matplotlib.cm as cmap
sns.set_style("white")
fig, ax = plt.subplots(1,1,figsize=(12,9))
ax = plotting.confusion_matrix(np.hstack([labels_val, labels_test]),
                      np.hstack([labels_lr_val, labels_lr_test]), 
                               labels_for_plotting, log=False, fig=fig, ax=ax,
                              cmap=cmap.Blues)
#fig.subplots_adjust(bottom=0.15, left=0.15)
plt.tight_layout()
plt.savefig("grs1915_supervised_cm.pdf", format="pdf")

labels_all = np.hstack([labels_lb["train"], labels_lb["val"], labels_lb["test"]])
lc_all = np.concatenate([lc_lb["train"], lc_lb["val"], lc_lb["test"]])
hr_all = np.concatenate([hr_lb["train"], hr_lb["val"], hr_lb["test"]])

lc_train = lc_lb["train"]
lc_val = lc_lb["val"]
lc_test = lc_lb["test"]

hr_train = hr_lb["train"]
hr_val = hr_lb["val"]
hr_test = hr_lb["test"]

np.vstack([fscaled_val, fscaled_test])

misclassifieds = []
for i,(f, lpredict, ltrue, proba, lc, hr) in enumerate(zip(np.vstack([fscaled_val, fscaled_test]),
                                                           np.hstack([labels_lr_val, labels_lr_test]), 
                                                           np.hstack([labels_val, labels_test]), 
                                                           np.vstack([proba_lr_val, proba_lr_test]),
                                                           np.concatenate([lc_val, lc_test]), 
                                                           np.concatenate([hr_val, hr_test]))):
    if lpredict == ltrue:
        continue
    else:
        misclassifieds.append([f, lpredict, ltrue, proba, lc, hr])

np.vstack([fscaled_val, fscaled_test]).shape

len(misclassifieds)

proba_df = pd.DataFrame(np.vstack([proba_lr_val, proba_lr_test]), columns=lr.classes_)

max_proba = np.max(np.vstack([proba_lr_val, proba_lr_test]), axis=1)

for j,m in enumerate(misclassifieds):
    #pos_human = np.random.choice([0,3], p=[0.5, 0.5])
    #pos_robot = int(3. - pos_human)

    f = m[0]
    lpredict = m[1]
    ltrue = m[2]
    proba = m[3]
    times = m[4][0]
    counts = m[4][1]
    hr1 = m[5][0]
    hr2 = m[5][1]
    df = pd.DataFrame(proba, index=lr.classes_)
    print("Human classified class is: " + str(ltrue) + ", p = " + str(df.loc[ltrue].values))
    print("Predicted class is: " + str(lpredict) + ", p = " + str(df.loc[lpredict].values))

    print(df)
    print("============================================\n\n")

import scipy.stats 

def logbin_periodogram(freq, ps, percent=0.01):
    df = freq[1]-freq[0]
    minfreq = freq[0]*0.5
    maxfreq = freq[-1]
    binfreq = [minfreq, minfreq+df]
    while binfreq[-1] <= maxfreq:
        binfreq.append(binfreq[-1] + df*(1.+percent))
        df = binfreq[-1]-binfreq[-2]
    binps, bin_edges, binno = scipy.stats.binned_statistic(freq, ps, statistic="mean", bins=binfreq)
    #binfreq = np.logspace(np.log10(freq[0]), np.log10(freq[-1]+epsilon), bins, endpoint=True)
    #binps, bin_edges, binno = scipy.stats.binned_statistic(freq[1:], ps[1:], 
    #                                                       statistic="mean", bins=binfreq)
    #std_ps, bin_edges, binno = scipy.stats.binned_statistic(freq[1:], ps[1:], 
    #                                                       statistic=np.std, bins=binfreq)

    nsamples = np.array([len(binno[np.where(binno == i)[0]]) for i in xrange(np.max(binno))])
    df = np.diff(binfreq)
    binfreq = binfreq[:-1]+df/2.
    #return binfreq, binps, std_ps, nsamples
    return np.array(binfreq), np.array(binps), nsamples

import powerspectrum

def plot_misclassifieds(features, trained_labels, real_labels, proba_test, lc_test, hr_test, labels_all, 
                        lc_all, hr_all, nexamples=6, namestr="misclassified"):

    """
    Find all mis-classified light curves and plot them with examples of the real and false classes.
    """
    misclassifieds = []
    for i,(f, lpredict, ltrue, proba, lc, hr) in enumerate(zip(features, trained_labels, real_labels, 
                                                               proba_test, lc_test, hr_test)):
        if lpredict == ltrue:
            continue
        else:
            misclassifieds.append([f, lpredict, ltrue, proba, lc, hr])

    for j,m in enumerate(misclassifieds):
        pos_human = np.random.choice([0,3], p=[0.5, 0.5])
        pos_robot = int(3. - pos_human)

        f = m[0]
        lpredict = m[1]
        ltrue = m[2]
        proba = m[3]
        times = m[4][0]
        counts = m[4][1]
        hr1 = m[5][0]
        hr2 = m[5][1]
        print("Predicted class is: " + str(lpredict))
        print("Human classified class is: " + str(ltrue))
        robot_all = [[lp, lc, hr] for lp, lc, hr in                      zip(labels_all, lc_all, hr_all)                     if lp == lpredict ]
        human_all = [[lp, lc, hr] for lp, lc, hr in                      zip(labels_all, lc_all, hr_all)                     if lp == ltrue ]

        np.random.shuffle(robot_all)
        np.random.shuffle(human_all)
        robot_all = robot_all[:6]
        human_all = human_all[:6]
        sns.set_style("darkgrid")
        current_palette = sns.color_palette()
        fig = plt.figure(figsize=(10,15))

        def plot_lcs(times, counts, hr1, hr2, xcoords, ycoords, colspan, rowspan):
            #print("plotting in grid point " + str((xcoords[0], ycoords[0])))
            ax = plt.subplot2grid((9,6),(xcoords[0], ycoords[0]), colspan=colspan, rowspan=rowspan)
            ax.plot(times, counts, lw=2, linestyle="steps-mid", rasterized=True)
            ax.set_xlim([times[0], times[-1]])
            ax.set_ylim([0.0, 12000.0])
            #print("plotting in grid point " + str((xcoords[1], ycoords[1])))

            ax = plt.subplot2grid((9,6),(xcoords[1], ycoords[1]), colspan=colspan, rowspan=rowspan)
            ax.scatter(hr1, hr2, facecolor=current_palette[1], edgecolor="none", rasterized=True)
            ax.set_xlim([.27, 0.85])
            ax.set_ylim([0.04, 0.7])

            #print("plotting in grid point " + str((xcoords[2], ycoords[2])))    
            ax = plt.subplot2grid((9,6),(xcoords[2], ycoords[2]), colspan=colspan, rowspan=rowspan)
            dt = np.min(np.diff(times))
            ps = powerspectrum.PowerSpectrum(times, counts=counts/dt, norm="rms")
            binfreq, binps, nsamples = logbin_periodogram(ps.freq[1:], ps.ps[1:])
            ax.loglog(binfreq, binps, linestyle="steps-mid", rasterized=True)
            ax.set_xlim([ps.freq[1], ps.freq[-1]])
            ax.set_ylim([1.e-6, 10.])

            return

        ## first plot misclassified:
        plot_lcs(times, counts, hr1, hr2, [0,0,0], [0,2,4], 2, 2)

        ## now plot examples
        for i in range(4):
            r = robot_all[i]
            h = human_all[i]
            plot_lcs(h[1][0], h[1][1], h[2][0], h[2][1], [i+2, i+2, i+2], [pos_human, pos_human+1, pos_human+2], 1, 1)
            plot_lcs(r[1][0], r[1][1], r[2][0], r[2][1], [i+2, i+2, i+2], [pos_robot, pos_robot+1, pos_robot+2], 1, 1)

        ax = plt.subplot2grid((9,6),(8,pos_human+1))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel("Human: %s"%ltrue, fontsize=20)
        ax = plt.subplot2grid((9,6),(8,pos_robot+1))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel("Robot: %s"%lpredict, fontsize=20)
        plt.savefig(namestr+"%i.pdf"%j, format="pdf")
        plt.close()

plot_misclassifieds(np.vstack([fscaled_val, fscaled_test]),
                    np.hstack([labels_lr_val, labels_lr_test]), 
                    np.hstack([labels_val, labels_test]), 
                    np.vstack([proba_lr_val, proba_lr_test]),
                    np.concatenate([lc_val, lc_test]), 
                    np.concatenate([hr_val, hr_test]), 
                    labels_all, lc_all, hr_all, nexamples=102, namestr="../../misclassified")

fscaled_full = np.concatenate([fscaled["train"], fscaled["val"], fscaled["test"]])
fscaled_unclass = np.vstack([fscaled["train"][labels["train"] == "None"],
                                  fscaled["val"][labels["val"] == "None"],
                                  fscaled["test"][labels["test"] == "None"]
    ])

labels_cls = np.hstack([labels_lb["train"],
                        labels_lb["val"],
                        labels_lb["test"]])

labels_trained_unclass = lr.predict(fscaled_unclass[:,:nfeatures])
proba_unclass = lr.predict_proba(fscaled_unclass[:,:nfeatures])

proba_unclass = pd.DataFrame(proba_unclass, columns=lr.classes_, index=range(proba_unclass.shape[0]))

labels_trained_full = np.hstack([labels_cls,
                                 labels_trained_unclass])

sns.set_style("whitegrid") 
plt.rc("font", size=24, family="serif", serif="Computer Sans")
plt.rc("axes", titlesize=20, labelsize=20) 
plt.rc("text", usetex=True)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 

st_human = pd.Series(labels_cls)
st_robot = pd.Series(labels_trained_unclass)
nstates_human = st_human.value_counts()
nstates_human /= np.sum(nstates_human.values)

nstates_robot = st_robot.value_counts()
nstates_robot /= np.sum(nstates_robot.values)

nstates_df = pd.concat([nstates_human, nstates_robot], axis=1)
nstates_df.columns = ["human", "computer"]
nstates_df

tstart_cls = np.hstack([tstart["train"][labels["train"] != "None"],
                        tstart["val"][labels["val"] != "None"],
                        tstart["test"][labels["test"] != "None"],])

tstart_unclass = np.hstack([tstart["train"][labels["train"] == "None"],
                        tstart["val"][labels["val"] == "None"],
                        tstart["test"][labels["test"] == "None"],])

np.max(tstart_cls)

np.max(tstart_unclass)

duration_cls = pd.Series(np.zeros(len(lr.classes_)), index=lr.classes_)
duration_unclass = pd.Series(np.zeros(len(lr.classes_)), index=lr.classes_)

for i,(l, ts) in enumerate(zip(labels_cls, tstart_cls)):
    if i == 0:
        duration_cls.loc[l] += 1024.
    else:
        dt = ts - tstart_cls[i-1]
        if np.isclose(dt, 256.0, rtol=0.1, atol=0.1):
            duration_cls.loc[l] += 256.
        else:
            duration_cls.loc[l] += 1024.


for i,(l, ts) in enumerate(zip(labels_trained_unclass, tstart_unclass)):
    if i == 0:
        duration_unclass.loc[l] += 1024.
    else:
        dt = ts - tstart_unclass[i-1]
        if np.isclose(dt, 256.0, rtol=0.1, atol=0.1):
            duration_unclass.loc[l] += 256.
        else:
            duration_unclass.loc[l] += 1024.




durations_df = pd.concat([duration_cls, duration_unclass], axis=1)
durations_df.columns = ["human", "computer"]
labels_for_plotting = [r"$\%s$"%u for u in unique_labels]
durations_df.index = labels_for_plotting

durations_df

durations_df /= durations_df.sum()

durations_sorted = durations_df.sort_values("human", ascending=False,)

import matplotlib.cm as cmap

fig, ax = plt.subplots(1,1,figsize=(9,6))
durations_sorted.plot(kind="bar", ax=ax, color=[sns.color_palette()[0], 
                                          sns.color_palette()[2]])

ax.set_ylim(0, np.max(durations_sorted.max())+0.01)
ax.set_title("Distribution of classified states from the supervised classification")
ax.set_xlabel("State")
ax.set_ylabel(r"Fraction of $T_\mathrm{obs}$ spent in state")
ax.set_xticklabels(durations_sorted.index, rotation=0)
plt.tight_layout()
plt.savefig(datadir+"grs1915_supervised_states_histogram.pdf", format="pdf")
#plt.close()


proba_beta = proba_unclass.loc[labels_trained_unclass == "beta"]

beta_second_class = np.array([proba_beta.loc[i].sort_values(ascending=False).index[1] for i in proba_beta.index])
beta_first_class_proba = np.array([proba_beta.loc[i].sort_values(ascending=False).loc[proba_beta.loc[i].sort_values(ascending=False).index[0]] for i in proba_beta.index])
beta_second_class_proba = np.array([proba_beta.loc[i].sort_values(ascending=False).loc[proba_beta.loc[i].sort_values(ascending=False).index[1]] for i in proba_beta.index])

len(beta_first_class_proba[beta_first_class_proba>0.8])/np.float(len(beta_first_class_proba))

plt.hist(beta_first_class_proba, bins=20);

log_ratio = np.log10(beta_first_class_proba/beta_second_class_proba)
frac_twice = len(log_ratio[log_ratio>np.log10(2)])/np.float(len(log_ratio))
print("For a fraction of %.4f cases, the probability for beta"%frac_twice + 
      " is twice as large as the second-highest probability.")

len(beta_second_class_proba[beta_second_class_proba<0.2])/np.float(len(beta_second_class_proba))

beta_df = pd.DataFrame({"state": beta_second_class, "proba":beta_second_class_proba,
                      "log ratio":log_ratio})

beta_df["state"].value_counts()

(16+4+4+1)/np.float(beta_df["state"].value_counts().sum())

beta_df["state"].value_counts().plot("bar")

sns.violinplot("state", "log ratio", data=beta_df, scale="count");

27./(27+68)

def state_analysis(proba_unclass, lb, labels):
    proba= proba_unclass.loc[labels == lb]
    second_class = np.array([proba.loc[i].sort_values(ascending=False).index[1] for i in proba.index])
    first_class_proba = np.array([proba.loc[i].sort_values(ascending=False).loc[proba.loc[i].sort_values(ascending=False).index[0]] for i in proba.index])
    second_class_proba = np.array([proba.loc[i].sort_values(ascending=False).loc[proba.loc[i].sort_values(ascending=False).index[1]] for i in proba.index])

    print("There are %i observations in state %s."%(len(proba), lb))
    
    n_fc = len(first_class_proba[first_class_proba>0.8])/np.float(len(first_class_proba))
    print("The fraction of %s observations with probability >0.8 is %.4f"%(lb, n_fc))
    
    n_sc = len(second_class_proba[second_class_proba<0.2])/np.float(len(second_class_proba))
    print("The fraction of observations with probability <0.2 in the second-most probably class is %.4f"%n_sc)

    log_ratio = np.log(first_class_proba/second_class_proba)
    frac_twice = len(log_ratio[log_ratio>np.log(2)])/np.float(len(log_ratio))
    print("For a fraction of %.4f cases, the probability for %s"%(frac_twice, lb) + 
          " is twice as large as the second-highest probability.")
    
    fig, ax1 = plt.subplots(1,1,figsize=(9,6))
    _,_,_ = ax1.hist(first_class_proba, bins=20)
    
    state_df = pd.DataFrame({"state": second_class, "proba": second_class_proba,
                        "log ratio":log_ratio})

    fig, ax2 = plt.subplots(1,1,figsize=(9,6))
    state_df["state"].value_counts().plot("bar", ax=ax2)
    
    
    fig, ax3 = plt.subplots(1,1,figsize=(16,6))
    sns.violinplot("state", "log ratio", data=state_df, ax=ax3, scale="count");
    ax3.hlines(np.log(2.0), -1.0, 10.0)
    ax3.hlines(np.log(2.0), -1.0, 10.0)

    return

state_analysis(proba_unclass, "phi", labels_trained_unclass)

def state_analysis_reverse(proba_unclass, lb, labels):
    #proba = proba_unclass.loc[labels == lb]
    idx = np.array([i for i in proba_unclass.index if proba_unclass.loc[i].sort_values(ascending=False).index[1] == lb])
    first_class = np.array([proba_unclass.loc[i].sort_values(ascending=False).index[0] for i in idx])
    second_class = np.array([proba_unclass.loc[i].sort_values(ascending=False).index[1] for i in idx])

    first_class_proba = np.array([proba_unclass.loc[i, proba_unclass.loc[i].sort_values(ascending=False).index[0]] for i in idx])
    second_class_proba = np.array([proba_unclass.loc[i, lb] for i in idx])

    print("There are %i observations in state %s."%(len(first_class_proba), lb))
    
    n_fc = len(first_class_proba[first_class_proba>0.8])/np.float(len(first_class_proba))
    print("The fraction of %s observations with probability >0.8 is %.4f"%(lb, n_fc))
    
    n_sc = len(second_class_proba[second_class_proba<0.2])/np.float(len(second_class_proba))
    print("The fraction of observations with probability <0.2 in the second-most probably class is %.4f"%n_sc)

    log_ratio = np.log(first_class_proba/second_class_proba)
    frac_twice = len(log_ratio[log_ratio>np.log(2)])/np.float(len(log_ratio))
    print("For a fraction of %.4f cases, the probability for %s"%(frac_twice, lb) + 
          " is twice as large as the second-highest probability.")
    
    fig, ax1 = plt.subplots(1,1,figsize=(9,6))
    _,_,_ = ax1.hist(first_class_proba, bins=20)
    
    state_df = pd.DataFrame({"state": first_class, "proba": second_class_proba,
                        "log ratio":log_ratio})

    fig, ax2 = plt.subplots(1,1,figsize=(9,6))
    state_df["state"].value_counts().plot("bar", ax=ax2)
    
    
    fig, ax3 = plt.subplots(1,1,figsize=(16,6))
    sns.violinplot("state", "log ratio", data=state_df, ax=ax3, scale="count");
    ax3.hlines(np.log(2.0), -1.0, 10.0)
    ax3.hlines(np.log(2.0), -1.0, 10.0)

    return



len(proba_unclass)

state_analysis_reverse(proba_unclass, "gamma", labels_trained_unclass)

state_analysis_reverse(proba_unclass, "kappa", labels_trained_unclass)

state_analysis_reverse(proba_unclass, "eta", labels_trained_unclass)

labels_all = np.hstack([labels["train"], labels["val"], labels["test"]])

sns.set_style("whitegrid")
plt.rc("font", size=24, family="serif", serif="Computer Sans")
plt.rc("axes", titlesize=20, labelsize=20)
plt.rc("text", usetex=True)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

unique_labels = np.unique(labels_all)
unique_labels = np.delete(unique_labels,
                          np.where(unique_labels == "None")[0])

# make a Figure object
fig, axes = plt.subplots(1,2,figsize=(16,6), sharey=True)

xlim = [np.min(fscaled_pca[:,0])-0.5, np.max(fscaled_pca[:,0])+3.5] # [-6.2, 8.0]
ylim = [np.min(fscaled_pca[:,1])-0.5, np.max(fscaled_pca[:,1])+0.5] # [-7.0, 8.0]
ax1, ax2 = axes[0], axes[1]

# first plot the unclassified examples:
ax1.scatter(fscaled_pca[labels_all == "None",0],
           fscaled_pca[labels_all == "None",1],
           color="grey", alpha=alpha)

# now make a color palette:
current_palette = sns.color_palette(palette, len(unique_labels))

for l, c in zip(unique_labels, current_palette):
    ax1.scatter(fscaled_pca[labels_all == l,0],
               fscaled_pca[labels_all == l,1], s=40,
               color=c, alpha=alpha, label=l)

ax1.set_xlabel("PCA Component 1")
ax1.set_ylabel("PCA Component 2")
ax1.set_xlim(xlim)
ax1.set_ylim(ylim)
#ax1.legend(loc="upper right", prop={"size":14})

fscaled_pca_plot = np.vstack([fscaled_pca[labels_all != "None"], 
                              fscaled_pca[labels_all == "None"]])

labels_plot = np.hstack([labels_cls, labels_trained_unclass])

# first plot the unclassified examples:
current_palette = sns.color_palette(palette, len(unique_labels))

for l, c in zip(unique_labels, current_palette):
    ax2.scatter(fscaled_pca_plot[labels_plot == l,0],
               fscaled_pca_plot[labels_plot == l,1], s=40,
               color=c, alpha=alpha, label=l)

ax2.set_xlabel("PCA Component 1")
ax2.set_ylabel("PCA Component 2")
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.legend(loc="upper right", prop={"size":14})

plt.savefig(datadir+"grs1915_supervised_pca_comparison.pdf", format="pdf")
plt.tight_layout()

labels_trained_all = np.zeros_like(labels_plot)

labels_trained_all[labels_all != "None"] = labels_cls
labels_trained_all[labels_all == "None"] = labels_trained_unclass

colors = sns.color_palette("Set3", 14)
unique_labels = np.unique(labels_trained_all)
print(unique_labels)
tstart_all = np.concatenate([tstart["train"], tstart["val"], tstart["test"]])

asm = np.loadtxt(datadir+"grs1915_asm_lc.txt",skiprows=5)
asm_time = asm[:,0]
asm_cr = asm[:,1]
asm_total = asm_time[-1]-asm_time[0]
print("The ASM light curve covers a total of %i days"%asm_total)

mjdrefi = 49353. 
tstart_all_days = tstart_all/(60.*60.*24.)
tstart_all_mjd = tstart_all_days + mjdrefi


get_ipython().magic('matplotlib notebook')
## each light curve covers 500 days
plot_len = 500.
start_time = asm_time[0]
end_time = start_time + plot_len
i = 0

fig = plt.figure(figsize=(12,15))

sns.set_style("white")

ax = fig.add_subplot(111)
# Turn off axis lines and ticks of the big subplot

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

sns.set_context("notebook", font_scale=1.0, rc={"axes.labelsize": 16})

sns.set_style("whitegrid")

plt.rc("font", size=16, family="serif", serif="Computer Sans")
plt.rc("axes", titlesize=16, labelsize=16) 
plt.rc("text", usetex=True)

plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.97, hspace=0.2)

current_palette = sns.color_palette(palette, len(unique_labels))
while end_time <= asm_time[-1]:
    print("I am on plot %i."%i)
    ax1 = fig.add_subplot(11,1,i+1)
    ax1.errorbar(asm[:,0], asm[:,1], yerr = asm[:,2], linestyle="steps-mid")
    for k, col in zip(unique_labels, colors):
        tstart_members = tstart_all_mjd[labels_trained_all == k]
        ax1.plot(tstart_members, np.ones(len(tstart_members))*240.,"o", color=col, label="state " + str(k))
    ax1.set_xlim([start_time, end_time])
    ax1.set_ylim([1.0, 299.0])
    plt.yticks(np.arange(3)*100.0+100.0, [100, 200, 300]);

    start_time +=plot_len
    end_time += plot_len
    i+=1

ax.set_xlabel("Time in MJD", fontsize=18)
ax.set_ylabel("Count rate [counts/s]", fontsize=18)

#plt.savefig(paperdir+"grs1915_asm_lc_all.pdf", format="pdf")
#plt.close()

from collections import Counter

def transition_matrix(labels, order="row"):
    unique_labels = np.unique(labels)
    nlabels = len(unique_labels)
    
    labels_numerical = np.array([np.where(unique_labels == l)[0][0] for l in labels])
    labels_numerical = labels_numerical.flatten()
    
    transmat = np.zeros((nlabels,nlabels))
    for (x,y), c in Counter(zip(labels_numerical, labels_numerical[1:])).iteritems():
        transmat[x,y] = c
    
    transmat_p = np.zeros_like(transmat)
    if order == "row":
        transmat_p = transmat/np.sum(transmat, axis=1)
    elif order == "column":
        transmat_p = transmat/np.sum(transmat, axis=0)
    else:
        raise Exception("input for keyword 'order' not recognized!")
        
    return unique_labels, transmat, transmat_p


unique_labels, transmat, transmat_p = transition_matrix(labels_trained_full)

reload(plotting)


fig, ax = plt.subplots(1,1, figsize=(13,12))

labels_for_plotting = [r"$\%s$"%l for l in np.unique(labels_trained_full)]

fig, ax = plotting.transition_matrix(labels_trained_full, 
                                     labels_for_plotting, fig=fig, 
                                     ax=ax, log=False, order="column")
plt.tight_layout()
plt.savefig(datadir+"grs1915_supervised_transmat.pdf", format="pdf")



sns.palplot(current_palette)

## each light curve covers 500 days
plot_len = 500.
start_time = asm_time[0]
end_time = start_time + plot_len
i = 0

fig = plt.figure(figsize=(12,15))

sns.set_style("white")

ax = fig.add_subplot(111)
# Turn off axis lines and ticks of the big subplot

ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

sns.set_context("notebook", font_scale=1.0, rc={"axes.labelsize": 16})

sns.set_style("whitegrid")

plt.rc("font", size=16, family="serif", serif="Computer Sans")
plt.rc("axes", titlesize=16, labelsize=16) 
plt.rc("text", usetex=True)

plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.97, hspace=0.2)


current_palette = sns.color_palette(palette, len(unique_labels))
while end_time <= asm_time[-1]:
    print("I am on plot %i."%i)
    ax1 = fig.add_subplot(11,1,i+1)
    asm_s = asm[:,0].searchsorted(start_time)
    asm_e = asm[:,0].searchsorted(end_time)
    ax1.errorbar(asm[asm_s:asm_e,0], asm[asm_s:asm_e,1], 
                 yerr = asm[asm_s:asm_e,2], linestyle="steps-mid")
    for k, col in zip(unique_labels, colors):
        if k == "chi":
            c = current_palette[0]
            label = k
        elif k == "alpha":
            c = current_palette[1]
            label = k
        elif k == "rho":
            c = current_palette[2]
            label = k
        else:
            c = "grey"
            label = "other"
        tstart_members = tstart_all_mjd[labels_trained_all == k]
        #ts_s = tstart_members.searchsorted(start_time)
        #ts_e = tstart_members.searchsorted(end_time)
        ts_s = 0
        ts_e = -1
        ax1.plot(tstart_members[ts_s:ts_e], np.ones(len(tstart_members[ts_s:ts_e]))*240.,
                 "o", color=c, label=label)
    ax1.set_xlim([start_time, end_time])
    ax1.set_ylim([1.0, 299.0])
    if i == 0:
        ax1.legend()
    plt.yticks(np.arange(3)*100.0+100.0, [100, 200, 300]);

    start_time +=plot_len
    end_time += plot_len
    i+=1

ax.set_xlabel("Time in MJD", fontsize=18)
ax.set_ylabel("Count rate [counts/s]", fontsize=18)

#plt.savefig(paperdir+"grs1915_asm_lc_all.pdf", format="pdf")
#plt.close()

labels_trained_reduced = np.zeros_like(labels_trained_all)
for l in unique_labels:
    if l == "chi" or l == "alpha" or l == "rho":
        labels_trained_reduced[labels_trained_all == l] = l
    else:
        labels_trained_reduced[labels_trained_all == l] = "other"

transmat = np.zeros((3,3))
for i in range(len(labels_trained_reduced)-1):
    if labels_trained_reduced[i] == "chi":
        if labels_trained_reduced[i+1] == "chi":
            transmat[0,0] += 1
        elif labels_trained_reduced[i+1] == "alpha":
            transmat[0,1] += 1
        elif labels_trained_reduced[i+1] == "rho":
            transmat[0,2] += 1
        else: continue
    elif labels_trained_reduced[i] == "alpha":
        if labels_trained_reduced[i+1] == "chi":
            transmat[1,0] += 1
        elif labels_trained_reduced[i+1] == "alpha":
            transmat[1,1] += 1
        elif labels_trained_reduced[i+1] == "rho":
            transmat[1,2] += 1
        else: continue
    elif labels_trained_reduced[i] == "rho":
        if labels_trained_reduced[i+1] == "chi":
            transmat[2,0] += 1
        elif labels_trained_reduced[i+1] == "alpha":
            transmat[2,1] += 1
        elif labels_trained_reduced[i+1] == "rho":
            transmat[2,2] += 1
        else: continue
    else: continue

            

transmat_p = transmat/np.sum(transmat, axis=1)

transmat_p

plt.figure()
plt.matshow(transmat_p, cmap=cmap.viridis)

ulr, tr, transmat_r = plotting._compute_trans_matrix(labels_trained_reduced, order="row")
ulc, tc, transmat_c = plotting._compute_trans_matrix(labels_trained_reduced, order="column")
print(transmat_r)
print(transmat_c)

transmat_r1 = np.zeros_like(tr)
for i, t in enumerate(tr):
    transmat_r1[i,:] = t/np.sum(t)

transmat_r2 = np.zeros_like(tr)
for i, t in enumerate(tr):
    transmat_r2[:,i] = tr[:,i]/np.sum(tr[:,i])

transmat_r1

transmat_r2

fig, ax = plt.subplots(1,1,figsize=(5,4))
fig.subplots_adjust(left=0.15, bottom=0.15)
ax.matshow(transmat_r1, cmap=cmap.viridis)
ax.set_xticks(np.arange(len(ulr)))
ax.set_xticklabels(ulr, rotation=90)

ax.set_yticks(np.arange(len(ulr)))
ax.set_yticklabels(ulr)

plt.tight_layout()

transmat_r1

fig, ax = plt.subplots(1,1,figsize=(5,4))
fig.subplots_adjust(left=0.15, bottom=0.15)
ax.matshow(transmat_r2, cmap=cmap.viridis)
ax.set_xticks(np.arange(len(ulr)))
ax.set_xticklabels(ulr, rotation=90)

ax.set_yticks(np.arange(len(ulr)))
ax.set_yticklabels(ulr)

plt.tight_layout()

transmat_r2

labels_train_phys = np.array(labels_phys["train"], dtype='|S16')
labels_test_phys = np.array(labels_phys["test"], dtype='|S16')
labels_val_phys = np.array(labels_phys["val"], dtype='|S16')

labels_train_phys[labels_train_phys == "deterministic"] = "chaotic+coloured"
labels_val_phys[labels_val_phys == "deterministic"] = "chaotic+coloured"
labels_test_phys[labels_test_phys == "deterministic"] = "chaotic+coloured"

labels_all_phys_withetaomega = np.hstack([labels_train_phys, labels_val_phys, labels_test_phys])

np.unique(labels_val_phys)

unphysical = ["eta", "omega"]

for i,l in enumerate(labels_train_phys):
    if l in unphysical:
        labels_train_phys[i] = "None"

for i,l in enumerate(labels_val_phys):
    if l in unphysical:
        labels_val_phys[i] = "None"

for i,l in enumerate(labels_test_phys):
    if l in unphysical:
        labels_test_phys[i] = "None"


labels_unique_phys = np.unique(labels_train_phys)
print("unique physical labels: " + str(labels_unique_phys))

labels_train = labels_lb["train"]
labels_test = labels_lb["test"]
labels_val = labels_lb["val"]

fscaled_train_phys = fscaled_train[(labels_train != "eta") & (labels_train != "omega")]
fscaled_test_phys = fscaled_test[(labels_test != "eta") & (labels_test != "omega")]
fscaled_val_phys = fscaled_val[(labels_val != "eta") & (labels_val != "omega")]

labels_train = labels_train_phys[labels_train_phys != "None"]
labels_test = labels_test_phys[labels_test_phys != "None"]
labels_val = labels_val_phys[labels_val_phys != "None"]

params = {'n_neighbors': [1, 3, 5, 10, 15, 20, 25, 30, 50, 60, 80, 100, 120, 150]}#, 'max_features': }

grid = GridSearchCV(KNeighborsClassifier(), param_grid=params, verbose=1, n_jobs=10)
grid.fit(fscaled_train_phys, labels_train)


print(grid.best_params_)
print(grid.score(fscaled_train_phys, labels_train))
print(grid.score(fscaled_val_phys, labels_val))
print(grid.score(fscaled_test_phys, labels_test))




lr = LogisticRegression(penalty="l2", class_weight="balanced",
                       multi_class="multinomial", solver="lbfgs")

#params = {"C":[0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}

#grid_lr = GridSearchCV(lr, param_grid=params,
#                        verbose=0, n_jobs=10)
#grid_lr.fit(fscaled_train_phys, labels_train)
params_c =  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2, 5, 10.0, 20, 50, 100.0, 1000.0]

scores = np.zeros_like(params_c)
for i,c in enumerate(params_c):
    lr = LogisticRegression(penalty="l2", class_weight="balanced", multi_class="multinomial",
                            C=c, solver="lbfgs")

    lr.fit(fscaled_train_phys, labels_train)
    scores[i] = lr.score(fscaled_val_phys, labels_val)



max_score = np.max(scores)
print(max_score)
max_ind = np.where(scores == max_score)[0][0]
print(max_ind)
lr_max_c = params_c[max_ind]
print(lr_max_c)
lr_best = LogisticRegression(penalty="l2", class_weight="balanced", multi_class="multinomial",
                            C=lr_max_c, solver="lbfgs")

lr_best.fit(fscaled_train_phys, labels_train)


labels_lr = lr_best.predict(fscaled_val_phys)
labels_lr_test = lr_best.predict(fscaled_test_phys)

#labels_lr_all = lr_best.predict(fscaled_full[:,:nfeatures])
print("Test data set: " + str(lr_best.score(fscaled_test_phys, labels_test)))

params_c =  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2, 5, 10.0, 20, 50, 100.0, 1000.0]

scores = np.zeros_like(params_c)
for i,c in enumerate(params_c):
    #lr = LogisticRegression(penalty="l2", class_weight="balanced", multi_class="multinomial",
    #                        C=c, solver="lbfgs")
    svc = LinearSVC(penalty="l2", class_weight="balanced", dual=False, multi_class="crammer_singer",
                    C=c)


    svc.fit(fscaled_train_phys, labels_train)
    scores[i] = svc.score(fscaled_val_phys, labels_val)


max_score = np.max(scores)
print(max_score)
max_ind = np.where(scores == max_score)[0][0]
print(max_ind)
svc_max_c = params_c[max_ind]
print(svc_max_c)
svc_best = LinearSVC(penalty="l2", class_weight="balanced", dual=False, multi_class="crammer_singer",
                    C=svc_max_c)


#lr_best = LogisticRegression(penalty="l2", class_weight="balanced", multi_class="multinomial",
#                            C=lr_max_c, solver="lbfgs")

svc_best.fit(fscaled_train_phys, labels_train)


labels_svc = svc_best.predict(fscaled_val_phys)
labels_svc_test = svc_best.predict(fscaled_test_phys)

#labels_lr_all = lr_best.predict(fscaled_full[:,:nfeatures])
print("Test data set: " + str(svc_best.score(fscaled_test_phys, labels_test)))

from sklearn.ensemble import RandomForestClassifier

params_max_depth = [7, 10, 20,40, 50, 100, 200, 500]
params_max_features = [2,4,6,8,10]

scores = np.zeros((len(params_max_depth), len(params_max_features)))

for i, md in enumerate(params_max_depth):
    for j, mf in enumerate(params_max_features):
        
        rfc = RandomForestClassifier(n_estimators=500, 
                                     max_features=mf, 
                                     max_depth=md)
        
        rfc.fit(fscaled_train_phys, labels_train)

        scores[i,j] = rfc.score(fscaled_val_phys, labels_val)
        
max_score = np.max(scores)
print(max_score)
max_ind = np.where(scores == max_score)

print(max_ind)
rfc_best =  RandomForestClassifier(n_estimators=500, 
                              max_depth=params_max_depth[max_ind[0][0]], 
                              max_features=params_max_features[max_ind[1][0]])

rfc_best.fit(fscaled_train_phys, labels_train)

labels_rfc = rfc_best.predict(fscaled_val_phys)
labels_rfc_test = rfc_best.predict(fscaled_test_phys)

print("Validation data set: " + str(rfc_best.score(fscaled_val_phys, labels_val)))
print("Test data set: " + str(rfc_best.score(fscaled_test_phys, labels_test)))

cm = sklearn.metrics.confusion_matrix(labels_test, labels_rfc_test, 
                                      labels=np.unique(labels_rfc_test))

fig, ax = plt.subplots(1,1,figsize=(10,7))
plotting.confusion_matrix(labels_test, labels_rfc_test, 
                          np.unique(labels_rfc_test), ax=ax, log=False, fig=fig,
                         cmap=cmap.Blues)

plt.tight_layout()
plt.savefig(datadir+"grs1915_supervised_phys_cm.pdf", format="pdf")


11./17

np.sum(c)

labels_phys_full = np.hstack([labels_train_phys, labels_val_phys, labels_test_phys])

fscaled_cls = fscaled_full[labels_phys_full != "None"]
fscaled_unclass = fscaled_full[labels_phys_full == "None"]
labels_cls = labels_phys_full[labels_phys_full != "None"]

labels_phys_unclass = rfc_best.predict(fscaled_unclass)
labels_phys_cls = labels_phys_full[labels_phys_full != "None"]

# make a set of labels with the human + machine labels
labels_trained_all = np.zeros_like(labels_phys_full)
labels_trained_all[labels_phys_full == "None"] = labels_phys_unclass
labels_trained_all[labels_phys_full != "None"] = labels_phys_full[labels_phys_full != "None"]

tstart_cls = tstart_all[labels_phys_full != "None"]

duration_cls = pd.Series(np.zeros(len(rfc_best.classes_)), index=rfc_best.classes_)
duration_unclass = pd.Series(np.zeros(len(rfc_best.classes_)), index=rfc_best.classes_)

for i,(l, ts) in enumerate(zip(labels_phys_cls, tstart_cls)):
    if i == 0:
        duration_cls.loc[l] += 1024.
    else:
        dt = ts - tstart_cls[i-1]
        if np.isclose(dt, 256.0, rtol=0.1, atol=0.1):
            duration_cls.loc[l] += 256.
        else:
            duration_cls.loc[l] += 1024.

for i,(l, ts) in enumerate(zip(labels_phys_unclass, tstart_unclass)):
    if i == 0:
        duration_unclass.loc[l] += 1024.
    else:
        dt = ts - tstart_unclass[i-1]
        if np.isclose(dt, 256.0, rtol=0.1, atol=0.1):
            duration_unclass.loc[l] += 256.
        else:
            duration_unclass.loc[l] += 1024.




durations_df = pd.concat([duration_cls, duration_unclass], axis=1)
durations_df.columns = ["human", "computer"]


durations_df /= durations_df.sum()
durations_sorted = durations_df.sort_values("human", ascending=False,)

plt.rc("font", size=24, family="serif", serif="Computer Sans")
plt.rc("axes", titlesize=20, labelsize=20) 
plt.rc("text", usetex=True)
plt.rc('xtick', labelsize=20) 
plt.rc('ytick', labelsize=20) 


fig, ax = plt.subplots(1,1,figsize=(9,6))
durations_sorted.plot(kind="bar", ax=ax, color=[sns.color_palette()[0], 
                                          sns.color_palette()[2]])

ax.set_ylim(0, np.max(durations_sorted.max())+0.01)
ax.set_title("Distribution of classified states from the supervised classification")
ax.set_xlabel("State")
ax.set_ylabel(r"Fraction of $T_\mathrm{obs}$ spent in state")

plt.tight_layout()
plt.savefig(datadir+"grs1915_supervised_phys_states_histogram.pdf", format="pdf")
#plt.close()




#sns.set_style("whitegrid") 
#plt.rc("font", size=24, family="serif", serif="Computer Sans")
#plt.rc("axes", titlesize=20, labelsize=20) 
#plt.rc("text", usetex=True)
#plt.rc('xtick', labelsize=20) 
#plt.rc('ytick', labelsize=20) 

#st = pd.Series(labels_trained_phys)
#nstates = st.value_counts()
#nstates.plot(kind='bar', color=sns.color_palette()[0])
#plt.ylim(0,1.05*np.max(nstates))
#plt.title("Distribution of classified states from the supervised classification")
#plt.savefig(datadir+"grs1915_supervised_phys_states_histogram.pdf", format="pdf")
#plt.close()


current_palette = sns.color_palette("Set3", len(np.unique(labels_all_phys_withetaomega)))

colours = [current_palette[0], current_palette[1], current_palette[4]]

reload(plotting)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,6), sharey=True)
ax1 = plotting.scatter(fscaled_pca, labels_all_phys_withetaomega, ax=ax1)
ax2 = plotting.scatter(fscaled_pca, labels_trained_all, ax=ax2, colours=colours, ylabel=False)

axes[1].set_ylabel("")
plt.tight_layout()
plt.savefig(datadir+"sgr1915_supervised_phys_features_pca.pdf", format="pdf")
#plt.close()

labels_trained_all[labels_all_phys_withetaomega == "eta"]

fig, axes = plt.subplots(1,2,figsize=(16,6))
paper_figures.plot_eta_omega(labels_all_phys_withetaomega, labels_trained_all, axes=axes)
plt.tight_layout()
plt.savefig(datadir+"grs1915_supervised_eta_omega.pdf", format="pdf")

reload(plotting)

fig, ax = plt.subplots(1,1,figsize=(10,7))
ax = plotting.transition_matrix(labels_trained_all, np.unique(labels_trained_all), 
                                ax=ax, log=False, fig=fig)
fig.subplots_adjust(bottom=0.15, left=0.15)
plt.tight_layout()
plt.savefig(datadir+"grs1915_supervised_phys_transmat.pdf", format="pdf")





