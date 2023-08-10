import pickle
import pandas as pd
from functools import partial

filename = "../data/training_data/official_test_train_5_class_013018.pkl"

X_train, X_test, y_train, y_test = pickle.load(open(filename, "rb"))

df = pd.concat([X_train, X_test])
df = df.apply(partial(pd.to_numeric, errors='ignore'))
df.info()

df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

X_train = df[df.index.year>=2006]
y_train = X_train.label.values

X_test = df[df.index.year<2006]
y_test = X_test.label.values

feature_list = ['area', 'convex_area', 'eccentricity', 
                'intense_area', 'convection_area',
                'convection_stratiform_ratio', 'intense_stratiform_ratio',
                'intense_convection_ratio', 'mean_intensity', 'max_intensity',
                'intensity_variance', 'major_axis_length', 'minor_axis_length',
                'solidity']

labels = ['MCS', 'UCC', 'TROP', 'SYNP', 'CLUT']

df.label.values

df['label_name'] = [labels[x] for x in df.label.values]

df.head(5)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = 20, 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20

labs = ['MCS', 'UCC', 'Tropical', 'Synoptic', 'Clutter']

intensities = ['mean_intensity', 'max_intensity', 'intensity_variance']
ratios = ['intense_stratiform_ratio', 'convection_stratiform_ratio', 'intense_convection_ratio']
shapes = ['solidity', 'minor_major_ratio', 'eccentricity']
areas = ['area', 'convection_area', 'intense_area']

meanpointprops = dict(marker='.', markeredgecolor='black',
                      markerfacecolor='black')

for pnum, (var_group, group_name) in enumerate(zip([areas, intensities, ratios, shapes], 
                                                   ['area', 'int', 'rat', 'shp'])):

    d = {'var':[], 'label':[], 'val':[]}

    for var in var_group:
    
        grouped = df.groupby('label')

        for gid, group in grouped:

            d['var'].append(var)
            d['label'].append(labs[gid])
            d['val'].append(group[var].values.astype(float))

        dists = pd.DataFrame.from_dict(d)

        ax = plt.subplot(2, 2, pnum+1)

        bplot = ax.boxplot(dists['val'].values, whis=[5, 95], showmeans=True, meanprops=meanpointprops)

        plt.setp(bplot['medians'], color='black')

        [item.set_color('black') for item in bplot['means']] 

        ax.set_xticklabels(dists['label'].values, rotation=45)

        ax.axvspan(5.5, 10.5, color='grey', alpha=.2)

        ax.grid(True)

        ticklines = ax.get_xticklines() + ax.get_yticklines()
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        ticklabels = ax.get_xticklabels() + ax.get_yticklabels()


        for line in ticklines:
            line.set_linewidth(.5)

        for line in gridlines:
            line.set_linestyle('--')
            line.set_color('k')
            line.set_linewidth(.5)

        if group_name == 'int':

            ax.set_ylabel('dBZ', fontsize=25)

        elif group_name == 'area':

            ax.set_ylabel("Area (" + r'$km^2$' + ")", fontsize=25)
            ax.set_yscale('symlog')

plt.rcParams['ytick.labelsize'] = 25
plt.rcParams['xtick.labelsize'] = 25

for color, label in zip(['r', 'g', 'b', 'y', 'k'], labels):
    t_df = df[df.label_name==label]
    plt.scatter(t_df.area, t_df.mean_intensity, c=color, label=label)


plt.xscale('symlog')
plt.legend()
plt.xlim(1000, 600000)
plt.title("Area vs Mean Intensity", fontsize=30)
plt.ylabel("Mean Intensity (dBZ)", fontsize=30)
plt.xlabel("Area (" + r'$km^2$' + ")", fontsize=30)

for color, label in zip(['r', 'g', 'b', 'y', 'k'], labels):
    t_df = df[df.label_name==label]
    plt.scatter(t_df.convection_stratiform_ratio, t_df.intense_convection_ratio, c=color, label=label)


plt.legend()
plt.ylabel("Intense / Convection Ratio", fontsize=30)
plt.xlabel("Convection / Stratiform Ratio", fontsize=30)

plt.title("Convection / Stratiform Ratio vs. Intense / Convection Ratio", fontsize=30)

plt.axes().set_aspect('equal')

from sklearn.ensemble import VotingClassifier
from sklearn import tree
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

rf_clf = pickle.load(open("../data/classifiers/best_rfc_020618.pkl", 'rb'))
gb_clf = pickle.load(open("../data/classifiers/best_gbc_020618.pkl", 'rb'))
#xb_clf = pickle.load(open("../data/classifiers/best_xgbc_092717.pkl", 'rb'))

#add , ('xgb', xb_clf) if on linux
vclf = VotingClassifier([('rf', rf_clf), ('gb', gb_clf)], voting='soft')

vclf.fit(X_train[feature_list].values, y_train)

predicted = vclf.predict(X_test[feature_list].values)
expected = y_test

print("Classification report for classifier %s:\n%s\n"
      % (vclf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

plt.rcParams['ytick.labelsize'] = 30
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['figure.figsize'] = 20, 20

y_train1 = 1*(y_train==0)
y_test1 = 1*(y_test==0)

rfc = pickle.load(open("../data/classifiers/best_rfc_binary_020618.pkl", 'rb'))
gbc = pickle.load(open("../data/classifiers/best_gbc_binary_020618.pkl", 'rb'))
#xb_clf = pickle.load(open("../classifiers/best_xgbc_binary_092717.pkl", 'rb'))

vclf = VotingClassifier([('rf', rf_clf), ('gb', gb_clf)], voting='soft')

vclf.fit(X_train[feature_list].values, y_train1)

predicted = vclf.predict(X_test[feature_list].values)
expected = y_test1

print("Classification report for classifier %s:\n%s\n"
      % (vclf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

dt_clf = tree.DecisionTreeClassifier(max_depth=3)
kn_clf = KNeighborsClassifier()
lr_clf = LogisticRegression()

linetype = ['-', '--', ':', '-.']

vclf = VotingClassifier([('rf', rfc), ('gb', gbc)], voting='soft')

classifier_list = [vclf, lr_clf, kn_clf, dt_clf]
name_list = ['Ensemble', 'Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree']

y_train1 = 1*(y_train==0)
y_test1 = 1*(y_test==0)

for l, (clf, lab) in enumerate(zip(classifier_list, name_list)):

    clf.fit(X_train[feature_list[:-1]].values, y_train1)

    preds = clf.predict_proba(X_test[feature_list[:-1]].values)[:,1]
        
    fpr, tpr, _ = metrics.roc_curve(y_test1, preds)
    
    plt.plot(fpr, tpr, label=lab + ' (AUC = %0.2f)' % metrics.auc(fpr, tpr), lw=5, color='k', linestyle=linetype[l])
    

plt.plot([0, 1], [0, 1], color='grey', lw=5, linestyle='--', label='All MCS (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Probability of False Detection', fontsize=45)
plt.ylabel('Probability of Detection', fontsize=45)
plt.legend(loc="lower right", prop={'size': 35})
plt.show()

for year in range(2003, 2006):

    X_t = X_test[X_test.index.year==year]
    y_t = X_t.label1.values

    preds = vclf.predict_proba(X_t[feature_list[:-1]].values)[:,1]
        
    fpr, tpr, _ = metrics.roc_curve(y_t, preds)
    
    cval = (year-2000)/5
    
    plt.plot(fpr, tpr, label=str(year) + ' (AUC = %0.2f)' % metrics.auc(fpr, tpr), 
             lw=5, color=plt.cm.Greys(cval), linestyle='-')
    

plt.plot([0, 1], [0, 1], color='grey', lw=5, linestyle='--', label='All MCS (AUC = 0.50)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Probability of False Detection', fontsize=45)
plt.ylabel('Probability of Detection', fontsize=45)
plt.legend(loc="lower right", prop={'size': 35})
plt.show()

from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

fig = plt.figure(0, figsize=(20, 20))
plt.rcParams['ytick.labelsize'] = 35
plt.rcParams['xtick.labelsize'] = 35
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

rfc = pickle.load(open("../data/classifiers/best_rfc_binary_020618.pkl", 'rb'))
gbc = pickle.load(open("../data/classifiers/best_gbc_binary_020618.pkl", 'rb'))
vclf = VotingClassifier([('rf', rfc), ('gb', gbc)], voting='soft')


linetype = ['-', '--', ':', '-.']
for l, (clf, name) in enumerate(zip([vclf, lr_clf, kn_clf, dt_clf], ['Ensemble', 'Logistic Regression', 'K-Nearest Neighbors', 'Decision Tree'])):
    
    
    clf.fit(X_train[feature_list[:-1]].values, y_train1)
    y_pred = clf.predict(X_test[feature_list[:-1]].values)

    prob_pos = clf.predict_proba(X_test[feature_list[:-1]].values)[:, 1]

    clf_score = brier_score_loss(y_test1, prob_pos, pos_label=1)

    fraction_of_positives, mean_predicted_value =         calibration_curve(y_test1, prob_pos, n_bins=11)

    ax1.plot(mean_predicted_value, fraction_of_positives, "-", color='k', linestyle=linetype[l],
             label="%s (%1.3f)" % (name, clf_score), lw=5)

    ax2.hist(prob_pos, range=(0, 1), bins=11, color='k', linestyle=linetype[l], label=name,
             histtype="step", lw=5)
    
ax1.plot([0, 1], [0, 1], "--", lw=5, color='grey', label="Perfectly\ncalibrated (0.000)")


ax1.set_ylabel("Fractional Frequency of MCS Label", fontsize=45)
ax1.set_xticks(np.linspace(0, 1, 11))
ax1.set_ylim([-0.05, 1.05])
ax1.legend(loc="lower right", prop={'size': 30})

ax2.set_xlabel("Predicted Probability of MCS Label", fontsize=45)
ax2.set_xticks(np.linspace(0, 1, 11))
ax2.set_ylabel("Count", fontsize=45)

plt.tight_layout()

fig = plt.figure(0, figsize=(20, 20))
ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
ax2 = plt.subplot2grid((3, 1), (2, 0))

for year in range(2003, 2006):

    print(year)
    X_t = X_test[X_test.index.year==year]
    y_t = X_t.label1.values

    prob_pos = vclf.predict_proba(X_t[feature_list[:-1]].values)[:,1]

    clf_score = brier_score_loss(y_t, prob_pos, pos_label=1)

    fraction_of_positives, mean_predicted_value =         calibration_curve(y_t, prob_pos, n_bins=11)
        
    cval = (year-2000)/5

    ax1.plot(mean_predicted_value, fraction_of_positives, color=plt.cm.Greys(cval), linestyle='-',
             label="%s (%1.3f)" % (str(year), clf_score), lw=5)

    ax2.hist(prob_pos, range=(0, 1), bins=11, color=plt.cm.Greys(cval), linestyle='-', label=name,
             histtype="step", lw=5)
    
ax1.plot([0, 1], [0, 1], "--", lw=5, color='grey', label="Perfectly\nCalibrated (0.000)")


ax1.set_ylabel("Fractional Frequency of MCS Label", fontsize=45)
ax1.set_xticks(np.linspace(0, 1, 11))
ax1.set_ylim([-0.05, 1.05])
ax1.legend(prop={'size': 30})

ax2.set_xlabel("Predicted Probability of MCS Label", fontsize=45)
ax2.set_xticks(np.linspace(0, 1, 11))
ax2.set_ylabel("Count", fontsize=45)

plt.tight_layout()

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train[feature_list].values, y_train1)

importances = rfc.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train[feature_list].values.shape[1]):
    print("%d. (%s) feature %d (%f)" % (f + 1, feature_list[indices[f]], indices[f], importances[indices[f]]))

gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, random_state=42)
gbc.fit(X_train[feature_list].values, y_train1)

importances = gbc.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train[feature_list].values.shape[1]):
    print("%d. (%s) feature %d (%f)" % (f + 1, feature_list[indices[f]], indices[f], importances[indices[f]]))

df_labeled = pd.read_csv("../data/slice_data/labeled_slices_020618.csv")

df_ = df_labeled.sample(50000)

sp = plt.scatter(df_.area, df_.mean_intensity, c=df_.mcs_proba, s=10)
plt.xscale('symlog')
plt.title("Area vs Mean Intensity", fontsize=30)
plt.ylabel("Mean Intensity (dBZ)", fontsize=30)
plt.xlabel("Area (" + r'$km^2$' + ")", fontsize=30)
plt.colorbar(sp, shrink=0.4, pad=0.1, orientation='horizontal', boundaries=[0, .25, 0.5, 0.9, 0.95, 1], label='MCS Probability')
plt.ylim(20, 45)

