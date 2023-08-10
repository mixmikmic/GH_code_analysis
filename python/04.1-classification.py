get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import qiime2 as q2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from scipy import interp
from skbio import TreeNode
from biom import load_table, Table

from featlib import Sculptor, load_mf
from featlib.extractor import abs_energy

plt.style.use('seaborn-whitegrid')

# rarefy before renaming
bt = load_table('otu-table.noblanks.biom')
bt = bt.subsample(4000)

mf = load_mf('mapping-file.alpha.txt')

# we only keep the samples that have sequences in the table
mf = mf.loc[bt.ids()].copy()

tree = TreeNode.read('gg-tree/97_otus.tree')

mf['days_since_epoch'] = pd.to_numeric(mf['days_since_epoch'], errors='coerce')

think = Sculptor(biom_table=bt, mapping_file=mf, tree=tree,
                 gradient='days_since_epoch', trajectory='host_subject_id',
                 name='gg')

lw = 2

N = 100
for N_samples in range(0, 12):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure()
    i = 0
    while i < N:
        ## BEGIN feature creation
        
        # zero samples means "traditional" classifier
        # with one sample per subject
        think.randomly_select(1 if N_samples == 0 else N_samples)

        # groups discovered by phylofactor
        features_to_keep = []
        for _, id_, md in think.biom_table.iter(axis='observation', dense=False):
            t = md['taxonomy']
            if (t[4].lower() == 'f__lachnospiraceae'
                or t[2].lower() == 'c__gammaproteobacteria'
                or t[2].lower() == 'c__betaproteobacteria'):
                features_to_keep.append(id_)

        # more than one sample
        if N_samples > 1:
            alpha = think.alpha_table(['faith_pd', 'chao1', 'brillouin_d'])
            beta = think.beta_table()
            features = think.microbes_over_time(ids=features_to_keep)
            
            # combine the data
            combined_features = pd.concat([features, alpha, beta], axis=1)

            # get a column with ibd status for all the subjects
            combined_features.dropna(axis=1, how='any', inplace=True)
            classes = think.mapping_file.groupby(['host_subject_id', 'ibd'],
                                        as_index=False).aggregate(np.sum).set_index('host_subject_id',
                                                                                    inplace=False)
            combined_features['ibd'] = classes['ibd']
        # one sample with our model
        elif N_samples == 1:
            alpha = think.alpha_table(['faith_pd', 'chao1', 'brillouin_d'], [abs_energy])
            features = think.biom_table.filter(ids_to_keep=features_to_keep, axis='observation')
            features = features.norm(inplace=False).to_dataframe().to_dense().T
            features['host_subject_id'] = think.mapping_file['host_subject_id']
            features['ibd'] = think.mapping_file['ibd']
            features.set_index('host_subject_id', inplace=True)
            combined_features = pd.concat([features, alpha], axis=1)
        # one sample with only relative abundances
        elif N_samples == 0:
            combined_features = think.biom_table.norm(inplace=False).to_dataframe().to_dense().T
            combined_features['ibd'] = think.mapping_file['ibd']

        # get a list of the features without the labels
        no_ibd = combined_features.columns.tolist()
        no_ibd.remove('ibd')
        ## END feature creation

        X_train, X_test, Y_train, Y_test = train_test_split(combined_features[no_ibd],
                                                            combined_features['ibd'],
                                                            test_size=0.35)

        clf = RandomForestClassifier(n_estimators=500, n_jobs=4)
        probas_ = clf.fit(X_train, Y_train).predict_proba(X_test)

        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(Y_test, probas_[:, 1], pos_label='Crohns')
        
        # skip any results with np.nan values as it means
        # the test/train split yieleded a bad selection
        if np.any(np.isnan(fpr)) or np.any(np.isnan(tpr)):
            continue 
        else:
            mean_tpr += interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            plt.plot(fpr, tpr, lw=0.1*lw, color='lightgray')

            i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Luck')

    mean_tpr /= N
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc-curves/%s/%d.samples.per.subject-%d.iterations.pdf' % (think.name, N_samples, N))



