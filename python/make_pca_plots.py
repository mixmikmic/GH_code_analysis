get_ipython().magic('matplotlib inline')
from sklearn.decomposition import PCA as sklearnPCA
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
from __future__ import division
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

counts = pd.read_table('/home/bay001/projects/human_HBOT_20161201/analysis/human_HBOT_merged/unused/counts.txt',index_col=0, skiprows=1)
def trunc(name):
    return name.replace('.polyATrim.adapterTrim.rmRep.sorted.rg.bam','').replace('_R1_001','')# [:name.find('_')]

def counts_to_rpkm(featureCountsTable):
    counts = featureCountsTable.ix[:,5:]
    lengths = featureCountsTable['Length']
    mapped_reads = counts.sum()
    return (counts * pow(10,9)).div(mapped_reads, axis=1).div(lengths, axis=0)
counts.head()

rpkms = counts_to_rpkm(counts)
"""
If we want a threshold.
"""
rpkm_threshold = 0
num_samples = rpkms.shape[1]*rpkm_threshold
rpkms = rpkms[rpkms.sum(axis=1)>=num_samples]
rpkms.to_csv('/home/bay001/projects/codebase/data/test_featurecounts_rpkms.txt',sep='\t')

rpkms = rpkms+1
rpkms_log2 = np.log2(rpkms)
rpkms_log2.to_csv('/home/bay001/projects/codebase/data/test_featurecounts_rpkms_log2.txt',sep='\t')
rpkms_log2.columns = [trunc(col) for col in rpkms_log2.columns]
print(rpkms_log2.shape)
rpkms_log2.head()

sklearn_pca = sklearnPCA(n_components=2)
sklearn_transf = sklearn_pca.fit_transform(rpkms_log2.T)

print(len(sklearn_pca.components_[1])) # number of genes in feature table
pc_components = pd.DataFrame(index = rpkms.index, columns=['PC1','PC2'])
for i,j in zip(rpkms.index, np.abs(sklearn_pca.components_[0])):
    pc_components.ix[i,'PC1'] = j
for i,j in zip(rpkms.index, np.abs(sklearn_pca.components_[1])):
    pc_components.ix[i,'PC2'] = j

pc_components.sort_values(by='PC1',ascending=False).to_csv('/home/bay001/projects/human_HBOT_20161201/analysis/human_HBOT_merged/PC_components.txt',
                                                          sep='\t')

pca = pd.DataFrame(sklearn_transf)
pca.index = rpkms_log2.columns
pca.to_csv('/home/bay001/projects/codebase/data/test_pcomp.txt',sep='\t')
pca.head()

ax = plt.figure(figsize=(15,15)) # .gca(projection='3d')

labels = rpkms_log2.columns
plt.scatter(pca[0], pca[1])

for label, x, y in zip(labels, pca[0], pca[1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
plt.xlabel('PC1 (%.2f %%) '%(sklearn_pca.explained_variance_ratio_[0]*100), fontsize=25)
plt.ylabel('PC2 (%.2f %%) '%(sklearn_pca.explained_variance_ratio_[1]*100), fontsize=25)
plt.show()

sns.palplot(sns.color_palette("hls", 14))
pca.to_csv('/ho')

conditions = pd.read_table('/home/bay001/projects/human_HBOT_20161201/analysis/human_HBOT_merged/unused/df.all.txt')
conditions.index = [i.replace('_R1_001','').replace('.','-') for i in conditions.index]
conditions.to_csv('/home/bay001/projects/codebase/data/test_conditions.txt',sep='\t')

cols = sns.color_palette("hls", 14)

def get_size(row):
    if row['day'] == 'day1':
        return 50
    else:
        return 200
def get_patient(row):
    return cols[int(row['patient'].replace("DHMC","").replace("UPMC",""))]

def get_responder(row):
    """
    if responder: cols[0] (red)
    if non: cols[8]
    """
    if row['type'] == 'responder':
        return cols[0]
    else:
        return cols[8]
def get_treatment(row):
    if row['factors'] == 'HBOT':
        return 200
    else:
        return 50

pca.columns = ['x','y']
merged = pd.merge(pca,conditions,how='left',left_index=True,right_index=True)
merged['daysize'] = merged.apply(get_size,axis=1)
merged['patientcol'] = merged.apply(get_patient,axis=1)
merged['respondcol'] = merged.apply(get_responder,axis=1)
merged['treatmentnum'] = merged.apply(get_treatment,axis=1)
merged

ax = plt.figure(figsize=(15,15)) # .gca(projection='3d')
plt.scatter(merged['x'],merged['y'],s=merged['daysize'],c=merged['patientcol'])
plt.xlabel('PC1 (%.2f %%) '%(sklearn_pca.explained_variance_ratio_[0]*100), fontsize=25)
plt.ylabel('PC2 (%.2f %%) '%(sklearn_pca.explained_variance_ratio_[1]*100), fontsize=25)
for label, x, y in zip(merged['patient'], merged['x'], merged['y']):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '-', connectionstyle='arc3,rad=0'))
smallcircle = Line2D(range(1), range(1), color="white", marker='o',markersize=7,markerfacecolor=cols[5], label='Day 1')
largecircle = Line2D(range(1), range(1), color="white", marker='o',markersize=15,markerfacecolor=cols[5], label='Day 10')

plt.legend(handles=[smallcircle,largecircle])

plt.show()

ax = plt.figure(figsize=(15,15)) # .gca(projection='3d')
plt.scatter(merged['x'],merged['y'],s=merged['treatmentnum'],c=merged['respondcol'])
plt.xlabel('PC1 (%.2f %%) '%(sklearn_pca.explained_variance_ratio_[0]*100), fontsize=25)
plt.ylabel('PC2 (%.2f %%) '%(sklearn_pca.explained_variance_ratio_[1]*100), fontsize=25)
for label, x, y in zip(merged.index, merged['x'], merged['y']):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '-', connectionstyle='arc3,rad=0'))

    
red_patch = mpatches.Patch(color=cols[0], label='Responder')
yellow_patch = mpatches.Patch(color=cols[8], label='Non Responder')
smallcircle = Line2D(range(1), range(1), color="white", marker='o',markersize=7,markerfacecolor=cols[5], label='sham')
largecircle = Line2D(range(1), range(1), color="white", marker='o',markersize=15,markerfacecolor=cols[5], label='HBOT')

plt.legend(handles=[red_patch,yellow_patch,smallcircle,largecircle])


plt.show()

merged

X[:, 0]

merged



