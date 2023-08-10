# Loading libraries
import numpy as np
import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = 6, 3
plt.rcParams['figure.dpi'] = 150

primary = pd.read_table('../data/PrimaryScreen.tsv')
OR = pd.read_table('../data/Receptors.tsv', index_col='OR')

odor = pd.read_table('../data/Odors.tsv', index_col='Odor')
odor.loc[9999,'OdorName'] = 'no odor'  # 9999 means there is no odorant

primary = primary.join(OR[['Gene']], on="OR")
primary = primary.join(odor[['OdorName']], on="Odor")

test_or = 'Olfr544' 

primary.dropna(inplace=True) #removing missing data

primary['log_Luc'] = np.log(primary.Luc)
primary['log_RL'] = np.log(primary.RL)
primary['log_normalized'] = np.log(primary.Luc/primary.RL)
primary['normalized_log'] = primary.log_Luc/primary.log_RL

print(primary.shape)
primary.head()

test = primary[primary.Gene == test_or].copy()

test.Concentration.value_counts()

g = test.groupby('Concentration')
g.log_Luc.hist(bins=20, normed=True, alpha=0.5)
plt.legend(g.groups.keys());
plt.title('Raw Histogram of Test Wells')
plt.xlabel("log(Luc)")
plt.savefig('../fig/raw_histogram.pdf', frameon=True)

test['c_is_1'] = (test.Concentration == 1)
tmp = pd.DataFrame(test.groupby(['Date', 'Plate']).c_is_1.max())
tmp.columns=['has_c_1']
#test.drop('c_is_1', axis=1, inplace=True)
primary = primary.join(tmp, on=['Date','Plate'])

test = primary[(~primary.has_c_1) & (primary.Gene == test_or)].copy()

g = test.groupby('Concentration')
g.log_Luc.hist(bins=20, normed=True, alpha=0.5)
plt.legend(g.groups.keys());
plt.title('Cleaned Histogram of Test Wells')
plt.xlabel("log(Luc)")
plt.savefig('../fig/cleaned_histogram.pdf')

g.log_normalized.hist(bins=20, normed=True, alpha=0.5)
plt.title('Histogram of Normalized Test Wells')
plt.xlabel("log(Luc/RL)")
plt.savefig('../fig/normalized_histogram.pdf')

test.loc[test.Concentration == 0,['Luc', 'RL']].corr(method='spearman')

test.loc[test.Concentration == 10,['Luc', 'RL']].corr(method='spearman')

# Load the classifiers

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Naive Bayes on log(Luc/R)

clf = GaussianNB()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) 

m = cross_val_score(clf, test[['log_normalized']], test.Concentration, cv=cv, scoring='accuracy')
print(m.mean(), m.std())

# Naive Bayes on log(Luc)

clf = GaussianNB()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) 

m = cross_val_score(clf, test[['log_Luc']], test.Concentration, cv=cv, scoring='accuracy')
print(m.mean(), m.std())

# Naive Bayes on log(Luc) and log(R) together. 

clf = GaussianNB()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) 

m = cross_val_score(clf, test[['log_Luc', 'log_RL']], test.Concentration, cv=cv, scoring='accuracy')
print(m.mean(), m.std())

cor = test[['Date', 'Plate', 'Concentration', 'Luc', 'RL']].groupby(['Date', 'Plate', 'Concentration']).mean().unstack().corr(method='spearman')
print(cor.to_latex())

tmp = test[['Date', 'Plate', 'Concentration', 'Luc']].groupby(['Date', 'Plate', 'Concentration']).mean().unstack()
#np.log(tmp.Luc).hist()
((tmp.Luc[10]/tmp.Luc[0])).min()
primary = primary.join(tmp, on=['Date','Plate'])

primary['Luc_0'] = primary[('Luc',0)]
primary['Luc_10'] = primary[('Luc',10)]

tmp.head()

primary.Gene.value_counts().head()

primary[primary.Gene == 'Olfr1341'].OdorName.value_counts().head()

primary[primary.Gene == 'Olfr73'].OdorName.value_counts().head()

primary[primary.Gene == 'OR2T5'].OdorName.value_counts().head()

primary[(primary.OdorName == 'no odor') & (primary.Gene == 'Olfr1341') & (~primary.has_c_1)].plot(kind='scatter', x=('Luc',0), y='Luc');

primary[(primary.OdorName == 'no odor') & (primary.Gene == 'Olfr73') & (~primary.has_c_1)].plot(kind='scatter', x=('Luc',0), y='Luc');
plt.title('Olfr73')
plt.savefig('../fig/Olfr73.pdf')

primary[(primary.OdorName == 'no odor') & (primary.Gene == 'OR2T5') & (~primary.has_c_1)].plot(kind='scatter', x=('Luc',0), y='Luc');

# Olfr1341 and no odor
primary.loc[(primary.OdorName == 'no odor') & (primary.Gene == 'Olfr1341') & (~primary.has_c_1), ['Luc', 'RL', 'Luc_0', 'Luc_10']].corr()

# Olfr73 and no odor
primary.loc[(primary.OdorName == 'no odor') & (primary.Gene == 'Olfr73') & (~primary.has_c_1), ['Luc', 'RL', 'Luc_0', 'Luc_10']].corr()

# OR2T5 and no odor
primary.loc[(primary.OdorName == 'no odor') & (primary.Gene == 'OR2T5') & (~primary.has_c_1), ['Luc', 'RL', 'Luc_0', 'Luc_10']].corr()

# Olfr1341 and heptaldehyde
primary.loc[(primary.OdorName == 'heptaldehyde') & (primary.Gene == 'Olfr1341') & (~primary.has_c_1), ['Luc', 'RL', 'Luc_0', 'Luc_10']].corr()

# Olfr73 and heptaldehyde
primary.loc[(primary.OdorName == 'heptaldehyde') & (primary.Gene == 'Olfr73') & (~primary.has_c_1), ['Luc', 'RL', 'Luc_0', 'Luc_10']].corr()

# OR2T5 and heptaldehyde
primary.loc[(primary.OdorName == 'heptaldehyde') & (primary.Gene == 'OR2T5') & (~primary.has_c_1), ['Luc', 'RL', 'Luc_0', 'Luc_10']].corr()

primary.to_csv('../data/PrimaryScreen_cleaned.tsv', sep='\t', index = False)



