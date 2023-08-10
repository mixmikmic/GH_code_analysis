get_ipython().magic('matplotlib inline')
from IPython.display import display
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import hdbscan
import seaborn as sns
sns.set_style("dark")

#pd.set_option("display.max_rows",500)
#pd.set_option("display.max_columns",100)
pd.set_option('max_colwidth', 1000)

bachelors = pd.read_csv('data/scorecard_reduced_bachelors.csv')
bachelors.set_index('UNITID', inplace=True)
bachelors.info()

df_dict = pd.read_csv('data/CollegeScorecardDataDictionary-09-08-2015.csv')
df_dict.columns

df_dict.set_index('VARIABLE NAME', inplace=True)
dcat = df_dict.groupby('dev-category')
dcat.groups.keys()

ac_vars = dcat.groups['academics']
ac_dict = dcat.get_group('academics')
len(ac_dict)

ac_dict[['developer-friendly name', 'NAME OF DATA ELEMENT']]

academics = bachelors[["INSTNM", "sch_deg", "year"]+ac_vars]

academics.info()

sns.set_style("darkgrid")
sns.distplot(academics.count(axis=1), bins=5, kde=False)
plt.xlabel("Number of entries")
plt.ylabel("Number of Schools")
plt.savefig("Academics_Data_histogram_all.png")

sns.distplot(academics.count(axis=0), bins=5, kde=False, rug=False)
plt.xlabel("Number of entries")
plt.ylabel("Number of Variables")

col_entry_counts = academics.count(axis=0)
col_entry_counts.sort_values()

col_entry_counts.unique()

col_entry_counts.unique()

from sklearn import preprocessing,decomposition

pcip_vars = ac_dict[~ac_dict.index.str.contains("CERT|ASSOC|BACH")].index
pcip = academics[pcip_vars].dropna(how='any')
pcip.columns

programs = ac_dict.loc[pcip.columns]['developer-friendly name']
programs = programs.str.split('.',expand=True)[1]
program_dict = programs.to_dict()
pcip.rename(columns=program_dict, inplace=True)
pcip.columns

f,ax = plt.subplots(figsize=(6,30))
m = sns.heatmap(pcip.sample(frac=.25, axis=0), vmax=.3, vmin=0, square=False, yticklabels=False, cbar_kws={"shrink": .5})
#p = plt.setp(m.axes.xaxis.get_majorticklabels(), rotation=45)

m = sns.clustermap(pcip.sample(frac=.25, axis=0), vmax=.3, vmin=0, square=False, yticklabels=False)
#p = plt.setp(m.axes.xaxis.get_majorticklabels(), rotation=45)

pcip.sample(frac=.015, axis=0).plot.barh(stacked=True, legend=False)

f,axarr = plt.subplots(8,5, figsize=(13,13))
axarr = axarr.flatten()
for i,field in enumerate(pcip.columns):
    pcip[field].hist(ax=axarr[i], bins=100)
    axarr[i].set_ylim(0,100)
    axarr[i].set_xlabel(field)
plt.tight_layout()

icorr = pcip.corr()
mask = np.zeros_like(icorr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f,ax = plt.subplots(figsize=(12,12))
sns.heatmap(icorr, vmax=.3,square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5})

icorr_cols = ['resources', 
              'parks_recreation_fitness',
              'ethnic_cultural_gender', 
              'psychology',
              'communication', 
              'language',
              'english',
              'biological',
              'social_science',
              'history',
              'mathematics',
              'physical_science',
              'engineering',
              'engineering_technology',
              'construction',
              'mechanic_repair_technology',
              'precision_production',
              'transportation',
              'science_technology',
              'agriculture', 
              'family_consumer_science',
              'education',
              'legal',
              'library',
              'military',
              'multidiscipline',
              'philosophy_religious',
              'theology_religious_vocation',
              'security_law_enforcement',
              'health',
              'business_marketing',
              'humanities',
              'architecture', 
              'visual_performing',
              'communications_technology',
              'personal_culinary',
              'computer',
              'public_administration_social_service',
              ]

icorr_sorted = pcip[icorr_cols].corr()
f,ax = plt.subplots(figsize=(13,13))
sns.heatmap(icorr_sorted, vmax=.3,square=True, xticklabels=True, yticklabels=True,
            linewidths=.5, cbar_kws={"shrink": .5})

cg = sns.clustermap(pcip.corr(), vmax=.5,square=True, xticklabels=True, yticklabels=True,
                    linewidths=.5, figsize=(13, 13), z_score=1)
p = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

cg = sns.clustermap(pcip.corr(), vmax=.5,square=True, xticklabels=True, yticklabels=True,
                    linewidths=.5, figsize=(13, 13), z_score=0)
p = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

#pcip = preprocessing.normalize(pcip, axis=0)
pca = decomposition.PCA(n_components=4).fit(pcip)
print(pca.explained_variance_ratio_, "-- Fractional variance explained by each component.")
print(pca.explained_variance_ratio_.cumsum(),"-- Cummulative variance explained for Successive components")

components = pd.DataFrame(pca.components_.transpose(), index=programs.tolist(), 
                          columns=['Principal Component 1','Principal Component 2','Principal Component 3',
                                   'Principal Component 4'])

cg = sns.clustermap(components, cmap=plt.cm.viridis, vmax=.1, vmin=-.1, figsize=(13,13),
                    square=True, xticklabels=False, yticklabels=True, linewidths=.5, col_cluster=False)
p = plt.setp(cg.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

components[np.abs(components) > .2].dropna(how='all')

