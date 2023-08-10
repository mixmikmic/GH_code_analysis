import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (10, 6)
import numpy as np
import pandas as pd
import scipy.stats as stats

gex = pd.DataFrame.from_csv('Sanger_molecular_data/gex.csv', header=0)
methyl = pd.DataFrame.from_csv('Sanger_molecular_data/methyl/CpG_isle_level/methyl_ilse_beta.csv', header=0)
locus_gene = pd.DataFrame.from_csv('Converted_data/results.csv', header=0, index_col=False)

# drop rows without gene names
# drop rows where gene is not in gene expression file
locus_gene = locus_gene.dropna()
locus_gene = locus_gene[locus_gene['grch38_gene_name'].isin(gex.index.values)].reset_index(drop=True)
locus_gene

locus_gene.shape

gex.shape

methyl.shape

gex.drop(['SW620', 'KMS-11'], axis=1, inplace=True)
methyl.drop('NCI-H1437', axis=1, inplace=True)

methyl.shape

gex.shape

methyl=methyl[gex.columns]

methyl

def get_correlation(methyl_file):
    corr = []
    all_x = []
    all_y = []
    for i in range(locus_gene.shape[0]): 
        x = methyl_file.ix[locus_gene.iloc[i]['loci']] # methylation values of each cell line (per gene index)
        y = gex.ix[locus_gene.ix[i]['grch38_gene_name']] # gene expr values of each cell line
        corr.append(np.asarray(stats.pearsonr(x, y))) # pearson correlation for that gene
        all_x.append(np.asarray(x))
        all_y.append(np.asarray(y))
    
    mean_pearson = np.array(corr)[:, 0].mean()
    mean_pval = np.array(corr)[:, 1].mean()
    mean_x = np.mean(np.array(all_x), axis=1) # average methylation per gene
    mean_y = np.mean(np.array(all_y), axis=1) # average gene expr per gene
    
    return {'pearson': mean_pearson, 'pval': mean_pval, 'mean_x': mean_x, 'mean_y': mean_y}

corr = get_correlation(methyl)

corr['pearson']

corr['pval']

stats.pearsonr(corr['mean_x'], corr['mean_y'])

plt.scatter(corr['mean_x'], corr['mean_y'])
plt.plot()

sample_x = methyl.ix[locus_gene.iloc[1]['loci']]
sample_y = gex.ix[locus_gene.ix[1]['grch38_gene_name']]
print(stats.pearsonr(sample_x, sample_y))
plt.scatter(sample_x, sample_y)
plt.plot

methyl_m = pd.DataFrame.from_csv('Sanger_molecular_data/methyl/CpG_isle_level/methyl_ilse_m.csv', header=0)

methyl_m.drop('NCI-H1437', axis=1, inplace=True)
methyl_m = methyl_m[gex.columns]

corr_m = get_correlation(methyl_m)

corr_m['pearson']

corr_m['pval']

