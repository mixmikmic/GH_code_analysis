import macosko2015
macosko2015.__version__

macosko2015.load_amacrine()

import os
import common

# Assign notebook and folder names
notebook_name = '02_robust_pca'
figure_folder = os.path.join(common.FIGURE_FOLDER, notebook_name)
data_folder = os.path.join(common.DATA_FOLDER, notebook_name)

# Make the folders
get_ipython().system(' mkdir -p $figure_folder')
get_ipython().system(' mkdir -p $data_folder')

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')

table1 = pd.read_table('/Users/olgabot/Downloads/GSE63473_RAW/GSM1626793_P14Retina_1.digital_expression.txt.gz', 
                       compression='gzip', index_col=0)
print(table1.shape)
table1.head()

data = table1.values
mask = data > 5

sns.distplot(data[mask].flat, kde=False)

mask = table1 == 0
sns.heatmap(table1, xticklabels=[], yticklabels=[], mask=mask)
fig = plt.gcf()
fig.savefig('table1_heatmap.png')

n_transcripts_per_cell = table1.sum()
n_transcripts_per_cell.head()

sns.distplot(n_transcripts_per_cell)

n_transcripts_per_cell.describe()

n_expressed_genes_per_cell = (table1 > 0).sum()
n_expressed_genes_per_cell.head()

sns.distplot(n_expressed_genes_per_cell)

greater500 = (table1 > 100).sum(axis=1) > 1
greater500.sum()

table1_t = table1.T
print(table1_t.shape)
table1_t.head()

n_transcripts_per_gene = table1_t.sum()
n_transcripts_per_gene.head()

n_transcripts_per_gene = table1_t.sum()
n_transcripts_per_gene.head()

sns.distplot(n_transcripts_per_gene)

(n_transcripts_per_gene > 1e3).sum()

n_transcripts_per_gene[n_transcripts_per_gene > 1e4]

median_transcripts_per_gene = table1_t.median()
median_transcripts_per_gene.head()

sns.distplot(median_transcripts_per_gene)
fig = plt.gcf()
fig.savefig('median_transcripts_per_gene.png')

data = median_transcripts_per_gene
mask = data > 0

sns.distplot(data[mask])
fig = plt.gcf()
fig.savefig('median_transcripts_per_gene_greater0.png')

gene_symbols = table1_t.columns.map(lambda x: x.split(':')[-1].upper())
gene_symbols.name = 'symbol'
table1_t.columnsmns = gene_symbols
table1_t.head()

barcodes = 'r1_' + table1_t.index
barcodes.name = 'barcode'
table1_t.index = barcodes
table1_t.head()

table1_t.to_csv('expression_table1.csv')



