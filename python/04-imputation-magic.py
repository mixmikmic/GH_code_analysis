import magic

# Plotting and miscellaneous imports
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

get_ipython().magic('matplotlib inline')

scdata = magic.mg.SCData.from_csv(os.path.expanduser('../output/exprs.norm.before.imputation.csv'),
                                  data_type='sc-seq', normalize=False)

scdata

fig, ax = scdata.plot_pca_variance_explained(n_components=600, random=True)

scdata.run_magic(n_pca_components=20, random_pca=True, t=4, k=9, 
                 ka=3, epsilon=1, rescale_percent=0)

fig, ax = scdata.scatter_gene_expression(['ENSMUSG00000029661', 'ENSMUSG00000020911'])
ax.set_xlabel('Col1a2')
ax.set_ylabel('Ck19')

fig, ax = scdata.magic.scatter_gene_expression(['MAGIC ENSMUSG00000029661', 'MAGIC ENSMUSG00000020911'])
ax.set_xlabel('Col1a2')
ax.set_ylabel('Ck19')

scdata.run_pca()

scdata.magic.run_pca()

gs = gridspec.GridSpec(2,2)
fig = plt.figure(figsize=[15, 12])
genes = ['ENSMUSG00000029661', 'ENSMUSG00000020911', 'ENSMUSG00000026185', 'ENSMUSG00000017969']
for i in range(len(genes)):
    ax = fig.add_subplot(gs[i//2, i%2])
    scdata.scatter_gene_expression(genes=['PC1', 'PC2'], color=genes[i], fig=fig, ax=ax)

gs = gridspec.GridSpec(2,2)
fig = plt.figure(figsize=[15, 12])
genes = ['MAGIC ENSMUSG00000029661', 'MAGIC ENSMUSG00000020911', 'MAGIC ENSMUSG00000026185', 'MAGIC ENSMUSG00000017969']
for i in range(len(genes)):
    ax = fig.add_subplot(gs[i//2, i%2])
    scdata.magic.scatter_gene_expression(genes=['MAGIC PC1', 'MAGIC PC2'], color=genes[i], fig=fig, ax=ax)

mat = scdata.magic.data
mat.to_csv(path_or_buf="../output/exprs.norm.after.imputation.csv")



