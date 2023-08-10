import itertools
from collections import namedtuple, OrderedDict
import numpy as np
import scanpy.api as sc
import networkx as nx
import matplotlib.pyplot as pl
import seaborn as sns
import pandas as pd
sc.settings.verbosity = 1                         # amount of logging output
sc.settings.set_figure_params(dpi=80)              # control size of inline figures via dots per inch
sc.logging.print_version_and_date()

adata = sc.read('krumsiek11_blobs')

sc.tl.aga(adata, n_neighbors=30, resolution=4)
ax = sc.pl.aga(adata, basis='draw_graph_fr',
               title='reference partitions/clusters $\mathcal{N}_1^{*}$',
               title_graph='reference abstracted graph $\mathcal{G}_1^{*}$',               
               layout='fr', save='_reference')
adata_reference = adata.copy()

adata_new = sc.tl.aga(adata, n_neighbors=10, resolution=0.8, recompute_graph=True, copy=True)
ax = sc.pl.aga(adata_new, basis='draw_graph_fr', layout='fr',
               title='new partitions/clusters $\mathcal{N}_2^{*}$',
               title_graph='new abstracted graph $\mathcal{G}_2^{*}$',
               save='_new')

adata.smp['louvain_groups_new'] = adata_new.smp['louvain_groups']
result = sc.utils.compute_association_matrix_of_groups(adata, 'louvain_groups_new', 'louvain_groups')
sc.pl.matrix(result.asso_matrix, 
             xlabel='reference groups $\mathcal{N}_1^{*}$',
             ylabel='new groups $\mathcal{N}_2^{*}$', 
             title='association matrix (norm. $\mathcal{N}_2^{*}$)',
             save='_norm_new')

result = sc.utils.compute_association_matrix_of_groups(adata, 'louvain_groups_new', 'louvain_groups', normalization='reference')
sc.pl.matrix(result.asso_matrix,
             xlabel='reference groups $\mathcal{N}_1^{*}$',
             ylabel='new groups $\mathcal{N}_2^{*}$',              
             title='association matrix (norm. $\mathcal{N}_1^{*}$)',
             save='_norm_reference')

asso_groups_dict = sc.utils.identify_groups(adata.smp['louvain_groups'], adata_new.smp['louvain_groups'])
adata.smp['associated_new_groups'] = [asso_groups_dict[g][0] for g in adata.smp['louvain_groups']]
axs = sc.pl.draw_graph(adata, color=['louvain_groups', 'associated_new_groups'], legend_loc='on data')
ax = sc.pl.draw_graph(adata, color='associated_new_groups', legend_loc='on data',
                      title='$\mathcal{N}_1^*$ colored by associated partitions in $\mathcal{N}_2^*$',
                      save='_associated')

sc.settings.verbosity = 5  # increase to 6 for more output
result = sc.tl.aga_compare_paths(adata_reference, adata_new)
print(result)

sc.settings.verbosity = 1
statistics = OrderedDict([('nr. neighbors single-cell graph' , []), ('resolution louvain', []),
                          ('nr. louvain groups', []), ('total nr. steps in paths', []),
                          ('fraction of correct steps', []), ('fraction of correct paths', [])])
np.random.seed(0)
for i in range(100):
    n_neighbors = np.random.randint(5, 50)
    resolution = np.random.rand() * 5
    adata_new = sc.tl.aga(adata_reference, n_neighbors=n_neighbors, resolution=resolution, copy=True, recompute_graph=True)
    result = sc.tl.aga_compare_paths(adata_reference, adata_new)
    # uncomment for visualization or output
    # axs = sc.pl.aga(adata_new, basis='draw_graph_fr', layout_graph='fr')
    # print('n_neighbors' , n_neighbors, 'resolution', resolution,
    #       'n_groups', len(adata_new.add['aga_groups_order']), 'frac_steps', result.frac_steps)
    statistics['nr. neighbors single-cell graph' ].append(n_neighbors)
    statistics['resolution louvain'].append(resolution)
    statistics['nr. louvain groups'].append(len(adata_new.add['aga_groups_order']))
    statistics['total nr. steps in paths'].append(result.n_steps)        
    statistics['fraction of correct steps'].append(result.frac_steps)
    statistics['fraction of correct paths'].append(result.frac_paths)

df = pd.DataFrame(statistics)
_, axs = pl.subplots(ncols=df.shape[1], figsize=(12, 4), gridspec_kw={'left': 0.07, 'wspace': 0.9})
for i, col in enumerate(df.columns):
    sns.boxplot(df[col], ax=axs[i], orient='vertical')
axs[0].set_title('distribution of input parameters', loc='left')
axs[2].set_title('distribution of output parameters', loc='left')
axs[4].set_title('robustness of topology inference', loc='left')
pl.savefig('./figures/robustness_summary.png', dpi=300)
pl.show()

sc.settings.verbosity = 1
statistics = OrderedDict([('nr. neighbors single-cell graph' , []), ('resolution louvain', []),
                          ('nr. louvain groups', []), ('total nr. steps in paths', []),
                          ('fraction of correct steps', []), ('fraction of correct paths', [])])
np.random.seed(0)
for i in range(100):
    n_neighbors = np.random.randint(5, 50)
    resolution = np.random.rand() * 5
    adata_new = sc.tl.aga(adata_reference, tree_detection='iterative_matching',
                          n_neighbors=n_neighbors, resolution=resolution, copy=True, recompute_graph=True)
    result = sc.tl.aga_compare_paths(adata_reference, adata_new)
    # uncomment for visualization or output
    # axs = sc.pl.aga(adata_new, basis='draw_graph_fr', layout_graph='fr')
    # print('n_neighbors' , n_neighbors, 'resolution', resolution,
    #       'n_groups', len(adata_new.add['aga_groups_order']), 'frac_steps', result.frac_steps)
    statistics['nr. neighbors single-cell graph' ].append(n_neighbors)
    statistics['resolution louvain'].append(resolution)
    statistics['nr. louvain groups'].append(len(adata_new.add['aga_groups_order']))
    statistics['total nr. steps in paths'].append(result.n_steps)        
    statistics['fraction of correct steps'].append(result.frac_steps)
    statistics['fraction of correct paths'].append(result.frac_paths)

df = pd.DataFrame(statistics)
_, axs = pl.subplots(ncols=df.shape[1], figsize=(12, 4), gridspec_kw={'left': 0.07, 'wspace': 0.9})
for i, col in enumerate(df.columns):
    sns.boxplot(df[col], ax=axs[i], orient='vertical')
axs[0].set_title('distribution of input parameters', loc='left')
axs[2].set_title('distribution of output parameters', loc='left')
axs[4].set_title('robustness of topology inference', loc='left')
pl.show()

