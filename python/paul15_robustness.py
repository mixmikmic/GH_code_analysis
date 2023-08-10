import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy.api as sc
from collections import OrderedDict
import pandas as pd
import seaborn as sns

sc.settings.verbosity = 1                # amount of output
sc.settings.set_figure_params(dpi=80)    # low pixel number yields small inline figures
sc.logging.print_version_and_date()

adata = sc.read('paul15')
sc.tl.aga(adata, n_neighbors=4, resolution=1.2)
adata_reference = adata.copy()

asso_names, asso_matrix = sc.utils.compute_association_matrix_of_groups(
    adata, prediction='aga_groups', reference='paul15_clusters', normalization='reference')
asso_colors = sc.utils.get_associated_colors_of_groups(sc.pl.palettes.default_26, asso_matrix)
axs = sc.pl.aga(adata, color='paul15_clusters', basis='draw_graph_fr', title_graph='abstracted graph',
                palette=sc.pl.palettes.default_26,
                root=0, layout='eq_tree', frameon=True,
                color_graph=asso_colors, groups_graph=asso_names,
                fontsize=10)

adata = adata[~(adata.smp['paul15_clusters'] == '19Lymph')]

sc.tl.aga(adata, n_neighbors=4, resolution=1.2, recompute_graph=True)

axs = sc.pl.aga(adata, root=10, basis='draw_graph_fr',
                color='aga_groups', groups_graph='aga_groups', layout='eq_tree', title='')

asso_names, asso_matrix = sc.utils.compute_association_matrix_of_groups(
    adata, prediction='aga_groups', reference='paul15_clusters', normalization='reference')
asso_colors = sc.utils.get_associated_colors_of_groups(sc.pl.palettes.default_26, asso_matrix)
axs = sc.pl.aga(adata, color='paul15_clusters', basis='draw_graph_fr', title='', palette=sc.pl.palettes.default_26,
                root=10, layout='eq_tree', 
                color_graph=asso_colors, groups_graph=asso_names,
                fontsize=10)

import re
asso_names, _ = sc.utils.compute_association_matrix_of_groups(
    adata, prediction='aga_groups', reference='paul15_clusters', normalization='reference', max_n_names=1)
adata.add['aga_groups_order'] = ['{}/{}'.format(name, re.sub('\d+', '', asso_names[iname])) # replace integers
                                 for iname, name in enumerate(adata.add['aga_groups_order'])]

axs = sc.pl.aga(adata, title='',
                basis='draw_graph_fr',
                root=10, color='aga_groups', groups_graph='aga_groups', layout='eq_tree', show=False)
axs[0].set_title('single-cell/ abstracted graph for Paul et al. (2015), lymphoids removed', loc='left', fontsize=16)
pl.savefig('./figures/aga_without_lymphoids.png', dpi=300)
pl.show()

adata = sc.read('paul15')
sc.tl.aga(adata, n_neighbors=4, resolution=0.5)

sc.settings.set_figure_params(dpi=80)
asso_names, asso_matrix = sc.utils.compute_association_matrix_of_groups(
    adata, prediction='aga_groups', reference='paul15_clusters', normalization='reference')
asso_colors = sc.utils.get_associated_colors_of_groups(sc.pl.palettes.default_26, asso_matrix)
axs = sc.pl.aga(adata, color='paul15_clusters', basis='draw_graph_fr', title_graph='abstracted graph',
                palette=sc.pl.palettes.default_26,
                root=4, layout='eq_tree', frameon=True,
                color_graph=asso_colors, groups_graph=asso_names,
                fontsize=10)
axs = sc.pl.aga(adata, root=4, layout='eq_tree', frameon=True,
                basis='draw_graph_fr', title_graph='abstracted graph')

asso_names, asso_matrix = sc.utils.compute_association_matrix_of_groups(
    adata_reference, prediction='aga_groups', reference='paul15_clusters', normalization='reference')
asso_colors = sc.utils.get_associated_colors_of_groups(sc.pl.palettes.default_26, asso_matrix)
axs = sc.pl.aga(adata_reference, color='paul15_clusters', basis='draw_graph_fr', title_graph='abstracted graph',
                layout='eq_tree', frameon=True,
                color_graph=asso_colors, groups_graph=asso_names,
                fontsize=10)
axs = sc.pl.aga(adata_reference, frameon=True, layout='eq_tree', basis='draw_graph_fr', title_graph='abstracted graph')

# sc.settings.verbosity = 5
result = sc.tl.aga_compare_paths(adata_reference, adata,
                                 adjacency_key='aga_adjacency_tree_confidence')
print(result)

adata = sc.read('paul15')
sc.tl.aga(adata, n_neighbors=4, resolution=2.4)

axs = sc.pl.aga(adata, root=21, color='aga_groups', groups_graph='aga_groups', layout='eq_tree', frameon=True)
asso_names, asso_matrix = sc.utils.compute_association_matrix_of_groups(
    adata, prediction='aga_groups', reference='paul15_clusters', normalization='reference')
asso_colors = sc.utils.get_associated_colors_of_groups(sc.pl.palettes.default_26, asso_matrix)
axs = sc.pl.aga(adata, color='paul15_clusters', basis='draw_graph_fr', title_graph='abstracted graph',
                root=21, layout='eq_tree', frameon=True,
                color_graph=asso_colors, groups_graph=asso_names,
                fontsize=10)

# sc.settings.verbosity = 5
result = sc.tl.aga_compare_paths(adata_reference, adata,
                                 adjacency_key='aga_adjacency_tree_confidence')
print(result)

sc.settings.verbosity = 1
statistics = OrderedDict([('nr. neighbors single-cell graph' , []), ('resolution louvain', []),
                          ('nr. louvain groups', []), ('total nr. steps in paths', []),
                          ('fraction of correct steps', []), ('fraction of correct paths', [])])
np.random.seed(0)
for i in range(100):
    n_neighbors = np.random.randint(3, 20)
    resolution = np.random.rand() * 3
    adata_new = sc.tl.aga(adata_reference, n_neighbors=n_neighbors, resolution=resolution, copy=True, recompute_graph=True)
    result = sc.tl.aga_compare_paths(adata_reference, adata_new, adjacency_key='aga_adjacency_tree_confidence')
    # uncomment for visualization or output
    # axs = sc.pl.aga(adata_new, basis='draw_graph_fr', layout='fr')
    # print('n_neighbors' , n_neighbors, 'resolution', resolution,
    #       'n_groups', len(adata_new.add['aga_groups_order']), 'frac_steps', result.frac_steps)
    statistics['nr. neighbors single-cell graph' ].append(n_neighbors)
    statistics['resolution louvain'].append(resolution)
    statistics['nr. louvain groups'].append(len(adata_new.add['aga_groups_order']))
    statistics['total nr. steps in paths'].append(result.n_steps)        
    statistics['fraction of correct steps'].append(result.frac_steps)
    statistics['fraction of correct paths'].append(result.frac_paths)

df = pd.DataFrame(statistics)
_, ax = pl.subplots(ncols=df.shape[1], figsize=(12, 4), gridspec_kw={'left': 0.07, 'wspace': 0.9})
for i, col in enumerate(df.columns):
    sns.boxplot(df[col], ax=ax[i], orient='vertical')
ax[0].set_title('distribution of input parameters', loc='left')
ax[2].set_title('distribution of output parameters', loc='left')
ax[4].set_title('robustness of topology inference', loc='left')
pl.savefig('./figures/robustness_summary.png', dpi=300)
pl.show()

