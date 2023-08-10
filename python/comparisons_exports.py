import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import scanpy.api as sc
sc.settings.verbosity = 1                          # verbosity = 3: errors, warnings, info, hints
sc.settings.set_figure_params(dpi=80)              # low dots per inch yields small inline figures
sc.logging.print_version_and_date()

adata = sc.read('krumsiek11_blobs')
ax = sc.pl.aga(adata, basis='draw_graph_fr', layout='rt', root=[7, 10, 11],
               color='aga_groups', groups_graph='aga_groups',)

# For Monocle 2, simply export the whole AnnData object.
sc.write('./comparisons/data/krumsiek11_blobs.csv', adata)
# For Eclair, we need it in tab-separated format.
df = pd.DataFrame(adata.X)
df.to_csv('./comparisons/eclair/X_krumsiek11_blobs.txt', sep='\t')
# For StemID, we need to shift it to remove negative values.
X_shifted = adata.X - np.min(adata.X)
df = pd.DataFrame(X_shifted)
df.to_csv('./comparisons/stemID/X_krumsiek11_blobs_shifted.csv')

adata = sc.AnnData(X_shifted)
sc.tl.draw_graph(adata)
sc.tl.aga(adata, resolution=2.5)
ax = sc.pl.aga(adata, basis='draw_graph_fr', layout='rt', root=['5', '0', '1'])

adata = sc.datasets.krumsiek11()
# For Eclair, we need it in tab-separeted format.
pd.DataFrame(adata.X).to_csv('./comparisons/eclair/X_krumsiek11.txt', sep='\t')
sc.pp.scale(adata)
pd.DataFrame(adata.X).to_csv('./comparisons/eclair/X_krumsiek11_scaled.txt', sep='\t')
# For StemID, we need to shift it to remove negative values.
X_shifted = adata.X - np.min(adata.X)
df = pd.DataFrame(X_shifted)
df.to_csv('./comparisons/stemID/X_krumsiek11_shifted.csv')

sc.tl.tsne(adata)
ax = sc.pl.tsne(adata)

sc.tl.aga(adata)
axs = sc.pl.aga(adata, layout='rt', root='1', show=False, title='')
axs[0].set_title('single-cell graph and abstracted graph for simple tree', loc='left')
pl.savefig('./figures/aga_simple_tree.png')
pl.show()

axs = sc.pl.tsne(adata, color='aga_pseudotime')

sc.write('./comparisons/data/krumsiek11.csv', adata)

