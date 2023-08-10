import numpy as np
import pandas as pd
import scanpy.api as sc
sc.settings.verbosity = 1                          # verbosity = 3: errors, warnings, info, hints
sc.settings.set_figure_params(dpi=80)              # low dots per inch yields small inline figures
sc.logging.print_version_and_date()

outdir = 'ECLAIR_instance/ECLAIR_ensemble_clustering_files/2017-09-15__09:09:31/'
labels = np.loadtxt(outdir + 'consensus_labels.txt').astype(int).astype(str)
adjacency = np.loadtxt(outdir + 'consensus_adjacency_matrix.txt')
variances = np.loadtxt(outdir + 'ensemble_distances_variances.txt')
distances = np.loadtxt(outdir + 'consensus_distances_matrix.txt')

adjacency

variances

distances

adata = sc.read('X_krumsiek11_scaled.txt')

sc.tl.draw_graph(adata)

adata.smp['eclair_clusters'] = labels

ax = sc.pl.draw_graph(adata, color='eclair_clusters', title='ECLAIR groups', legend_loc='on data', save='_eclair_clusters')

adata.add['eclair_adjacency'] = adjacency

ax = sc.pl.aga_graph(adata, solid_edges='eclair_adjacency', groups='eclair_clusters', frameon=True,
                     dashed_edges=None, title='ECLAIR tree', save='_eclair_tree')

