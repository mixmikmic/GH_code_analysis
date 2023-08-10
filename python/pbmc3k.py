import numpy as np
import scanpy.api as sc

sc.settings.verbosity = 2                # verbosity: 0=errors, 1=warnings, 2=info, 3=hints, ...
sc.settings.set_dpi(70)                  # dots (pixels) per inch determine size of inline figures
sc.logging.print_version_and_date()

adata = sc.read('pbmc3k_corrected')

sc.tl.tsne(adata, n_pcs=10)
sc.write('pbmc3k_corrected', adata)

axs = sc.pl.tsne(adata, color='louvain_groups')

sc.tl.aga(adata, node_groups='louvain_groups')
sc.write('pbmc3k_corrected', adata)

adata = sc.read('pbmc3k_corrected')
axs = sc.pl.aga(adata, title='Cells colored by cell types (Seurat).', 
          title_graph='Abstracted graph $\mathcal{G}^*$.',
          legend_fontsize=9, legend_fontweight='bold', fontsize=9,
          save='_pbmc3k')

