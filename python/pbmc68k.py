import numpy as np
import scanpy.api as sc
import matplotlib.pyplot as pl

sc.settings.verbosity = 2                # verbosity: 0=errors, 1=warnings, 2=info, 3=hints, ...
sc.settings.set_dpi(80)                  # dots (pixels) per inch determine size of inline figures
sc.logging.print_version_and_date()

adata = sc.read('zheng17')

ax = sc.pl.tsne(adata, color='bulk_labels', legend_loc='on data', legend_fontsize=10, legend_fontweight='bold')

adata = sc.read('zheng17')
sc.tl.aga(adata, node_groups='bulk_labels')
sc.write('zheng17', adata)

adata = sc.read('zheng17')
axs = sc.pl.aga(adata, legend_fontsize=9, legend_fontweight='bold', fontsize=9,
          title='Bulk labels for 68K PBMCs.',
          title_graph='Abstracted graph.', frameon=True, show=True, save='_pbmc68k')

adata = sc.read('zheng17')
sc.tl.aga(adata, node_groups='bulk_labels', recompute_graph=True)

sc.pl.aga(adata, legend_fontsize=9, legend_fontweight='bold', fontsize=9,
          title_graph='abstracted graph', frameon=True)

