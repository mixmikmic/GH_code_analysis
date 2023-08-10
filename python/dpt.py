import numpy as np
import pandas as pd
import scanpy.api as sc
sc.settings.verbosity = 1                          # verbosity = 3: errors, warnings, info, hints
sc.settings.set_figure_params(dpi=80)              # low dots per inch yields small inline figures
sc.logging.print_version_and_date()

adata = sc.datasets.krumsiek11()
sc.tl.tsne(adata)
sc.tl.draw_graph(adata)
sc.tl.dpt(adata, n_branchings=2, n_neighbors=5, knn=False)

ax = sc.pl.tsne(adata, color='dpt_groups', title='DPT groups', legend_loc='on data', save=True)
ax = sc.pl.draw_graph(adata, color='dpt_groups', title='DPT groups', save=True)
ax = sc.pl.diffmap(adata, color='dpt_groups', title='DPT groups', components=['1,2', '2,3'], save=True)

ax = sc.pl.diffmap(adata, color='dpt_groups', title='DPT groups', projection='3d', legend_loc='center left', save=True)

