import numpy as np
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy.api as sc

sc.settings.verbosity = 3                # increase for more output
sc.settings.set_dpi(60)                  # low pixel number yields small inline figures
sc.logging.print_version_and_date()

adata = sc.read('nestorowa16_171011')
for n_branchings in [1, 2]:
    print('n_branchings:', n_branchings)
    sc.tl.dpt(adata, n_branchings=n_branchings, n_neighbors=4, allow_kendall_tau_shift=False)
    ax = sc.pl.draw_graph(adata, color=['exp_groups', 'dpt_groups'],
                    title=['Experimental groups.', 'DPT groups.'],
                    palette=[sc.pl.palettes.vega_20, sc.pl.palettes.zeileis_26],
                    legend_loc='on data', legend_fontsize=14)
    if n_branchings == 1:
        ax = sc.pl.draw_graph(adata, color='dpt_groups', 
                        legend_loc='on data', legend_fontsize=14,
                        title='DPT groups.',
                        save='_compare_with_DPT' )

adata = sc.read('nestorowa16_170420')
for n_branchings in [1, 2, 3]:
    print('n_branchings:', n_branchings)
    sc.tl.dpt(adata, n_branchings=n_branchings, n_neighbors=4, allow_kendall_tau_shift=False)
    ax = sc.pl.draw_graph(adata, color=['exp_groups', 'dpt_groups'],
                    title=['Experimental groups', 'DPT groups'],
                    palette=[sc.pl.palettes.vega_20, sc.pl.palettes.zeileis_26],
                    legend_loc='on data', legend_fontsize=14)
    if n_branchings == 1:
        ax = sc.pl.draw_graph(adata, color='dpt_groups', 
                        legend_loc='on data', legend_fontsize=14,
                        title='DPT groups.',
                        save='_compare_with_DPT' )

