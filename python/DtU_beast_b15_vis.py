from astropy.table import Table

import numpy as np

stats = Table.read('b15/b15_stats_toothpick_v1.1.fits')

print(sorted(stats.colnames))

from beast.plotting.plot_param_vs_chi2min import make_param_vs_chi2min_plots

make_param_vs_chi2min_plots(stats)

from beast.plotting.plot_good_bad import make_good_bad_plots

make_good_bad_plots(stats)

make_good_bad_plots(stats, chi2min=100)

make_good_bad_plots(stats, chi2min=3)

make_good_bad_plots(stats, xparam='Rv', yparam='Av')

make_good_bad_plots(stats, xparam='Rv', yparam='Av', chi2min=100)

make_good_bad_plots(stats, xparam='Rv', yparam='Av', chi2min=3.)

from beast.plotting.plot_region_diags import make_region_diag_plots

make_region_diag_plots(stats, xparam='logT', pxrange=[3.8,4.2], yparam='logL', pyrange=[5.0,6.0])

make_region_diag_plots(stats, xparam='Rv', pxrange=[2.8,3.4], yparam='Av', pyrange=[7.0,9.5])



