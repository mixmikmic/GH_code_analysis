from dedop.ui.compare import compare_l1b_products
get_ipython().magic('matplotlib inline')

comp = compare_l1b_products(__L1B_FILE_PATH_1__,
                            __L1B_FILE_PATH_2__)

comp.plot.locations()

comp.plot.waveforms_delta_im()

comp.plot.waveforms_hist()

comp.plot.waveforms_delta_hist()

comp.plot.waveforms_hexbin()

comp.plot.waveforms_scatter()

comp.waveforms

comp.waveforms_delta

comp.waveforms_delta_range

import numpy as np

x = comp.waveforms_delta
x_min, x_max = comp.waveforms_delta_range

mean = x.mean()
std = x.std()

k_max = (x_max - x_min) / std if std else 1

print()
print('mean =', mean, ' std =', std, '  k_max = ', k_max)
print()

for k in [1., 2., 3., 4., 5.]:
    num_outliers = np.logical_or(x < mean - k * std, x > mean + k * std).sum()
    num_within = x.size - num_outliers
    ratio_outliers = num_outliers / x.size
    ratio_within = 1.0 - ratio_outliers 

    print('k =', k, ':')
    print('  num_within =', num_within, ' ratio_within =', 100 * ratio_within, '%')
    print('  num_outliers =', num_outliers, ' ratio_outliers =', 100 * ratio_outliers, '%')

(x < x_min + 1e7).sum()

(x > x_max - 1e7).sum()



