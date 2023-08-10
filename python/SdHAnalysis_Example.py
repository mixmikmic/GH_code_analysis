from sdhanalysis import *

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

sweep1 = SdHDataSet('Example name', 'Example date')

sweep1.load_data(15,5)

sweep1.subtract_background(5, 10, 40, save=True)

sweep1.get_fft_peaks(mph=1.3e6, mpd=10, threshold=2, xmax=400, keep_ind=[2,4,5,6,8], save=True)

center_freq, passband, orbit = 594, 100, '594 T'
sweep1.isolate_orbit(orbit, center_freq, passband, save=True)

center_freq, passband, orbit = 1188, 100, '1188 T'
sweep1.isolate_orbit(orbit, center_freq, passband, save=True)

center_freq, passband, orbit = 3253, 100, '3253 T'
sweep1.isolate_orbit(orbit, center_freq, passband, save=True)

center_freq, passband, orbit = 3846, 100, '3846 T'
sweep1.isolate_orbit(orbit, center_freq, passband, save=True)

center_freq, passband, orbit = 4454, 100, '4454 T'
sweep1.isolate_orbit(orbit, center_freq, passband, save=True)

sweep1.get_peak_amplitudes('594 T', save=True)

sweep1.get_peak_amplitudes('1188 T', save=True)

sweep1.get_peak_amplitudes('3253 T', save=True)

sweep1.get_peak_amplitudes('3846 T', save=True)

sweep1.get_peak_amplitudes('4454 T', save=True)

sweep1.plot_orbit_amplitudes()



