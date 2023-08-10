from skdiscovery.visualization import fourier_plot, emd_plot, wavelets_plot, linear_decomposition_plot
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = 16, 9

from skdaccess.framework.param_class import *
from skdaccess.astro.kepler import DataFetcher as Kepler_DF
from skdaccess.utilities.kepler_util import normalize
from skdiscovery.utilities.patterns.trend_tools import medianFilter

from skdaccess.geo.pbo import DataFetcher as PBO_DF

# Latitude and Longitude range around Akutan Volcano
lat_range = AutoList((53.9,54.35))
lon_range = AutoList((-166.1, -165.5))
start_time = '2006-01-01'
end_time = '2015-06-01'

akutan_DF = PBO_DF(start_time, end_time, [lat_range, lon_range],mdyratio=.5)
akutan_data     = akutan_DF.output().get()

akudE0607 = akutan_data['AV06']['dE'] - akutan_data['AV07']['dE']

akuS = 'Akutan dE [meters]'

plt.plot(akudE0607);

fourier_plot.run_plotDFT(akudE0607);

emd_plot.run_plotImfs(akudE0607);

akuSplit25 = emd_plot.run_plotImfsSplitComp(akudE0607, highNums = [2, 5], mainTitle = akuS)

akuLow5 = akuSplit25[3]
akuLow5Lin = linear_decomposition_plot.plot_lin_interp(akuLow5, mainTitle = akuS)

akuLow5Slopes = linear_decomposition_plot.plot_lin_slope(akuLow5, mainTitle = akuS)

wavelets_plot.run_plotWPDecon(akudE0607, wavelet = 'db5')
wavelets_plot.run_plotWPShowall(akudE0607, wavelet = 'db5')
wavelets_plot.run_plotWPRecon(akudE0607, wavelet = 'db5')

