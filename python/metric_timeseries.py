import iris
import iris.plot as iplt
import matplotlib.pyplot as plt

pe_file = '/g/data/r87/dbi599/drstree/CMIP5/GCM/CCCMA/CanESM2/historical/yr/atmos/pe/r1i1p1/pe-global-abs_Ayr_CanESM2_historical_r1i1p1_all.nc'
sos_file = '/g/data/r87/dbi599/drstree/CMIP5/GCM/CCCMA/CanESM2/historical/yr/ocean/sos/r1i1p1/sos-global-amp_Oyr_CanESM2_historical_r1i1p1_all.nc'

pe_cube = iris.load_cube(pe_file, 'precipitation minus evaporation flux')
sos_cube = iris.load_cube(sos_file, 'sea_surface_salinity')

pe_cube.data = pe_cube.data * 86400
pe_mean = pe_cube.data.mean()
pe_std = pe_cube.data.std()
pe_cube.data = (pe_cube.data - pe_mean) / pe_std

sos_mean = sos_cube.data.mean()
sos_std = sos_cube.data.std()
sos_cube.data = (sos_cube.data - sos_mean) / sos_std

get_ipython().magic('matplotlib inline')

iplt.plot(pe_cube, label='P-E')
iplt.plot(sos_cube, label='salinity')
plt.legend(loc=2)
plt.show()

iplt.plot(pe_cube.rolling_window('time', iris.analysis.MEAN, 5), label='P-E')
iplt.plot(sos_cube.rolling_window('time', iris.analysis.MEAN, 5), label='salinity')
plt.legend(loc=2)
plt.show()

iplt.plot(pe_cube.rolling_window('time', iris.analysis.MEAN, 10), label='P-E')
iplt.plot(sos_cube.rolling_window('time', iris.analysis.MEAN, 10), label='salinity')
plt.legend(loc=2)
plt.show()

iplt.plot(pe_cube.rolling_window('time', iris.analysis.MEAN, 40), label='P-E')
iplt.plot(sos_cube.rolling_window('time', iris.analysis.MEAN, 40), label='salinity')
plt.legend(loc=2)
plt.show()

hist_file = '/g/data/r87/dbi599/drstree/CMIP5/GCM/CCCMA/CanESM2/historical/yr/atmos/pe/r1i1p1/pe-global-abs_Ayr_CanESM2_historical_r1i1p1_all.nc'
aa_file = '/g/data/r87/dbi599/drstree/CMIP5/GCM/CCCMA/CanESM2/historicalMisc/yr/atmos/pe/r1i1p4/pe-global-abs_Ayr_CanESM2_historicalMisc_r1i1p4_all.nc'
ghg_file = '/g/data/r87/dbi599/drstree/CMIP5/GCM/CCCMA/CanESM2/historicalGHG/yr/atmos/pe/r1i1p1/pe-global-abs_Ayr_CanESM2_historicalGHG_r1i1p1_all.nc'

hist_cube = iris.load_cube(hist_file, 'precipitation minus evaporation flux')
aa_cube = iris.load_cube(aa_file, 'precipitation minus evaporation flux')
ghg_cube = iris.load_cube(ghg_file, 'precipitation minus evaporation flux')

iplt.plot(hist_cube, label='historical', color='black')
iplt.plot(ghg_cube, label='GHG', color='red')
iplt.plot(aa_cube, label='aa', color='blue')
plt.legend(loc=2)
plt.show()

iplt.plot(hist_cube.rolling_window('time', iris.analysis.MEAN, 10), label='historical', color='black')
iplt.plot(ghg_cube.rolling_window('time', iris.analysis.MEAN, 10), label='GHG', color='red')
iplt.plot(aa_cube.rolling_window('time', iris.analysis.MEAN, 10), label='aa', color='blue')
plt.legend(loc=2)
plt.show()

