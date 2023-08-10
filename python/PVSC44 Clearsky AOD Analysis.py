# imports and settings
import os

import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import seaborn as sns
import statsmodels.api as sm

from pvsc44_clearsky_aod import ecmwf_macc_tools

get_ipython().magic('matplotlib inline')

sns.set_context('notebook', rc={'figure.figsize': (8, 6)})
sns.set(font_scale=1.5)

# get the "metadata" that contains the timezone, latitude, longitude, elevation and station id for SURFRAD
METADATA = pd.read_csv('metadata.csv', index_col=0)

# get SURFRAD data
station_id = 'bon'  # choose a station ID from the metadata list

# CONSTANTS
DATADIR = 'surfrad'  # folder where SURFRAD data is stored relative to this file
NA_VAL = '-9999.9'  # value SURFRAD uses for missing data
USECOLS = [0, 6, 7, 8, 9, 10, 11, 12]  # omit 1:year, 2:month, 3:day, 4:hour, 5:minute and 13:jday columns

# read 2003 data for the station, use first column as index, parse index as dates, and replace missing values with NaN
DATA = pd.read_csv(
    os.path.join(DATADIR, '%s03.csv' % station_id),
    index_col=0, usecols=USECOLS, na_values=NA_VAL, parse_dates=True
)

# append the data from 2003 to 2012 which coincides with the ECMWF data available
for y in xrange(2004, 2013):
    DATA = DATA.append(pd.read_csv(
        os.path.join(DATADIR, '%s%s.csv' % (station_id, str(y)[-2:])),
        index_col=0, usecols=USECOLS, na_values=NA_VAL, parse_dates=True
    ))

DATA['press'] = DATA['press'] * 100.0  # convert mbars to Pascals
DATA.index.rename('timestamps', inplace=True)  # rename the index to "timestamps"
DATA = DATA.tz_localize('UTC')  # set timezone to UTC

# plot some temperatures
DATA['ta']['2006-07-16T00:00:00':'2006-07-16T23:59:59'].tz_convert(METADATA['tz'][station_id]).plot()
plt.ylabel('ambient temperature $[\degree C]$')

# plot some irradiance
DATA[['dni', 'dhi', 'ghi']]['2006-07-16T00:00:00':'2006-07-16T23:59:59'].tz_convert(METADATA['tz'][station_id]).plot()
plt.ylabel('irradiance $[W/m^2]$')

# get AOD and water vapor for the SURFRAD station
ATMOS = ecmwf_macc_tools.Atmosphere()  # get the downloaded ECMWF-MACC data for the world

# create a datetime series for the ECMWF-MACC data, it's available from 2003 to 2012 in 3 hour increment
# pandas creates timestamps at the *beginning* of the each interval,shifting the timestamps from the end
# to match the SURFRAD data
TIMES = pd.DatetimeIndex(start='2003-01-01T00:00:00', freq='3H', end='2012-12-31T23:59:59').tz_localize('UTC')

# get the atmospheric data for the SURFRAD station as a pandas dataframe, and append some addition useful calculations
# like AOD at some other wavelengths (380-nm, 500-nm, and 700nm) and convert precipitable water to cm.
STATION_ATM = pd.DataFrame(
    ATMOS.get_all_data(METADATA['latitude'][station_id], METADATA['longitude'][station_id]), index=TIMES
)

# plot some of the atmospheric data
atm_data = STATION_ATM[['tau380', 'tau500', 'aod550', 'tau700', 'aod1240']]['2006-07-13':'2006-07-17']
atm_data['2006-07-16T00:00:00':'2006-07-16T23:59:59'].tz_convert(METADATA['tz'][station_id]).plot()
plt.title('Atmospheric Data')
plt.ylabel('aerosol optical depth')
plt.savefig('%s_AOD_2006-07-16.png' % station_id)

# solarposition, airmass, pressure-corrected & water vapor
SOLPOS = pvlib.solarposition.get_solarposition(
    time=DATA.index,
    latitude=METADATA['latitude'][station_id],
    longitude=METADATA['longitude'][station_id],
    altitude=METADATA['elevation'][station_id],
    pressure=DATA['press'],
    temperature=DATA['ta']
)
AIRMASS = pvlib.atmosphere.relativeairmass(SOLPOS.apparent_zenith)  # relative airmass
AM_PRESS = pvlib.atmosphere.absoluteairmass(AIRMASS, pressure=DATA['press'])  # pressure corrected airmass
PWAT_CALC = pvlib.atmosphere.gueymard94_pw(DATA['ta'], DATA['rh'])  # estimate of atmospheric water vapor in cm
ETR = pvlib.irradiance.extraradiation(DATA.index)  # extra-terrestrial total solar radiation

# append these to the SOLPOS dataframe to keep them together more easily
SOLPOS.insert(6, 'am', AIRMASS)
SOLPOS.insert(7, 'amp', AM_PRESS)
SOLPOS.insert(8, 'pwat_calc', PWAT_CALC)
SOLPOS.insert(9, 'etr', ETR)

# plot solar position for a day
SOLPOS[['apparent_zenith', 'azimuth']]['2006-07-16T00:00:00':'2006-07-16T23:59:59'].tz_convert(METADATA['tz'][station_id]).plot()
plt.ylabel('solar position $[\deg]$')

# compare calculated solar position to SURFRAD
(DATA['solzen'] / SOLPOS['apparent_zenith'] - 1)['2006-07-16T00:00:00':'2006-07-16T23:59:59'].tz_convert(METADATA['tz'][station_id]).plot()

# concatenate atmospheric parameters so they're the same size
# pd.concat fills missing timestamps from low frequency datasets with NaN
atm_params = pd.concat([DATA, STATION_ATM, SOLPOS], axis=1)

# then fill in NaN in atmospheric data by padding with the previous value
atm_params['alpha'].fillna(method='pad', inplace=True)
atm_params['aod1240'].fillna(method='pad', inplace=True)
atm_params['aod550'].fillna(method='pad', inplace=True)
atm_params['pwat'].fillna(method='pad', inplace=True)
atm_params['tau380'].fillna(method='pad', inplace=True)
atm_params['tau500'].fillna(method='pad', inplace=True)
atm_params['tau700'].fillna(method='pad', inplace=True)
atm_params['tcwv'].fillna(method='pad', inplace=True)

# compare measured and calculated precipitable water
# Pad the ECMWF-MACC data since it's at 3-hour intervals, but SURFRAD is at 1-minute or 3-minute intervals.
# Then rollup daily averages to make long term trends easier to see.
pwat = pd.concat([atm_params['pwat'], atm_params['pwat_calc']], axis=1).resample('D').mean()
pwat['2006-01-01':'2006-12-31'].tz_convert(METADATA['tz'][station_id]).plot()
plt.title('Comparison of measured and calculated $P_{wat}$.')
plt.ylabel('Precipitable Water [cm]')
plt.legend(['Measured', 'Calculated'])

# lookup Linke turbidity
# use 3-hour intervals to speed up lookup
LT = pvlib.clearsky.lookup_linke_turbidity(
    time=TIMES,
    latitude=METADATA['latitude'][station_id],
    longitude=METADATA['longitude'][station_id],
)

# calculate Linke turbidity using Kasten pyrheliometric formula
LT_CALC = pvlib.atmosphere.kasten96_lt(
    atm_params['amp'], atm_params['pwat'], atm_params['tau700']
)

# calculate broadband AOD using Bird & Hulstrom approximation
AOD_CALC = pvlib.atmosphere.bird_hulstrom80_aod_bb(atm_params['tau380'], atm_params['tau500'])

# recalculate Linke turbidity using Bird & Hulstrom broadband AOD
LT_AOD_CALC = pvlib.atmosphere.kasten96_lt(
    atm_params['amp'], atm_params['pwat'], AOD_CALC
)

# insert Linke turbidity to atmospheric parameters table
atm_params.insert(25, 'lt', LT)
atm_params.insert(26, 'lt_calc', LT_CALC)
atm_params.insert(27, 'lt_aod_calc', LT_AOD_CALC)
atm_params.insert(28, 'aod_calc', AOD_CALC)

# Linke turbidity should be continuous, it only depends on time
# fill in Linke turbidity by padding previous values
atm_params['lt'].fillna(method='pad', inplace=True)

# compare Bird & Hulstrom approximated broadband AOD to AOD at 700-nm
# downsample to monthly intervals to show long term trend for entire range of ECMWF-MACC data
aod_bb = atm_params[['tau700', 'aod_calc']].resample('D').mean()
aod_bb['2006-01-01':'2006-12-31'].tz_convert(METADATA['tz'][station_id]).plot()
plt.ylabel('broadband AOD')
plt.title('Comparison of calculated and measured broadband AOD')

# compare historic Linke turbidity to calculated
# downsample to monthly averages to show long term trends
atm_params[['lt', 'lt_calc', 'lt_aod_calc']].resample('M').mean().tz_convert(METADATA['tz'][station_id]).plot()
plt.ylabel('Linke turbidity')
plt.title('Comparison of measured and historical Linke turbidity')

# compare historic Linke turbidity to calculated
# downsample to monthly averages to show long term trends
lt = atm_params[['lt', 'lt_calc', 'lt_aod_calc']].resample('M').mean()
lt['2003-01-01':'2003-12-31'].tz_convert(METADATA['tz'][station_id]).plot()
plt.ylabel('Linke turbidity')
plt.title('Comparison of measured and historical Linke turbidity')
plt.savefig('%s_Linke_turbidity_2003_monthly.png' % station_id)

# compare historic Linke turbidity to calculated
# downsample to monthly averages to show long term trends
atm_params['lt'].tz_convert(METADATA['tz'][station_id]).groupby(lambda x: x.month).mean().plot(linewidth=5)
atm_params['lt_calc'].tz_convert(METADATA['tz'][station_id]).groupby(lambda x: x.month).mean().plot(linewidth=5)
for y in xrange(2003, 2013):
    lt = atm_params['lt_calc'][('%d-01-01' % y):('%d-12-31' % y)].tz_convert(METADATA['tz'][station_id]).groupby(lambda x: x.month).mean()
    lt.plot(linestyle=':')
plt.ylabel('Linke turbidity')
plt.xlabel('month')
plt.legend(['static', 'average', 'years'])
plt.title('Comparison of measured and historical Linke turbidity')
plt.savefig('%s_Linke_turbidity_allyears_monthly.png' % station_id)

# how is atmosphere changing over time?
# Calculate the linear regression of the relative difference between historical and measured Linke turbidity
lt_diff = (atm_params['lt_calc'] / atm_params['lt']).resample('A').mean() - 1.0  # yearly differences
x = np.arange(lt_diff.size)  # years
x = sm.add_constant(x)  # add y-intercept
y = lt_diff.values  # numpy array of yearly relative differences
results = sm.OLS(y, x).fit()  # fit linear regression
results.summary()  # output summary

# plot yearly relative difference versus trendline
plt.plot(lt_diff)
plt.plot(lt_diff.index, results.fittedvalues)
plt.title('Yearly atmospheric trend between calculated and historic Linke turbidity')
plt.legend(['yearly relative differnce', 'trendline'])
plt.ylabel('relative difference')
plt.xlabel('years')
plt.savefig('%s_Linke_turbidity_trend.png' % station_id)

# calculate clear sky
INEICHEN_LT = pvlib.clearsky.ineichen(
    atm_params['apparent_zenith'], atm_params['amp'], atm_params['lt'], altitude=METADATA['elevation'][station_id],
    dni_extra=atm_params['etr']
)
INEICHEN_CALC = pvlib.clearsky.ineichen(
    atm_params['apparent_zenith'], atm_params['amp'], atm_params['lt_calc'], altitude=METADATA['elevation'][station_id],
    dni_extra=atm_params['etr']
)
INEICHEN_AOD_CALC = pvlib.clearsky.ineichen(
    atm_params['apparent_zenith'], atm_params['amp'], atm_params['lt_aod_calc'], altitude=METADATA['elevation'][station_id],
    dni_extra=atm_params['etr']
)
SOLIS = pvlib.clearsky.simplified_solis(
    atm_params['apparent_elevation'],
    atm_params['tau700'],
    atm_params['pwat'],
    pressure=atm_params['press'],
    dni_extra=atm_params['etr']
)
BIRD = pvlib.clearsky.bird(
    atm_params['apparent_zenith'],
    atm_params['am'],
    atm_params['tau380'],
    atm_params['tau500'],
    atm_params['pwat'],
    pressure=atm_params['press'],
    dni_extra=atm_params['etr']
)

# plot direct normal irradiance
LEGEND = ['Ineichen-Linke', 'Ineichen-ECMWF-MACC', 'Ineichen-Bird-Hulstrom', 'SOLIS', 'BIRD', 'SURFRAD']
pd.concat([
    INEICHEN_LT['dni'], INEICHEN_CALC['dni'], INEICHEN_AOD_CALC['dni'],
    SOLIS['dni'], BIRD['dni'], atm_params['dni']
], axis=1)['2006-07-16T00:00:00':'2006-07-16T23:59:59'].resample('H').mean().tz_convert(METADATA['tz'][station_id]).plot()
plt.legend(LEGEND)
plt.title('Comparison of clear sky irradiance at %s 2003-2012' % METADATA['station name'][station_id])
plt.ylabel('$DNI [W/m^2]$')
plt.savefig('%s_DNI_2006-07-16.png' % station_id)

# plot diffuse irradiance
pd.concat([
    INEICHEN_LT['dhi'], INEICHEN_CALC['dhi'], INEICHEN_AOD_CALC['dhi'],
    SOLIS['dhi'], BIRD['dhi'], atm_params['dhi']
], axis=1)['2006-07-16T00:00:00':'2006-07-16T23:59:59'].resample('H').mean().tz_convert(METADATA['tz'][station_id]).plot()
plt.legend(LEGEND)
plt.title('Comparison of clear sky irradiance at %s 2003-2012' % METADATA['station name'][station_id])
plt.ylabel('$DHI [W/m^2]$')
plt.savefig('%s_DHI_2006-07-16.png' % station_id)

# plot global horizontal irradiance
pd.concat([
    INEICHEN_LT['ghi'], INEICHEN_CALC['ghi'], INEICHEN_AOD_CALC['ghi'],
    SOLIS['ghi'], BIRD['ghi'], atm_params['ghi']
], axis=1)['2006-07-16T00:00:00':'2006-07-16T23:59:59'].resample('H').mean().tz_convert(METADATA['tz'][station_id]).plot()
plt.legend(LEGEND)
plt.title('Comparison of clear sky irradiance at %s 2003-2012' % METADATA['station name'][station_id])
plt.ylabel('$GHI [W/m^2]$')
plt.savefig('%s_GHI_2006-07-16.png' % station_id)

# add clear sky calc to atmospheric parameters
atm_params.insert(29, 'solis_dhi', SOLIS['dhi'])
atm_params.insert(30, 'solis_dni', SOLIS['dni'])
atm_params.insert(31, 'solis_ghi', SOLIS['ghi'])
atm_params.insert(32, 'lt_dhi', INEICHEN_LT['dhi'])
atm_params.insert(33, 'lt_dni', INEICHEN_LT['dni'])
atm_params.insert(34, 'lt_ghi', INEICHEN_LT['ghi'])
atm_params.insert(35, 'macc_dhi', INEICHEN_CALC['dhi'])
atm_params.insert(36, 'macc_dni', INEICHEN_CALC['dni'])
atm_params.insert(37, 'macc_ghi', INEICHEN_CALC['ghi'])
atm_params.insert(38, 'bird_dhi', BIRD['dhi'])
atm_params.insert(39, 'bird_dni', BIRD['dni'])
atm_params.insert(40, 'bird_ghi', BIRD['ghi'])

# make even intervals at 1-minute and 3-minutes
atm_params_3min = atm_params.resample('3T').mean() # down using averages
atm_params_1min = atm_params.resample('T').pad() # up padding previous value

# add columns for year and month for pivot and groupby plots
atm_params_3min.insert(0, 'month', atm_params_3min.index.month)
atm_params_3min.insert(0, 'year', atm_params_3min.index.year)

# detect clear sky indices

# 1-minute data with 10-minute window
is_clear_1min = pvlib.clearsky.detect_clearsky(atm_params_1min['ghi'], atm_params_1min['solis_ghi'], atm_params_1min.index, 10)
# 3-minute data with 30 minute window
is_clear_3min = pvlib.clearsky.detect_clearsky(atm_params_3min['ghi'], atm_params_3min['solis_ghi'], atm_params_3min.index, 10)

# filter out low light
is_bright = atm_params_3min['ghi'] > 200

# save data
np_atm_params_3min_clear = atm_params_3min.loc[is_clear_3min].to_records()
np_atm_params_3min_clear.index = np.array([dt.isoformat() for dt in np_atm_params_3min_clear.index.tolist()], dtype=str)
np_atm_params_3min_clear = np_atm_params_3min_clear.astype(np.dtype(
    [('index', str, 25)]
    + [(str(n), dt) for n, dt in np_atm_params_3min_clear.dtype.descr if n != 'index']
))
with h5py.File('%s_3min_clear_atm_params.h5' % station_id, 'w') as f:
    f['data'] = np_atm_params_3min_clear

# check hdf5 file loads back into pandas
with h5py.File('%s_3min_clear_atm_params.h5' % station_id, 'r') as f:
    np_atm_params_3min_clear = pd.DataFrame(np.array(f['data']))
np_atm_params_3min_clear['index'] = pd.DatetimeIndex(np_atm_params_3min_clear['index'])
np_atm_params_3min_clear.set_index('index', inplace=True)
np_atm_params_3min_clear.index.rename('timestamps', inplace=True)
np_atm_params_3min_clear

