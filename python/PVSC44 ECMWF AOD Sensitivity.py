# imports and settings
import os

import h5py
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pvlib
import seaborn as sns
import statsmodels.api as sm
from scipy.optimize import minimize_scalar

from pvsc44_clearsky_aod import ecmwf_macc_tools

get_ipython().magic('matplotlib inline')

sns.set_context('notebook', rc={'figure.figsize': (8, 6)})
sns.set(font_scale=1.5)

# get the "metadata" that contains the station id codes for the SURFRAD data that was analyzed
METADATA = pd.read_csv('metadata.csv', index_col=0)

# load calculations for each station
atm_params_3min_clear = {}
for station_id in METADATA.index:
    with h5py.File('%s_3min_clear_atm_params.h5' % station_id, 'r') as f:
        np_atm_params_3min_clear = pd.DataFrame(np.array(f['data']))
    np_atm_params_3min_clear['index'] = pd.DatetimeIndex(np_atm_params_3min_clear['index'])
    np_atm_params_3min_clear.set_index('index', inplace=True)
    np_atm_params_3min_clear.index.rename('timestamps', inplace=True)
    atm_params_3min_clear[station_id] = np_atm_params_3min_clear

# filter out low light

# CONSTANTS
MODELS = {'solis': 'SOLIS', 'lt': 'Linke', 'macc': 'ECMWF-MACC', 'bird': 'Bird'}
CS = ['dni', 'dhi', 'ghi']
LOW_LIGHT = 200  # threshold for low light in W/m^2

is_bright = {}
for station_id, station_atm_params_3min_clear in atm_params_3min_clear.iteritems():
    is_bright[station_id] = station_atm_params_3min_clear['ghi'] > LOW_LIGHT

TL_SENS = pd.read_csv('TL_sensitivity.csv')
TL_SENS

# compare historic Linke turbidity to calculated
# downsample to monthly averages to show long term trends
f, ax = plt.subplots(2, 4, figsize=(24, 8), sharex=False)
rc = 0
for station_id, station_atm_params_3min_clear in atm_params_3min_clear.iteritems():
    r, c = rc // 4, rc % 4
    station_tl = station_atm_params_3min_clear[['lt', 'lt_calc']][is_bright[station_id]]
    station_tl['lt'].groupby(lambda x: x.month).mean().plot(linewidth=5, ax=ax[r][c])
    station_tl['lt_calc'].groupby(lambda x: x.month).mean().plot(linewidth=5, ax=ax[r][c])
    for y in xrange(2003, 2013):
        lt = station_tl['lt_calc'][('%d-01-01 00:00:00' % y):('%d-12-31 23:59:59' % y)].groupby(lambda x: x.month).mean()
        lt.plot(linestyle=':', ax=ax[r][c])
    ax[r][c].set_ylabel('$T_L$')
    ax[r][c].set_xlabel('month')
    ax[r][c].legend(['static', 'average', 'yearly'])
    ax[r][c].set_title('$T_L$ at %s' % station_id)
    ax[r][c].set_ylim([2, 6])
    rc += 1
ax[1][3].axis('off')
f.tight_layout()
plt.savefig('Linke_turbidity_allyears_monthly.png')

bon2003 = atm_params_3min_clear['bon'][['lt', 'lt_calc']][is_bright['bon']]['2003-01-01 00:00:00':'2003-12-31 23:59:59']
monthly_2003_tl = bon2003.resample('M').mean()
monthly_2003_tl.plot()

mean_2003_tl = monthly_2003_tl.mean()
mean_2003_tl['lt']/mean_2003_tl['lt_calc']

monthly_2003_tl['scaled'] = monthly_2003_tl['lt_calc']*mean_2003_tl['lt']/mean_2003_tl['lt_calc']
monthly_2003_tl.plot()

mean_monthly_2003_tl = monthly_2003_tl['lt'] / monthly_2003_tl['lt_calc']
mean_monthly_2003_tl

atm_params_2003 = atm_params_3min_clear['bon'][['amp', 'pwat', 'tau700', 'lt']][is_bright['bon']]['2003-01-01 00:00:00':'2003-12-31 23:59:59']
def _poop(x, amp=atm_params_2003['amp'], pwat=atm_params_2003['pwat'], bbaod=atm_params_2003['tau700']):
    lt_calc = pvlib.atmosphere.kasten96_lt(amp, pwat, (x * bbaod))
    lt_calc_monthly = lt_calc.resample('M').mean()
    lt_monthly = atm_params_2003['lt'].resample('M').mean()
    return np.sum((lt_calc_monthly - lt_monthly)**2)
res = minimize_scalar(_poop)
res

monthly_2003_tl['scaled_monthly'] = pvlib.atmosphere.kasten96_lt(atm_params_2003['amp'], atm_params_2003['pwat'], res['x']*atm_params_2003['tau700']).resample('M').mean()
monthly_2003_tl.plot()

solis_scaled = pvlib.clearsky.simplified_solis(
   atm_params_3min_clear['bon']['apparent_elevation'],
   atm_params_3min_clear['bon']['tau700']*res['x'],
   atm_params_3min_clear['bon']['pwat'],
   pressure=atm_params_3min_clear['bon']['press'],
   dni_extra=atm_params_3min_clear['bon']['etr']
)
solis_scaled.rename(columns={'ghi': 'scaled_ghi', 'dni': 'scaled_dni', 'dhi': 'scaled_dhi'}, inplace=True)
solis_scaled = pd.concat([solis_scaled, atm_params_3min_clear['bon'][['solis_ghi', 'solis_dni', 'solis_dhi', 'ghi', 'dni', 'dhi']]], axis=1)
solis_scaled['solis_ghi_err'] = solis_scaled['solis_ghi'] - solis_scaled['ghi']
solis_scaled['solis_dni_err'] = solis_scaled['solis_dni'] - solis_scaled['dni']
solis_scaled['solis_dhi_err'] = solis_scaled['solis_dhi'] - solis_scaled['dhi']
solis_scaled['ghi_err'] = solis_scaled['scaled_ghi'] - solis_scaled['ghi']
solis_scaled['dni_err'] = solis_scaled['scaled_dni'] - solis_scaled['dni']
solis_scaled['dhi_err'] = solis_scaled['scaled_dhi'] - solis_scaled['dhi']
solis_scaled['ghi_norm'] = solis_scaled['ghi_err']**2
solis_scaled['dni_norm'] = solis_scaled['dni_err']**2
solis_scaled['dhi_norm'] = solis_scaled['dhi_err']**2
solis_scaled_annual = solis_scaled.resample('A').mean()
solis_scaled_annual['ghi_rel'] = solis_scaled_annual['ghi_err'] / solis_scaled_annual['ghi']
solis_scaled_annual['dni_rel'] = solis_scaled_annual['dni_err'] / solis_scaled_annual['dni']
solis_scaled_annual['dhi_rel'] = solis_scaled_annual['dhi_err'] / solis_scaled_annual['dhi']
solis_scaled_annual['solis_ghi_rel'] = solis_scaled_annual['solis_ghi_err'] / solis_scaled_annual['ghi']
solis_scaled_annual['solis_dni_rel'] = solis_scaled_annual['solis_dni_err'] / solis_scaled_annual['dni']
solis_scaled_annual['solis_dhi_rel'] = solis_scaled_annual['solis_dhi_err'] / solis_scaled_annual['dhi']
solis_scaled_annual[['ghi_rel', 'dni_rel', 'solis_ghi_rel', 'solis_dni_rel']].plot()



