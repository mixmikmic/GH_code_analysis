get_ipython().run_line_magic('run', '../../utils/notebook_setup.py')

from datascience import Table
import numpy as np

# custom functions that will help do some simple tasks
from datascience_utils import *
from datascience_stats import *

# Load lahman_teams.csv obtained from the Lahman databank.  We only need a selection of the columns.
lahman = Table.read_table("lahman_teams.csv", usecols=[0, 3, 6] + list(range(14, 28)))

# Restrict to after the year 2000
lahman = lahman.where(lahman['yearID'] >= 2000).copy()

# Need to add two fields, singles and PA (which is only approximate)
lahman['1B'] = lahman['H'] - lahman['2B'] - lahman['3B'] - lahman['HR']
lahman['HBP'] = fill_null(lahman, fill_column='HBP', fill_value=0)
lahman['PA'] = lahman['AB'] + lahman['BB'] + lahman['HBP'] + lahman['SF']

lahman.show(5)

# Batting Average
lahman['BA'] = lahman['H'] / lahman['AB']
# On-Base Percentage
lahman['OBP'] = (lahman['H'] + lahman['BB'] + lahman['HBP']) / lahman['PA']
# Slugging Percentage
lahman['SLG'] = (lahman['1B'] + 2 * lahman['2B'] +
                 3 * lahman['3B'] + 4 * lahman['HR']) / lahman['AB']

lahman.hist('BA', bins=20)

lahman.hist('OBP', bins=20)

lahman.hist('SLG', bins=20)

stats = ['BA', 'OBP', 'SLG']
scatterplot_by_x(lahman, stats, 'R', title='Classical Stats vs Runs')

linear_relationships = {}
errors = {}

linear_fits = Table().with_column('R', lahman['R'])

for i, stat in enumerate(stats):
    # Linear fit
    params, predictions, error = linear_fit(lahman[stat], lahman['R'])
    linear_relationships[stat] = params
    linear_fits = linear_fits.with_column(stat, lahman[stat])
    linear_fits = linear_fits.with_column(stat + '_pred', predictions)
    linear_fits = linear_fits.with_column(stat + '_err', error)

linear_fits.show(10)

correlations = {}

for i, stat in enumerate(stats):
    # Correlation
    correlations[stat] = correlation(lahman[stat], lahman['R'])

stat = 'BA'
linear_fits.scatter(stat, select='R', fit_line=True, color='C0')
linear_fits.scatter(stat, select=stat + '_err', color='C1')

stat = 'OBP'
linear_fits.scatter(stat, select='R', fit_line=True, color='C0')
linear_fits.scatter(stat, select=stat + '_err', color='C1')

stat = 'SLG'
linear_fits.scatter(stat, select='R', fit_line=True, color='C0')
linear_fits.scatter(stat, select=stat + '_err', color='C1')

def stat_summary_print(stat, corr, err_std):
    print(f"Stat: {stat}")
    print("=" * 20)
    print(f"Correlation with Runs: {corr:.3f}")
    print(f"Std dev of errors (in Runs): {err_std:.3f}")
    print()
    
for stat in stats:
    corr = correlations[stat]
    err_std = np.std(linear_fits[stat + '_err'])
    # Print summary
    stat_summary_print(stat, corr, err_std)

rho_obp_ba = correlation(lahman["OBP"], lahman['BA'])
rho_obp_slg = correlation(lahman["OBP"], lahman['SLG'])
rho_ba_slg = correlation(lahman["BA"], lahman['SLG'])

print(f" BA and OBP: {rho_obp_ba:.3f}")
print(f" BA and SLG: {rho_ba_slg:.3f}")
print(f"OBP and SLG: {rho_obp_slg:.3f}")

# Team OPS
lahman['OPS'] = lahman['OBP'] + lahman['SLG']

lahman.hist('OPS', bins=20)

# Team wOBA
lahman['wOBA']= (
    .72 * lahman['BB'] + .75 * lahman['HBP'] + 
    .9 * lahman['1B'] + 1.24 * lahman['2B'] +  
    1.56 * lahman['3B'] + 1.95 * lahman['HR']
) / lahman['PA']

lahman.hist('wOBA', bins=20)

adv_stats = ['OPS', 'wOBA']

for i, stat in enumerate(adv_stats):
    # Linear fit
    params, predictions, error = linear_fit(lahman[stat], lahman['R']) 
    linear_relationships[stat] = params
    correlations[stat] = correlation(lahman[stat], lahman['R'])
    linear_fits = linear_fits.with_column(stat, lahman[stat])
    linear_fits = linear_fits.with_column(stat + '_pred', predictions)
    linear_fits = linear_fits.with_column(stat + '_err', error)

stat = 'OPS'
linear_fits.scatter(stat, select='R', fit_line=True, color='C0')
linear_fits.scatter(stat, select=stat + '_err', color='C1')

stat = 'wOBA'
linear_fits.scatter(stat, select='R', fit_line=True, color='C0')
linear_fits.scatter(stat, select=stat + '_err', color='C1')

for stat in adv_stats:
    corr = correlations[stat]
    err_std = np.std(linear_fits[stat + '_err'])
    # Print summary
    stat_summary_print(stat, corr, err_std)

rho_ops_wOBA = correlation(lahman["OPS"], lahman['wOBA'])
print(f"wOBA and OPS: {rho_ops_wOBA:.3f}")

