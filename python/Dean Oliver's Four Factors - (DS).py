get_ipython().run_line_magic('run', '../../utils/notebook_setup.py')

get_ipython().run_line_magic('matplotlib', 'inline')
import datascience as ds
import numpy as np
from scipy.stats import zscore
from datascience_helpers import correlation, linear_fit
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
plt.style.use('fivethirtyeight')

teams = ds.Table.read_table('team_season_data.csv', sep=',')

teams.show(5)

teams['efg'] = (teams['fg'] + .5 * teams['fg3']) / teams['fga']
teams['to'] = teams['tov'] / (teams['tov'] + teams['fga'] + .44 * teams['fta'])
teams['oreb'] = teams['orb'] / (teams['orb'] + teams['opp_drb'])
teams['ftr'] = teams['ft'] / teams['fga']

teams['opp_efg'] = (teams['opp_fg'] + .5 * teams['opp_fg3']) / teams['opp_fga']
teams['opp_to'] = teams['opp_tov'] /     (teams['opp_tov'] + teams['opp_fga'] + .44 * teams['opp_fta'])
teams['opp_oreb'] = teams['opp_orb'] / (teams['opp_orb'] + teams['drb'])
teams['opp_ftr'] = teams['opp_ft'] / teams['opp_fga']

teams['eFG'] = zscore(teams['efg'] - teams['opp_efg'])
teams['Tov'] = zscore(teams['to'] - teams['opp_to'])
teams['Reb'] = zscore(teams['oreb'] - teams['opp_oreb'])
teams['Ftr'] = zscore(teams['ftr'] - teams['opp_ftr'])

teams['eFG_diff'] = teams['efg'] - teams['opp_efg']
teams.hist(['eFG_diff', 'eFG'], overlay=False, bins=50)

print(f"""
Compare Std Deviations for eFG:
Original: {np.std(teams['eFG_diff']):.03f}
Z-Scored: {np.std(teams['eFG']):.03f}
""")

teams['Four Factors'] = .4 * teams['eFG'] - .25 * teams['Tov'] +     .20 * teams['Reb'] + .15 * teams['Ftr']

teams['win_pct'] = teams['wins'] / (teams['wins'] + teams['losses'])

teams['pr'] = teams['off_rtg'] / teams['def_rtg']
teams['log_pr'] = np.log(teams['pr'])

teams.scatter('Four Factors', 'win_pct')

teams.scatter('Four Factors', 'log_pr')

t = teams.select('win_pct', 'log_pr', 'eFG', 'Tov', 'Reb', 'Ftr', 'Four Factors')
t.to_df().corr()

scatter_matrix(t.to_df(), figsize=(10, 10));

params, predictions, errors = linear_fit(
    teams['Four Factors'], teams['log_pr'], constant=False)

beta = params[0]
print("Computed Linear Fit:")
print("====================")
s = f"xLogPR = {beta:.3f} * FourFactorModel"
print(s)

teams.scatter('Four Factors', 'log_pr', fit_line=True)

teams['four_factors_errors'] = errors

corr = correlation(teams['four_factors_errors'], teams['log_pr'])
print(f"""
Correlation of errors and log PR: {corr:.03f}
""")
teams.scatter('log_pr', 'four_factors_errors', fit_line=True)

games = ds.Table.read_table('four_factor_game_data.csv', sep=',')
games.show(5)

games['eFG'] = zscore(games['EFG_PCT'] - games['OPP_EFG_PCT'])
games['Tov'] = zscore(games['TOV_PCT'] - games['OPP_TOV_PCT'])
games['Reb'] = zscore(games['OREB_PCT'] - games['OPP_OREB_PCT'])
games['Ftr'] = zscore(games['FTA_RATE'] - games['OPP_FTA_RATE'])

games['Four Factors'] = .4 * games['eFG'] - .25 * games['Tov'] +     .20 * games['Reb'] + .15 * games['Ftr']

games['LOG_PR'] = np.log(games['OFF_RATING'] / games['DEF_RATING'])

games.scatter('Four Factors', 'LOG_PR')

t = games.select('LOG_PR', 'eFG', 'Tov', 'Reb', 'Ftr', 'Four Factors')
t.to_df().corr()

scatter_matrix(t.to_df(), figsize=(10, 10));

params, predictions, errors = linear_fit(
    games['Four Factors'], games['LOG_PR'], constant=False)

beta = params[0]
print("Computed Linear Fit:")
print("====================")
s = f"xLogPR = {beta:.3f} * FourFactorModel"
print(s)

games.scatter('Four Factors', 'LOG_PR', fit_line=True)

games['four_factors_errors'] = errors

corr = correlation(errors, games['LOG_PR'])
print(f"""
Correlation of errors and log PR: {corr:.03f}
""")
games.scatter('LOG_PR', 'four_factors_errors', fit_line=True)

