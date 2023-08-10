get_ipython().run_line_magic('run', '../../utils/notebook_setup.py')

import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
mpl.rcParams['figure.facecolor'] = (0.941, 0.941, 0.941, 1.0)

import pandas as pd
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from datascience_helpers import correlation, linear_fit
from pandas.tools.plotting import scatter_matrix

pd.set_option('precision', 2)

teams = pd.read_csv('team_season_data.csv')

teams.head()

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

fig, ax = plt.subplots(nrows=2)
tmp = teams['efg'] - teams['opp_efg']
tmp.plot.hist(ax=ax[0], bins=50)
teams['eFG'].plot.hist(ax=ax[1], bins=50)

print(f"""
Compare Std Deviations for eFG:
Original: {np.std(tmp):.03f}
Z-Scored: {np.std(teams['eFG']):.03f}
""")

teams['Four Factors'] = .4 * teams['eFG'] - .25 * teams['Tov'] +     .20 * teams['Reb'] + .15 * teams['Ftr']

teams['win_pct'] = teams['wins'] / (teams['wins'] + teams['losses'])

teams['pr'] = teams['off_rtg'] / teams['def_rtg']
teams['log_pr'] = np.log(teams['pr'])

fig, ax = plt.subplots()
teams.plot.scatter('Four Factors', 'win_pct', ax=ax)
ax.set_ylim(0, 1)
ax.set_xlim(-2, 2);

fig, ax = plt.subplots()
teams.plot.scatter('Four Factors', 'log_pr', ax=ax)
ax.set_ylim(-0.175, 0.175)
ax.set_xlim(-2, 2);

df = teams[['win_pct', 'log_pr', 'eFG', 'Tov', 'Reb', 'Ftr', 'Four Factors']]
df.corr()

scatter_matrix(df, figsize=(10, 10));

params, predictions, errors = linear_fit(
    teams['Four Factors'], teams['log_pr'], constant=False)

beta = params['Four Factors']
print("Computed Linear Fit:")
print("====================")
s = f"xLogPR = {beta:.3f} * FourFactorModel"
print(s)

fig, ax = plt.subplots()
teams.plot.scatter('Four Factors', 'log_pr', ax=ax)

teams['xLogPR'] = predictions
ax.plot(teams['Four Factors'], predictions, color='C1', label='xLogPR')

ax.legend()
ax.set_ylim(-0.175, 0.175)
ax.set_xlim(-2, 2);

corr = correlation(errors, teams['log_pr'])
print(f"""
Correlation of errors and log PR: {corr:.03f}
""")
fig, ax = plt.subplots()
ax.plot(teams['log_pr'], errors, '.')
ax.set_xlabel('log_pr')
ax.set_ylabel('error');

games = pd.read_csv('four_factor_game_data.csv')
games.head()

games['eFG'] = zscore(games['EFG_PCT'] - games['OPP_EFG_PCT'])
games['Tov'] = zscore(games['TOV_PCT'] - games['OPP_TOV_PCT'])
games['Reb'] = zscore(games['OREB_PCT'] - games['OPP_OREB_PCT'])
games['Ftr'] = zscore(games['FTA_RATE'] - games['OPP_FTA_RATE'])

games['Four Factors'] = .4 * games['eFG'] - .25 * games['Tov'] +     .20 * games['Reb'] + .15 * games['Ftr']

games['LOG_PR'] = np.log(games['OFF_RATING'] / games['DEF_RATING'])

fig, ax = plt.subplots()
games.plot.scatter('Four Factors', 'LOG_PR', ax=ax)
ax.set_ylim(-0.6, 0.6)
ax.set_xlim(-1.75, 1.75)
ax.set_yticks(np.arange(-.45, .46, .15))
ax.set_xticks(np.arange(-1.5, 1.6, .5));

df = games[['LOG_PR', 'eFG', 'Tov', 'Reb', 'Ftr', 'Four Factors']]
df.corr()

scatter_matrix(df, figsize=(10, 10));

params, predictions, errors = linear_fit(
    games['Four Factors'], games['LOG_PR'], constant=False)

beta = params['Four Factors']
print("Computed Linear Fit:")
print("====================")
s = f"xLogPR = {beta:.3f} * FourFactorModel"
print(s)

fig, ax = plt.subplots()
games.plot.scatter('Four Factors', 'LOG_PR', ax=ax)

games['xLogPR'] = predictions
ax.plot(games['Four Factors'], predictions, color='C1', label='xLogPR')

ax.legend()
ax.set_ylim(-0.6, 0.6)
ax.set_xlim(-1.75, 1.75)
ax.set_yticks(np.arange(-.45, .46, .15))
ax.set_xticks(np.arange(-1.5, 1.6, .5));

corr = correlation(errors, games['LOG_PR'])
print(f"""
Correlation of errors and log PR: {corr:.03f}
""")
fig, ax = plt.subplots()
ax.plot(games['LOG_PR'], errors, '.')
ax.set_xlabel('LOG_PR')
ax.set_ylabel('error');

