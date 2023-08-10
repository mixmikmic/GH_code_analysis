import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

import os
import sys

module_path = os.path.abspath(os.path.join('..'))

if module_path not in sys.path:
    sys.path.append(module_path)

from nba.database import Database

plt.rc('font', size=14)        # controls default text sizes
plt.rc('axes', titlesize=16)   # fontsize of the axes title
plt.rc('axes', labelsize=16)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=14)  # fontsize of the tick labels
plt.rc('ytick', labelsize=14)  # fontsize of the tick labels
plt.rc('legend', fontsize=14)  # legend fontsize

conn = sqlite3.connect('../nba.db')

data = pd.read_sql('''
    SELECT SEASON,
           100.0 * SUM(CASE WHEN HOME_WL = 'W' THEN 1 ELSE 0 END) / COUNT(HOME_WL) AS HomeWinPct
    FROM games
    WHERE SEASON >= 1990
    GROUP BY SEASON''', conn)

pct = data.HomeWinPct
plt.figure(figsize=(10, 6))
plt.plot(data.SEASON, pct)
plt.text(1990.5, 83,
         '$\mu=${0:.1f}%\n$\sigma=${1:.1f}%\nRange = {2:.1f}%'
         .format(np.mean(pct), np.std(pct), np.ptp(pct)))
plt.xlabel('Season')
plt.ylabel('NBA Home Team Win Percentage')
plt.title('NBA Home Teams Win Predictably Often')
plt.xlim(1990, 2015)
plt.ylim(0, 100)
plt.show()

database = Database('../nba.db')
season_stats = database.season_stats()

teams = pd.read_sql('SELECT * FROM TEAMS', conn)

off_lim = 114
def_lim = 98
net_lim = 9
off_flag = (season_stats.TEAM_OFF_RTG > off_lim) & (season_stats.SEASON >= 1990)
def_flag = (season_stats.TEAM_DEF_RTG < def_lim) & (season_stats.SEASON >= 1990)
net_flag = (season_stats.TEAM_NET_RTG > net_lim) & (season_stats.SEASON >= 1990)
stats = ['SEASON', 'TEAM_ID', 'TEAM_OFF_RTG', 'TEAM_DEF_RTG', 'TEAM_NET_RTG', 'TEAM_SRS']

# Isolate desired teams
best_off = season_stats.loc[off_flag, stats]

# Merge with teams table to add team information
best_off = teams.merge(best_off, left_on='ID', right_on='TEAM_ID')

# Remove ID columns
best_off = best_off[[c for c in best_off.columns if 'ID' not in c]]

# Sort by descending offensive rating
best_off.sort_values(by='TEAM_OFF_RTG', ascending=False)

# Isolate desired teams
best_def = season_stats.loc[def_flag, stats]

# Merge with teams table to add team information
best_def = teams.merge(best_def, left_on='ID', right_on='TEAM_ID')

# Remove ID columns
best_def = best_def[[c for c in best_def.columns if 'ID' not in c]]

# Sort by ascending defensive rating
best_def.sort_values(by='TEAM_DEF_RTG')

# Isolate desired teams
best_net = season_stats.loc[net_flag, stats]

# Merge with teams table to add team information
best_net = teams.merge(best_net, left_on='ID', right_on='TEAM_ID')

# Remove ID columns
best_net = best_net[[c for c in best_net.columns if 'ID' not in c]]

# Sort by descending net rating
best_net.sort_values(by='TEAM_NET_RTG', ascending=False)

# Create champions table that identifies which team won the NBA championship each season
champs = pd.DataFrame({'SEASON': range(1990, 2016),
                       'ABBREVIATION': ['CHI', 'CHI', 'CHI', 'HOU', 'HOU', 'CHI', 'CHI', 'CHI', 'SAS',
                                        'LAL', 'LAL', 'LAL', 'SAS', 'DET', 'SAS', 'MIA', 'SAS', 'BOS',
                                        'LAL', 'LAL', 'DAL', 'MIA', 'MIA', 'SAS', 'GSW', 'CLE']})
# Isolate desired stats from all teams
champ_stats = season_stats[stats]

# Merge with teams table to add team information
champ_stats = teams.merge(champ_stats, left_on='ID', right_on='TEAM_ID')

# Merge with champs table created above
columns = ['ABBREVIATION', 'SEASON']
champ_stats = champs.merge(champ_stats, left_on=columns, right_on=columns)

# Remove ID columns
champ_stats = champ_stats[[c for c in best_net.columns if 'ID' not in c]]
champ_stats

# Add line of constant net rating
x = np.array([95, 120])
plt.figure(figsize=(10, 8))
plt.plot(x, x-net_lim, 'black', label='Net Rating = %d' % net_lim)

# Isolate teams that do not make the offensive/defensive/net rating cutoffs
others = season_stats.loc[~off_flag & ~def_flag & ~net_flag & (season_stats.SEASON >= 1990), stats]
others = teams.merge(others, left_on='ID', right_on='TEAM_ID')

# Remove the champions with an outer join
columns = ['ABBREVIATION', 'SEASON']
others = champs.merge(others, left_on=columns, right_on=columns, how='outer', indicator=True)
others = others[others._merge == 'right_only']

# Plot data
plt.scatter(x='TEAM_OFF_RTG', y='TEAM_DEF_RTG', data=others, label='Everyone Else')
plt.scatter(x='TEAM_OFF_RTG', y='TEAM_DEF_RTG', data=best_off, label='Off Rating > %d' % off_lim)
plt.scatter(x='TEAM_OFF_RTG', y='TEAM_DEF_RTG', data=best_def, label='Def Rating < %d' % def_lim)
plt.scatter(x='TEAM_OFF_RTG', y='TEAM_DEF_RTG', data=best_net, label='Net Rating > %d' % net_lim)
plt.scatter(x='TEAM_OFF_RTG', y='TEAM_DEF_RTG', data=champ_stats, label='Champions')

# Label axes and add legend
plt.legend()
plt.xlabel('Offensive Rating')
plt.ylabel('Defensive Rating')
plt.xlim(90, 120)
plt.ylim(90, 120)
plt.show()

srs = season_stats['TEAM_SRS']
(mu, sigma) = norm.fit(srs)

plt.figure(figsize=(12, 6))
ax = sns.distplot(srs, fit=norm, kde=True, kde_kws={'label': 'KDE', 'color': 'green', 'linewidth': 3},
                  hist_kws={'label': 'SRS', 'edgecolor': 'k', 'linewidth': 2},
                  fit_kws={'label': 'Normal Dist ($\mu=${0:.2g}, $\sigma=${1:.2f})'.format(mu, sigma),
                           'linewidth': 3})
ax.legend()
plt.xlabel('SRS')
plt.ylabel('Frequency')
plt.title('Most NBA Teams are Average')
plt.xlim(-20, 20)
plt.ylim(0, 0.1)
plt.show()

seasons = season_stats.filter(regex='SEASON|TEAM')
games = pd.read_sql('SELECT * FROM games', conn)
games = games.merge(seasons, left_on=['SEASON', 'HOME_TEAM_ID'], right_on=['SEASON', 'TEAM_ID'])
games = games.merge(seasons, left_on=['SEASON', 'AWAY_TEAM_ID'], right_on=['SEASON', 'TEAM_ID'], 
                    suffixes=('', '_AWAY'))

ax = sns.jointplot(x='TEAM_SRS', y='TEAM_SRS_AWAY', data=games, kind='kde',
                   shade_lowest=False, stat_func=None, xlim=(-15, 15), ylim=(-15, 15), size=8)
ax.set_axis_labels(xlabel='Home Team SRS', ylabel='Away Team SRS')
plt.show()

def kde(data, stat, label, title, ax):
    stat = 'TEAM_' + stat
    sns.kdeplot(data[stat], data[stat + '_AWAY'], cmap='Blues', shade=True, shade_lowest=False, ax=ax)
    ax.plot(0, 0, 'or')
    ax.set_xlabel('Home Team ' + label)
    ax.set_ylabel('Away Team ' + label)
    ax.set_title(title)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)

plt.figure(figsize=(14, 6))

# Find games where the home team won
kde(games[games.HOME_WL=='W'], 'SRS', 'SRS', 'KDE of Home Team Wins', plt.subplot(121))

# Find games where the home team lost
kde(games[games.HOME_WL=='L'], 'SRS', 'SRS', 'KDE of Home Team Losses', plt.subplot(122))

plt.show()

plt.figure(figsize=(14, 6))

# Find games where the home team won
kde(games[games.HOME_WL=='W'], 'NET_RTG', 'Net Rating', 'KDE of Home Team Wins', plt.subplot(121))

# Find games where the home team lost
kde(games[games.HOME_WL=='L'], 'NET_RTG', 'Net Rating', 'KDE of Home Team Losses', plt.subplot(122))

plt.show()

