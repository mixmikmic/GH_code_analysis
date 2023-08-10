import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pystan

get_ipython().magic('matplotlib inline')

def stan_map(vector):
    unique_items = np.unique(vector)
    number_of_unique_items = len(unique_items)
    return dict(zip(unique_items, range(1, number_of_unique_items + 1)))

data = (
    pd.read_csv('data/2006-07.csv')
    .assign(goal_difference=lambda df: df['FTHG'] - df['FTAG'])
    .rename(columns={'HomeTeam': 'home_team', 'AwayTeam': 'away_team'})
    [['home_team', 'away_team', 'goal_difference']]
)

team_map = stan_map(data['home_team'])
data['home_team_id'] = data['home_team'].replace(team_map)
data['away_team_id'] = data['away_team'].replace(team_map)

data.head()

model_data = {
    'n_teams': len(data['home_team_id'].unique()),
    'n_games': len(data['goal_difference']),
    'home_team': data['home_team_id'],
    'away_team': data['away_team_id'],
    'goal_difference': data['goal_difference']
}

get_ipython().run_cell_magic('time', '', "model = pystan.StanModel('model.stan')\n\nfit = model.sampling(\n    data=model_data,\n    iter=1000,\n    chains=4\n)\n\noutput = fit.extract()")

# Get final league table order to reproduce the paper's plot
data['home_points'] = np.where(data['goal_difference'] > 0,
                               3, np.where(data['goal_difference'] == 0, 1, 0))
data['away_points'] = np.where(data['goal_difference'] < 0,
                               3, np.where(data['goal_difference'] == 0, 1, 0))

home_points = (
    data.groupby('home_team', as_index=False)
    .agg({'home_points': np.sum})
    .rename(columns={'home_team': 'team'})
)
away_points = (
    data.groupby('away_team', as_index=False)
    .agg({'away_points': np.sum})
    .rename(columns={'away_team': 'team'})
)

total_points = pd.merge(
    home_points, away_points
)
total_points['total_points'] = total_points['home_points'] + total_points['away_points']
total_points = total_points.sort_values(by='total_points', ascending=False).reset_index(drop=True)

# Reverse the order for plotting (lowest at the bottom)
ordered_teams = total_points['team'][::-1]

# Map ids : names for parsing Stan output
reverse_map = {v: k for k, v in team_map.items()}

def plot_coefficients(data, ordered_teams, title, alpha=0.05, axes_colour='black'):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    y_lookup = {i: team for i, team in enumerate(ordered_teams)}
    
    upper = 1 - (alpha / 2)
    lower = 0 + (alpha / 2)

    for i, team in y_lookup.items():
        x_mean = np.mean(data[team])
        x_lower = np.percentile(data[team], lower * 100)
        x_upper = np.percentile(data[team], upper * 100)
        
        ax.scatter(x_mean, i, alpha=1, color='black', s=25)
        ax.hlines(i, x_lower, x_upper, color='black')

    ax.set_ylim([-1, len(ordered_teams)])
    ax.set_yticks(list(y_lookup.keys()))
    ax.set_yticklabels(list(y_lookup.values()))

    # Add title
    fig.suptitle(title, ha='left', x=0.125, fontsize=18, color='k')

    # Change axes colour
    ax.spines["bottom"].set_color(axes_colour)
    ax.spines["left"].set_color(axes_colour)
    ax.tick_params(colors=axes_colour)

    # Remove top and bottom spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    
    return fig

reverse_map = {v: k for k, v in team_map.items()}

offense = pd.DataFrame(output['offense'])
offense.columns = [reverse_map[id_ + 1] for id_ in offense.columns]

fig_offense = plot_coefficients(offense, ordered_teams, 'Offense')
fig_offense.savefig('figures/offense.png')

defense = pd.DataFrame(output['defense']) * -1
defense.columns = [reverse_map[id_ + 1] for id_ in defense.columns]

fig_defense = plot_coefficients(defense, ordered_teams, 'Defense')
fig_defense.savefig('figures/defense.png')



