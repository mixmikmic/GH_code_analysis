from __future__ import division
from pandas import concat, read_csv, to_datetime
from ggplot import *
from sklearn import linear_model
import pandas as pd
import numpy as np
from numpy import floor, histogram
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn import linear_model, svm

teams = read_csv('./data/2015/team.csv', index_col=0)
statistics = read_csv('./data/2015/team-game-statistics.csv', index_col=0)
games = teams.join(statistics)

for column in games:
    if 'Unnamed' in column:
        del games[column]

games.Date = to_datetime(games.Date, format='%Y%m%d')

winning_scores = games.groupby('Game Code')['Points'].max()
losing_scores = games.groupby('Game Code')['Points'].min()
scores = pd.DataFrame(data={'Losing Points': losing_scores.values, 'Winning Points': winning_scores.values}, index=winning_scores.index)
games = games.join(scores, on='Game Code')

def is_win(game):
    return game['Points'] > game['Losing Points']

def margin(game):
    if is_win(game):
        return game['Points'] - game['Losing Points']
    else:
        return game['Points'] - game['Winning Points']

games['Margin'] = games.apply(margin, axis=1)
games['Is Win'] = games.apply(is_win, axis=1)

def range_bin(array, step):
    return range(int(array.min() / step) * step, int(array.max() / step) * step + step, step)

def histogram_random(data, bins, samples):
    hist, bins = np.histogram(data, bins=bins*2)

    bin_midpoints = bins[:-1] + np.diff(bins)/2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    values = np.random.rand(samples)
    value_bins = np.searchsorted(cdf, values)
    return bin_midpoints[value_bins]

def kde_random(data, samples):
    def kde(x, x_grid):
        kde = gaussian_kde(x)
        return kde.evaluate(x_grid)
    try:
        x_grid = np.linspace(min(data), max(data), samples)
        pdf = kde(data, x_grid)
        cdf = np.cumsum(pdf)
        cdf = cdf / cdf[-1]
        values = np.random.rand(samples)
        value_bins = np.searchsorted(cdf, values)
        random_from_cdf = x_grid[value_bins]
        return random_from_cdf
    except:
        return [0] * samples
    
def centiles(outcomes):
    return outcomes.quantile([x / 100 for x in range(0, 100, 10)])

feature_columns = ['Rush Att', 'Rush Yard',
       'Rush TD', 'Pass Att', 'Pass Comp', 'Pass Yard', 'Pass TD',
       'Pass Int', 'Pass Conv', 'Kickoff Ret', 'Kickoff Ret Yard',
       'Kickoff Ret TD', 'Punt Ret', 'Punt Ret Yard', 'Punt Ret TD',
       'Fum Ret', 'Fum Ret Yard', 'Fum Ret TD', 'Int Ret', 'Int Ret Yard',
       'Int Ret TD', 'Misc Ret', 'Misc Ret Yard', 'Misc Ret TD',
       'Field Goal Att', 'Field Goal Made', 'Off XP Kick Att',
       'Off XP Kick Made', 'Off 2XP Att', 'Off 2XP Made', 'Def 2XP Att',
       'Def 2XP Made', 'Safety', 'Punt', 'Punt Yard',
       'Kickoff', 'Kickoff Yard', 'Kickoff Touchback',
       'Kickoff Out-Of-Bounds', 'Kickoff Onside', 'Fumble', 'Fumble Lost',
       'Tackle Solo', 'Tackle Assist', 'Tackle For Loss',
       'Tackle For Loss Yard', 'Sack', 'Sack Yard', 'QB Hurry',
       'Fumble Forced', 'Pass Broken Up', 'Kick/Punt Blocked',
       '1st Down Rush', '1st Down Pass', '1st Down Penalty',
       'Time Of Possession', 'Penalty', 'Penalty Yard', 'Third Down Att',
       'Third Down Conv', 'Fourth Down Att', 'Fourth Down Conv',
       'Red Zone Att', 'Red Zone TD', 'Red Zone Field Goal',
       'First Down Total']

def point_estimator(team):
    clf = linear_model.Lasso()
    train = team[feature_columns]
    model = clf.fit(train, team['Points'])
    return model

def margin_estimator(team):
    clf = linear_model.Lasso()
    train = team[feature_columns]
    model = clf.fit(train, team['Margin'])
    return model

def predict_scores(team, model, iterations, debug=False):
    simulations = pd.DataFrame([kde_random(team[feature], iterations) for feature in feature_columns]).transpose()

    if debug:
        coefficients = pd.DataFrame(model.coef_, feature_columns)
        print coefficients[abs(coefficients[0]) > 0]

    predicted_scores = model.predict(simulations)
    return [max([0, score]) for score in predicted_scores]

def measure_accuracy(spread, projected_winner, actual_winner, correct, against_spread):
    if projected_winner == actual_winner[0]:
        correct += 1
        if actual_winner[3] < 0 and spread < actual_winner[3]:
            against_spread += 1
        elif actual_winner[3] > 0:
            against_spread += 1

    return (correct, against_spread)

k = 10000

bowls = [
    ('Arizona', 'New Mexico'),
    ('Utah', 'BYU'),
    ('Appalachian State', 'Ohio'),
    ('San Jose State', 'Georgia State'),
    ('Louisiana Tech', 'Arkansas State'),
    ('Western Kentucky', 'South Florida'),
    ('Akron', 'Utah State'),
    ('Toledo', 'Temple'),
    ('Boise State', 'Northern Illinois'),
    ('Georgia Southern', 'Bowling Green'),
    ('Western Michigan', 'Middle Tennessee'),
    ('San Diego State', 'Cincinnati'),
    ('Marshall', 'Connecticut'),
    ('Washington State', 'Miami (Florida)'),
    ('Washington', 'Southern Mississippi'),
    ('Duke', 'Indiana'),
    ('Virginia Tech', 'Tulsa'),
    ('Nebraska', 'UCLA'),
    ('Pittsburgh', 'Navy'),
    ('Central Michigan', 'Minnesota'),
    ('Air Force', 'California'),
    ('North Carolina', 'Baylor'),
    ('Nevada', 'Colorado State'),
    ('Texas Tech', 'LSU'),
    ('Memphis', 'Auburn'),
    ('Mississippi State', 'North Carolina State'),
    ('Louisville', 'Texas A&M'),
    ('Wisconsin', 'USC'),
    ('Houston', 'Florida State'),
    ('Clemson', 'Oklahoma'),
    ('Alabama', 'Michigan State'),
    ('Northwestern', 'Tennessee'),
    ('Notre Dame', 'Ohio State'),
    ('Michigan', 'Florida'),
    ('Iowa', 'Stanford'),
    ('Oklahoma State', 'Mississippi'),
    ('Penn State', 'Georgia'),
    ('Kansas State', 'Arkansas'),
    ('Oregon', 'TCU'),
    ('West Virginia', 'Arizona State'),
    ('Clemson', 'Alabama')
]

winners = [
    ('Arizona', 45, 37, -12.0),
    ('Utah', 35, 28, -3.0),
    ('Appalachian State', 31, 29, -9.5),
    ('San Jose State', 27, 16, -5.0),
    ('Louisiana Tech', 47, 28, -1.5),
    ('Western Kentucky', 45, 35, -3.5),
    ('Akron', 23, 21, 7.0),
    ('Toledo', 32, 17, 1.0),
    ('Boise State', 55, 7, -8.5),
    ('Georgia Southern', 58, 27, 7.5),
    ('Western Michigan', 45, 31, -3.5),
    ('San Diego State', 42, 7, 2.0),
    ('Marshall', 16, 10, -4.0),
    ('Washington State', 20, 14, -3.0),
    ('Washington', 44, 31, -8.5),
    ('Duke', 44, 41, 2.0),
    ('Virginia Tech', 55, 52, -14.0),
    ('Nebraska', 37, 29, 7.0),
    ('Navy', 44, 28, -5.0),
    ('Minnesota', 21, 14, -6.0),
    ('California', 55, 36, -7.0),
    ('Baylor', 49, 38, -2.5),
    ('Nevada', 28, 23, 3.5),
    ('LSU', 56, 27, -7.0),
    ('Auburn', 31, 10, -2.5),
    ('Mississippi State', 51, 28, -7.0),
    ('Louisville', 27, 21, 3.0),
    ('Wisconsin', 23, 21, 3.0),
    ('Houston', 38, 24, 6.5),
    ('Clemson', 37, 17, 3.5),
    ('Alabama', 38, 0, -9.5),
    ('Tennessee', 45, 6, -9.0),
    ('Ohio State', 44, 28, -6.5),
    ('Michigan', 41, 7, -4.0),
    ('Stanford', 45, 16, -6.5),
    ('Mississippi', 48, 20, -6.5),
    ('Georgia', 24, 17, -7.0),
    ('Arkansas', 45, 23, -12.0),
    ('TCU', 47, 41, -1.5),
    ('West Virginia', 43, 42, 1.5),
    ('Alabama', 45, 40, -7.0)
]

correct = 0
against_spread = 0

for i, bowl in enumerate(bowls):
    winner = bowl[0]
    team1 = games[games.Name == bowl[0]]
    team2 = games[games.Name == bowl[1]]

    outcome = pd.DataFrame([predict_scores(team1, point_estimator(team1), k), predict_scores(team2, point_estimator(team2), k)]).transpose()
    
    team1_probability = len(outcome[outcome[0] > outcome[1]]) / k
    team2_probability = len(outcome[outcome[1] > outcome[0]]) / k
    spreads = centiles(outcome[0] - outcome[1])

    if team1_probability > team2_probability:
        spreads = centiles(outcome[1] - outcome[0])
        print ', '.join([str(x) for x in [bowl[0], team1_probability, bowl[1], team2_probability]])
    else:
        winner = bowl[1]
        print ', '.join([str(x) for x in [bowl[1], team2_probability, bowl[0], team1_probability]])
    spread = spreads.iloc[5]
    print spread
    
    if i < len(winners):
        actual_winner = winners[i]
        correct, against_spread = measure_accuracy(spread, winner, actual_winner, correct, against_spread)

print 'Accuracy:', correct / len(winners)
print 'Against Spread:', against_spread / len(winners)

correct = 0
against_spread = 0

for i, bowl in enumerate(bowls):
    winner = bowl[0]
    team1 = games[games.Name == bowl[0]]
    team2 = games[games.Name == bowl[1]]

    outcome = pd.DataFrame([predict_scores(team1, margin_estimator(team1), k), predict_scores(team2, margin_estimator(team2), k)]).transpose()
    
    team1_probability = len(outcome[outcome[0] >= outcome[1]]) / k
    team2_probability = len(outcome[outcome[1] >= outcome[0]]) / k
    spreads = centiles(outcome[0])

    if team1_probability > team2_probability:
        spreads = centiles(outcome[1])
        print ', '.join([str(x) for x in [bowl[0], team1_probability, bowl[1], team2_probability]])
    else:
        winner = bowl[1]
        print ', '.join([str(x) for x in [bowl[1], team2_probability, bowl[0], team1_probability]])

    spread = spreads.iloc[5] * -1
    print spread
    
    if i < len(winners):
        actual_winner = winners[i]
        correct, against_spread = measure_accuracy(spread, winner, actual_winner, correct, against_spread)

print 'Accuracy:', correct / len(winners)
print 'Against Spread:', against_spread / len(winners)



