get_ipython().magic('matplotlib inline')

# Standard imports.
import numpy as np
import pylab
import scipy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Resize plots.
pylab.rcParams['figure.figsize'] = 8, 4

# Simulate 1000 series
game_lengths = []
for i in range(10000):
    wins_a = 0
    wins_b = 0
    for j in range(7):
        winning_team = np.random.rand() > .5
        if winning_team:
            wins_b += 1
        else:
            wins_a += 1
    
        if wins_a >= 4 or wins_b >= 4:
            break
    game_lengths.append(j + 1)
    continue
        
game_lengths = np.array(game_lengths)
plt.hist(game_lengths)
_ = plt.title('Game lengths under null hypothesis')
plt.xlabel('Game lengths')

print game_lengths.mean()

game_lengths_historical = np.hstack(([4] * 8, [5] * 16, [6] * 24, [7] * 19))
plt.hist(game_lengths_historical)
_ = plt.title('Historical game lengths')

print game_lengths_historical.mean()

dfs = pd.read_html('http://www.oddsshark.com/nba/nba-finals-historical-series-odds-list')
df = dfs[0]

df.columns = pd.Index(['year', 'west', 'west_moneyline', 'east', 'east_moneyline'])
df

def moneyline_to_odds(val):
    if val < 0:
        return -val / (-val + 100.0)
    else:
        return 100 / (val + 100.0)

mean_odds = (df.west_moneyline.map(moneyline_to_odds) - 
             df.east_moneyline.map(moneyline_to_odds)
             + 1) / 2.0

plt.hist(mean_odds, np.arange(21) / 20.0)
plt.title('Implied west conference winning odds')

# Remove the West conference bias by flipping the sign of the odds at random.
alphas = np.zeros(25)
for n in range(25):
    flip_bits = np.random.rand(mean_odds.size) > .5
    mean_odds_shuffled = mean_odds * flip_bits + (1 - flip_bits) * (1 - mean_odds)
    alpha, beta, _, _ = scipy.stats.beta.fit(mean_odds_shuffled)
    # Symmetrize the result to have a mean of .5
    alphas[n] = ((alpha + beta)/2)

alpha = np.array(alphas).mean()
plt.hist(mean_odds_shuffled, np.arange(21) / 20.0)
x = np.arange(101) / 100.0
y = scipy.stats.beta.pdf(x, 1 + alpha, 1 + alpha) * 1.0
plt.plot(x, y, 'r-')
plt.legend(['observations', 'Fitted distribution'])
plt.title('Implied winning odds (east/west randomized)')

game_win_percentages = np.arange(.5, 1, .01)
series_win_percentages = np.zeros_like(game_win_percentages)

Nsims = 4000
for k, frac in enumerate(game_win_percentages):
    wins = 0
    for i in range(Nsims):
        wins_a = 0
        wins_b = 0
        for j in range(7):
            winning_team = np.random.rand() > frac
            if winning_team:
                wins_b += 1
            else:
                wins_a += 1
                
            if wins_a == 4 or wins_b == 4:
                break
        wins += (wins_a == 4) * 1
    series_win_percentages[k] = wins / float(Nsims)
    

def logit(p):
    return np.log(p / (1 - p))

def logistic(x):
    """Returns the logistic of the numeric argument to the function.""" 
    return 1 / (1 + np.exp(-x))

plt.plot(game_win_percentages, series_win_percentages)

# Fit a logistic function by eye balling.
plt.plot(game_win_percentages, logistic(2.3*logit(game_win_percentages)))
plt.xlabel('game win percentage')
plt.ylabel('series win percentage')
plt.legend(['empirical', 'curve fit'])

def inverse_map(y):
    x = logistic(1 / 2.3 * logit(y))
    return x
assert np.allclose(inverse_map(logistic(2.3*logit(game_win_percentages))),
                   game_win_percentages)

plt.hist(np.random.beta(alpha + 1, alpha + 1, 1000))

# Simulate 1000 series
game_lengths = []
for i in range(10000):
    # Pick a per series win percentage
    series_win_percentage = np.random.beta(alpha + 1, alpha + 1)
    # Transform to a per-game win percentage
    game_win_percentage = inverse_map(series_win_percentage)
    wins_a = 0
    wins_b = 0
    for j in range(7):
        winning_team = np.random.rand() > game_win_percentage
        if winning_team:
            wins_b += 1
        else:
            wins_a += 1
    
        if wins_a >= 4 or wins_b >= 4:
            break
    game_lengths.append(j + 1)
    continue
        
game_lengths = np.array(game_lengths)
plt.hist(game_lengths)
_ = plt.title('Game lengths under (more sophisticated) null hypothesis')
plt.xlabel('Game length')

game_lengths.mean()

m = game_lengths_historical.mean()
ci = game_lengths_historical.std() / np.sqrt(float(game_lengths_historical.size))

print "Simulated series length is %.2f" % game_lengths.mean()
print "95%% CI for observed series length is [%.2f, %.2f]" % (m - 1.96*ci, m + 1.96*ci)

plt.subplot(121)
plt.hist(np.random.binomial(4, 0.5, size = (10000)), np.arange(6) - .5)
plt.title('Games won by team A\nno home team advantage')
plt.ylim([0, 4500])

plt.subplot(122)
plt.hist(np.random.binomial(2, 0.7, size = (10000)) + 
         np.random.binomial(2, 0.3, size = (10000)), np.arange(6) - .5)
plt.title('Games won by team A\nwith home team advantage')
plt.ylim([0, 4500])

