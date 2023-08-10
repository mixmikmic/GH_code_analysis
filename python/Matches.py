# Python 3
get_ipython().magic('matplotlib inline')

import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import axelrod as axl

plt.rcParams['figure.figsize'] = (10, 10)

# Two Zero-Determinant Strategies
player1 = axl.ZDGTFT2()
player2 = axl.ZDSet2()

matches = 1000
turns = 100

scores = []
for i in range(matches):
    match = axl.Match((player1, player2), turns)
    results = match.play()
    scores.append(list(zip(*match.scores())))

scores1, scores2 = zip(*scores)

sns.tsplot(scores1)
sns.tsplot(scores2, color="y")
sns.plt.xlabel("Turns")
sns.plt.ylabel("Mean Scores per turn")

# Two Zero-Determinant Strategies
player1 = axl.ZDGTFT2()
player2 = axl.ZDSet2()

scores = []

for i in range(matches):
    match = axl.Match((player1, player2), turns)
    results = match.play()
    # Sum the scores from the match for each player
    scores.append(np.sum(match.scores(), axis=0) / float(turns))    

df = pd.DataFrame(scores)
df.columns = ["Player1", "Player2"]
df.mean()

plt.violinplot(df[["Player1", "Player2"]].as_matrix(), showmedians=True)
plt.xticks([1,2], [str(player1), str(player2)])
plt.ylabel("Mean Scores over all matches")
plt.xlabel("Players")

# Two typically deterministic strategies
player1 = axl.OmegaTFT()
player2 = axl.TitForTat()

scores = []
for i in range(matches):
    match = axl.Match((player1, player2), turns, noise=0.05) # 5% noise
    results = match.play()
    scores.append(list(zip(*match.scores())))

scores1, scores2 = zip(*scores)

sns.tsplot(scores1)
sns.tsplot(scores2, color="y")
sns.plt.xlabel("Turns")
sns.plt.ylabel("Mean Scores per turn")

# Two typically deterministic strategies
player1 = axl.WinStayLoseShift()
player2 = axl.WinStayLoseShift()

scores = []
for i in range(matches):
    match = axl.Match((player1, player2), turns, noise=0.01) # 5% noise
    results = match.play()
    scores.append(list(zip(*match.scores())))

scores1, scores2 = zip(*scores)

sns.tsplot(scores1)
sns.tsplot(scores2, color="y")
sns.plt.xlabel("Turns")
sns.plt.ylabel("Mean Scores per turn")

