get_ipython().magic('matplotlib inline')
import itertools
import random

import matplotlib.pyplot as plt
import axelrod as axl
plt.rcParams['figure.figsize'] = (10, 10)

# Create a population of size 30
N = 20
players = []
for _ in range(N):
    player = random.choice(axl.basic_strategies)
    players.append(player())

# Run the process. Eventually there will be only
# one player type left.
mp = axl.MoranProcess(players=players, turns=200)
mp.play()
print("The winner is:", mp.winning_strategy_name)

# Plot the results

player_names = mp.populations[0].keys()

plot_data = []
labels = []
for name in player_names:
    labels.append(name)
    values = [counter[name] for counter in mp.populations]
    plot_data.append(values)
    domain = range(len(values))

plt.stackplot(domain, plot_data, labels=labels)
plt.legend()
plt.xlabel("Rounds")
plt.ylabel("Number of Individuals")
plt.show()

# Create a population of size 30
N = 25
players = []
for _ in range(N):
    player = random.choice([axl.TitForTat, axl.Cooperator, axl.Defector])
    players.append(player())

rounds = 1000
mp = axl.MoranProcess(players=players, turns=200, mutation_rate=0.05)
list(itertools.islice(mp, rounds))
print("Completed {} rounds.".format(rounds))

# Plot the results

player_names = mp.populations[0].keys()

plot_data = []
labels = []
for name in player_names:
    labels.append(name)
    values = [counter[name] for counter in mp.populations]
    plot_data.append(values)
    domain = range(len(values))

plt.stackplot(domain, plot_data, labels=labels)
plt.legend()
plt.xlabel("Rounds")
plt.ylabel("Number of Individuals")
plt.show()



