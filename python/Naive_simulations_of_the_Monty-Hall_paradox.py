import random

M = 3
allocation = [False] * (M - 1) + [True]  # Only 1 treasure!
assert set(allocation) == {True, False}  # Check: only True and False
assert sum(allocation) == 1              # Check: only 1 treasure!

allocation

def randomAllocation():
    r = allocation[:]
    random.shuffle(r)
    return r

for _ in range(10):
    print(randomAllocation())

def last(r, i):
    # Select a random index corresponding of the door we keep
    if r[i]:  # She found the treasure, returning a random last door
        return random.choice([j for (j, v) in enumerate(r) if j != i])
    else:     # She didn't find the treasure, returning the treasure door
        # Indeed, the game only removes door that don't contain the treasure
        return random.choice([j for (j, v) in enumerate(r) if j != i and v])

for _ in range(10):
    r = randomAllocation()
    i = random.randint(0, M - 1)
    j = last(r, i)
    print("- r =", r, "i =", i, "and last(r, i) =", j)
    print("  Stay on", r[i], "or go to", r[j], "?")

def firstChoice():
    global M
    # Uniform first choice
    return random.randint(0, M - 1)

def simulate(stayOrNot):
    # Random spot for the treasure
    r = randomAllocation()
    # Initial choice
    i = firstChoice()
    # Which door are remove, or equivalently which is the last one to be there?
    j = last(r, i)
    assert {r[i], r[j]} == {False, True}  # There is still the treasure and only one
    stay = stayOrNot()
    if stay:
        return r[i]
    else:
        return r[j]

N = 10000

def simulateManyGames(stayOrNot):
    global N
    results = [simulate(stayOrNot) for _ in range(N)]
    return sum(results)

def keep():
    return True  # True == also stay on our first choice

rate = simulateManyGames(keep)
print("- For", N, "simulations, the strategy 'keep' has won", rate, "of the trials...")
print("  ==> proportion = {:.2%}.".format(rate / float(N)))

def change():
    return False  # False == never stay, ie always chose the last door

rate = simulateManyGames(change)
print("- For", N, "simulations, the strategy 'change' has won", rate, " of the trials...")
print("  ==> proportion = {:.2%}.".format(rate / float(N)))

def bernoulli(p=0.5):
    return random.random() < p

rate = simulateManyGames(bernoulli)
print("- For", N, "simulations, the strategy 'bernoulli' has won", rate, " of the trials...")
print("  ==> proportion = {:.2%}.".format(rate / float(N)))

import numpy as np
import matplotlib.pyplot as plt

values_p = np.linspace(0, 1, 500)

def stratBernoulli(p):
    def stayOrNot():
        return bernoulli(p=p)
    return stayOrNot

chance_of_winning = [simulateManyGames(stratBernoulli(p)) / float(N) for p in values_p]

plt.figure()
plt.plot(values_p, chance_of_winning, 'r')
plt.title("Monty-Hall paradox with {} doors ({} random simulation)".format(M, N))
plt.xlabel("Probability $p$ of staying on our first choice (Bernoulli strategy)")
plt.ylabel("Probability of winning")
plt.ylim(0, 1)
plt.yticks(np.linspace(0, 1, 11))
plt.show()

def completeSimu():
    global M
    global N
    allocation = [False] * (M - 1) + [True]  # Only 1 treasure!
    
    def randomAllocation():
        r = allocation[:]
        random.shuffle(r)
        return r
    
    def last(r, i):
        # Select a random index corresponding of the door we keep
        if r[i]:  # She found the treasure, returning a random last door
            return random.choice([j for (j, v) in enumerate(r) if j != i])
        else:     # She didn't find the treasure, returning the treasure door
            # Indeed, the game only removes door that don't contain the treasure
            return random.choice([j for (j, v) in enumerate(r) if j != i and v])
    
    def simulate(stayOrNot):
        # Random spot for the treasure
        r = randomAllocation()
        # Initial choice
        i = firstChoice()
        # Which door are remove, or equivalently which is the last one to be there?
        j = last(r, i)
        stay = stayOrNot()
        if stay:
            return r[i]
        else:
            return r[j]

    def simulateManyGames(stayOrNot):
        global N
        results = [simulate(stayOrNot) for _ in range(N)]
        return sum(results)

    values_p = np.linspace(0, 1, 300)
    chance_of_winning = [simulateManyGames(stratBernoulli(p)) / float(N) for p in values_p]
    plt.figure()
    plt.plot(values_p, chance_of_winning, 'r')
    plt.title("Monty-Hall paradox with {} doors ({} random simulation)".format(M, N))
    plt.xlabel("Probability $p$ of staying on our first choice (Bernoulli strategy)")
    plt.ylabel("Probability of winning")
    plt.ylim(0, 1)
    plt.yticks(np.linspace(0, 1, 11))
    plt.show()

M = 4
completeSimu()

M = 100
completeSimu()

