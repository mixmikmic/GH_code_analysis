import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from scipy.special import binom
from numpy.random import choice
from numpy.random import permutation

get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

C = 2
n = 10**5
population = np.arange(C)    # [0, 1] == ['girl', 'boy']
child1 = choice(population, size=n, replace=True)    # the gender of the elder child in each of n families
child2 = choice(population, size=n, replace=True)    # the gender of the younger child in each of n families

n_b = np.sum(child1 == 0)    # N(B): the number of families where the elder is a girl
n_ab = np.sum(np.all([child1 == 0, child2 == 0], axis=0))    # N(A \cap B): the number of families where both childeren are girls and the elder is a girl
print(n_ab / float(n_b))

n_b = np.sum(np.any([child1 == 0, child2 == 0], axis=0))    # N(B): the number of families where at least one of the children is a girl
n_ab = np.sum(np.all([child1 == 0, child2 == 0], axis=0))    # N(A \cap B): the number of families where both childeren are girls and the elder is a girll
print(n_ab / float(n_b))

# Assume the contestant always chooses door 0
C = 3
n = 10**5   # Number of trials
population = np.arange(C)   # [0, 1, 2]
cardoor = choice(population, n, replace=True)
print(np.sum(cardoor == 0) / float(n))   # The fraction of times when the never-switch strategy succeeds

def monty(simulate=True):
    doors = np.arange(3)   # [0, 1, 2]
    # Randomly pick where the car is
    cardoor = choice(doors, 1)[0]
    
    if not simulate:
        # Prompt player - 
        # Receive the player's choice of door (should be 0, 1, or 2)
        chosen = int(input("Monty Hall says 'Pick a door, any door!'"))
    else:
        chosen = 0
    
    # Pick Monty's door (can't be the player's door or the car door)
    if chosen != cardoor:
        montydoor = doors[np.all([doors != chosen, doors != cardoor], axis=0)]
    else:
        montydoor = choice(doors[doors != chosen])
        
    if not simulate:
        # Find out whether the player wants to switch doors
        print('Monty opens door {}!'.format(montydoor))
        reply = str(input('Would you like to switch (y/n)?'))
        
        # Interpret what player wrote as 'yes' if it starts with 'y'
        if reply[0] == 'y':
            chosen = doors[np.all([doors != chosen, doors != montydoor], axis=0)]
    else:
        # FIXME: always change
        chosen = doors[np.all([doors != chosen, doors != montydoor], axis=0)]
    
    # Announce the result of the game!
    if (chosen == cardoor): 
        if not simulate: print('You won!')
        return True
    else:
        if not simulate: print('You lost!')
        return False

n = 10**5   # Number of trials
results = []
for i in range(n):
    results.append(monty(simulate=True))
print(np.sum(results)/float(n))

