import matplotlib.pyplot as plt
import numpy as np
import random as random

# initialize all the weights to 1
wb = 1
wa = 1
wd = 1
wn = 1
wee = 1
wv = 1
wel = 1
weg = 1

weights = [wb,wa,wd,wn,wee,wv,wel,weg]

# returns a vector describing all of the weights
def weights():
    return [wb,wa,wd,wn,wee,wv,wel,weg]

"""initialize energy terms"""

bond = 49.7829
angle = 138.4463
dihed = 217.8041
nb = 39.1752
eel = 695.8385
vdw = -83.5197
el = -1162.1836
egb = -162.3717

energy_terms = np.array([[bond], [angle], [dihed], [nb], [eel], [vdw], [el], [egb]])

# original weights of 1
original_energy = np.dot(np.ones(len(energy_terms)), energy_terms)
print('The original energy before applying the genetic algorithm was as follows', original_energy) 

last_energy = original_energy
end = 1000
list_total_energy = []

for a in range(end):

    population = []

    # random initialization of an attempt at the solution, an "individual"
    individual = []

    # total amount of weight left to be redistributed 
    total_weight_left = float(len(weights()))

    # distribute random weights to an individual, while keeping total weight constant
    for weight in (range(len(weights())-1)):    
        distribute = random.uniform(0,1) * total_weight_left
        total_weight_left -= distribute
        new_weight = distribute
        individual.append(new_weight)

    # add what's left to be distributed, as the final weight 
    individual.append(total_weight_left)

    # total weight is unchanged 
    # weights have been randomly distributed

    # randomly shuffle the weights in the list
    random.shuffle(individual)

    """matrix multiplication to compute total_energy"""

    np.array(individual)

    # calculate dot product of the arrays
    total_energy = np.dot(individual, energy_terms)

    if total_energy < last_energy:
        last_energy = total_energy
        best_weights = individual
        
    list_total_energy.append(last_energy)       

# plots the performance of the G.A.
plt.plot(range(end), list_total_energy)
plt.suptitle('Performance of Genetic Algorithm')
plt.xlabel('Iteration')
plt.ylabel('Energy')
plt.show()

def print_weights_of(weights):
    print()
    print("Here are the weights for the following energy terms...")
    print()
    print(" bond =", weights[0])
    print(" angle =", weights[1])
    print(" dihed =", weights[2])
    print(" nb =", weights[3])
    print(" eel =", weights[4])
    print(" vdw =", weights[5])
    print(" el =", weights[6])
    print(" egb =", weights[7])
    print()
    
print_weights_of(best_weights)
print('This results in the following energy', last_energy)

