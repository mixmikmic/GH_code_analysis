import numpy as np
import random as random

# returns a vector that has all the energy terms 
def energy_terms():
    # initial values of the energy terms
    bond = 49.7829
    angle = 138.4463
    dihed = 217.8041
    nb = 39.1752
    eel = 695.8385
    vdw = -83.5197
    el = -1162.1836
    egb = -162.3717
    
    return np.array([bond, angle, dihed, nb, eel, vdw, el, egb])

# https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html
    
print(sum(energy_terms()))

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

print(sum(weights()))

def total_energy():    
    return wb*bond + wa*angle + wd*dihed + wn*nb + wee*eel + wv*vdw + wel*el + weg*egb

print(total_energy())



# random initialization of an attempt at the solution, an "individual"
individual = []

# total amount of weight left to be redistributed 
total_weight_left = float(len(weights()))
print(total_weight_left)

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
print(individual)
print(sum(individual))

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

"""matrix multiplication to computer total_energy"""

np.array(individual)

# calculate dot product of the arrays
total_energy = np.dot(individual, energy_terms)

print(total_energy)

print_weights_of(individual)

# repeat this, but with weights all equal to 1 just to verfiy that it's working
print(np.dot(np.ones(len(energy_terms)), energy_terms))

population = []

# random initialization of an attempt at the solution, an "individual"
individual = []

# total amount of weight left to be redistributed 
total_weight_left = float(len(weights()))
# print(total_weight_left)

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
print_weights_of(individual)

# print(individual)
# print(sum(individual))

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

"""matrix multiplication to computer total_energy"""

np.array(individual)

# calculate dot product of the arrays
total_energy = np.dot(individual, energy_terms)

# original weights of 1
original_energy = np.dot(np.ones(len(energy_terms)), energy_terms)

# print sum of all the weights in this attempt at the solution, notice it is constant
print(sum(individual))
print(individual)
print(original_energy)
print(total_energy)
# the objective is to return the most negative total_energy

# original weights of 1
original_energy = np.dot(np.ones(len(energy_terms)), energy_terms)
last_energy = original_energy
end = 100

for a in range(end):

    population = []

    # random initialization of an attempt at the solution, an "individual"
    individual = []

    # total amount of weight left to be redistributed 
    total_weight_left = float(len(weights()))
    # print(total_weight_left)

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

    """matrix multiplication to computer total_energy"""

    np.array(individual)

    # calculate dot product of the arrays
    total_energy = np.dot(individual, energy_terms)

    if total_energy < last_energy:
        last_energy = total_energy
        best_weights = individual
    
print(last_energy)
print(best_weights)
print_weights_of(best_weights)
print('This results in the following energy', last_energy)

import matplotlib.pyplot as plt
import numpy as np

# original weights of 1
original_energy = np.dot(np.ones(len(energy_terms)), energy_terms)
last_energy = original_energy
end = 100

list_total_energy = []

for a in range(end):

    population = []

    # random initialization of an attempt at the solution, an "individual"
    individual = []

    # total amount of weight left to be redistributed 
    total_weight_left = float(len(weights()))
    # print(total_weight_left)

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

    """matrix multiplication to computer total_energy"""

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

print_weights_of(best_weights)
print('This results in the following energy', last_energy)

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

# # initialization of the energy and weight vectors
# evector = [49.7829, 138.4463, 217.8041, 39.1752, 691.3034, -87.4435, -1053.1789, -272.7384]
# weights = [1]*8

# # definition of the magnitude of perturbation
# pert = .001

# # perturbation of the weight vector
# def perturb(wvector, index):
#     wvector[index] += pert
#     for i in list(range(8)):
#         if index != i:
#             wvector[i] -= pert/7.
            
# # calculation of the total energy
# def sum_energies(wvector):
#     sum_components = 0
#     for i in list(range(8)):
#         sum_components += evector[i] * wvector[i]
#     return sum_components

# energy = sum_energies(weights)

# # performs perturbation until energy reaches a threshold
# while energy > -400:
#     # iterates over all elements in the weight vector
#     for i in list(range(8)):
#         # performs the perturbation at the current element
#         perturb(weights, i)
#         # if energy decreases, prints the value
#         if sum_energies(weights) < energy:
#             energy = sum_energies(weights)
#             print(energy)

