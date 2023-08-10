# initialization of the energy and weight vectors
evector = [49.7829, 138.4463, 217.8041, 39.1752, 691.3034, -87.4435, -1053.1789, -272.7384]
weights = [1]*8

# definition of the magnitude of perturbation
pert = .001

# perturbation of the weight vector
def perturb(wvector, index):
    wvector[index] += pert
    for i in list(range(8)):
        if index != i:
            wvector[i] -= pert/7.
            
# calculation of the total energy
def sum_energies(wvector):
    sum_components = 0
    for i in list(range(8)):
        sum_components += evector[i] * wvector[i]
    return sum_components

energy = sum_energies(weights)

# performs perturbation until energy reaches a threshold
while energy > -400:
    # iterates over all elements in the weight vector
    for i in list(range(8)):
        # performs the perturbation at the current element
        perturb(weights, i)
        # if energy decreases, prints the value
        if sum_energies(weights) < energy:
            energy = sum_energies(weights)
            print(energy)


