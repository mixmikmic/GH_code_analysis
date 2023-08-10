from pycav.quantum import SpinSystem
import numpy as np
import matplotlib.pyplot as plt

# create a triangle of spins (1/2, 1/2, 1)
spins = [0.5, 0.5, 1.0]
couplings = [[0,1,1],[1,0,1],[1,1,0]]
spin_triangle = SpinSystem(spins, couplings)

# calculate the energy levels
spin_triangle.get_energies()
print("Energy levels of the spin triangle")
print(spin_triangle.energies)

# alternatively, find the multiplicities
# and non-repeated energies
spin_triangle.count_multiplicities()
print(spin_triangle.multiplicities)
print(spin_triangle.reduced_energies)

# create the familiar spin-1/2 pair and find the energy levels
spins = [0.5, 0.5]
couplings = [[0,1],[1,0]]

spin_pair1 = SpinSystem(spins, couplings)
spin_pair1.get_energies()
print("Energy levels with no magnetic field")
print(spin_pair1.energies)

# then apply a magnetic field in the x-direction to both spins
B= [[0.1,0.0,0.0], [0.1,0.0,0.0]]
spin_pair2 = SpinSystem(spins, couplings, B_field = B)
spin_pair2.get_energies()
print("Energy level in a homogeneous magnetic field")
print(spin_pair2.energies)

N = 6
spins = [0.5 for i in range(N)]

#coupling between adjacent spins only
couplings = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        if abs(i-j) == 1 or abs(i-j) == N-1:
            couplings[i,j] = 1.0

#coupling between x components only            
scaling = [1.0, 0.0, 0.0]

spin_ring = SpinSystem(spins, couplings, scaling = scaling)
spin_ring.count_multiplicities()
print("Multiplicity, Energy")
for i in range(len(spin_ring.multiplicities)):
    print(spin_ring.multiplicities[i], ", %.2f" % spin_ring.reduced_energies[i])


from bokeh.plotting import figure, show, output_notebook
output_notebook()

angular_momenta = [0.5, 1.0]
couplings = [[0,-1],[-1,0]]

B_increment = 0.04
fig = figure(title = "Hydrogen 2p Splitting", x_axis_label = "Magnetic Field", y_axis_label = "Energy")
for i in range(100):
    B = np.array([0,0,0.05*i])
    pair = SpinSystem(angular_momenta, couplings, B_field = [2*B,B])
    pair.count_multiplicities()
    fig.scatter([i*B_increment for energy in pair.reduced_energies], pair.reduced_energies, color = 'indigo')

show(fig)

