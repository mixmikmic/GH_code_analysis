import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib notebook')

gas1 = ct.Solution('gri30.xml')

gas1.TP = 300, 101325   # [K], [Pa]
phi = 1
gas1.set_equivalence_ratio(phi,'CH4','O2:1,N2:3.76')

width = 0.03  # domain size [m], need to increase for lower pressures and decrease for higher pressures
loglevel = 1  # amount of diagnostic output (0 to 8)

# Set up flame object
f = ct.FreeFlame(gas1, width=width)
f.set_refine_criteria(ratio=3, slope=0.06, curve=0.12)
f.show_solution()

# Solve with mixture-averaged transport model
f.transport_model = 'Mix'
f.solve(loglevel=loglevel, auto=True)
f.show_solution()
print('Initial baseline mixture-averaged flamespeed = {0:7f} m/s'.format(f.u[0]))

fig, ax1 = plt.subplots() # intialize a plot on axes 'ax1' in figure 'fig'

# now, plot the species mole fractions as a function of space:
ax1.plot(f.grid, f.X[gas1.species_index('CH4')], 'b-', label='CH4')
ax1.plot(f.grid, f.X[gas1.species_index('O2')], 'b--', label='O2')
ax1.plot(f.grid, f.X[gas1.species_index('CO')], 'k-', label='CO')
ax1.plot(f.grid, f.X[gas1.species_index('H2')], 'k--', label='H2')
ax1.plot(f.grid, f.X[gas1.species_index('CO2')], 'g-', label='CO2')
ax1.plot(f.grid, f.X[gas1.species_index('H2O')], 'g--', label='H2O')

ax1.set_ylabel('Mole Fraction', color='k')
ax1.set_xlabel('Distance [m]')
ax1.tick_params('y', colors='k')

ax2 = ax1.twinx()
ax2.plot(f.grid, f.T, 'r-')
ax2.set_ylabel('Temperature [K]', color='r')
ax2.tick_params('y', colors='r')

plt.xlim((0.009,0.012))
fig.tight_layout()

ax1.legend() # add a legend
plt.show() # display the plot

# Solve with multi-component transport properties
f.transport_model = 'Multi'
f.solve(loglevel) # don't use 'auto' on subsequent solves
f.show_solution()
print('Initial baseline multicomponent flamespeed = {0:7f} m/s'.format(f.u[0]))

# calculate sensitivities
sens1 = f.get_flame_speed_reaction_sensitivities()

# print the reaction number followed by the associated sensitivity 
print()
print('Rxn #   k/S*dS/dk    Reaction Equation')
print('-----   ----------   ----------------------------------')
for m in range(gas1.n_reactions):
    print('{: 5d}   {: 10.3e}   {}'.format(
          m, sens1[m], gas1.reaction_equation(m)))
    

# use argsort to obtain an array of the *indicies* of the values of the sens1 array sorted by absolute value 
newOrder = np.argsort(np.abs(sens1))
# argsort ranks from small to large, but we want large to small, so we flip this around 
newOrder = newOrder[::-1] 

# make some storage variables so that we can plot the results later in a bar graph
newOrder2 = np.zeros(len(newOrder))
sens2 = np.zeros(len(newOrder))
reactionList = []

# using the same method above, print the sensitivties but call the new indicies that we defined 
print()
print('Rxn #   k/S*dS/dk    Reaction Equation')
print('-----   ----------   ----------------------------------')
for ii in range(gas1.n_reactions):
    print('{: 5d}   {: 10.3e}   {}'.format(
          newOrder[ii], sens1[newOrder[ii]], gas1.reaction_equation(newOrder[ii])))
    # assign new variables values for plot use
    newOrder2[ii] = ii
    sens2[ii] = sens1[newOrder[ii]]
    reactionList.append(gas1.reaction_equation(newOrder[ii]))

# generate horizontal bar graph 
numTopReactions = 10; # how many of the top reactions do we want to look at?

plt.rcdefaults()
fig, ax = plt.subplots()
# plot results
ax.barh(newOrder2[0:numTopReactions-1], sens2[0:numTopReactions-1], align='center',
        color='green', ecolor='black')
# make sure that every single tick on the y-axis is marked
ax.set_yticks(np.arange(len(reactionList[0:numTopReactions-1])))
# label these y-ticks with the reaction in question
ax.set_yticklabels(reactionList[0:numTopReactions-1])
# invert the y-axis so that the most sensitive reaction is at the top instead of the bottom
ax.invert_yaxis()
fig.tight_layout()
plt.show()



