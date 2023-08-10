import numpy as np
import copy, sys
import sys     # max float

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def cost(x):    
    # This computes the cost (value of objective function) for a given point in our objective space
    # Any function which returns a scalar can be used here, as long as we seek to minimize it.
    dim = len(x)
    cost = 10*dim + np.sum( np.multiply(x,x) - 10*np.cos(2*np.pi*x))
    return cost

class Particle:
    def __init__(self, minx, maxx, seed):
        # Assumes that minx and maxx are arrays or vectors with len()=n
        self.rnd = np.random.seed(seed)
        dim = len(maxx)

        self.position = (maxx - minx) * np.random.rand(dim) + minx
        self.velocity = (maxx - minx) * (np.random.rand(dim)-0.5)

        self.cost = cost(self.position) # Cost of current position
        self.best_part_pos = copy.copy(self.position)  # Position and cost of the particle's own best position
        self.best_part_cost = self.cost # Particle's own best cost

def Solve(max_epochs, n, minx, maxx):
    # max_epochs: Number of simulation epochs, i.e. flight time steps
    # n : Number of particles
    # minx, maxx: Assuming that the simulation is in a hypercube defined by the range (minx, maxx) in each dimension

    ## Initialization
    w = 0.729    # inertia
    c1 = 1.49445 # cognitive (particle)
    c2 = 1.49445 # social (swarm)
    dim = len(minx)

    # create n random particles, stored in an array named Swarm
    swarm = [Particle( minx, maxx, i) for i in range(n)] 

    ## Identify the best value reported from the initial batch
    best_swarm_cost = 1000 # High initial value    
    for i in range(n): # See what the actual best position is in the swarm
        if swarm[i].cost < best_swarm_cost:
            best_swarm_cost = swarm[i].cost
            best_swarm_pos = copy.copy(swarm[i].position) 

    epoch = 0
    
    # Save position of a selected particle
    track = 5
    trackData = np.zeros([max_epochs+1, dim])
    trackData[epoch] = swarm[track].position
    
    while epoch < max_epochs:
        
        for i in range(n): # process each particle

            # compute new velocity of curr particle, in each dimension
            r1 = np.random.rand(dim)    # uniform randomizations in the range 0-1
            r2 = np.random.rand(dim)

                # New velocity = w * inertia + c1 * own best + c2 * swarm best
            swarm[i].velocity = ( (w * swarm[i].velocity) + 
                                  (c1 * r1 * (swarm[i].best_part_pos - swarm[i].position)) +  
                                  (c2 * r2 * (best_swarm_pos - swarm[i].position))  )

            # compute new position using new velocity
            swarm[i].position = swarm[i].position + swarm[i].velocity

            # compute error of new position
            swarm[i].cost = cost(swarm[i].position)

            # is new position a new best for the particle?
            if swarm[i].cost < swarm[i].best_part_cost:
                swarm[i].best_part_cost = swarm[i].cost
                swarm[i].best_part_pos = copy.copy(swarm[i].position)

            # is new position a new best overall?
            if swarm[i].cost < best_swarm_cost:
                best_swarm_cost = swarm[i].cost
                best_swarm_pos = copy.copy(swarm[i].position)
            # END OF PARTICLE LOOP
        
        if epoch % 5 == 0:
            print("Epoch = " + str(epoch) + " best cost = %.3f" % best_swarm_cost)

        # END OF EPOCH LOOP
        epoch += 1
        trackData[epoch] = swarm[track].position
    
    return (best_swarm_pos, trackData)
# end Solve

## This is the main execution

num_particles = 50
max_epochs = 50
minx = np.array( [-4, -4] )
maxx = np.array( [ 4,  4] )

print("Starting PSO algorithm")

(best_position, trackData) = Solve(max_epochs, num_particles, minx, maxx)

print("PSO completed \nBest solution found:")
print(best_position)
print("Cost of best solution = %.6f" % cost(best_position))

# Plot the Rastrigin function and the path taken by the tracked particle
lim = 4  # We will plot the Rastrigin function in an x-y square centered at 0 with this as the max value 
stepsize = 0.01
x_axis = np.linspace(-lim,lim,lim/stepsize+1)
y_axis = np.linspace(-lim,lim,lim/stepsize+1)
x,y = np.meshgrid(x_axis, y_axis)  # This produces arrays of the same size which will vary in x and y appropriately
f = 20 + np.multiply(x,x) + np.multiply(y,y) - 2*np.cos(2*np.pi*x) - 2*np.cos(2*np.pi*y)
plt.plot(trackData[:,0],trackData[:,1],'r',lw=2)
plt.imshow(f, extent = [-lim, lim, -lim, lim])
plt.xticks(np.linspace(-lim,lim,2*lim+1))
plt.yticks(np.linspace(-lim,lim,2*lim+1))
plt.colorbar()



