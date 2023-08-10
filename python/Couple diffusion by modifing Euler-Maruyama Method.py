get_ipython().magic('pylab inline')
import IPython.display

import random

# Euler - Maruyama method on coupled diffusion SDE
#
# SDE is dX = mu*dt + sigma*dW
#   The system can switch between two states
#   Where mu: two values mu1 = 5 (slow diffusion) or mu2 = 50 (fast diffusion)
#         sigma: two values, but for simplicity set to 0 (No drift)
#         rate of switching: q12 = 1 (state 1 to 2) and q21 = 1 (state 2 to 1)

random.seed(100)                # Set initial state of random number generator to reproduce experiments

## Problem parameters
N = 10000
tfinal = 25.
mu1 = 0.
mu2 = 0.
sigma1 = 5.
sigma2 = 50.
q12 = 1.
q21 = 1.

dt = tfinal/N                    # Set desired step size from parameters
X = zeros(N)                     # Store the position of the particle
X[0] = 1.                        # Inital position of the particle
stimes = zeros(N)                # A fixed array to store switching times
                                 # Assumed number of switching times less than N

# Calculate switching times
i=0
while(stimes[i] < tfinal):       # Loop provided latest calculated switching
                                 # time is less than final time
    if i%2 == 0:                
        stimes[i + 1] = stimes[i] - (1/q12)*log(rand())
    else: 
        stimes[i + 1] = stimes[i] - (1/q21)*log(rand())
    i+=1
    
# Remove excess number of switching times
swtimes =trim_zeros(stimes)

current_state = 1                # Set initial state of the system
tn = 0
dT = dt
tnxt = tn + dt
times = zeros(N)                 # Stores the elapsed time

j = 0                            # Position in the array of switching times
for i in range(N - 1):
    if tnxt < tfinal and j < len(swtimes) - 1 and tnxt > swtimes[j]:  # Switching occurs if the next time is more
        dT = swtimes[j] - tn                                          # than the next time; recalulate time step
        tnxt = tn + dT                                                # Update next time to be the switching time
        
        # Switch states
        current_state = (current_state + 2) % 2 + 1                   # Switch from 1 to 2 or vice versa
        j += 1                                                        # Increment position in the array of switching
                                                                      # times
    # Update position of the particle depending on the state
    if(current_state == 1):
        X[i+1] = X[i] + dT*mu1 + (sigma1*sqrt(dT)*randn())
    else:
        X[i+1] = X[i] + dT*mu2 + (sigma2*sqrt(dT)*randn())
    
    
    tn = tnxt                    # Update current time to next time
    times[i +1] = tn             # Store current time
    dT = dt                      # Redefine to be the desired increment
    tnxt = tn + dT               # Calculate the next time

figure(figsize = (20,10))
plot(times, X)
rcParams.update({'font.size': 20})
xlabel("Time", fontsize = 20)
ylabel("Position of the particle", fontsize = 20)
title("Coupled Diffusion simulation", fontsize = 25)
for marker in delete(swtimes,-1):                                     # Inidicates the time of switching
    plt.axvline(x = marker,color = 'black', linestyle = '--')
savefig('Cdiff.png',transparent=True, bbox_inches='tight', pad_inches=0)

