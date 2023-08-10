from matplotlib import pyplot
from scipy.stats  import invgauss
from random import random

mu =  0.00500 
sigma =  0.00500 
dt = 1
S0 = 1

time = [0]
density = [0]
dS = [0]
S = [S0]

time.append(time[0] + dt)
density.append(0.00500) 
dS.append(S[0] * density[1])
S.append(S[0] + dS[1])

for i in range(1, 12):
    time.append(time[i] + dt)
    
    randvar = invgauss.rvs(mu=random(), loc=0 ,scale=dt)
    walk = mu * dt + sigma * randvar
    density.append(walk)
    
    dS.append(S[i] * density[i+1])
    S.append(S[i] + dS[i+1])

print("Time \t dS/S \t dS \t S")

for j in range(12):
    print(time[j]," \t ", "%.5f" % density[j], " \t ", "%.5f" % dS[j], " \t ", "%.5f" % S[j])


pyplot.plot(time,S,color='green', marker='o', linestyle='solid')
pyplot.title("Share Price - Wiener Stochastic Process - Single Name")
pyplot.show()

