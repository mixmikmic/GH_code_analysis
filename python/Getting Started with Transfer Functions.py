get_ipython().system('pip install slycot')
get_ipython().system('pip install control')

get_ipython().magic('matplotlib inline')

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import control.matlab as control

G = control.tf([4.3],[3.2, 1])
print(G)

y,t = control.step(G)
plt.plot(t,y)
plt.xlabel('Time')
plt.title('Step Response')

G1 = control.tf([12.3],[10,1])
G2 = control.tf([4],[15,1])

G = G2 * G1

print(G)

y,t = control.step(G)
plt.plot(t,y)
plt.xlabel('Time')
plt.title('Step Response of Two First Order Transfer Functions in Series')

R = control.tf([3.],[4.,1.,1.])
print(R)

y,t = control.step(R)
plt.plot(t,y)
plt.xlabel('Time')
plt.title('Response of a Second Order System')

