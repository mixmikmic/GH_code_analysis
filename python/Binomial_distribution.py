get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas
from scipy import special
import matplotlib.pyplot as plt

nb = 12960 
# nb = 432 #24h
nw = 1080
# nw = 18
tot = nb + nw 

p=nb/tot
q=1-p
print('p:',p, 'q:',q)

nsim=10000000

r = np.random.binomial(nb,p,size=nsim)
accuracy =  pandas.Series(r/nb)
print('acc:', accuracy.mean(), '\t\t\tcorrects:', r.mean(), '\tincorrs:', (nb-r).mean())

accuracy_norm = accuracy.value_counts().sort_index()/nsim
# print(accuracy_norm.tail(50))
cs=accuracy_norm.cumsum()
print(cs.tail(100))

fig, ax1 = plt.subplots()

ax1.plot(accuracy_norm,'b')
ax1.set_xlabel('accuracy')
ax1.set_ylabel('pmf', color='b')

ax2 = ax1.twinx()
ax2.plot(cs,'r')
ax2.set_ylabel('CDF', color='r')
# ax2.set_ylim([0.9,1])
ax1.xaxis.grid(True)
plt.grid(True)
plt.savefig('binomial_chance_accuracy')



