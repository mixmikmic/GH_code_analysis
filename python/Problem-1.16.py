from Ulam import ulam

results = dict()
N = 100000
for n in range(1,N):
    results[n] = ulam(n,0,False)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

X = range(1,N)
Y = [results[x] for x in X]
fig = plt.figure(figsize=(14,10))
plt.xlabel('Input')
plt.ylabel('Number of Iterations')
plt.suptitle('Ulam\'s Algorithm')
plt.plot(X,Y,'.')
plt.tight_layout()
plt.show()

#organize data
Xe = [x for x in range(2,N,2)]
Ye = [results[x] for x in Xe]
Xo = [x for x in range(1,N,2)]
Yo = [results[x] for x in Xo]

#plot
fig,(axe,axo) = plt.subplots(1,2,figsize=(14,7))
axe.set_title('Even inputs')
axo.set_title('Odd inputs')
axe.plot(Xe,Ye,'.')
axo.plot(Xo,Yo,'.')
axe.set_xlabel('Input')
axe.set_ylabel('Number of Iterations')
axo.set_xlabel('Input')
axo.set_ylabel('Number of Iterations')
plt.tight_layout()
plt.show()

