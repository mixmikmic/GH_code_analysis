# libraries required for plotting and maths
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib
import matplotlib.pyplot as plt

import matplotlib.patches as patches
ns = 400
#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111, aspect='equal')
#ax1.add_patch(patches.Circle((0.3, 0.3),0.3,fill=False))
#ax1.add_patch(patches.Circle((-0.5, 0.5),0.3,fill = False))

x=2 * np.random.rand(ns,1) -1
y=2 *np.random.rand(ns,1) -1

X=np.column_stack([x,y])

#plt.subplot(121)
#plt.plot(x,y,".")
#plt.title("y=x^2")
#plt.xlabel("x")
#plt.ylabel("y")
#ptsA = np.around(np.random.rand(ns, 2), decimals =3)

def in_circle(x,y):
    if (x-0.3)**2+(y-0.3)**2<0.09:
        return 1
    if (x+0.5)**2+(y-0.5)**2<0.09:
        return 1
    return 0
T=np.array([in_circle(X[i,0],X[i,1]) for i in range(ns)])
#print(T)
#T = (X[:,1]>(m*X[:,0]+κ))
#T = T[:,None]

plt.scatter(X[:,0], X[:,1],c=T)

#plt.axis([-1,1,-1,1])

def predict(Xb, w, v):
    assert Xb.shape[1]==w.shape[0]
    ns=Xb.shape[0]
    Z=np.matmul(Xb,w)
    H=1./(1+np.exp(-Z))
    H=1./(1+np.exp(-Z))

    return P

def cost(P, T):
    C = -np.sum(T*np.log(P)+(1-T)*np.log(1-P))/ns
    return C









