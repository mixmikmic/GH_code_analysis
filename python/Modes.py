import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

x = np.linspace(0,1,200)
X,Y = np.meshgrid(x,x)

def seeMode(n,m,X,Y):
    phi_nm = lambda n,m: np.sin(n*np.pi*X)*np.sin(m*np.pi*Y)
    plt.figure()
    plt.contourf(X,Y,phi_nm(n,m))
    plt.colorbar()
    title = '$\phi_{'+'{},{}'.format(n,m)+'}$'
    plt.title(title)
    plt.axis('equal')
    plt.show()

seeMode(1,1,X,Y)
seeMode(1,2,X,Y)
seeMode(2,1,X,Y)
seeMode(2,2,X,Y)
seeMode(10,20,X,Y)




