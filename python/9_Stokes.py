def B(I,x,y):                                         
    mu0 = 1.26 * 10**(-6) 
    r = np.sqrt((x)**2+(y)**2)
    c = mu0*I/(2*np.pi) 
    Bx = -y*c/r**2
    By = x*c/r**2         
    Bz = z*0                     
    return Bx,By,Bz

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'notebook')

X = np.linspace(-1,1,12)
Y = np.linspace(-1,1,12)
Z = np.linspace(-1,1,12)
x,y,z = np.meshgrid(X,Y,Z)

I = 200000
Bx,By,Bz = B(I,x,y)                            

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x,y,z,Bx,By,Bz)                                   
ax.plot([0, 0],[0, 0],[-1,1],linewidth=3,color='r')
plt.xlabel('x')
plt.ylabel('y')
plt.show()



