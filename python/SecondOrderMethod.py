#math and linear algebra stuff
import numpy as np

#plots
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (15.0, 15.0)
#mpl.rc('text', usetex = True)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

x=np.linspace(-5,10,100)
f = lambda x : 0.5*x**2-3
f1 = lambda x : x
def ftayl(a):
    def fa(x):
        return f(a)+(x-a)*f1(a)
    return fa

fig=plt.figure(0,figsize=(8,8))
ax=fig.add_subplot(111)
plt.title("Overview of newton method")
plt.xlabel('X axis')
plt.ylabel('Y axis')
ax.grid(True, which='both')
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.xlim(0,12)
plt.ylim(-5,50)

#Plot original function
plt.plot(x,f(x))

#Initial estimate
xi = 8

for i in range(4):
    #Plot estimate
    plt.scatter(xi,0)
    plt.plot([xi,xi],[0,f(xi)],'k')
    ax.annotate('$x_'+str(i)+'$',(xi,0))
    #plot linearization in xi
    plt.plot(x,ftayl(xi)(x),'g')
    #Compute next estimate
    xi = xi-f(xi)/f1(xi)



