import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
# change scale of all figures to make them bigger
import matplotlib as mpl
mpl.rcParams['savefig.dpi']=100 

# The plot command takes the x coordinates as first argument
# then the y coordinates. The third argument controls the
# way points look on the plot

xr = np.arange(0,3,0.1)
yr1 = np.exp(xr) * np.sin(5*xr)
yr2 = np.exp(xr) * np.cos(5*xr)

plt.plot(xr, yr1, '-r', lw=3, label = 'red line')
plt.plot(xr, yr2, 'xb', label = 'blue line')
plt.legend()

xr = np.arange(0,1,0.01)
yr = np.sin(xr)
plt.plot(xr,yr,'--b',lw=4)
plt.plot([-0.5,1.5],[0.0,0.4],'-g^',label='a line')
plt.legend()
plt.xlabel('time $t$')
plt.ylabel('$\int \,dt\, \cos(t)$')
plt.axis([-1,2,-0.2,1.1])
#xlim(0,2)

xr = np.arange(0,10,0.01)
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(xr,np.sin(xr),'b')
plt.title("subplot 1")
plt.ylabel('sin')
plt.subplot(2,1,2)
plt.plot(xr,np.cos(xr),'r')
plt.title("subplot 2")
plt.ylabel('cos')
plt.xlabel(r'$\omega$',)
plt.figure(2)
plt.plot(xr,np.exp(-0.1*xr**2),label='some function here')
plt.legend()
plt.title("figure 2")

mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)
xr = np.arange(50,150,0.1)

plt.hist(x, 50, normed=1, facecolor='r', alpha=0.3)
plt.plot(xr,0.028*np.exp(-0.0025*(xr-100)**2),'b',lw=3)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram')
plt.text(45, .025, r'$\mu=100,\ \sigma=15$',fontsize=20)
plt.axis([40, 160, 0, 0.03])
plt.grid(True)

xr = np.arange(0,3,0.2)
yr = np.tanh(xr)

fig = plt.figure(1)
ax = fig.add_axes([0., 0.8, 1.5, 0.9])
ax.set_title("My axes",fontsize=20)
ax.set_xlabel(r'$x$',fontsize=20)
ax.set_ylabel(r'$\tanh(x)$',fontsize=20)
ax.plot(xr,yr,'o')
subax = fig.add_axes([0.45,0.85,1.,0.5])
subax.plot(xr,np.sin(xr),'r',label='sinus')
plt.legend()

xr = np.arange(0,3,.1)
yr1 = np.exp(xr)*(np.sin(5*xr))
yr2 = np.exp(xr)*(np.cos(5*xr))
plt.figure(1)
plt.plot(xr,yr1,'-r',lw=3, label='a first curve')
plt.plot(xr,yr2,'--b',lw=3, label='$e^{x} \cos(5 x)$')
plt.legend()
plt.xlabel('time $t$')
plt.ylabel('$\int \, \cos(t) $')

plt.figure(2)
plt.plot(xr,yr1*3.,'-.b^',lw=1, label='a second curve')
plt.legend()
plt.xlabel('time $t$')
plt.ylabel('$\int \, \cos(t) $')


plt.figure(3)
plt.subplot(211)
plt.plot(xr,yr1,'-r',lw=3, label='a first curve')
plt.legend()
plt.ylabel('$\int \, \cos(t) $')
plt.subplot(212)
plt.plot(xr,yr2,'--b',lw=3, label='$e^{x} \cos(5 x)$')
plt.legend()
plt.xlabel('time $t$')
plt.ylabel('$\int \, \cos(t) $')


plt.figure(4)

plt.subplot(121)
plt.title('My title')
plt.plot(xr,yr1,'-r',lw=3, label='a first curve')
plt.legend()
plt.ylabel('$\int \, \cos(t) $')
plt.xlabel('time $t$')
plt.subplot(122)
plt.grid(True)
plt.plot(xr,yr2,'--b',lw=3, label='$e^{x} \cos(5 x)$')
plt.legend()
plt.xlabel('time $t$')

plt.figure(5)

a = np.loadtxt("my_data.dat")
plt.plot(a[:,0],a[:,1],'-o')



