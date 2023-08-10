import numpy as np

a = np.array([1,2,0,0,4,0],dtype=int)
np.nonzero(a)

for i in range(1,6):
    N = 10**i
    r = 1.0 + 2*np.random.random(N)
    print('%12s\t%f'%(N,np.average(r)))

M = np.array([[j +i for i in range(5)] for j in range(0,41,10)])
print(M)

M[0,2:4]

M[1:3,1:3]

M[:,2]

M[2::2,::3]

M[2::2,::3] = -1000
M

M2 = np.copy(M)
M2

M2[2::2,::3] = 1000
M2

M

# loadtxt ignores comments with '#' by default
data = np.loadtxt('data/sho_energy.dat')

# create a new array that also includes the total energy
energies = np.zeros([data.shape[0],data.shape[1]+1])
energies[:,:2] = data
energies[:,-1] = data[:,0] + data[:,1]

# create the header string
cols = ('Kinetic [K]','Potential [K]','Total Energy [K]')
headers = '%14s\t%16s\t%16s' % cols

# save to disk
np.savetxt('data/sho_energy_total.dat', energies, fmt='%+16.8E', 
           delimiter='\t', comments='# ', header=headers)

get_ipython().system('cat data/sho_energy_total.dat')

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')
# or %matplotlib notebook
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

plt.style.use('notebook')

get_ipython().system('cat /Users/agdelma/.matplotlib/stylelib/notebook.mplstyle')

# Our first plot, a list
plt.plot([0,1,4,9,15])

# can use all the python tricks we have learned so far
x = list(range(6))
plt.plot(x,[xi**2 for xi in x])

# We can use numpy array operations to simplify things
x = np.arange(0,6,0.001)
plt.plot(x,x**2)

plt.figure(num=3)
plt.plot(x,x**1.5)
plt.plot(x,x**2)
plt.plot(x,x**2.5)

# we could have included them all in a single line
plt.plot(x,x**1.5,x,x**2,x,x**2.5)

# grids
plt.plot(x,0*x,x,x,x,-x)
plt.grid(True)

# specifying axes bounds, labels and titles
y = x - 4*x**2 + x**3
plt.plot(x,y)
plt.xlim(1,3)
plt.ylim(-8,0)
plt.xlabel('x')
plt.ylabel('y')
plt.title(r'$y = x - 4x^2 + x^3$')

# adding a legend
plt.plot(x,0*x,label=r'$y=0$')
plt.plot(x,y,label=r'$y=x-4x^2+x^3$')
plt.plot(x,-y,label=r'$y=-x+4x^2-x^3$')
plt.ylim(-20,20)
plt.legend(loc='upper left', fontsize=16)

data = np.loadtxt('data/ligo_data.dat')
pred = np.loadtxt('data/nr_prediction.dat')
plt.plot(data[:,0],data[:,1],label='H1 Strain')
plt.plot(data[:,0],data[:,2],label='L1 Strain')
plt.plot(pred[:,0],pred[:,1],label='NR Prediction')
plt.xlabel('time (s)')
plt.ylabel('Strain Waveform')
plt.legend()
plt.ylim(-4,4);
plt.xlim(-0.05,0.05)



