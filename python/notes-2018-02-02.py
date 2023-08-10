import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

x = np.array([1,2,3,4,5])

print(x)

type(x)

x.shape

x.ndim

M = np.array([[5,7],[1,-1]])

print(M)

type(M)

M.shape

M.ndim

x = np.arange(0,1,0.1)

print(x)

y = np.linspace(0,1,11)

print(y)

np.zeros((5,10))

np.ones((3,4))

np.eye(3)

u = np.array([1,0,-3])
v = np.array([3,-2,1])

u + v

u - v

u * v

u / v

w = np.arange(0,10)

print(w)

w**2

x = np.linspace(-1,1,11)
print(x)

y = np.exp(x)
print(y)

np.cos(np.pi / 3)

x = np.linspace(-2,2,100)
y = x * (x - 1) * (x + 1)
plt.plot(x,y)
plt.xlim([-1.5,1.5])
plt.grid('on')
plt.show()

x = np.linspace(0,20,1000)
y = np.cos(4*x) + np.cos(np.pi * x)
plt.plot(x,y,c=(0,0.1,1,0.5))
plt.title('Beating Vibrations')
plt.show()

x = np.linspace(-10,10,1000)
y = np.exp(-0.1*x**2) * np.cos(x)
for A in np.arange(-2,2.1,0.1):
    plt.plot(x,A*y)
plt.show()

t = np.linspace(0,1,100)
x = np.cos(2*np.pi*t)
y = np.sin(2*np.pi*t)
plt.plot(x,y)
plt.axis('equal')
plt.show()

