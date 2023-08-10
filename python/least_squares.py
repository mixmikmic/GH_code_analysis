import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set_context('talk')
get_ipython().magic('matplotlib inline')

data = np.genfromtxt('grace_data.txt', skip_header = 10, delimiter=',')
t = data[:, 0]
mass = data[:, 1]

plt.figure(figsize=(10,6))
plt.plot(t, mass)
plt.xlabel('Time (years)')
plt.ylabel('Mass (centimeters water equivelent)')
plt.title('Mass change over time of Greenland periphery')

G = np.matrix([np.ones(len(t)), t, 
               np.cos(2 * np.pi * t), np.sin(2 * np.pi * t), 
               np.cos(4 * np.pi * t), np.sin(4 * np.pi * t)]).T

mass = np.matrix(mass).T  # Change dimension and change to a matrix 
m = np.linalg.inv(G.T*G) * G.T * mass

fwd = G * m

plt.figure(figsize=(10,6))
plt.plot(t, fwd, label='Forward model')
plt.plot(t, mass, label='Data')
plt.xlabel('Time (years)')
plt.ylabel('Mass (centimeters water equivelent)')
plt.title('Forward model vs. data')
plt.legend(loc='upper right')

plt.figure(figsize=(10,6))
plt.plot(t, mass - fwd)
plt.xlabel('Time (years)')
plt.ylabel('Mass (centimeters water equivelent)')
plt.title('Residual mass signal')

trend_scale = G[:, 0:2] * m[0:2, :]

plt.figure(figsize=(10,6))
plt.plot(t, (mass-fwd) + trend_scale)
plt.xlabel('Time (years)')
plt.ylabel('Mass (centimeters water equivelent)')
plt.title('Deseasoned mass signal')

