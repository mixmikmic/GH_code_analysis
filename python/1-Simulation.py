import matplotlib.pyplot as plt
import numpy as np

numSamples = 50 * 2

mu0, sigma0 = 10, 2 # mean (signal) and standard deviation (noise) for 0
mu1, sigma1 = 20, 4 # mean (signal) and standard deviation (noise) for 1
samples0 = np.random.normal(mu0, sigma0, int(numSamples/2))
samples1 = np.random.normal(mu1, sigma1, int(numSamples/2))

x = np.concatenate((np.zeros(len(samples0)), np.ones(len(samples1)))) # intermediate variables
y = np.concatenate((samples0, samples1)) # simulated leakages

columnOfOnes = np.concatenate((np.ones(len(samples0)), np.ones(len(samples1))))
A = np.vstack((x, columnOfOnes))

beta = np.linalg.lstsq(A.T,y)[0]
print(beta[0], beta[1])

xnew = np.arange(-1,3) # range somewhat wider to ensure 0 and 1 are not on the edges of the plot
line = beta[0] * xnew + beta[1] # fitted line
plt.plot(xnew, line, 'r-', x, y, '.')
plt.xlim((-1, 2))
plt.ylim((0, 30))
plt.show()

