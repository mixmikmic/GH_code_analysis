from scipy.stats import beta
import numpy as np
from matplotlib import pyplot as plt

# theta is the random variable which can range between 0 and 1
theta = np.linspace(0,1, num=200)
fig = plt.figure(figsize=(12,6))
# beta distribution has two parameters a and b
a = 3
b = 2
pdfvals = beta.pdf(theta,a,b)
plt.plot(theta, pdfvals, lw=2)
plt.fill_between(theta, pdfvals, alpha = .1)
plt.ylabel(r'PDF at $\theta$')
plt.xlabel(r'$\theta$')

