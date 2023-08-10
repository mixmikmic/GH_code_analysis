import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt 
import seaborn as sns

belief=0.44 

k = 0

(belief**k)*(1-belief)**(1-k)

sns.set_palette("Reds")
x = np.linspace(0, 1, 100)
params = [
    (0.5, 0.5),
    (1, 1),
    (1, 2),
    (5, 2),
    (6, 6)
]
for p in params:
    y = beta.pdf(x, p[0], p[1])
    plt.plot(x, y, label="$\\alpha=%s$, $\\beta=%s$" % p, lw=3)
plt.xlabel("$\\theta$, Belief")
plt.ylabel("Density")
plt.legend(title="Parameters")
plt.show()

a =12
b = 12
prior = (a.numerator, b.numerator)

#Obervations
N = 50
success = 22
n_alpha = success+a
n_beta = N - success + b
posterior = (n_alpha.numerator, n_beta.numerator)

sns.set_palette(sns.color_palette("Reds", n_colors=2))
y = beta.pdf(x, prior[0], prior[1])
plt.plot(x, y, label="$\\alpha=%s$, $\\beta=%s$" % prior, lw=3)
y = beta.pdf(x, posterior[0], posterior[1])
plt.plot(x, y, label="$\\alpha=%s$, $\\beta=%s$" % posterior, lw=3)

plt.xlabel("$\\theta$, Belief")
plt.ylabel("Density")
plt.legend(title="Parameters")
plt.show()

#Mean
n_alpha/(n_alpha+n_beta)



