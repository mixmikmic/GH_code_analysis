get_ipython().magic('pylab inline')
import numpy as np
from scipy import stats

n = 10
p = 0.3
k = np.arange(0, 21)

binomial = stats.binom.pmf(k, n, p)

plot(k, binomial, 'o-')
title(f'Binomial: n={n}, p={p:.2}', fontsize=15)
xlabel('Number of Successes')
ylabel('Probability of Success')

rate = 2
n = np.arange(0, 10)
poisson = stats.poisson.pmf(n, rate)

plot( poisson)
title(f'Poisson: $\lambda $={rate}')
xlabel('Number of accidents')
ylabel('Probability of number of accidents')

