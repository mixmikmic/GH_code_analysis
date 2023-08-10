import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
get_ipython().run_line_magic('matplotlib', 'inline')

pop1 = np.random.binomial(10, 0.2, 10000)
pop2 = np.random.binomial(10, 0.5, 10000)

sample1 = np.random.choice(pop1, 100, replace=True)
sample2 = np.random.choice(pop2, 100, replace=True)

print("Mean for sample1: ", sample1.mean())
print("Mean for sample2: ", sample2.mean())
print("Standard deviation for sample1: ", sample1.std())
print("Standard deviation for sample2: ", sample2.std())

sample_increase_pop1 = np.random.choice(pop1, 1000, replace=True)
sample_increase_pop2 = np.random.choice(pop2, 1000, replace=True)

print("Mean for increased sample of pop1: ", sample_increase_pop1.mean())
print("Mean for increased sample of pop2: ", sample_increase_pop2.mean())
print("Standard deviation for increased sample of pop1: ", sample_increase_pop1.std())
print("Standard deviation for increased sample of pop2: ", sample_increase_pop2.std())

sample_decrease_pop1 = np.random.choice(pop1, 20, replace=True)
sample_decrease_pop2 = np.random.choice(pop2, 20, replace=True)

print("Mean for decreased sample of pop1: ", sample_decrease_pop1.mean())
print("Mean for decreased sample of pop2: ", sample_decrease_pop2.mean())
print("Standard deviation for decreased sample of pop1: ", sample_decrease_pop1.std())
print("Standard deviation for decreased sample of pop2: ", sample_decrease_pop2.std())

pop1 = np.random.binomial(10, 0.3, 10000)
pop2 = np.random.binomial(10, 0.5, 10000)

sample1 = np.random.choice(pop1, 1500, replace=True)
sample2 = np.random.choice(pop2, 1500, replace=True)

plt.hist(sample1, alpha=0.5, label='sample 1') 
plt.hist(pop1, alpha=0.5, label='pop 1') 
plt.legend(loc='upper right') 
plt.show()

print(ttest_ind(sample2, sample1, equal_var=False))

pop1 = np.random.logseries(0.3, 10000)
pop2 = np.random.logseries(0.5, 10000)

sample1 = np.random.choice(pop1, 1500, replace=True)
sample2 = np.random.choice(pop2, 1500, replace=True)

plt.hist(sample1, alpha=0.5, label='sample 1') 
plt.hist(pop1, alpha=0.5, label='pop 1') 
plt.legend(loc='upper right') 
plt.show()
print(ttest_ind(sample2, sample1, equal_var=False))

