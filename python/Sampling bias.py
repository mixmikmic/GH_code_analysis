import numpy as np
from scipy.stats import pearsonr

# Generate completely random numbers
randos = [np.random.rand(100) for i in range(100)]
y = np.random.rand(100)

# Compute correlation coefficients (Pearson r) and record their p-values (2nd value returned by pearsonr)
ps = [pearsonr(x,y)[1] for x in randos]

# Print the p-values of the significant correlations, i.e. those that are less than .05
print [p for p in ps if p < .05]

