from scipy.stats import chi2_contingency
import numpy as np

obs = np.array([[7, 87, 12,9], [4, 102, 7,8]])

chi2, p, dof, expected = chi2_contingency(obs)

print (p)

print (chi2)

print (dof)

print (expected)



