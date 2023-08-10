import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

hospital_prefs = {'X': ['B', 'A', 'C'], 'Y': ['A', 'B', 'C'], 'Z': ['A', 'B', 'C']}

hospital_prefs

# Creating Random Permutations
np.random.permutation(5)

# Preference Dictionary
n = 5
preferences = {}
for i in range(n):
    preferences[i] = list(np.random.permutation(n))

print(preferences)

# ValueError Function
for i in range(3):
    raise ValueError('Error')

# Get Hospital Match Ranks
# Hint: hospital_ranks[<hospital>] = hospital_prefs[<hospital>].index(<matched_student>) + 1

# Mean Values using NumPy
d = {'Y': 2, 'X': 2, 'Z': 3}

np.mean(list(d.values()))



