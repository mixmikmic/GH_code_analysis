import numpy as np

x = np.arange(1,1000) #array of 1 to 999

x[(x % 3 == 0) | (x % 5 == 0)].sum()

