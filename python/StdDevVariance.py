get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt

incomes = np.random.normal(150.0, 50.0, 10000)

plt.hist(incomes, 50)
plt.show()

incomes.std()

incomes.var()



