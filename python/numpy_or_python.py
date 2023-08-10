import timeit
import numpy as np
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns # if you don't have seaborn just uncomment this and the next line
sns.set(style='white', context='talk') 

def measure(stmt, sizes):
    return [timeit.Timer(
        stmt, 
        'import numpy as np; x = np.random.random({:d})'.format(int(size))
    ).timeit(number=1000) for size in sizes]

sizes = np.logspace(1, 6, 6)
vanila = measure('sum(x) / len(x)', sizes)
numpy =  measure('x.mean()', sizes)

plt.plot(sizes, vanila, label='Python')
plt.plot(sizes, numpy, label='NumPy')
plt.axvline(50, ymax=0.23, color='k', ls='--')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc=2)
plt.xlabel('# elements in a list')
plt.ylabel('time to average 1000 lists');

