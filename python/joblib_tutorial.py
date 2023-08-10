from joblib import Memory
import numpy as np

mem = Memory(cachedir='/tmp/joblib')
a = np.vander(np.arange(3)).astype(np.float)
square = mem.cache(np.square)

get_ipython().run_cell_magic('time', '', '# The call below did not trigger an evaluation\nc = square(a)')

from joblib import Parallel, delayed
from math import sqrt

get_ipython().run_cell_magic('time', '', 'r = Parallel(n_jobs=4)(delayed(sqrt)(i) for i in range(10**3))')

get_ipython().run_cell_magic('time', '', 'r2 = [sqrt(i) for i in range(10**3)]')





