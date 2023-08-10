import numpy as np
import pandas as pd
from numba import jit
import time

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from matplotlib import rcParams
from matplotlib import pyplot as plt
rcParams['figure.figsize'] = 16, 8

import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

from utilities.ewma import ewma  # this is the function we're going to test versus pandas

# create some sample data and introduce a few NaNs
data = np.arange(1e5) * 1.0
data[3] = np.nan
data[4] = np.nan

expected = pd.Series(data).ewm(alpha=0.1, adjust=True, ignore_na=False).mean().values

output = ewma(data, alpha=0.1, adjust=True, ignore_na=False)

np.allclose(expected, output)  # assert output is as per Pandas equivalent

def benchmarks():
    
    res = []
    
    for exponent in range(3, 7):
        n = 10**exponent
        data = np.arange(n).astype(float)
        data[3] = np.nan
        data[4] = np.nan
        data[-1] = np.nan
        s = pd.Series(data)

        t1 = time.time()
        pandas_output = s.ewm(alpha=0.1, adjust=True, ignore_na=False).mean().values
        t2 = time.time()
        res.append(('pandas', n, (t2 - t1)))
    
        t1 = time.time()
        ewma_output = ewma(data, alpha=0.1, adjust=True, ignore_na=False)
        t2 = time.time()
        res.append(('ewma', n, (t2 - t1)))
        
        assert np.allclose(pandas_output, ewma_output)
        
    return res

data = benchmarks()
df = pd.DataFrame(data, columns = ['fn', 'population', 'time (ms)'])

df['time (ms)'] = df['time (ms)'].apply(lambda x: x * 1000.) 
df = pd.pivot_table(df, values='time (ms)', index=['population'], columns=['fn'], aggfunc=np.sum)
df

df.plot(logx=True)
plt.ylabel('time (ms)')



