from numba import jit
import pandas as pd
import numpy as np
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

from utilities.rolling_stats import rolling_mean  # this is the function we're going to test versus pandas

x = np.arange(30).astype(float)

s = pd.Series(x)
s[0] = np.nan
s[6] = np.nan
s[12:18] = np.nan
s[-1] = np.nan
s.values  # arbitrary but small input data

s.rolling(window=3).mean().values  # pandas output

rolling_mean(s.values, 3)  # rolling_sum output

a = s.rolling(window=3).mean().values
b = rolling_mean(s.values, 3)
np.allclose(a, b, equal_nan=True)

def benchmarks():
    
    res = []
    
    for exponent in range(3, 7):
        n = 10**exponent
        data = np.arange(n).astype(float)
        data[3] = np.nan
        data[4] = np.nan
        data[-1] = np.nan
        s = pd.Series(data)
        
        window = int(max(1000, n * 0.1))  # cap window size at 1,000
        
        t1 = time.time()
        pandas_output = s.rolling(window=window).mean().values
        t2 = time.time()
        res.append(('pandas', n, (t2 - t1)))
    
        t1 = time.time()
        rmean_output = rolling_mean(s.values, window)
        t2 = time.time()
        res.append(('rolling_mean', n, (t2 - t1))) 
        
        assert np.allclose(pandas_output, rmean_output, equal_nan=True)
        
    return res

data = benchmarks()
df = pd.DataFrame(data, columns = ['fn', 'population', 'time (ms)'])

df['time (ms)'] = df['time (ms)'].apply(lambda x: x * 1000.) 
df = pd.pivot_table(df, values='time (ms)', index=['population'], columns=['fn'], aggfunc=np.sum)
df

df.plot(logx=True)
plt.ylabel('time (ms)')

