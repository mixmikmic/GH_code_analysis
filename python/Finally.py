get_ipython().magic('matplotlib inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import seaborn
except ImportError:
    pass

dft = pd.DataFrame(dict(A = np.random.rand(3),
                        B = 1,
                        C = 'foo',
                        D = pd.Timestamp('20010102'),
                        E = pd.Series([1.0]*3).astype('float32'),
                        F = False,
                        G = pd.Series([1]*3,dtype='int8'),
                        H = pd.Series(['a', 'b', 'a'], dtype='category')))
 

dft.dtypes



