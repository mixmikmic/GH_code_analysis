import numpy as np
from pandas import Series, DataFrame
import pandas as pd

dframe1 = DataFrame(np.arange(4).reshape((2, 2)), columns = list('AB'), index=['NYC', 'LA'])

dframe1

dframe2 = DataFrame(np.arange(9).reshape((3, 3)), columns = list('ADC'), index=['NYC', 'SF', 'LA'])

dframe2

dframe1 + dframe2

dframe1.add(dframe2, fill_value = 0)

dframe2['A']

