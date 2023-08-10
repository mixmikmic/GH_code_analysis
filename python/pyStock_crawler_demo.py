import datetime as dt
import pandas as pd
import numpy as np
from pandas_datareader import data
import statsmodels.formula.api as sm
import time
import wmcm

import finsymbols

sp500 = pd.DataFrame(finsymbols.get_sp500_symbols())

sp500.head()

