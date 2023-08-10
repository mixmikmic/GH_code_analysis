# HIDDEN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().magic('matplotlib inline')

sns.set()
sns.set_context('talk')
pd.options.display.max_rows = 7

import pandas as pd

students = pd.read_csv('roster.csv')
students

students['Name'] = students['Name'].str.lower()
students

