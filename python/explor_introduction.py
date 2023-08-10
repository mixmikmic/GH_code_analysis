from explor import EDA

help(EDA)

import numpy as np
import pandas as pd

data = np.random.randint(0, 100, 100)  # generate an array of 1000 random integers between 0 and 100
data = data.reshape(25, 4)  # change the shape of the array to 250 rows, and 4 columns
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])  # convert our data to a pandas DataFrame, and add column labels

eda = EDA()

eda.set_data(df)

# eda.analyze()

get_ipython().run_line_magic('matplotlib', 'inline')

# eda.analyze(plot=True)

data = np.random.normal(0, 1, 1000)  # generate 1000 random observations from a standard normal distribution.

series = pd.Series(data)

series.name = "Normal Distribution"

eda.set_data(series)

eda.analyze(plot=True)

data = np.random.randint(0, 100, 100)
data = data.reshape(20, 5)
df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E'])

eda.set_data(df)

eda.corr_plot(adv=True);



