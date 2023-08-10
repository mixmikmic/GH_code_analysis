import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# make a list of random numbers
a_python_list = list(np.random.randn(5))
a_python_list

a_pandas_series = pd.Series(data=a_python_list)
a_pandas_series

a_simple_index = ['a', 'b', 'c', 'd', 'e']
a_pandas_series = pd.Series(data=a_python_list, index=a_simple_index)
a_pandas_series

# index by label
a_pandas_series['a']

# index by location
a_pandas_series[1]

a_python_dictionary = {'a' : 0., 'b' : 1., 'c' : 2.}
pd.Series(a_python_dictionary)

a_big_series = pd.Series(np.random.randn(1000))
a_big_series

a_big_series * 2

a_big_series.sum() / len(a_big_series)

a_big_series.mean()

a_big_series.describe()

a_dictionary = {'one' : [1., 2., 3., 4.],
                'two' : [4., 3., 2., 1.]}

a_dataframe = pd.DataFrame(a_dictionary)
a_dataframe    

a_dataframe = pd.DataFrame(a_dictionary,
                           index=['a', 'b', 'c', 'd'])
a_dataframe

a_list_of_dictionaries = [{'a': 1, 'b': 2}, {'a': 5, 'b': 10, 'c': 20}]
a_dataframe = pd.DataFrame(a_list_of_dictionaries)
a_dataframe

a_dataframe['a']

a_dataframe = pd.DataFrame({'a': np.random.randn(1000),
                            'b': np.random.randn(1000),
                            'c': np.random.randn(1000),
                            'd': np.random.randn(1000),
                            'e': 'hi'})
a_dataframe

a_dataframe.dtypes

a_dataframe.describe()

a_dataframe.to_csv("random-data.csv") 

a_dataframe.to_excel("random-data.xls")

a_dataframe.to_excel("random-data.xls", index=False)
a_dataframe.to_csv("random-data.csv", index=False) 

a_new_dataframe = pd.read_csv("random-data.csv")
a_new_dataframe

a_new_dataframe.dtypes

a_new_dataframe.plot()

a_new_dataframe['a'].plot()

a_new_dataframe.plot(kind="box")

a_new_dataframe.plot(kind="density")

a_new_dataframe.hist()



