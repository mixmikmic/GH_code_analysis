import pandas as pd

city = pd.DataFrame(pd.read_csv('Data/GlobalLandTemperaturesByCity.csv'))

city.memory_usage(index=True).sum()



# Convert dataframe data types into a dictionary, for referencing
types_dict = city.dtypes.to_dict()
types_dict

city = pd.read_csv('Data/GlobalLandTemperaturesByCity.csv', low_memory=True, dtype=types_dict)

city.to_feather('Data/city')

city_feather = pd.read_feather('Data/city')

get_ipython().run_cell_magic('timeit', '', "city_feather = pd.read_feather('Data/city')")

get_ipython().run_cell_magic('timeit', '', "city = pd.read_csv('Data/GlobalLandTemperaturesByCity.csv')")

city_feather.memory_usage(index=True).sum()



