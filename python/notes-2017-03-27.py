import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import matplotlib
matplotlib.rcParams['figure.figsize'] = (12, 6) # Sets default figure size for plots

import pandas as pd

pop_data = pd.read_csv("canada_population_data.csv",index_col=0) # Set the first column in the .csv file as the index

type(pop_data)

pop_data.head(10) # Display top 10 rows of the DataFrame

pop_data.tail(5) # Display the last 10 rows of the DataFrame

pop_data.info()

canada_pop_data = pop_data['Canada']
canada_pop_data.head()

maritimes_pop_data = pop_data[['Nova Scotia','New Brunswick','Newfoundland and Labrador','Prince Edward Island']]
maritimes_pop_data.head()

type(canada_pop_data)

nineties_pop_data = pop_data.loc[1990:2000]
nineties_pop_data.head()

pop_data.loc[1990:2000,['Northwest Territories','Nunavut']]

pop_data['Maritimes'] = pop_data['Nova Scotia'] + pop_data['New Brunswick'] + pop_data['Newfoundland and Labrador'] + pop_data['Prince Edward Island']

pop_data.head()

pop_data['West'] = pop_data['British Columbia'] + pop_data['Alberta'] + pop_data['Saskatchewan']

pop_data.head()

pop_data.plot()

pop_data[['Ontario','Quebec','West','Maritimes']].plot()

rel_pop_data = pop_data.div(pop_data['Canada'],axis=0)*100 # Divide the values in each column by the 'Canada' column

rel_pop_data[['Ontario','Quebec','West','Maritimes']].plot(title='Percent of Total Population')

