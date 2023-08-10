import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
get_ipython().magic('matplotlib inline')

FAO_sec=pd.read_csv('data/foodsecurity.csv')
FAO_sec.head(5)

# available items
set(FAO_sec.ItemName)

FAO_sec[(FAO_sec.ItemName == 'Average protein supply (g/capita/day) (3-year average)') & (FAO_sec.AreaName == 'Yemen')].head(5)

FAO_liv=pd.read_csv('data/livestock_data.csv')
FAO_liv.head(2)

# available items
set(FAO_liv.ElementName)

# Plot evolution country (data year by year)
AreaName='Yemen'
ItemName='Meat, Total'

values=FAO_liv[(FAO_liv.ItemName == ItemName) & (FAO_liv.AreaName == AreaName) & (FAO_liv.ElementName =='Production')].Value
year=FAO_liv[(FAO_liv.ItemName == ItemName) & (FAO_liv.AreaName == AreaName) & (FAO_liv.ElementName =='Production')].Year

plt.plot(year,values)
plt.xlabel('Year')
plt.ylabel(ItemName)
plt.title(AreaName)



