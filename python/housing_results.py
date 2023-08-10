import requests as req
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


mainfile = os.path.join("..","Project1_AmazonSites.xlsx")
# read places 
xls = pd.ExcelFile(mainfile) 
places_df=xls.parse('AmazonCities', dtype=str) 

# Housing data from Census

housingfile = os.path.join("Results","housing.csv")
housing_df = pd.read_csv(housingfile)
housing_df = housing_df.merge(places_df,on='NAME')
housing_df


plt.figure(figsize=(8,5))
housing_df = housing_df.sort_values('home owner afford')
sns.barplot(y='City',x='home owner afford',data=housing_df)
plt.xlabel("% if income to cover housing costs")

plt.title("Affordable House Ownership")

housingplot = os.path.join("Plots",'affordhouse.png')
plt.savefig(housingplot)
plt.show()

ranking_own = pd.DataFrame(housing_df['City'])
#ranking_own=ranking_own.rename(columns={'City':'Housing Owner Affordability'})
ranking_own
ranking= ranking_own
ordered = np.arange(8,0,-1)
ordered
ranking['own'] = ordered 

        

ranking = ranking.set_index('City')

ranking

plt.figure(figsize=(8,5))
housing_df = housing_df.sort_values('rent afford')
sns.barplot(y='City',x='rent afford',data=housing_df)
plt.xlabel("% if income to cover renting costs")
plt.title("Affordable Rent")
rentplot = os.path.join("Plots","affordrent.png")
plt.savefig(rentplot)
plt.show()

ranking_rent = pd.DataFrame(housing_df['City'])
ranking_rent['rent']=ordered
r = ranking_rent.set_index('City')

ranking['rent']=r['rent']
ranking

# combined plot
housing_df_temp_own = housing_df.copy()
housing_df_temp_rent = housing_df.copy()
housing_df_temp_own['RentOwn'] = 'own'
housing_df_temp_own['Afford'] = housing_df_temp_own['home owner afford']
housing_df_temp_rent['RentOwn'] = 'rent'
housing_df_temp_rent['Afford'] = housing_df_temp_own['rent afford']
frames= [housing_df_temp_own, housing_df_temp_rent]
housing_df_rentOwn = pd.concat(frames, ignore_index= 'True')#By City
sns.barplot(x="Afford", y="City", hue="RentOwn", data=housing_df_rentOwn)
plt.title("Housing Affordability")
plt.show()

house_rent_ranking_file = os.path.join("Results","house_rent_ranking.csv")
ranking.to_csv(house_rent_ranking_file)





