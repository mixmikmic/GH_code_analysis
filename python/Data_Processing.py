import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset from Starbucks 
Data_ST = pd.read_csv("Data/starbucks.csv")
Data_ST = pd.DataFrame(Data_ST.loc[:, 'Country'])

# Calculating the frequency of locations per country 
Data_ST['freq'] = Data_ST.groupby('Country')['Country'].transform('count')

# Drop duplicates
Data_ST = Data_ST.drop_duplicates()
Data_ST = Data_ST.reset_index(drop=True)

# Renaming columns
Data_ST.columns = ['Code', 'Starbucks locations']

# Importing the country code dataset
Data_code = pd.read_csv('https://raw.githubusercontent.com/datasets/country-list/master/data.csv')

# Importing the GEI index dataset
Data_gei = pd.read_csv('Data/gei.csv')

# Renaming columns
Data_code.columns = ["Country", 'Code']
Data_gei.columns = ['Rank', 'Country', "GEI Score"]
Data_gei.head()

# Merging datasets
Data_label = pd.merge(Data_code,Data_gei, on="Country", how="inner" )
final = pd.merge(Data_ST,Data_label, on="Code", how="inner" )

# Importing the Mobile dataset
data_mobile = pd.read_csv('Data/mobile.csv')
data_mobile.columns = ["Country", "Total Subscribers"]

# According to Gartner's SAMSUNG vendor rating Report, Apple's global market share is 27%
data_mobile.loc[:,'Total Subscribers'] * 0.27
final = pd.merge(data_mobile,final, on="Country", how="inner" )
final.rename(columns={'Total Subscribers_x': 'iPhone Users', 'Starbucks locations': 'Starbucks Locations'}, inplace=True)

# Importing the ATM dataset
data_atm = pd.read_csv("Data/ATM.csv")
data_atm.columns = ["Country", "ATMs per 1000 Adults"]
data_atm = data_atm.dropna()
data_atm = data_atm.reset_index(drop=True)

# Merging all datasets together 
final = pd.merge(data_atm, final, on="Country", how="inner" )
final = final.dropna()
final = final.reset_index(drop=True)
final.rename(columns={'ATMs per 1000 Adults_x': 'ATMs per 1000 Adults', 'Total Subscribers': 'iPhone Users'}, inplace=True)
col = ["Country", "Code", "ATMs per 1000 Adults", "iPhone Users", "Starbucks Locations", "GEI Score" ]
final = final[col]

# Print final dataset
final

final.describe()

final.corr()

