import pandas as pd # We know this one...
import requests # This is usefull with the API
import numpy as np # For performing numerical analysis
import matplotlib.pyplot as plt # Plotting
import weightedcalcs as wc # This allows for "weighted" calculations

years = range(1969,2018)

years = "".join(str(list(years)))

years = years[1:-1]

BEA_ID = "6BF79D8C-8042-4196-88DC-0E0C55B0C3B6" # This is my Key

my_key = "https://bea.gov/api/data?&UserID=" + BEA_ID + "&method=GetData&"

data_set = "datasetname=RegionalIncome&" # This access the Regional Income dataset

table_and_line_income = "TableName=CA1&LineCode=3&" # This grabs the income data

table_and_line_population = "TableName=CA1&LineCode=2&" # This grabs the populaiton data

year = "Year=" + years + "&" # Makes the years

location = "GeoFips=COUNTY&" # This is the location. I'm going to do this at the county level.

form = "ResultFormat=json" # The format.

API_URL = my_key + data_set + table_and_line_income + year + location + form

r = requests.get(API_URL)

df_income = pd.DataFrame(r.json()["BEAAPI"]["Results"]["Data"])

df_income.drop(['CL_UNIT', 'Code',"NoteRef", "UNIT_MULT"], axis=1, inplace = True)

#df["DataValue"].column = "IncomePC"

df_income.rename(columns={"DataValue":"IncomePC"}, inplace=True)

API_URL = my_key + data_set + table_and_line_population + year + location + form

r = requests.get(API_URL)

population = pd.DataFrame(r.json()["BEAAPI"]["Results"]["Data"])

population.drop(['CL_UNIT', 'Code',"NoteRef", "UNIT_MULT", "GeoName"], axis=1, inplace = True)

#df["DataValue"].column = "IncomePC"

population.rename(columns={"DataValue":"Population"}, inplace=True)

population.head()

combo = pd.merge(population, df_income,   # left df, right df
                 how='inner',      # Try the different options, inner, outer, left, right...what happens.
                 on=['GeoFips',"TimePeriod"],       # link with cntry
                 indicator=True)  # Tells us what happend

combo.info()

combo[combo.TimePeriod == "2015"].head(10)

combo[combo.TimePeriod == "2015"].tail(10)

