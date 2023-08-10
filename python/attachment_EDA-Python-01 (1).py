your_local_path="C:/sandbox/python"

import urllib.request

# In real world we need to fetch data from different source systems and put it in a single place

tb_deaths_url_csv = 'https://docs.google.com/spreadsheets/d/12uWVH_IlmzJX_75bJ3IH5E-Gqx6-zfbDKNvZqYjUuso/pub?gid=0&output=CSV'
tb_existing_url_csv = 'https://docs.google.com/spreadsheets/d/1X5Jp7Q8pTs3KLJ5JBWKhncVACGsg5v4xu6badNs4C7I/pub?gid=0&output=csv'
tb_new_url_csv = 'https://docs.google.com/spreadsheets/d/1Pl51PcEGlO9Hp4Uh0x2_QM0xVb53p2UDBMPwcnSjFTk/pub?gid=0&output=csv'

local_tb_deaths_file = your_local_path+'tb_deaths_100.csv'
local_tb_existing_file = your_local_path+'tb_existing_100.csv'
local_tb_new_file = your_local_path+'tb_new_100.csv'

deaths_f = urllib.request.urlretrieve(tb_deaths_url_csv, local_tb_deaths_file)
existing_f = urllib.request.urlretrieve(tb_existing_url_csv, local_tb_existing_file)
new_f = urllib.request.urlretrieve(tb_new_url_csv, local_tb_new_file)

import pandas as pd

# Let us load the data into our Python environment for getting started with EDA

tb_deaths_file = your_local_path+'tb_deaths_100.csv'
tb_existing_file = your_local_path+'tb_existing_100.csv'
tb_new_file = your_local_path+'tb_new_100.csv'

deaths_df = pd.read_csv(tb_deaths_file, index_col = 0, thousands  = ',').T
existing_df = pd.read_csv(tb_existing_file, index_col = 0, thousands  = ',').T
new_df = pd.read_csv(tb_new_file, index_col = 0, thousands  = ',').T

#First step that we always do in EDA is that we examine the data 
#(first few row to understand what kind of data we are dealing with)

df_summary = existing_df.describe()
df_summary

#In order to access individual columns

#df_summary[['Afghanistan','Zambia','Zimbabwe']]
df_summary[['Spain','United Kingdom']]

#What can you infer from the data above

#If you want to check percentage change in exsiting cases over the years
tb_pct_change = existing_df.pct_change()
tb_pct_change

#Let us look at curious case of Spain. What do you infer?

tb_pct_change_spain = existing_df.Spain.pct_change()
tb_pct_change_spain

#existing_df.Spain

tb_pct_change_spain = deaths_df.Spain.pct_change()
tb_pct_change_spain

tb_pct_change_spain.max()         # It always decreased

existing_df['United Kingdom']                               # 1992, 2003, 2005, 2007 shows increase
#existing_df['United Kingdom'].pct_change()        
#existing_df['United Kingdom'].pct_change().max()

existing_df['United Kingdom'].pct_change().argmax()         # Returns the indices of the maximum values along an axis
#existing_df['United Kingdom'].pct_change()

#Let us go ahead and do some plotting
get_ipython().magic('matplotlib inline')
existing_df[['United Kingdom', 'Spain', 'Colombia']].plot()

#How about box-plots
existing_df[['United Kingdom', 'Spain', 'Colombia']].boxplot()
#existing_df[['Spain']].sort_values(["Spain"])                  # Lets discuss spread of Spain data

#Now let us ask some questions to the data
#Which country has the highest number of existing and new TB cases Year wise.

# Solution: Get a Series data with index values as Years & Country with max existing cases
existing_df.apply(pd.Series.argmax, axis=1)

# Info: Djibouti country located in the Horn of Africa

#What about world trends? 
#What is following code doing?
deaths_df.head()
deaths_df.sum(axis=1)
deaths_total_per_year_df = deaths_df.sum(axis=1)
existing_total_per_year_df = existing_df.sum(axis=1)
new_total_per_year_df = new_df.sum(axis=1)

world_trends_df = pd.DataFrame({
           'Total deaths per 100K' : deaths_total_per_year_df, 
           'Total existing cases per 100K' : existing_total_per_year_df, 
           'Total new cases per 100K' : new_total_per_year_df}, 
       index=deaths_total_per_year_df.index)

world_trends_df.plot(figsize=(12,6)).legend(
    loc='center left', 
    bbox_to_anchor=(1, 0.5))

#What inferences can we derive?

#What about specific countries?
deaths_by_country_mean = deaths_df.mean()
deaths_by_country_mean_summary = deaths_by_country_mean.describe()
existing_by_country_mean = existing_df.mean()
existing_by_country_mean_summary = existing_by_country_mean.describe()
new_by_country_mean = new_df.mean()
new_by_country_mean_summary = new_by_country_mean.describe()

deaths_by_country_mean.v().plot(kind='bar', figsize=(24,6))

#Let us think about outlier countries  -- Get all data datapoints that are more than 1.5 times the median of all the countries
print(deaths_by_country_mean.describe())

deaths_outlier = deaths_by_country_mean_summary['50%']*1.5
existing_outlier = existing_by_country_mean_summary['50%']*1.5
new_outlier = new_by_country_mean_summary['50%']*1.5

outlier_countries_by_deaths_index = deaths_by_country_mean > deaths_outlier
outlier_countries_by_existing_index =  existing_by_country_mean > existing_outlier
outlier_countries_by_new_index = new_by_country_mean > new_outlier

#Proportions of countries as outliers

print ('Consider countries whose Mean TB Deaths are greater than: ',deaths_outlier)
print ('Consider countries whose Mean Existing Cases are greater than: ',existing_outlier)
print ('Consider countries whose Mean New Cases are greater than: ',new_outlier)

print ('--------------------------------------------------------------------------------')
num_countries = len(deaths_df.T)
print ('Number of countries: ',num_countries)
print ('Number (Mean TB Deaths) of outlier countries: ',sum(outlier_countries_by_deaths_index))
print ('Number (Mean Existing Cases) of outlier countries: ',sum(outlier_countries_by_existing_index))
print ('Number (Mean New Cases) of outlier countries: ',sum(outlier_countries_by_new_index))

# What if you change outlier criteria? ~ As a Data Scientist always keep questioning the data and find ways to get more insights

outlier_deaths_df = deaths_df.T[ outlier_countries_by_deaths_index ].T
outlier_existing_df = existing_df.T[ outlier_countries_by_existing_index ].T
outlier_new_df = new_df.T[ outlier_countries_by_new_index ].T

#Filter the data frame
print(outlier_deaths_df.head())

outlier_new_df.plot(figsize=(12,4)).legend(loc='center left', bbox_to_anchor=(1, 0.5))

#What do you infer from above dataset? Can we somehow combine all of that information?

#print(outlier_new_df.head())
average_outlier_country = outlier_new_df.mean(axis=1)
average_outlier_country

#Compare this with rest of world

#print(new_df.T[ - outlier_countries_by_new_index ].T.head())     # Gets all the left over Countries that are not outlier
avearge_better_world_country = new_df.T[ - outlier_countries_by_new_index ].T.mean(axis=1)
avearge_better_world_country

# Let us plot the 2 data set and see how it behaves for Outliers Countries and Better World Countries

two_world_df = pd.DataFrame({ 
            'Average Better World Country': avearge_better_world_country,
            'Average Outlier Country' : average_outlier_country},
        index = new_df.index)
two_world_df
#two_world_df.plot(title="Estimated new TB cases per 100K",figsize=(12,8))

two_world_df.pct_change().plot(title="Percentage change in estimated new TB cases", figsize=(12,8))
#What do you infer here?

# China: DOTS strategy

existing_df.China.plot(title="Estimated existing TB cases in China")


#print(new_df[['China']].index)
'''
death_china_study = deaths_df[['China']]
exist_china_study = existing_df[['China']]
new_china_study = new_df[['China']]
china_study = pd.concat([death_china_study,exist_china_study,new_china_study],axis=1)
china_study.columns = ['Death','Existing','New']
print(china_study)
'''

new_df.apply(pd.Series.argmax, axis=1)['2007']

