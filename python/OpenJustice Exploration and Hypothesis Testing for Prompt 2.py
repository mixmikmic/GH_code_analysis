#Load DOJ arrests data from AzureML Datasets
from azureml import Workspace
ws = Workspace()
ds = ws.datasets['ca_doj_arrests_deidentified_2014_05-07-2016.csv']
frame = ds.to_dataframe()

#View metadata
print(ds.name)
print(ds.data_type_id)
print(ds.size)
print(ds.created_date)

#Need to fix this code to bring in all files at once, not just 2014
import glob
test = glob.glob('ca_doj_arrests_deidentified_20[00-14]_05-07-2016.csv')
#dsarrests = ws.datasets['ca_doj_arrests_deidentified_*.csv']
#framearrests = dsarrests.to_dataframe()

test

#Load population data from AzureML Datasets in case it's needed later
dspop = ws.datasets['ca_county_population_by_race_gender_age_2005-2014_02-05-2016.csv']
framepop = dspop.to_dataframe()

#View metadata
print(dspop.name)
print(dspop.data_type_id)
print(dspop.size)
print(dspop.created_date)

framepop.dtypes

#Load contextual data from AzureML Datasets in case it's needed later
dscontex = ws.datasets['ca_county_agency_contextual_indicators_2009-2014_05-03-2016.csv']
framecontex = dscontex.to_dataframe()

#View metadata
print(dscontex.name)
print(dscontex.data_type_id)
print(dscontex.size)
print(dscontex.created_date)

framecontex.dtypes

#Load offense code dictionary data from AzureML Datasets in case it's needed later
dsoffensecodes = ws.datasets['offense_codes.csv']

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from calendar import isleap

# fix rows with arrest_day == 0
frame.loc[(frame['arrest_day'] == 0), 'arrest_day'] = 1

# fix months with 30 days that have arrest_day == 31
month_30 = [4, 6, 9, 11]
for month in month_30:
    frame.loc[(frame['arrest_month'] == month) & (frame['arrest_day'] == 31), 'arrest_day'] = 30

# Roll February arrest_day past 28 (i.e., 29, 30, 31) or 29th (i.e., 30, 31) back to max number of days
frame.loc[(frame['arrest_year'].apply(lambda x: isleap(x))) & (frame['arrest_month'] == 2) & (frame['arrest_day'] > 29), 'arrest_day'] = 29
frame.loc[~(frame['arrest_year'].apply(lambda x: isleap(x))) & (frame['arrest_month'] == 2) & (frame['arrest_day'] > 28), 'arrest_day'] = 28

get_ipython().magic('matplotlib inline')

frameage = frame.loc[:,['age_group']].groupby('age_group').size().order(ascending=False)

frameage

#Select juvenile arrests data only
framejuv = frame.loc[frame['age_group'] == 'juvenile']

framejuv.dtypes

#Juvenile arrest by month of year
framejuv.arrest_month.hist(grid=False, bins=12)

#Juvenile arrests by day of month
framejuv.arrest_day.hist(grid=False, bins=31)

#[2] returns the week day (1-7)
#todayWeekDay = date.today().isocalendar()[2]
#todayWeekDay

#Adding datetime and date formats
framejuv['datetime'] = pd.to_datetime(framejuv.arrest_year*10000 + framejuv.arrest_month*100 + framejuv.arrest_day, format="%Y%m%d")
framejuv['date'] = pd.DatetimeIndex(framejuv.datetime).normalize()

#Adding day of week
framejuv['week'] = pd.DatetimeIndex(framejuv['datetime']).dayofweek

#Juvenile arrests by day of week
framejuv.week.hist(grid=False, bins=7)

#Potential to model seasonality here or include seasonality as contextual factor in later predictive model
#import statsmodels.api as sm

#Group by offense level
framejuvbyoffense = framejuv.loc[:,['offense_level']].groupby('offense_level').size().order(ascending=False)

#View arrests by offense level
framejuvbyoffense

#Group by gender
framejuvbygender = framejuv.loc[:,['gender']].groupby('gender').size().order(ascending=False)

#View arrests by gender
framejuvbygender

#Group by race/ethnicity
framejuvbyrace = framejuv.loc[:,['race_or_ethnicity']].groupby('race_or_ethnicity').size().order(ascending=False)

#View arrests by race/ethnicity
framejuvbyrace

#Identify BCS summary offense codes with highest frequency
framejuv.bcs_summary_offence_code.hist(grid=False, bins=66)

#Identify BCS offense codes with highest frequency
framejuv.bcs_offense_code.hist(grid=False, bins=226)

#Select offense code 30
framejuv30 = framejuv.loc[framejuv['bcs_summary_offence_code'] == 30]

framejuvexplore0 = framejuv30.groupby(['race_or_ethnicity']).size()

#View race grouping for misdemeanor with BCS summary offense code 30
framejuvexplore0ind = framejuvexplore0.reset_index(name='count')

framejuvexplore0ind

framejuvexplore1 = framejuv30.groupby(['bcs_summary_offence_code', 'disposition', 'race_or_ethnicity']).size()

framejuvexplore1ind = framejuvexplore1.reset_index(name='count')

framejuvexplore1ind

framejuvexplore2 = framejuv30.groupby(['bcs_summary_offence_code', 'status_type', 'race_or_ethnicity']).size()

framejuvexplore2ind = framejuvexplore2.reset_index(name='count')

framejuvexplore2ind

framejuvexplore3 = pd.merge(framejuvexplore0ind, framejuvexplore2ind, how='inner', on=['race_or_ethnicity', 'race_or_ethnicity'])

framejuvexplore3['rate'] = framejuvexplore3['count_y'] / framejuvexplore3['count_x']

framejuvexplore3

framejuvexplore4 = framejuv30.groupby(['race_or_ethnicity', 'arrest_month']).size()
framejuvexplore4ind = framejuvexplore4.reset_index(name='count')
framejuvexplore5 = framejuv30.groupby(['bcs_summary_offence_code', 'status_type', 'race_or_ethnicity', 'arrest_month']).size()
framejuvexplore5ind = framejuvexplore5.reset_index(name='count')
framejuvexplore6 = pd.merge(framejuvexplore4ind, framejuvexplore5ind, how='inner', on=['race_or_ethnicity', 'arrest_month'])
framejuvexplore6['rate'] = framejuvexplore6['count_y'] / framejuvexplore6['count_x']

framejuvexplore6asian = framejuvexplore6.loc[framejuvexplore6['race_or_ethnicity'] == 'Asian/Pacific Islander']
framejuvexplore6asianbooked = framejuvexplore6asian.loc[framejuvexplore6asian['status_type'] == 'booked']

framejuvexplore6asianbooked

framejuvexplore6black = framejuvexplore6.loc[framejuvexplore6['race_or_ethnicity'] == 'Black']
framejuvexplore6blackbooked = framejuvexplore6black.loc[framejuvexplore6black['status_type'] == 'booked']

framejuvexplore6blackbooked

framejuvexplore6hispanic = framejuvexplore6.loc[framejuvexplore6['race_or_ethnicity'] == 'Hispanic']
framejuvexplore6hispanicbooked = framejuvexplore6hispanic.loc[framejuvexplore6hispanic['status_type'] == 'booked']

framejuvexplore6hispanicbooked

framejuvexplore6white = framejuvexplore6.loc[framejuvexplore6['race_or_ethnicity'] == 'White']
framejuvexplore6whitebooked = framejuvexplore6white.loc[framejuvexplore6white['status_type'] == 'booked']

framejuvexplore6whitebooked

from scipy import stats
stats.shapiro(framejuvexplore6asianbooked.rate)

stats.shapiro(framejuvexplore6blackbooked.rate)

stats.shapiro(framejuvexplore6hispanicbooked.rate)

stats.shapiro(framejuvexplore6whitebooked.rate)

np.mean(framejuvexplore6asianbooked.rate)

np.mean(framejuvexplore6blackbooked.rate)

np.mean(framejuvexplore6hispanicbooked.rate)

np.mean(framejuvexplore6whitebooked.rate)

stats.ttest_ind(a= framejuvexplore6asianbooked.rate,
                b= framejuvexplore6blackbooked.rate,
                equal_var=False) 

stats.ttest_ind(a= framejuvexplore6hispanicbooked.rate,
                b= framejuvexplore6blackbooked.rate,
                equal_var=False) 

stats.ttest_ind(a= framejuvexplore6whitebooked.rate,
                b= framejuvexplore6blackbooked.rate,
                equal_var=False) 

stats.ttest_ind(a= framejuvexplore6whitebooked.rate,
                b= framejuvexplore6asianbooked.rate,
                equal_var=False) 

framejuv397 = framejuv.loc[framejuv['bcs_offense_code'] == 397]
framejuv397explore1 = framejuv397.groupby(['race_or_ethnicity', 'arrest_month']).size()
framejuv397explore1ind = framejuv397explore1.reset_index(name='count')
framejuv397explore2 = framejuv397.groupby(['bcs_offense_code', 'status_type', 'race_or_ethnicity', 'arrest_month']).size()
framejuv397explore2ind = framejuv397explore2.reset_index(name='count')
framejuv397merged = pd.merge(framejuv397explore1ind, framejuv397explore2ind, how='inner', on=['race_or_ethnicity', 'arrest_month'])
framejuv397merged['rate'] = framejuv397merged['count_y'] / framejuv397merged['count_x']

framejuv397asian = framejuv397merged.loc[framejuv397merged['race_or_ethnicity'] == 'Asian/Pacific Islander']
framejuv397asianbooked = framejuv397asian.loc[framejuv397asian['status_type'] == 'booked']

framejuv397black = framejuv397merged.loc[framejuv397merged['race_or_ethnicity'] == 'Black']
framejuv397blackbooked = framejuv397black.loc[framejuv397black['status_type'] == 'booked']

framejuv397hispanic = framejuv397merged.loc[framejuv397merged['race_or_ethnicity'] == 'Hispanic']
framejuv397hispanicbooked = framejuv397hispanic.loc[framejuv397hispanic['status_type'] == 'booked']

framejuv397white = framejuv397merged.loc[framejuv397merged['race_or_ethnicity'] == 'White']
framejuv397whitebooked = framejuv397white.loc[framejuv397white['status_type'] == 'booked']

np.mean(framejuv397asianbooked.rate)

np.mean(framejuv397blackbooked.rate)

np.mean(framejuv397hispanicbooked.rate)

np.mean(framejuv397whitebooked.rate)

stats.ttest_ind(a= framejuv397asianbooked.rate,
                b= framejuv397blackbooked.rate,
                equal_var=False) 

stats.ttest_ind(a= framejuv397whitebooked.rate,
                b= framejuv397blackbooked.rate,
                equal_var=False) 

stats.ttest_ind(a= framejuv397whitebooked.rate,
                b= framejuv397hispanicbooked.rate,
                equal_var=False)

stats.ttest_ind(a= framejuv397whitebooked.rate,
                b= framejuv397asianbooked.rate,
                equal_var=False)

stats.ttest_ind(a= framejuv397blackbooked.rate,
                b= framejuv397hispanicbooked.rate,
                equal_var=False)

