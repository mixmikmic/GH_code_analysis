import pandas as pd
import numpy as np
from datetime import datetime
import os

#Read the csv file that was downloaded from the City of Austin: "https://data.austintexas.gov/"
austin_311_df = pd.read_csv('data/austin_311_service_requests.csv',low_memory=False)
austin_311_df.head()

austin_311_df.shape

austin_311_df.columns

#Keep ['city', 'complaint_description', 'complaint_type', 'county', 'created_date', 
#'incident_zip', 'latitude', 'longitude', 'owning_department']

Austin_311_df = austin_311_df[["city", "county", "incident_zip", 
                            "created_date", "owning_department", 
                            "complaint_description", "complaint_type",
                           "latitude","longitude"]]


Austin_311_df.head()

Austin_311_df.shape

# Check to see if there are any null cells in the rows. 
Austin_311_df.isnull().sum()

#Getting the counts that are not null 
Austin_311_df.count()

#Replace the empty rows with 'NaN'.
Austin_311_df.replace('', np.nan, inplace=True)

#Drop rows with 'NaN'.
clean_Austin_311_df = Austin_311_df.dropna(how="any")
clean_Austin_311_df.head()

# Check to see if there are any null cells in the rows. 
clean_Austin_311_df.isnull().sum()

#Getting the counts for each row.  
clean_Austin_311_df.count()

# Check to see if what cities are in the df.
clean_Austin_311_df['city'].value_counts()

# Check to see if what counties are in the df. 
clean_Austin_311_df['county'].value_counts()

#Keep all cities in Travis county.

Austin_311_Travis = clean_Austin_311_df[clean_Austin_311_df.county.isin(['TRAVIS'])]
Austin_311_Travis['county'].value_counts()

Austin_311_Travis['city'].value_counts()

# Keep 'AUSTIN', 'Austin', and 'austin'. 

All_Austin_311 = Austin_311_Travis[Austin_311_Travis.city.isin(['AUSTIN', 'Austin', 'austin'])]

# Get the new counts
All_Austin_311.count()

All_Austin_311['city'].value_counts()

All_Austin_311.head()

# Change the zip to an integer
All_Austin_311["incident_zip"] = All_Austin_311["incident_zip"].astype(int)

#Drop the City and County since we don't need them anymore. 
All_Austin_311.drop('city',axis=1, inplace=True)
All_Austin_311.drop('county',axis=1, inplace=True)

#Change the date to a datetime format so we can extract month and year for new columns.
All_Austin_311['created_date'] = pd.to_datetime(All_Austin_311['created_date'], format='%Y/%m/%d')

#Create columns for year and month. 
All_Austin_311['year'] = All_Austin_311['created_date'].dt.year
All_Austin_311['month'] = All_Austin_311['created_date'].dt.month

#Drop the Date since we don't need it anymore. 
All_Austin_311.drop('created_date',axis=1, inplace=True)

All_Austin_311.head()

#Save the clean df as .csv file
All_Austin_311.to_csv('data/All_Austin_311.csv', encoding='utf-8', index=False)



