import pandas as pd
import numpy as np
from os import listdir
from datetime import datetime

#Read in names extracted from Appellate opinions
doc = pd.read_csv('full_name_extraction.csv')

#Create a copy of the extracted names file
working_doc = doc.copy()

working_doc.head()

#Create a year column
working_doc['year'] = working_doc.File.str[:4]

#Drop unnecessary for the analysis columns
working_doc.drop(['Text','Law_Number',
      'Appellate_Name_Match','Full_Name'],axis=1,inplace=True)

#Combine all NYS DOC files into a single dataframe for processing
doc = pd.concat( [ pd.read_csv('NYS DOC/'+f) for f in listdir('NYS DOC') ] )

doc.head()

#Remove blank spaces from column names in DOC file
for value in doc.columns:
    name = value.strip()
    doc.rename(columns= {value:name},inplace=True)

#Read in the original criminal data file to get more data for matching
full = pd.read_csv('archive/criminal_main.csv',encoding='latin-1')

#There are 36,314 files in the original dataframe
full.shape[0]

#Remove white spaces from dates
doc['Date Received (Current)'] = doc['Date Received (Current)'].str.strip()

def str_to_date(dat,typ=1):
    '''
    Function to convert string date to date format
    '''
    try: 
        if typ == 1:
            out = datetime.strptime(dat, '%m/%d/%Y').date()
        else:
            out = datetime.strptime(dat, '%m/%d/%y').date()
    except ValueError:
        print(dat)
        out = None
    except TypeError:
        out = None
    return out

#Convert current date received to date format
doc['Date_received'] = doc['Date Received (Current)'].map(lambda x: str_to_date(x))

#Remove any spaces in the date received original
doc['Date_received2'] = doc['Date Received (Original)'].str.strip()

#Convert date received current to date format
doc['Date_received2'] = doc['Date_received2'].map(lambda x: str_to_date(x))

#Function to pick the earliest of two dates while ensuring the date is not null
def min_date(date1,date2):
    try:
        if date1<date2 and date1.year>1900:
            return date1
        elif date2.year>1900:
            return date2
        else:
            return None
    except TypeError:
        if date1 == None:
            return date2
        else:
            return date1

#Pick the earliest date between the current and original one
doc['earlier_date'] = doc.apply(lambda x: min_date(x['Date_received'],x['Date_received2']),axis=1)

#Check earliest date
min(doc.earlier_date.dropna())

doc.columns

#Check for duplicated rows and drop them
dedup = doc.drop_duplicates()

#Ensure there're no duplicates
dedup.duplicated().any()

#Drop rows with missing last name, and clean white spaces
dedup.dropna(axis=0,subset = ['Inmate name'],inplace=True)
dedup['Inmate name'] = dedup['Inmate name'].apply(lambda x: x.strip() if isinstance(x,str) else '')

#Remove white spaces from first names
dedup['First name'] = dedup['First name'].apply(lambda x: x.strip() if isinstance(x,str) else '')

#Clean spaces in working doc
working_doc['Defendent_Last_Name'] = working_doc['Defendent_Last_Name'].apply(lambda x: x.strip())

#Remove white spaces
working_doc['Defendent_First_Name'] = working_doc['Defendent_First_Name'].apply(lambda x: x.strip() 
                                                                                if isinstance(x,str) else '')

#Merge records by first and last name - exact match
merged = working_doc.merge(dedup,how='inner',
                           left_on=['Defendent_Last_Name',
                                    'Defendent_First_Name'],
                          right_on=['Inmate name','First name'])

#We have duplicated entries still - need to fix that
sum(merged.File.duplicated())

#To address the issue of duplicates, I'll check the full name
merged['full_name'] = merged['Defendent_First_Name'] + merged['Defendent_Last_Name']

#We also have duplicated file numbers
merged[merged.File.duplicated()].head()

#Create unique id for each inmate for identifying duplicates
merged['id1'] = merged['full_name']+merged['DIN']

dup_file = merged[merged.File.duplicated()]

#There're 8025 duplicates by ID
sum(dup_file.id1.duplicated())

#Add first date and county to df
to_merge = full[['File','County', 'First_Date']]

#Fill missing values
to_merge.County.fillna('missing',inplace=True)
to_merge['county'] = to_merge.County.map(lambda x: x.replace(' County',''))

to_merge.county.replace('King','Kings',inplace=True)

#Convert county to uppercase
to_merge.county = to_merge.county.str.upper()

#Convert string date to date time
to_merge['date'] = to_merge['First_Date'].map(lambda x: str_to_date(x,2))

#Map values on File
keys = to_merge.merge(merged, how='right', on='File')

#Rename columns
keys.rename(columns={'date':'trial_date'},inplace=True)

#Check dates: earlier date is admission to prison, so has to be after trial 
keys['years_from_trial'] = (keys['earlier_date']-keys['trial_date']).dt.days/365

#Convert years from trial to integer
keys.years_from_trial.fillna(0,inplace=True)
keys['rounded_years_frm_trial'] = keys['years_from_trial'].astype(int)

#absolute value
keys['abs_years'] = abs(keys['rounded_years_frm_trial'])

keys.shape

#Removing duplicates round 1: sort by absolute years between the trial date and prison admission date.
#We assume the dates should be close to each other. Then remove non-unique ids keeping the first (with the closest
#date between trial date and admission to prison date)
df1 = keys.sort_values('abs_years').drop_duplicates(subset=['id1'],keep='first')

df1.File.duplicated().any()

df1.shape

#Removing duplicates round 2: again, sort by absolute years and drop duplicated file ID's. Again, we assume
#the correct person has the date of admission to prison that is the closest to the trial date.
df2 = df1.sort_values('abs_years').drop_duplicates(subset=['File'],keep='first')

#Unique values remaining
df2.shape

#No more duplicates
df2.File.duplicated().any()

#Exclude values with too many missing values
output = df2.drop(['Maximum Expiration Date for Parole Supervision','Maximum Expiration Date for Parole Supervision',
             'Aggregate Max Sentence', 'Aggregate Min Sentence', 'Conditional Release Date','Maximum Expiration Date',
             'Maximum Expiration Date', 'Crime/class3','Crime/class4', 'Earliest Release Date',
                  'Earliest Release Type','Post Release Supervision Maximum Expiration Date',
                  'Parole Hearing Type','Parole Hearing Date','Parole Eligibility Date', 'County', 'abs_years', 
                  'rounded_years_frm_trial','earlier_date','full_name'],axis=1)

output.columns

output = output.drop(['First name', 'Inmate name','id1','county'],axis=1)

output.head()

output.shape

output.File.isnull().any()

#output.to_csv('matched_DOC.csv',index=False)

