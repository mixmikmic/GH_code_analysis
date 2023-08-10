# Import the datasets and merge them to perform the first preprocessing steps
import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

first_health_camp_attended = pd.read_csv('data/first_health_camp_attended.csv')
first_health_camp_attended.drop('Unnamed: 4', axis=1, inplace=True)
second_health_camp_attended = pd.read_csv('data/second_health_camp_attended.csv')
third_health_camp_attended = pd.read_csv('data/third_health_camp_attended.csv')
health_camp_detail = pd.read_csv('data/health_camp_detail.csv')
patient_profile = pd.read_csv('data/patient_profile.csv')

train['is_test'] = np.zeros(train.shape[0])
test['is_test'] = np.ones(test.shape[0])
df = pd.concat([train, test])

df = pd.merge(df, first_health_camp_attended, on=['Patient_ID', 'Health_Camp_ID'], how='left')
df = pd.merge(df, second_health_camp_attended, on=['Patient_ID', 'Health_Camp_ID'], how='left')
df = pd.merge(df, third_health_camp_attended, on=['Patient_ID', 'Health_Camp_ID'], how='left')
df = pd.merge(df, health_camp_detail, on='Health_Camp_ID', how='left')
df = pd.merge(df, patient_profile, on='Patient_ID', how='left')

column_types = []
for column in df.columns:
     column_types.append(str(df[column].dtype))

columns = pd.concat([pd.Series(list(df.columns)), pd.Series(column_types)], axis=1)
columns.columns = ['feature', 'type']
columns

# Convert date feature from object to datetime
for column in ['Registration_Date', 'Camp_Start_Date', 'Camp_End_Date', 'First_Interaction']:
    df[column] = pd.to_datetime(df[column], format="%d-%b-%y")
    
# Convert CategoryX features to int and drop Category3 due to too low variance
column_types = []
for column in df.columns:
     column_types.append(str(df[column].dtype))

columns = pd.concat([pd.Series(list(df.columns)), pd.Series(column_types)], axis=1)
columns.columns = ['feature', 'type']
columns

df['Category1'] = pd.Categorical(df['Category1']).codes
df['Category2'] = pd.Categorical(df['Category2']).codes
df.drop('Category3', axis=1, inplace=True)

# Replace 'None' fields with NA
# Convert 'Income', 'Education_Score', 'Age' to float
to_replace_none = ['Income', 'Education_Score', 'Age']

for column in to_replace_none:
    df.loc[df[column] == 'None', column] = np.nan

df[to_replace_none] = df[to_replace_none].astype(float)

# Convert City_Type and Employer_Category to numerical
df.loc[:, ['City_Type', 'Employer_Category']] =     pd.Categorical(df.loc[:, ['City_Type', 'Employer_Category']]).codes
    
# Generate the Outcome feature
outcomes = df.loc[:, ['Health_Score', 'Health Score', 'Number_of_stall_visited']]
df['Outcome'] = (outcomes.notnull().sum(axis=1) > 0).astype(int)

# Separate the datasets
train = df[df['is_test'] == 0].copy()
test = df[df['is_test'] == 1].copy()

test.drop(['Donation', 'Health_Score', 'Health Score', 'Number_of_stall_visited', 'Last_Stall_Visited_Number',
          'is_test', 'Outcome'], axis=1, inplace=True)

train['Camp_Start_Date'].describe()









