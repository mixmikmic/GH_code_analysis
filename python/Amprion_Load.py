import pandas as pd
import numpy as np
import os

filepath = "../raw_data/load_data/Amprion Load"
filenames = os.listdir(filepath)
oldformat_filenames = filenames[0]
newformat_filenames = filenames[1:]

def extract_one_year(filename, old_format=True):
    if old_format:
        df = pd.read_csv(filename, sep=';')
        df.columns = ['date', 'time', 'forecast', 'actual']
        #split time column
        df['from'] = df['time'].apply(lambda x: x.split(' - ')[0]) 
        df['to'] = df['time'].apply(lambda x: x.split(' - ')[1])
        # add date to from and to column
        df['from'] = df.apply(lambda row: pd.to_datetime(row['date'] + '-' + row['from'], format='%d.%m.%Y-%H:%M'),axis=1)
        df['to']   = df.apply(lambda row: pd.to_datetime(row['date'] + '-' + row['to'], format='%d.%m.%Y-%H:%M'),axis=1)
        # change str date to datetime date
        df['date'] = df['date'].apply(lambda x: pd.to_datetime(x, format='%d.%m.%Y'))

        # drop unnecessary cols
        df.drop(['time'], axis=1, inplace=True)


    else:
        df = pd.read_csv(filename)
        df.columns = ['from-to', 'forecast', 'actual']
        df['from'], df['to'] = df['from-to'].str.split(' - ').str
        df['date'] = df['from-to'].str.split(' ').str[0]
        
        # change str date to datetime date
        # change dtype to datetime
        print("Processing pd.to_datetime('date')...")
        df['date'] = df['date'].apply(lambda x: pd.to_datetime(x, format='%d.%m.%Y'))
        print("Processing pd.to_datetime('from')...")
        df['from'] = df['from'].apply(lambda x: pd.to_datetime(x, format='%d.%m.%Y %H:%M'))
        print("Processing pd.to_datetime('to')...")
        df['to'] = df['to'].apply(lambda x: pd.to_datetime(x, format='%d.%m.%Y %H:%M'))
        
        #drop unnecessary cols
        df.drop(['from-to'], axis=1, inplace=True)
        
    # reorder cols
    df = df[['date', 'from', 'to', 'actual', 'forecast']]
    return df


print('Extracting oldformat files...')
oldformat_df = extract_one_year(os.path.join(filepath,oldformat_filenames), old_format=True)

print('Extracting newformat files...')
newformat_df = pd.DataFrame()
for filename in newformat_filenames:
    print('Processing {}'.format(filename))
    newformat_df = newformat_df.append(extract_one_year(os.path.join(filepath,filename), old_format=False))

# Merge old and new
oldformat_df = oldformat_df.append(newformat_df)
oldformat_df.reset_index(drop=True,inplace=True)

oldformat_df.to_csv('../input/Load_Amprion_2010-2017_cleaned.csv', index=None)

    

h = pd.read_csv('../input/Load_Amprion_2010-2017_cleaned.csv')
h['from'] = h['from'].apply(lambda x: pd.to_datetime(x))
h.loc[(h['from'].diff(1) != pd.Timedelta(minutes=15))]



