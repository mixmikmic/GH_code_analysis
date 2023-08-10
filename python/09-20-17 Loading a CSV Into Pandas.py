import pandas as pd
import numpy as np

raw_data = {'first_name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'], 
        'last_name': ['Miller', 'Jacobson', ".", 'Milner', 'Cooze'], 
        'age': [42, 52, 36, 24, 73], 
        'preTestScore': [4, 24, 31, ".", "."],
        'postTestScore': ["25,000", "94,000", 57, 62, 70]}
df = pd.DataFrame(raw_data, columns = ['first_name', 'last_name', 'age', 'preTestScore', 'postTestScore'])
df

df.to_csv('Data/example.csv')

df = pd.read_csv('Data/example.csv')
df

df = pd.read_csv('Data/example.csv', header=None)
df

df = pd.read_csv('Data/example.csv', 
                 names=['UID', 'First Name', 'Last Name', 
                        'Age', 'Pre-Test Score', 
                        'Post-Test Score'])
df

df = pd.read_csv('Data/example.csv', 
                 index_col='UID', 
                 names=['UID', 'First Name', 
                        'Last Name', 'Age', 
                        'Pre-Test Score', 
                        'Post-Test Score'])
df

df = pd.read_csv('Data/example.csv', 
                 index_col=['First Name', 'Last Name'], 
                 names=['UID', 'First Name', 'Last Name', 'Age', 
                        'Pre-Test Score', 'Post-Test Score'])
df

df = pd.read_csv('Data/example.csv', na_values=['.'])
pd.isnull(df)

sentinels = {'Last Name': ['.', 'NA'], 'Pre-Test Score': ['.']}

df = pd.read_csv('Data/example.csv', na_values=sentinels)
df

df = pd.read_csv('Data/example.csv', na_values=sentinels, skiprows=3)
df

df = pd.read_csv('Data/example.csv', thousands=',')
df

