import pandas as pd

df = pd.read_csv('criminal_main.csv',encoding='latin-1')

df.head()

grounds = df.GroundsForAppeal

num_rows = grounds.shape[0]

num_missing = grounds.isnull().sum()

share_missing = num_missing/num_rows
print('Share of missing values:', str(share_missing))

unique  = grounds.unique()

grounds_list = []

for line in grounds:
    if isinstance(line, float): #skip nan
        continue
    words = line.split(';')
    grounds_list.extend(words)

unique_grounds = set(grounds_list)

unique_grounds

#Count number of unique occurrences per category - this does
for ground in unique_grounds:
    print(ground, grounds_list.count(ground))

len(grounds)

mode = df.ModeOfConviction

unique_modes = mode.unique()

num_rows = mode.shape[0]
num_missing = mode.isnull().sum()

share_missing_mode = num_missing/num_rows
print("Share of missing mode of conviction:",str(share_missing_mode))

#Count number of unique occurrences per category
mode_list = mode.tolist()
for ex_mode in unique_modes:
    if isinstance(ex_mode,float):
        continue #ignore nan
    print(ex_mode, mode_list.count(ex_mode))

