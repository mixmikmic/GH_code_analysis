import pandas as pd

df = pd.read_csv('../data/gators.csv')

df.head()

# get the info()
df.info()

# what's the year range, with counts?
df['Year'].value_counts()

# let's also peep the carcass size values to get the pattern
df['Carcass Size'].unique()

def get_inches(row):
    carcass_size = row['Carcass Size']
    size_split = carcass_size.split('ft.')
    feet = int(size_split[0].strip())
    inches = int(size_split[1].replace('in.', '').strip())
    return inches + (feet * 12)

df['length_in'] = df.apply(get_inches, axis=1)

# check the output with head()
df.head()

# sort by length descending, check it out with head()
df.sort_values('length_in', ascending=False).head()

df['Year'].value_counts()

# get average length harvested by year
length_by_year = pd.pivot_table(df,
                                values='length_in',
                                index='Year',
                                aggfunc='mean')

print(length_by_year)

df['Harvest Date'] = pd.to_datetime(df['Harvest Date'], format='%m-%d-%Y')

df['Harvest Date'] = pd.to_datetime(df['Harvest Date'],
                                    format='%m-%d-%Y',
                                    errors='coerce')

df.head()

df.dtypes

df['month'] = df['Harvest Date'].apply(lambda x: x.month)

df['month'].unique()

df['month'].value_counts().sort_index()

by_month_by_year = pd.pivot_table(df,
                                  values='month',
                                  columns='Year',
                                  aggfunc='count')

by_year_by_month

by_year_by_month.fillna(0)

