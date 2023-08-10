import re
import pandas as pd
import numpy as np

pd.set_option('display.float_format', lambda x: '%.3f' % x)

nominations = pd.read_csv('../data/nominations.csv')

# clean out some obvious mistakes...
nominations = nominations[~nominations['film'].isin(['2001: A Space Odyssey', 'Oliver!', 'Closely Observed Train'])]
nominations = nominations[nominations['year'] >= 1980]

# scraper pulled in some character names instead of film names...
nominations.loc[nominations['film'] == 'Penny Lane', 'film'] = 'Almost Famous'
nominations.loc[nominations['film'] == 'Sister James', 'film'] = 'Doubt'

wins = pd.pivot_table(nominations, values='winner', index=['year', 'category', 'film', 'name'], columns=['award'], aggfunc=np.sum)
wins = wins.fillna(0) # if a nominee wasn't in a specific ceremony, we just fill it as a ZERO.

wins.reset_index(inplace=True) # flattens the dataframe
wins.head()

oscars = nominations[nominations['award'] == 'Oscar'][['year', 'category', 'film', 'name']]
awards = pd.merge(oscars, wins, how='left', on=['year', 'category', 'name', 'film'])
awards.head()

films = pd.read_csv('../data/films.csv')

relevant_fields = [
    'film',
    'country',
    'release_date', 
    'running_time', 
    'mpaa',
    'box_office',
    'budget',
    'imdb_score', 
    'rt_audience_score', 
    'rt_critic_score', 
    'stars_count', 
    'writers_count'
]

df = pd.merge(awards, films[relevant_fields], how='left', on='film')

print "Total Observations:", len(df)
print
print "Observations with NaN fields:"

for column in df.columns:
    l = len(df[df[column].isnull()])
    if l != 0:
        print len(df[df[column].isnull()]), "\t", column

### FIX RUN TIME ###
# df[df['running_time'].isnull()] # Hilary and Jackie
df.loc[df['film'] == 'Hilary and Jackie', 'running_time'] = '121 minutes'
df.loc[df['film'] == 'Fanny and Alexander', 'running_time'] = '121 minutes'

### FIX MPAA RATING ###
df = df.replace('NOT RATED', np.nan)
df = df.replace('UNRATED', np.nan)
df = df.replace('M', np.nan)
df = df.replace('NC-17', np.nan)
df = df.replace('APPROVED', np.nan)

# df[df['mpaa'].isnull()]
df.loc[df['film'].isin(['L.A. Confidential', 'In the Loop']), 'mpaa'] = 'R'
df.loc[df['film'].isin(['True Grit', 'A Room with a View']), 'mpaa'] = 'PG-13'

### FIX COUNTRY ###
# df[df['country'].isnull()] # Ulee's Gold, The Constant Gardner, Dave
df.loc[df['film'].isin(["Ulee's Gold", "Dave"]), 'country'] = 'United States'
df.loc[df['country'].isnull(), 'country'] = 'United Kingdom'
df.loc[df['country'] == 'Germany\\', 'country'] = 'Germany'
df.loc[df['country'] == 'United States & Australia', 'country'] = 'United States'
df['country'].unique()

### FIX STARS COUNT ###
# df[df['stars_count'].isnull()]

df.loc[df['film'].isin(['Before Sunset', 'Before Midnight']), 'stars_count'] = 2
df.loc[df['film'] == 'Dick Tracy', 'stars_count'] = 10
df.loc[df['stars_count'].isnull(), 'stars_count'] = 1

df = df[~df['release_date'].isin(['1970'])]

def to_numeric(value):
    multiplier = 1
    
    try:
        value = re.sub(r'([$,])', '', str(value)).strip() 
        value = re.sub(r'\([^)]*\)', '', str(value)).strip()
        
        if 'million' in value:
            multiplier = 1000000  
        elif 'billion' in value:
            multiplier = 10000000
        
        for replace in ['US', 'billion', 'million']:
            value = value.replace(replace, '')
            
        value = value.split(' ')[0]
        
        if isinstance(value, str):
            value = value.split('-')[0]
        
        value = float(value) * multiplier
    except:
        return np.nan
    
    return value

def to_runtime(value):
    try:
        return re.findall(r'\d+', value)[0]
    except:
        return np.nan

                   
### Apply function to appropriate fields ###
for field in ['box_office', 'budget']:
    df[field] = df[field].apply(to_numeric)
    
df['release_month'] = df['release_date'].apply(lambda y: int(y.split('-')[1]))
df['running_time'] = df['running_time'].apply(to_runtime)

### FIX BOX OFFICE ###
list(df[df['mpaa'].isnull()]['film'].unique())

# cleaned_box_offices = {
#     'Mona Lisa': 5794184, 
#     'Testament': 2044982, 
#     'Pennies from Heaven': 9171289, 
#     'The Year of Living Dangerously': 10300000
# }

# for key, value in cleaned_box_offices.items():
#     df.loc[df['film'] == key, 'box_office'] = value
    
# ### FIX BUDGET ###
# # df[(df['budget'].isnull())]['film'].unique()

# cleaned_budgets = {'Juno': 6500000, 'Blue Sky': 16000000, 'Pollock': 6000000 }

# for key, value in cleaned_budgets.items():
#     df.loc[df['film'] == key, 'budget'] = value

df = df[~df['mpaa'].isnull()]

df['produced_USA'] = df['country'].apply(lambda x: 1 if x == 'United States' else 0)

for column in df['mpaa'].unique():
    df[column.replace('-', '')] = df['mpaa'].apply(lambda x: 1 if x == column else 0)

df['q1_release'] = df['release_month'].apply(lambda m: 1 if m <= 3 else 0)
df['q2_release'] = df['release_month'].apply(lambda m: 1 if m > 3 and m <= 6 else 0)
df['q3_release'] = df['release_month'].apply(lambda m: 1 if m > 6 and m <= 9 else 0)
df['q4_release'] = df['release_month'].apply(lambda m: 1 if m > 9 else 0)

df.to_csv('../data/analysis.csv', index=False)

del df['mpaa']
del df['country']
del df['release_date']
del df['release_month']
del df['budget']

for column in df.columns:
    df = df[~df[column].isnull()]

df.to_csv('../data/prepared.csv', index=False)

