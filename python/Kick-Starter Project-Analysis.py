import re
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('DSI_kickstarterscrape_dataset.csv')
df.head()

df.shape

df.status.value_counts()

df.dtypes

df.isnull().sum()

df.category.describe()

# Drop status rows labeled as live, canceled, suspended.

df = df[~df['status'].isin(['live', 'canceled', 'suspended'])]

# Replace space in clomun with underscore.
df.columns = df.columns.str.replace(' ', '_')

df.shape

df.head(1)

# Split up the date to day, time, year, and month.

df['funded_day'] = df.funded_date.apply(lambda x: x.split(',')[0])

df['funded_date'] = pd.to_datetime(df['funded_date'])
df['funded_time'] = [d.time() for d in df['funded_date']]
df['funded_Newdate'] = [d.date() for d in df['funded_date']]

# Convert to String and then to int

df['funded_year'] = df['funded_Newdate'].apply(lambda date: str(date).split('-')[0]).astype(str).astype(int)
df['funded_month'] = df['funded_Newdate'].apply(lambda date: str(date).split('-')[1]).astype(str).astype(int)
df['day_of_month'] = df['funded_Newdate'].apply(lambda date: str(date).split('-')[2]).astype(str).astype(int)

df.head(2)

df.funded_year.value_counts()

df.shape

# Month(s) with more saturday is better to launch a project?
df.funded_day.value_counts().plot(figsize=(20, 6))

# Drop irrelevant columns.

df.drop(['project_id', 'name', 'url', 'reward_levels' ], axis = 1, inplace = True)
df.head()

# Drop NaN values

df.dropna(inplace = True)

# Drop a particular row because it location is unavailable.

df.drop([7762], inplace = True)

df.head()

# Split up City and States because some of them contain State and Country.

df['city'] = df.location.apply(lambda x: x.split(',')[0])
df['state'] = df.location.apply(lambda x: x.split(',')[1])

df.state.unique()

df.head()

df.shape

df.status.value_counts(normalize = True)

df.head(2)

# Split up States and create a new column for that.

def split_it(state):
     return re.findall('([A-Z]{2,})', state)

df['STATE'] = df['state'].apply(split_it)

def split_it(state):
     return re.findall('[A-Z][a-z]+', state)

df['Country'] = df['state'].apply(split_it)

df.head(2)

# Remove [] in STATE Column and Country

df['STATE'] = df['STATE'].str[0]
df['Country'] = df['Country'].str[0]

df.head()

# Fill in NaN values for Country (United State)
df.Country.fillna(value = 'United States', inplace = True)

df.head(3)

# Assigning country names using country listed in state column
df.loc[df.Country == 'United', 'Country'] = df.loc[df.Country == 'United', 'state']
df.loc[df.Country == 'South', 'Country'] = df.loc[df.Country == 'South', 'state']
df.loc[df.Country == 'Saint', 'Country'] = df.loc[df.Country == 'Saint', 'state']
df.loc[df.Country == 'Saudi', 'Country'] = df.loc[df.Country == 'Saudi', 'state']
df.loc[df.Country == 'Mt', 'Country'] = df.loc[df.Country == 'Mt', 'state']
df.loc[df.Country == 'Viet', 'Country'] = df.loc[df.Country == 'Viet', 'state']
df.loc[df.Country == 'Kyoto', 'Country'] = df.loc[df.Country == 'Kyoto', 'state']
df.loc[df.Country == 'Sri', 'Country'] = df.loc[df.Country == 'Sri', 'state']
df.loc[df.Country == 'Sierra', 'Country'] = df.loc[df.Country == 'Sierra', 'state']
df.loc[df.Country == 'Isle', 'Country'] = df.loc[df.Country == 'Isle', 'state']
df.loc[df.Country == 'Central', 'Country'] = df.loc[df.Country == 'Central', 'state']
df.loc[df.Country == 'Argent', 'Country'] = df.loc[df.Country == 'Argent', 'state']
df.loc[df.Country == 'El', 'Country'] = df.loc[df.Country == 'El', 'state']
df.loc[df.Country == 'Costa', 'Country'] = df.loc[df.Country == 'Costa', 'state']
df.loc[df.Country == 'Svalbard', 'Country'] = df.loc[df.Country == 'Svalbard', 'state']
df.loc[df.Country == 'Virgin', 'Country'] = df.loc[df.Country == 'Virgin', 'state']
df.loc[df.Country == 'Czech', 'Country'] = df.loc[df.Country == 'Czech', 'state']
df.loc[df.Country == 'New', 'Country'] = df.loc[df.Country == 'New', 'state']
df.loc[df.Country == 'Hong', 'Country'] = df.loc[df.Country == 'Hong', 'state']
df.loc[df.Country == 'Argentinaina', 'Country'] = df.loc[df.Country == 'Argentinaina', 'state']
df.loc[df.Country == 'Puerto', 'Country'] = df.loc[df.Country == 'Puerto', 'state']
df.loc[df.Country == 'Papua', 'Country'] = df.loc[df.Country == 'Papua', 'state']
df.loc[df.Country == 'Dominican', 'Country'] = df.loc[df.Country == 'Dominican', 'state']
df.loc[df.Country == 'Falkland', 'Country'] = df.loc[df.Country == 'Falkland', 'state']
df.loc[df.Country == 'Palestinian', 'Country'] = df.loc[df.Country == 'Palestinian', 'state']
df.loc[df.Country == 'Palestinian', 'Country'] = df.loc[df.Country == 'Palestinian', 'state']
df.loc[df.Country == 'Libyan', 'Country'] = df.loc[df.Country == 'Libyan', 'state']
df.loc[df.Country == 'Syrian', 'Country'] = df.loc[df.Country == 'Syrian', 'state']

# Rename Countries
df.Country = df.Country.str.replace('Mt', 'United States ')
df.Country = df.Country.str.replace('Kyoto', 'Japan')
df.Country = df.Country.str.replace('Argent', 'Argentina')
df.Country = df.Country.str.replace('Argentinaina', 'Argentina')
df.Country = df.Country.str.replace('Dominican Re', 'Dominican Republic')
df.Country = df.Country.str.replace('Congo', 'DR Congo')

# Rename City
df.city = df.city.str.replace('Ciudad Aut���_noma De Buenos Aires', 'Buenos Aires')
df.city = df.city.str.replace('Lim���_n', 'Lima')
df.city = df.city.str.replace('Panam��΍', 'San Jose')
df.city = df.city.str.replace('H�Ċܢibiny, ', 'Hradec Králové')

# Replace the space infront of the changed countries.
df.Country = df.Country.str.strip()

df.head(5)

df.drop(['state', 'location', 'funded_date', 'funded_time', 'funded_year'], axis =1, inplace = True)

df.head()

# Funded Percentage of each categories
df.groupby('category').funded_percentage.mean()

#bins = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
#group_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
#categories = pd.cut(df.funded_month, bins, labels=group_names)
#df.funded_month = categories

# df.to_csv('cleaned_kickstarter.csv', index = False)

#df['status'] = df.status.str.contains('successful').astype(int)

df.rename(columns={'STATE':'state'}, inplace =True)

df.head(2)

df.Country.unique()

# df.state = pd.to_numeric(df.state, errors='coerce').fillna(0).astype(np.int64)

def classifier(row):
    if row.Country in ['United States', 'Canada', 'Guatemala', 'Mexico', 'Puerto Rico', 'Nicaragua', 'El Salvador', 'Panama', 'Bolivia', 'Middleburg', 'Guam']:
        return 'America'
    elif row.Country in ['Nigeria', 'Ghana', 'South Africa', 'Kenya', 'Ethiopia', 'DR Congo', 'Morocco', 'Tanzania', 'Zambia', 'Liberia', 'Rwanda', 'Mali', 'Cameroon', 'Namibia', 'Zimbabwe', 'Tunisia', 'Sierra Leone', 'Central African Republic', 'Uganda', 'Sudan', 'Senegal', 'Malawi', 'Mozambique', 'Libyan Arab Jamahiriya', 'Guinea', 'Swaziland']:
        return 'Africa'
    elif row.Country in ['United Kingdom', 'Norway', 'Germany', 'Sweden', 'Bosnia', 'Iceland', 'Hungary', 'Italy', 'Netherlands','France', 'United Kingdom', 'Austria',
       'Turkey','Finland', 'Czech Republic','Armenia', 'Portugal','Denmark','Switzerland', 'Svalbard and Jan Mayen', 'Russia', 'Ukraine', 'Bulgaria','Spain','Poland', 'Georgia','Ireland','Greece','Serbia','Slovenia','Belgium','Greenland','Romania','Lithuania', 'Micronesia','Estonia','Cyprus', 'Macedonia','Kyrgyzstan',]:
        return 'Europe'
    elif row.Country in ['Jamaica', 'Haiti','Bahamas','Dominican Republic','Saint Lucia', 'Dominican Republicpublic', 'Trinidad']:
        return 'Carribean'
    elif row.Country in ['China', 'Taiwan', 'Hong Kong', 'Nepal', 'Indonesia', 'Singapore', 'India', 'Japan', 'Lebanon', 'Kazakhstan', 'South Korea', 'Philippines', 'Cambodia', 'Thailand','Malaysia','Bhutan','Sri Lanka','Bermuda','Viet Nam','Bangladesh', 'Laos','Guam']:
        return 'Asia'
    elif row.Country in ['Israel','Qatar', 'Afghanistan','Kazakhstan','United Arab Emirates','Palestinian Territories','Syrian Arab Republic','Saudi Arabia', 'Iraq','Iran','Tajikistan',]:
        return 'Arab'
    else:
        return "Oceania"   
df["continent"] = df.apply(classifier, axis=1)

df.continent.value_counts()

# from sklearn import preprocessing
# def encode_features(df):
#     features = ['category', 'status', 'subcategory', 'state', 'pledged', 'backers', 'duration', 'funded_month', 'city']
#     df_combined = pd.concat([df])
    
#     for feature in features:
#         le = preprocessing.LabelEncoder()
#         le = le.fit(df_combined[feature])
#         df[feature] = le.transform(df[feature])
#     return df
    
# data = encode_features(df)
# data.head()

df.head()

df.shape

df.continent.value_counts()

sns.countplot(x='continent',data=df,hue='status')

df.status.value_counts(normalize = True)

sns.countplot(x='status',data=df)  # 55% of projects launched were successful, and 45% failed

# .sort_index() was used to keep "successful" and "failed" in their respective order
color = ['r', 'g']

df.groupby('continent').status.value_counts(normalize = True).sort_index().plot(kind = 'bar', color = color)

color = ['r', 'g']

df.groupby('state').status.value_counts(normalize = True).sort_index().plot(kind = 'bar', color = color, figsize=(20, 6))

color = ['r', 'g']

df.groupby('funded_month').status.value_counts(normalize = True).sort_index().plot(kind = 'bar', color = color, figsize=(20, 6))

color = ['r', 'g']

df.groupby('category').status.value_counts(normalize = True).sort_index().plot(kind = 'bar', color = color, figsize=(20, 6))

color = ['r', 'g']

df.groupby('subcategory').status.value_counts(normalize = True).sort_index().plot(kind = 'bar', color = color, figsize=(20, 6))

sns.factorplot(x='category', y='goal',data=df, hue='continent',kind='bar', size = 5, aspect=3)

df.goal.mean()

df.pledged.mean()

color = ['r', 'g', 'b', 'y', 'm']
df.groupby('continent').goal.mean().sort_index().plot(kind = 'bar', color = color)
plt.title('Continent goal mean')
plt.ylabel('goal_mean')
plt.xlabel('continent')

color = ['r', 'g', 'b', 'y', 'm']
df.groupby('continent').pledged.mean().sort_index().plot(kind = 'bar', color = color)
plt.title('Continent pleadge mean')
plt.ylabel('pledged_mean')
plt.xlabel('continent')

# # df['status'] = df.status.str.contains('successful').astype(int)
# # 1 = Succesful
# # 0 = Failed

# color = ['r', 'g', 'b', 'y', 'm']
# df.groupby('continent').status.mean().sort_index().plot(kind = 'bar', color = color)

df[df['goal']>100000].shape

# only 289 projects had goals over $100,000.

df[df['goal']>100000]['status'].value_counts(normalize = True) 
# and only 22 (less than 1%) of those projects got funded 

# only 7296 projects had goals over $100,000.
df[df['goal']>10000]['status'].value_counts(normalize = True) 
# and only 2322 (less than 32%) of those projects got funded 

# Over 98% of projects with a pledged of over $100,000 were succesful
df[df['pledged']>100000]['status'].value_counts(normalize = True) 

df['loggoal'] = np.log10(df['goal'])
sns.lmplot(x = 'loggoal', y = 'pledged', col ='status', data = df, fit_reg = False)

df.head()

plt.figure(figsize = (6,6))
sns.boxplot(x ='status', y = 'loggoal', data = df)
plt.title('Successful Kickstarters have on average lower Goals')

# Number of Projects lunch each month

sns.countplot(x='funded_month',data=df)

color = ['r', 'g']

df.groupby('funded_month').status.value_counts(normalize = True).sort_index().plot(kind = 'bar',color = color, figsize=(20, 6))
plt.title('funded month vs status(percentage)')
plt.ylabel('status percentage')
plt.xlabel('funded_month')

df.category.value_counts(normalize = True).plot(kind = 'pie', figsize=(9, 6))

df.head()

# sum of successful project
df[df['status']=='successful']['pledged'].sum()

# Avg updates for failed projects
df[df['status']=='failed']['updates'].mean()

# Avg duration for successful projects
df[df['status']=='successful']['duration'].mean()

df.backers.mean()

df[df['status']=='successful']['backers'].mean()

df.funded_month.mean()

df.groupby('status').duration.mean().sort_index().plot(kind = 'box', figsize=(10, 6))

df[df['status']=='successful']['category'].value_counts().sort_index().plot(kind = 'bar', figsize=(12, 6))

df[df['status']=='failed']['category'].value_counts().sort_index().plot(kind = 'bar', figsize=(12, 6))

df[df['status']=='successful']['pledged'].mean()

df.pledged.mean()

df[df['status']=='successful']['state'].value_counts().sort_index().plot(kind = 'bar', figsize=(12, 6))

df['status_'] = df.status.map(lambda x:1 if x=='successful'else 0)

df.head()

# Success percentage per category
df.groupby('category').status_.mean().sort_index().plot(kind = 'bar', figsize=(12, 6))

df.groupby('category').status_.mean()

# Success percentage by states
df.groupby('state').status_.mean().sort_index().plot(kind = 'bar', figsize=(15, 6))

# Success percentage by states
df.groupby(['state', 'category']).status_.mean().sort_index()



