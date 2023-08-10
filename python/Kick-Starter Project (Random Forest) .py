import re
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, ExtraTreeClassifier

get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('DSI_kickstarterscrape_dataset.csv')
df.head(1)

df.shape

df.status.value_counts()

# Drop status rows labeled as live, canceled, suspended.

df = df[~df['status'].isin(['live', 'canceled', 'suspended'])]

# Replace space in clomun with underscore.
df.columns = df.columns.str.replace(' ', '_')

df.shape

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

df.shape

# Drop irrelevant columns.

df.drop(['name', 'url', 'reward_levels' ], axis = 1, inplace = True)
df.head()

# Drop NaN values

df.dropna(inplace = True)

# Drop a particular row because it location is unavailable.

df.drop([7762], inplace = True)

df.head()

# Split up City and States because some of them contain State and Country.

df['city'] = df.location.apply(lambda x: x.split(',')[0])
df['state'] = df.location.apply(lambda x: x.split(',')[1])

df.head()

# Split up States and create a new column for that.

def split_it(state):
     return re.findall('([A-Z]{2,})', state)

df['STATE'] = df['state'].apply(split_it)

def split_it(state):
     return re.findall('[A-Z][a-z]+', state)

df['Country'] = df['state'].apply(split_it)

# Remove [] in STATE Column and Country

df['STATE'] = df['STATE'].str[0]
df['Country'] = df['Country'].str[0]

# Fill in NaN values for Country (United State)
df.Country.fillna(value = 'United States', inplace = True)

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
#df.Country = df.Country.str[].replace(' ', '')

df.head(5)

df.drop(['state', 'location', 'funded_date', 'funded_time', 'funded_year', 'funded_Newdate'], axis =1, inplace = True)

df.head()

# df.to_csv('cleaned_kickstarter.csv', index = False)

df['status'] = df.status.str.contains('successful').astype(int)

df.rename(columns={'STATE':'state'}, inplace =True)

df.head(2)

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

df.head()

df.rename(columns = {'Country': 'country'}, inplace = True)

# Convert Features (columns) values to number.

from sklearn import preprocessing
def encode_features(df):
    features = ['category', 'status', 'subcategory', 'state', 'pledged', 'backers', 'funded_month', 'city', 'country', 'continent']
    df_combined = pd.concat([df])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df[feature] = le.transform(df[feature])
    return df
    
data = encode_features(df)
data.head()

df.continent.value_counts()

X = df.drop(['status', 'funded_day', 'day_of_month', 'project_id', 'pledged', 'funded_percentage', 'backers', 'funded_month', 'updates', 'comments'], axis=1)
y = df['status']

X.shape

X.head()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33) 

rf = RandomForestClassifier(n_estimators=1000, max_depth=None, max_features='auto')

dtc = DecisionTreeClassifier(max_depth=None, max_features='auto')
dtc.fit(X_train, y_train)
print 'dtc acc:', dtc.score(X_test, y_test)

rf.fit(X_train, y_train)
yhat = rf.predict(X_test)
print 'rf acc:', accuracy_score(y_test, yhat)

list(zip(X_train.columns, rf.feature_importances_))

# test_preds = rf.predict(X_test)

# print (cross_val_score(rf, X_train, y_train, cv = 3).mean())

# print (cross_val_score(rf, X_test, y_test, cv = 3).mean())

# project_id = X_test.project_id
# predictions = predictions


# output = pd.DataFrame({ 'project_id' : project_id, 'predictions': predictions })
# # output.to_csv('Kickstarter-predictions.csv', index = False)
# output.head()





