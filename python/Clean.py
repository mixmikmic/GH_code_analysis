import pandas as pd
import numpy as np

# import raw data file
df = pd.read_json('yelp_academic_dataset_business.json', lines=True)

# remove permanently closed
df = df[df['is_open'] == 1]
df = df.drop(['is_open'], axis=1)

# remove non restaurants
df = df[df['categories'].apply(str).str.contains("Restaurants")]

# combine Montreal cities
df['city'] = df['city'].replace(u'Montr\xe9al', 'Montreal')

# drop unnecessary columns for now
df = df.drop(['address', 'neighborhood', 'postal_code', 'type'], axis=1)

# strip latitude, longitude, hours, attributes for now too
df = df.drop(['latitude', 'longitude', 'hours', 'attributes'], axis=1)

# save cleaned data
df.to_csv('cleaned.csv', index=False)

