import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
pd.set_option('display.max_columns', 500)

df = pd.read_csv('data/listings.csv')

len(df)

candy = df[df['id'] == 7095125]

candy

df.columns

table_names = ['name',            'summary',            'description',            'host_since',            'host_is_superhost',            'neighbourhood_cleansed',             'neighbourhood_group_cleansed',             'city',             'room_type',            'accommodates',            'bathrooms',             'bedrooms',             'beds',             'amenities',            'square_feet',            'extra_people',            'number_of_reviews',            'review_scores_value',            'review_scores_rating',            'price'
           ]

df_clean = df[table_names].copy()

df_clean.price = df_clean['price'].apply(lambda x: float(x.replace('$', '').replace(',','')))
df_clean.extra_people = df_clean.extra_people.apply(lambda x: float(x.replace('$', '').replace(',','')))

df_clean['host_is_superhost'] = df_clean['host_is_superhost'].apply(lambda x: True if x == 't' else False)
df_clean['number_of_amenities'] = df_clean['amenities'].apply(lambda x: len(x))

from pandas.tools.plotting import scatter_matrix

scatter_matrix(df_clean[['accommodates','bedrooms','beds','extra_people','number_of_reviews','review_scores_rating', 'number_of_amenities', 'square_feet','price']], figsize=(12,12))

from patsy import dmatrices

y,X = dmatrices('price ~ host_is_superhost + neighbourhood_group_cleansed + room_type + accommodates + bathrooms +                     bedrooms + beds + number_of_amenities + number_of_reviews + review_scores_rating',                 data = df_clean, return_type='dataframe')

import statsmodels.api as sm

model = sm.OLS(y,X)
results = model.fit()
results.summary()

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

avg_MSE = []
alphas = np.linspace(-2, 8, 20, endpoint=False)
alphas
for alpha in alphas:
    MSE = []
#     for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)
    #model = sm.OLS(X_train, y_train)
    model = Lasso(alpha=alpha)
    model.fit(X_test, y_test)
    test_error = mean_squared_error(y_test, model.predict(X_test))
    MSE.append(test_error)
    avg_MSE.append(np.mean(MSE))
print avg_MSE

plt.figure(figsize=(12,8))
plt.xlabel('alpha', fontsize=14)
plt.ylabel('Cross Validation MSE', fontsize=14)
plt.title('alpha vs. Cross Validation MSE', fontsize=14)
plt.plot(alphas, avg_MSE)

relevant = ['id',            'host_id',            'host_name',            'host_is_superhost',            
            'host_since',\
            
            'name',\
            'neighbourhood_cleansed', \
            'neighbourhood_group_cleansed', \
            'city', \
            
            'price',\
            
            'room_type',\
            'accommodates',\
            'bathrooms', \
            'bedrooms', \
            'beds', \
            'amenities',\
            'description',\
            
            'first_review',\
            'last_review',\
            'number_of_reviews',\
            'review_scores_value',\
            'review_scores_rating'\
           ]

df[relevant].to_csv('data/listing_info.csv', index=False)

# df.groupby(['name','month'])['text'].apply(lambda x: ','.join(x)).reset_index()

# df_clean['neighbourhood_cleansed']

# df_clean.neighbourhood_cleansed.apply(pd.value_counts).plot(kind='bar')





