import pandas as pd
dc_listings = pd.read_csv('dc_airbnb.csv')
print(dc_listings.columns, '\n','\n', dc_listings.shape, '\n','\n', dc_listings.head())

import numpy as np
our_acc_value = 3
first_living_space_value = dc_listings.iloc[0]['accommodates']
first_distance = np.abs(first_living_space_value - our_acc_value)
print(first_distance)

#heres one way
#dc_listings['distance'] = [abs(x-3) for x in dc_listings.accommodates]    

#or heres another!
new_listing = 3
dc_listings['distance'] = dc_listings['accommodates'].apply(lambda x: np.abs(x-new_listing))
print(dc_listings['distance'].value_counts())

    

import numpy as np
np.random.seed(1)

rand_order = np.random.permutation(len(dc_listings))
dc_listings = dc_listings.loc[rand_order]
dc_listings = dc_listings.sort_values('distance')
print(dc_listings.iloc[0:10]['price'])

no_commas = dc_listings['price'].str.replace(',','')
no_dolla_sign = no_commas.str.replace('$','').astype('float')
dc_listings.price = no_dolla_sign

mean_price = dc_listings.price[0:5].mean()
mean_price

# Brought along the changes we made to the `dc_listings` Dataframe.
import numpy as np
np.random.seed(1)
dc_listings = pd.read_csv('dc_airbnb.csv')
stripped_commas = dc_listings['price'].str.replace(',', '')
stripped_dollars = stripped_commas.str.replace('$', '')
dc_listings['price'] = stripped_dollars.astype('float')
dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]

def predict_price(new_listing,k):
    temp_df = dc_listings
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbors = temp_df.iloc[0:k]['price']
    predicted_price = nearest_neighbors.mean()
    return(predicted_price)

acc_one = predict_price(1,5)
acc_two = predict_price(2,5)
acc_four = predict_price(4,5)
print(acc_one)
print(acc_two)
print(acc_four)

