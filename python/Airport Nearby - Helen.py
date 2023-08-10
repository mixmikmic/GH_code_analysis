import requests
import json
import numpy as np
import pandas as pd

from config import gkey, fb_key

csv_path = "Project1_AmazonSites.csv"
city_csv = pd.read_csv(csv_path)
# to remove all null values
city_df = city_csv.iloc[:39,:9]
# change zip to only integer
city_df['zip code'] = city_df['zip code'].astype(int)
for index, zip in city_df['zip code'].iteritems():
    if len(str(zip)) == 4:
        city_df['zip code'].iloc[index] = "0"+str(zip)
city_df.head()

city_param = city_df[['latitude', 'longitude', 'address','site name','city','amazon city']]
city_param

airport_list = []

## Airports
for city in city_param.values:
    target_coordinates = f"{city[0]}, {city[1]}"
    #radius of within 31 miles or 50000 meters
    target_radius = 50000
    target_type = "airport"
    # base url
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    # set up a parameters dictionary
    params = {
        "location": target_coordinates,
        "radius": target_radius,
        "type": target_type,
        "key": gkey
    }

    # run a request using our params dictionary
    response = requests.get(base_url, params=params)
    airport_json = response.json()
    for airport_num in range(len(airport_json['results'])):
        airport_str = airport_json['results'][airport_num]['name']
        if 'rating' in airport_json['results'][airport_num].keys():
            airport_rating = airport_json['results'][airport_num]['rating']
        else:
            airport_rating = None
        airport_list.append({'Site Name':city[3],
                     'City':city[4],
                     'Amazon City':city[5],
                     'Airport':airport_str,
                     'Airport Rating':airport_rating})
airport_df = pd.DataFrame(airport_list)
len(set(airport_df['Airport']))

#clean up any NaN
airport_df.replace(["NaN", 'NaT'], np.nan, inplace = True)
airport_df = airport_df.dropna()
print(airport_df.head())
len(set(airport_df['Airport']))

amazon_city_group = airport_df.groupby(['Amazon City','City','Site Name','Airport'])
final_amazon_city_df = amazon_city_group.mean()

final_amazon_city_df

#Filter rows without aiport in the name
final_amazon_city_df = final_amazon_city_df.reset_index()

# final_amazon_city_df = final_amazon_city_df.drop(any,axis = 1)
# final_amazon_city_df

airport_avg_rating_by_site_df = final_amazon_city_df[final_amazon_city_df["Airport"].str.contains("Airport") == True]
airport_avg_rating_by_site_df = airport_avg_rating_by_site_df[airport_avg_rating_by_site_df["Airport"].str.contains("Office") == False]
airport_avg_rating_by_site_df = airport_avg_rating_by_site_df[airport_avg_rating_by_site_df["Airport"].str.contains("Support") == False]
airport_avg_rating_by_site_df = airport_avg_rating_by_site_df[airport_avg_rating_by_site_df["Airport"].str.contains("Parking") == False]
airport_avg_rating_by_site_df = airport_avg_rating_by_site_df[airport_avg_rating_by_site_df["Airport"].str.contains(".com") == False]

airport_avg_rating_by_site_df.to_csv('site_airport_avg_rating.csv')

airport_avg_rating_by_site_df

#rating average per amazon city
# Need to dedupe the airport per city since multiple sites are within 31 miles of each other
#DataFrame with only airport, amaozon city, and airport rating
condensed_airport_df = airport_avg_rating_by_site_df[['Amazon City','Airport','Airport Rating']]
condensed_airport_df = condensed_airport_df.drop_duplicates()
condensed_airport_df
#df.drop_duplicates(subset=['A', 'C'], keep=False)

airport_by_city_df = condensed_airport_df.groupby(['Amazon City'])
avg_airport_rating_by_city_df = airport_by_city_df.mean()
avg_airport_rating_by_city_df

airport_count_by_city = airport_by_city_df['Amazon City'].count()
airport_count_by_city_df = pd.DataFrame(airport_count_by_city)
airport_count_by_city_df = airport_count_by_city_df.rename(columns={"Amazon City": "Count"})
airport_count_by_city_df = airport_count_by_city_df.reset_index()

airport_count_by_city_df.join(avg_airport_rating_by_city_df, on = "Amazon City")

condensed_airport_df[condensed_airport_df['Amazon City'].values == "Chicago"]

# splitting municpal airports, internationl airports, and other
for index, airport in enumerate(condensed_airport_df['Airport'].values):
    condensed_airport_df['Is International'] = 0

for index, airport in enumerate(condensed_airport_df['Airport'].values):
    if "International" in str(airport):
        condensed_airport_df['Is International'].iloc[index] = 1
    else:
        condensed_airport_df['Is International'].iloc[index] = 0

condensed_airport_df

con_airport_groupby = condensed_airport_df.groupby('Amazon City')
cond_airport_group_df = pd.DataFrame(con_airport_groupby['Amazon City'].count())
cond_airport_group_df = cond_airport_group_df.rename( columns={"Amazon City": "Airport Count"})
cond_airport_group_df['International Airports'] = con_airport_groupby['Is International'].sum()
cond_airport_group_df = cond_airport_group_df.rename( columns={"Is International": "International Airport Count"})
cond_airport_group_df = cond_airport_group_df.sort_values(by=['Airport Count', 'International Airports'], ascending=False)
cond_airport_group_df
order_list = [7,6,5,5,5,4,3,2,1]
cond_airport_group_df['Rank'] = order_list
cond_airport_group_df
cond_airport_group_df  = cond_airport_group_df.drop('Northern Virginia Area')
cond_airport_group_df
cond_airport_group_df.to_csv('Airport ranking.csv')
cond_airport_group_df

json_path = 'airports.json'
airport_json = pd.read_json(json_path)
airport_json = airport_json.rename(index=str, columns={"name": "Airport"})

airportcheck = list(set(airport_df['Airport']))
airport_list = airport_json['Airport']

for airportcheckrow in airportcheck:
    for airport in airport_list.values:
        if airport == airportcheckrow:
            print(f"{airportcheckrow} is in the airport_json")

# try:
#     final_amazon_city_df.join(airport_json, on='Airport', how='left', lsuffix='_left')
# except:
#     print('no airport')
# final_amazon_city_df.head()

json_path2 = 'airport2.json'
airport2_json = pd.read_json(json_path2)
airport2_json.head()

# airportcheck = list(set(airport_df['Airport']))
# airport2_list = airport2_json['Airport']

# for airportcheck in airportcheck:
#     for airport in airport2_list.values:
#         if airport == airportcheck:
#             print(f"{airportcheck} is in the airport2_json")

#from https://github.com/jbrooksuk/JSON-Airports/blob/master/airports.json
json_path3 = 'airport3.json'
airport3_json = pd.read_json(json_path3)
airport3_json = airport3_json.rename(index=str, columns={"name": "Airport"})
airport3_json.head()

airportcheck = list(set(airport_df['Airport']))
airport3_list = airport3_json['Airport']

for airportcheckrow in airportcheck:
    for airport in airport3_list.values:
        if  airportcheckrow == airport:
            print(f"{airportcheckrow} is in the airport3_json")

#Facebook API on airports
#https://developers.facebook.com/docs/places/web/search

# test out FB api

fb_base_url = "https://graph.facebook.com/search"
fb_params = {
            "type":"place",
            "center":"38.96,-77.42",
            "distance":50000,
            "q":"airport",
            "fields":"name",
            "access_token":fb_key
             }
response = requests.get(fb_base_url, params=fb_params)
fb_airport_json = response.json()

fb_airport_json

# Can't extract any facebook data because of the cambridge analytica.        





