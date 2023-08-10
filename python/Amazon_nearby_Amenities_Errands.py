# Dependencies
import requests
import json
import pandas as pd
from pprint import pprint
import time

# Google developer API key
from config import g_key

# read places 
xls = pd.ExcelFile('BC_Project1/Project1_AmazonSites.xlsx') 
places_df=xls.parse('AmazonSites', dtype=str)
places_df = places_df[['Amazon City','Site','Site Name','Latitude','Longitude']]
places_df.head()
#len(places_df)

errands = {'supermarket':'supermarket', 'postoffice':'Post Office'}
all_errands_rating = []

for key in errands.keys():
    errands_rating = []
    for site in places_df.values:

        # geocoordinates
        target_coordinates = str(site[3]) + ',' + str(site[4])
        target_search = errands[key]
        target_radius = 2500
        target_type = key
        print("{} For {}: {}, {}".format(key, site[0], site[1], target_coordinates))
        print("----------")
        # set up a parameters dictionary
        params = {
            "location": target_coordinates,
            "keyword": target_search,
            "radius": target_radius,
            "type": target_type,
            "key": g_key
        }

        # base url
        base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

        # run a request using our params dictionary
        response = requests.get(base_url, params=params).json()
        results = response.get('results')
        total_counts = len(results)
        print(total_counts)

        # Print the name and address of the first restaurant that appears
        for x in range(total_counts):
            if "rating" in results[x].keys():
                rating = results[x]["rating"]
            else:
                rating = 'NAN'
    
            errands_rating.append({"Site Name":site[2],
                                       key+" Total Count":total_counts,
                                       "Latitude":site[3],
                                       "Longitude":site[4],
                                       "Facility "+key:results[x]["name"],
                                       "Rating":rating})
            
            time.sleep(2)
        time.sleep(2)
    all_errands_rating.append(errands_rating)
    print("ALL Done!!!!!")
#all_eatingout_rating

all_errands_rating

all_supermarket_rating_df = pd.DataFrame(all_errands_rating[0])
all_supermarket_rating_df.head()

all_postoffice_rating_df = pd.DataFrame(all_errands_rating[1])
all_postoffice_rating_df.head()

all_supermarket_rating_df.to_csv("SuperMarket_Rating.csv")

all_postoffice_rating_df.to_csv("PostOffice_Rating.csv")

# geocoordinates
target_coordinates = '38.96,-77.42'
target_search = 'restaurant'
target_radius = 2500
target_type = 'Restaurant'
#print("{} For {}: {}, {}".format(key, site[0], site[1], target_coordinates))
print("----------")
# set up a parameters dictionary
params = {
    "location": target_coordinates,
    "keyword": target_search,
    "radius": target_radius,
    "type": target_type,
    "key": g_key
}

# base url
base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

# run a request using our params dictionary
response = requests.get(base_url, params=params).json()
results = response.get('results')

results



