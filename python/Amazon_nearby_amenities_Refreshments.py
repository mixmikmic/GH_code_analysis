# Dependencies
import requests
import json
import pandas as pd
from pprint import pprint
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Google developer API key
from config import g_key

# read places 
xls = pd.ExcelFile('BC_Project1/Project1_AmazonSites.xlsx') 
places_df=xls.parse('AmazonSites', dtype=str)
places_df = places_df[['Amazon City','Site','Site Name','Latitude','Longitude']]
places_df.head()
#len(places_df)

# eating_out = {'restaurant':'restaurant', 'cafe':'cafe'}
# all_eatingout_rating = []

# for key in eating_out.keys():
#     eatingout_rating = []
#     for site in places_df.values:

#         # geocoordinates
#         target_coordinates = str(site[3]) + ',' + str(site[4])
#         target_search = eating_out[key]
#         target_radius = 2500
#         target_type = key
#         print("{} For {}: {}, {}".format(key, site[0], site[1], target_coordinates))
#         print("----------")
#         # set up a parameters dictionary
#         params = {
#             "location": target_coordinates,
#             "keyword": target_search,
#             "radius": target_radius,
#             "type": target_type,
#             "key": g_key
#         }

#         # base url
#         base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"

#         # run a request using our params dictionary
#         response = requests.get(base_url, params=params).json()
#         results = response.get('results')
#         total_counts = len(results)
#         print(total_counts)

#         # Print the name and address of the first restaurant that appears
#         for x in range(total_counts):
#             if "rating" in results[x].keys():
#                 rating = results[x]["rating"]
#             else:
#                 rating = 'NAN'
#             if "price_level" in results[x].keys():
#                 price_level = results[x]["price_level"]
#             else:
#                 price_level = 'NAN'
#             eatingout_rating.append({"Site Name":site[2],
#                                        key+" Total Count":total_counts,
#                                        "Latitude":site[3],
#                                        "Longitude":site[4],
#                                        "Facility "+key:results[x]["name"],
#                                        "Rating":rating,
#                                        "price_level":price_level})
            
#             time.sleep(2)
#         time.sleep(2)
#     all_eatingout_rating.append(eatingout_rating)
#     print("ALL Done!!!!!")
# #all_eatingout_rating

all_eatingout_rating

all_restaurant_rating_df = pd.DataFrame(all_eatingout_rating[0])
all_restaurant_rating_df

all_cafe_rating_df = pd.DataFrame(all_eatingout_rating[1])
all_cafe_rating_df

all_cafe_rating_df.to_csv("Cafe_Rating.csv")

all_restaurant_rating_df.to_csv("Restaurant_Rating.csv")

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

restaurant = pd.read_csv('Restaurant_Rating.csv')
del restaurant['Unnamed: 0']
restaurant['Rating']=restaurant['Rating'].astype(float)
restaurant.replace('NAN', value=0, inplace=True)
restaurant.head()

restaurant_grouped = restaurant.groupby('City Name')
site_avg_count = restaurant_grouped['restaurant Total Count'].median()
site_avg_rating = restaurant_grouped['Rating'].median()
site_avg_count_df = pd.DataFrame(site_avg_rating)
# site_avg_rating_df =pd.DataFrame({'City Name': restaurant['City Name'],
#                                 'Median Rating':site_avg_rating})

site_avg_count_df['Restaurant Median Count']=site_avg_count
site_avg_count_df = site_avg_count_df.rename(columns={'Rating':'Median Rating'})
site_avg_count_df['Median Rating'] = site_avg_count_df['Median Rating'].astype(float)
site_avg_count_df = site_avg_count_df.reset_index()
site_avg_count_df['Restaurant Median Count'] = site_avg_count_df['Restaurant Median Count'].astype(int)
site_avg_count_df

# Create legend for colors
colors = ['lightblue', 'green', 'red', 'blue', 'yellow']

# Use seaborn to make the scatter plot
ax = sns.lmplot(x='City Name', y='Median Rating', data=site_avg_count_df, fit_reg=False, aspect=2.5, 
                hue='City Name', legend=False, size=8,
                scatter_kws={"s":site_avg_count_df['Median Rating']*500,'alpha':1, 'edgecolors':'black', 'linewidths':1})

# Make the grid, set x-limit and y-limit
plt.grid()
plt.ylim(3.5,4.5)

# Set scale for all the fonts of the plot
sns.set(font_scale=1.4)

# Make x-axis, y-axis & title labels
# plt.title("SENTIMENT ANALYSIS OF MEDIA TWEETS (03/25/2018)")
# plt.xlabel("Tweets Ago")
# plt.ylabel("Tweets Polarity")

# Set the plot baclground color
sns.set_style("dark")

# Format the legend and plot
plt.legend(loc='upper right', title='City Types')
# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

x=site_avg_count_df['City Name']
y=site_avg_count_df['Median Rating']
z=site_avg_count_df['Median Rating']
plt.scatter(x, y, s=z*1000, alpha=0.4, edgecolors="grey", linewidth=2, c=y, cmap="PuBuGn")
plt.grid()
# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print ("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 20
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size
plt.show()

sns.boxplot(x='City Name', y='Rating', data=restaurant )
plt.grid()

plt.show()



