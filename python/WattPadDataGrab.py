# Import Dependencies
import requests
import json
import numpy as np
import csv
import yaml
import os
from pandas.io.json import json_normalize

# Load the config.yaml file to get the api keys and other parameters
with open("./config.yaml") as y:
    cfg = yaml.load(y)

header = {
    "Authorization": "Basic {}".format(cfg["keys"]["API_KEY"]),
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.186 Safari/537.36",

    }

# Files to save our data
categories_file_name = "data/categories.csv"
languages_file_name = "data/languages.csv"

################################################################################
# This function makes a Wattpad api call to get a list of all the categories
# It writes all the categories data into a csv file to be used later
################################################################################
def get_categories():
    category_url = "https://www.wattpad.com/v4/categories"
    
    # Make the api call
    req = requests.get(category_url, headers=header)
    category_response = req.json()
    
    # Write to the csv file
    with open(categories_file_name,'w') as csvfile:
        write=csv.writer(csvfile, delimiter=',')
        
        # Write the header row
        write.writerow(["ID","NAME"])
        
        # Loop through the data and write
        for category in category_response["categories"]:
            write.writerow([category["id"],category["name"]])
            

# Call the function to get all the categories from Whatpad and then view the data 
# from the csv file that is created to make sure we have usable data
get_categories()

# Open the csv file and read its contents to see if we got all the data right
with open(categories_file_name) as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        print(row)

################################################################################
# This function makes a Wattpad api call to get a list of all the languages
# It writes all the language code data into a csv file to be used later
################################################################################
def get_languages():
    language_url = "https://www.wattpad.com/v4/languages"
    
    # Make the api call
    req = requests.get(language_url, headers=header)
    category_response = req.json()
    
    # Write to the csv file
    with open(languages_file_name,'w') as csvfile:
        write=csv.writer(csvfile, delimiter=',')
        
        # Write the header row
        write.writerow(["LANGUAGE_CODE"])
        
        # Loop through the data and write 
        for category in category_response["languages"]:
            write.writerow([category["code"]])
            

# Make the call to get the languages and then view the data from the csv file that
# is created to make sure we have usable data
get_languages()

# Open the csv file and read its contents to see if we got all the data right
with open(languages_file_name) as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    for row in reader:
        print(row)

def get_stories(x):
    BASE_URL = "https://www.wattpad.com/v4/stories?limit=100offset%3D0&offset=" + str(x) + "&filter=new"

    req = requests.get(BASE_URL.format("stories"), headers=header)
    json_response = req.json()
    return(json_response)

#number of stories
N = 10000
json_list = []
for x in np.arange(0, N, 100):
    json_list.append(get_stories(x))

pages_of_stories = [x['stories'] for x in json_list]

################################################################################
# Creates a single array of all stories downloaded, parses each json element
# into its own column, then changes the values of the categories column to be
# a single integer instead of an array.
################################################################################

flat_list=[x for y in pages_of_stories for x in y]

stories_df = json_normalize(flat_list)

for i in range(len(stories_df['categories'])):
    stories_df.loc[i, 'categories'] = stories_df['categories'][i][0]
    
stories_df.to_csv(os.path.join('Data', 'stories_3_12_2018_new.csv'))

stories_df.categories.unique()



