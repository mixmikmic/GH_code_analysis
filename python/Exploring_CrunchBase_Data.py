import pandas as pd
import csv
from geopy.geocoders import Nominatim
import requests

closed_seed_angel_us_df = pd.read_csv("../data/raw/US_Angel-Seed-SeriesA_Closed_CrunchBase/closed-companies-angel-seed-6-9-2017 (1).csv")

closed_seriesa_us_df = pd.read_csv("../data/raw/US_Angel-Seed-SeriesA_Closed_CrunchBase/series-a-closed-us-6-9-2017.csv")

closed_seed_angel_us_df.info()

closed_seed_angel_us_df['Total Funding Amount']

closed_seed_angel_us_df.head()

closed_seriesa_us_df .info()

#with open('closed_rounds.csv','w') as fp:
with open('closed_rounds.csv','w') as fp:
    csv_writer = csv.writer(fp)
    for index,row in closed_seed_angel_us_df.loc[:,('Company Name','Website','Number of Employees')].iterrows():
        csv_writer.writerow([index,row])


closed_seed_angel_us_df.loc[:,('Company Name','Website','Number of Employees')].to_csv('closed_rounds.csv')

#test 
geolocator = Nominatim()
location = geolocator.geocode("San Francisco")
print(location.address)

print((location.latitude, location.longitude))

def convert_city_to_lat_long(city):
    # convert city into latitude and longtitude
    geolocator = Nominatim()
    location = geolocator.geocode(city)
    return location.latitude, location.longitude

convert_city_to_lat_long(closed_seed_angel_us_df['Headquarters Location'][2].split(',')[0]) # test

lat,long = zip(*closed_seed_angel_us_df[
    'Headquarters Location'].apply(lambda x: convert_city_to_lat_long(x.split(',')[0])))

closed_seed_angel_us_df



from bs4 import BeautifulSoup
soup = BeautifulSoup(html_doc, 'html.parser')

company='bitinstant'
r = requests.get('view-source:https://www.crunchbase.com/organization/bitinstant#/entity')

r











