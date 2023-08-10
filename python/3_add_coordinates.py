import pandas as pd

df_ebola = pd.read_csv("data/out/ebola-outbreaks-before-2014-country-codes.csv", encoding="utf-8", index_col=False)

df_eb = df_ebola.drop(df_ebola.columns[[0, 1]], axis=1) # remove 2 useless colums of indexes

df_eb.head()

all_countries = list(df_eb["Country"])
print all_countries

cleaned_countries_list = [u'Uganda', u'Democratic Republic of the Congo', u'Uganda', u'Uganda', 
                          u'Democratic Republic of the Congo', u'Philippines', u'Uganda', 
                          u'Democratic Republic of the Congo', u'Russia', u'Sudan (South Sudan)', 
                          u'Democratic Republic of the Congo', u'Democratic Republic of the Congo', 
                          u'Democratic Republic of the Congo', u'Gabon', u'Uganda', u'Russia', 
                          u'Philippines', u'USA', u'South Africa', u'Gabon', u'Gabon', 
                          u'Democratic Republic of the Congo', u"C\xf4te d'Ivoire (Ivory Coast)", 
                          u'Gabon', u'Italy', u'Philippines', u'USA', u'USA', u'Sudan (South Sudan)', 
                          u'Democratic Republic of the Congo', u'England', u'Sudan (South Sudan)', 
                          u'Democratic Republic of the Congo'] 

countries_list = list(set(cleaned_countries_list))

print len(countries_list)
print countries_list

from geopy.geocoders import Nominatim
geolocator = Nominatim()

locations = [geolocator.geocode(country) for country in countries_list]

longitudes = [l.longitude for l in locations]

print longitudes

latitudes = [l.latitude for l in locations]

print latitudes

countries_lat = [latitudes[countries_list.index(country)] for country in cleaned_countries_list]

print countries_lat

countries_lon = [longitudes[countries_list.index(country)] for country in cleaned_countries_list]

print countries_lon

df_eb2 = df_eb.drop(df_ebola.columns[9], axis=1)

df_eb2.head()

df_eb2.insert(7, "Country name", cleaned_countries_list)

df_eb2.head()

df_eb2.insert(8, "Latitude", countries_lat)
df_eb2.insert(9, "Longitude", countries_lon)

df_eb2.head()

df_eb2.to_csv("data/out/ebola-outbreaks-before-2014-coordinates.csv", encoding="utf-8", index_col=False)



