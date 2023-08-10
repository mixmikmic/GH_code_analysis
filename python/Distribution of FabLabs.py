import requests
from collections import OrderedDict
import pycountry
from incf.countryutils import transformations
import json

# From http://stackoverflow.com/a/7423575/2237113
def autolabel(rects):
# attach some text labels
    for ii,rect in enumerate(rects):
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.02*height, '%s'% (name[ii]),
                ha='center', va='bottom')

# Load FabLab list
url = "https://api.fablabs.io/v0/labs.json"
fablab_list = requests.get(url, verify=False).json()

# Print a beautified version of the FabLab list for debug
# print json.dumps(fablab_list, sort_keys=True, indent=4)

labs = {}
print "There are",len(fablab_list["labs"]),"FabLabs."

# Load data for reordering by continent - country later
groupedlabs = {}

for i in fablab_list["labs"]:
    labs[i["name"]] = {}
    labs[i["name"]]["name"] = i["name"]
    labs[i["name"]]["city"] = i["city"]
    country = pycountry.countries.get(alpha2=i["country_code"].upper())
    labs[i["name"]]["country"] = country.name
    continent = transformations.cca_to_ctn(i["country_code"])
    labs[i["name"]]["continent"] = continent
    
    # Save by continent and country
    if continent not in groupedlabs:
        groupedlabs[continent] = {}
    if country.name not in groupedlabs[continent]:
        groupedlabs[continent][country.name] = {}
    groupedlabs[continent][country.name][i["name"]] = labs[i["name"]]
        

# Order alphabetically

# Get list from continents and countries in the data
continents = []
countries = []
for m in groupedlabs:
    continents.append(m)
    for j in groupedlabs[m]:
        countries.append(j)  
continents = sorted(continents)
countries = sorted(countries)

# Order continents and countries alphabetically
sortedcontinents = OrderedDict(sorted(groupedlabs.items(), key=lambda t: t[0]))
for k in sortedcontinents:
    sortedcontinents[k] = OrderedDict(sorted(sortedcontinents[k].items(), key=lambda t: t[0]))

# Print output for debug
for k in sortedcontinents:
    print ""
    print k
    for h in sortedcontinents[k]:
        print "-",h

# Let's start the graphics
get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')

import matplotlib.pyplot as plt
import seaborn

# Get distribution of FabLabs per country

countries_stats = {}

for k in sortedcontinents:
    for h in sortedcontinents[k]:
        countries_stats[h] =  len(sortedcontinents[k][h])
        
countries_stats2 = OrderedDict(sorted(countries_stats.items(), key=lambda t: t[1]))

figure(figsize=(20,8))
rect1 = plt.bar(range(len(countries_stats2)), countries_stats2.values(), align='center')
plt.xticks(range(len(countries_stats2)), countries_stats2.keys(), rotation=90)
plt.title("Distribution of Fab Labs by country")
name = list(countries_stats2.values())
autolabel(rect1)
plt.savefig('FabLabs-CountryDistribution.pdf')

# Get distribution of FabLabs per country, per continent

countries_stats = {}

continent_name = "Europe"

for k in sortedcontinents:
    if k == continent_name:
        for h in sortedcontinents[k]:
            countries_stats[h] =  len(sortedcontinents[k][h])
    else:
        pass

countries_stats2 = OrderedDict(sorted(countries_stats.items(), key=lambda t: t[1]))

figure(figsize=(20,8))
rect1 = plt.bar(range(len(countries_stats2)), countries_stats2.values(), align='center')
plt.xticks(range(len(countries_stats2)), countries_stats2.keys(), rotation=90)
plt.title("Distribution of Fab Labs by country in "+continent_name)
name = list(countries_stats2.values())
autolabel(rect1)
plt.savefig('FabLabs-CountryContinent'+continent_name+'Distribution.pdf')

# Get distribution of FabLabs per continent

continents_stats = {}

for k in sortedcontinents:
    continents_stats[k] = 0
    for h in sortedcontinents[k]:
        continents_stats[k] += len(sortedcontinents[k][h])

continents_stats2 = OrderedDict(sorted(continents_stats.items(), key=lambda t: t[1]))    
    
rect1 = plt.bar(range(len(continents_stats2)), continents_stats2.values(), align='center')
plt.xticks(range(len(continents_stats2)), continents_stats2.keys(), rotation=90)
plt.title("Distribution of Fab Labs by continent")
name = list(continents_stats2.values())
autolabel(rect1)
plt.savefig('FabLabs-ContinentDistribution.pdf')

# Get distribution of FabLabs per city

# Get list from cities in the data
cities = {}

for j,m in enumerate(labs):
    if labs[m]["city"] not in cities:
        cities[labs[m]["city"]] = {}
    cities[labs[m]["city"]][j] = 1
    
# Delete cities without a name
if "" in cities:
    del cities[""]
    
cities_stats = {}

for i in cities:
    cities_stats[i] = len(cities[i])

cities_stats2 = OrderedDict(sorted(cities_stats.items(), key=lambda t: t[1]))    

figure(figsize=(60,8))
rect1 = plt.bar(range(len(cities_stats2)), cities_stats2.values(), align='center')
plt.xticks(range(len(cities_stats2)), cities_stats2.keys(), rotation=90)
plt.title("Distribution of FabLa bs by city")
name = list(cities_stats2.values())
autolabel(rect1)
plt.savefig('FabLabs-CityDistribution.pdf')



