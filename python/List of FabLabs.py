import requests
from collections import OrderedDict
import pycountry
from incf.countryutils import transformations
import json

# Load FabLab list
#url = "https://api.fablabs.io/v0/labs.json"
#fablab_list = requests.get(url).json()
json_data=open('labs.json')

fablab_list = json.load(json_data)

# Print a beautified version of the FabLab list for debug
# print json.dumps(fablab_list, sort_keys=True, indent=4)

labs = {}
print "There are",len(fablab_list["labs"]),"FabLabs."

# Print an analysis of the data
for i in fablab_list["labs"]:
    print ""
    print "Name:",i["name"]
    print "E-mail:",i["email"]
    print "Links:"
    for l in i["links"]:
        print l["url"]
    print "Address:"
    if i["address_1"] != None:
        print i["address_1"]
    if i["address_2"] != None:
        print i["address_2"]
    print "City:",i["city"]
    if i["county"] != None:
        print "County:",i["county"]
    country = pycountry.countries.get(alpha2=i["country_code"].upper())
    print "Country:",country.name
    print "Continent:",transformations.cca_to_ctn(i["country_code"])

# Load data for reordering by continent - country later
groupedlabs = {}

for i in fablab_list["labs"]:
    labs[i["name"]] = {}
    labs[i["name"]]["name"] = i["name"]
    labs[i["name"]]["email"] = i["email"]
    labs[i["name"]]["links"] = {}
    for f,l in enumerate(i["links"]):
        labs[i["name"]]["links"][f] = l["url"]
    labs[i["name"]]["address_1"] = i["address_1"]
    labs[i["name"]]["address_2"] = i["address_2"]
    labs[i["name"]]["city"] = i["city"]
    labs[i["name"]]["county"] = i["county"]
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

# Get finally the sorted list of labs

for k in sortedcontinents:
    print "*************************************************************"
    print ""
    print k
    for h in sortedcontinents[k]:
        print ""
        print "----------------------------------------------"
        print h
        for u in sortedcontinents[k][h]:
            print ""
            print sortedcontinents[k][h][u]["name"]
            if sortedcontinents[k][h][u]["address_1"] != None and sortedcontinents[k][h][u]["address_1"] != "":
                print sortedcontinents[k][h][u]["address_1"]
            if sortedcontinents[k][h][u]["address_2"] != None and sortedcontinents[k][h][u]["address_2"] != "":
                print sortedcontinents[k][h][u]["address_2"]
            print sortedcontinents[k][h][u]["city"]
            if sortedcontinents[k][h][u]["county"] != None and sortedcontinents[k][h][u]["county"] != "":
                print sortedcontinents[k][h][u]["county"]
            for l in sortedcontinents[k][h][u]["links"]:
                print sortedcontinents[k][h][u]["links"][l]
            print sortedcontinents[k][h][u]["email"]



