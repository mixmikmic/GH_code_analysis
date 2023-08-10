# import libraries
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

url = "https://en.wikipedia.org/wiki/List_of_countries_by_GDP_sector_composition"

response = requests.get(url)

wiki = BeautifulSoup(response.text, 'lxml')

print(wiki.prettify)

all_tables = wiki.find_all('table')

print(all_tables)

correctTable = all_tables[3]
print(correctTable.prettify)

country = []
totalNaturalResources = []
oil = []
naturalGas = []
coal = []
mineral = []
forest = []

for row in correctTable.findAll('tr')[1:]:
    cells = row.findAll('td')
    if (len(cells) > 0):
        country.append(cells[0].find('a')['title'])
        totalNaturalResources.append(cells[1].find(text = True))
        oil.append(cells[2].find(text = True))
        naturalGas.append(cells[3].find(text = True))
        coal.append(cells[4].find(text = True))
        mineral.append(cells[5].find(text = True))
        forest.append(cells[6].find(text = True))
        
resources_df = pd.DataFrame(country, columns = ["Country"])
resources_df["TotalNaturalResources"] = totalNaturalResources
resources_df["Oil"] = oil
resources_df["NaturalGas"] = naturalGas
resources_df["Coal"] = coal
resources_df["Mineral"] = mineral
resources_df["Forest"] = forest

resources_df.head()

country = []
agriculture = []
industry = []
services = []
year = [] 

pct = wiki.findAll('table')[4]

for row in pct.findAll('tr')[1:]:
    cells = row.findAll('td')
    if (len(cells) > 1):
        country.append(cells[0].find('a').find(text = True))
        agriculture.append(cells[1].find(text = True))
        industry.append(cells[2].find(text = True))
        services.append(cells[3].find(text = True))
        year.append(cells[4].find(text = True))
        
gdp = pd.DataFrame(country, columns = ['Country'])
gdp["Agriculture"] = agriculture
gdp["Industry"] = industry
gdp["Services"] = services
gdp["Year"] = year

gdp.head()

combined = pd.merge(resources_df, gdp, how = "outer", on = "Country")

print(combined.shape)

print(resources_df.shape)

print(gdp.shape)

combined.head()

combined = combined.replace("..", np.NaN)
combined.head()



