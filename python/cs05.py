import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from plotnine import *
import plotnine as pln

# Read in the data

buildings = pd.read_csv("data/buildings.csv")
climate_zone_fips = pd.read_csv("data/climate_zone_fips.csv")
permits = pd.read_csv("data/permits.csv")
restaurants = pd.read_csv("data/restaurants.csv")

# Clean up the data
# First, we merge the buildings data with the climate data
building_pop = pd.merge(buildings, climate_zone_fips, on=['FIPS.state', 'FIPS.county'])
building_pop = building_pop[building_pop['Type'] == 'Food_Beverage_Service']

# These are lists with which we will categorize the data
not_restaurants = ["development","Food preperation center", "Food Services center",
                   "bakery","Grocery","conceession","Cafeteria", "lunchroom","school",
                   "facility"," hall "]
standalone_retail = ["Wine","Spirits","Liquor","Convenience","drugstore","Flying J", 
                     "Rite Aid ","walgreens ","Love's Travel "]
full_service_type = ["Ristorante","mexican","pizza ","steakhouse"," grill ","buffet",
                     "tavern"," bar ","waffle","italian","steak house"]
quick_service_type = ["coffee"," java "," Donut ","Doughnut"," burger ","Ice Cream ","custard ",
                      "sandwich ","fast food "," bagel "]

# The next four lines take the lists defined above and strip
# them of leading/trailing whitespace and make all letters lowercase. 
not_restaurants = [x.strip().lower() for x in not_restaurants]
standalone_retail = [x.strip().lower() for x in standalone_retail]
full_service_type = [x.strip().lower() for x in full_service_type]
quick_service_type = [x.strip().lower() for x in quick_service_type]

# Next, we create a subgroup column and populate it based off
# the project title using the lists created above. 
building_pop['subgroup'] = building_pop['ProjectTitle'].str.strip()                                                        .str.lower()                                                        .apply(lambda x: 'Not Restaurant' if any(word in x for word in not_restaurants)                                                                 else 'Standalone Retail' if any(word in x for word in standalone_retail)                                                                 else 'Full Service'      if any(word in x for word in full_service_type)                                                                 else 'Quick Service'     if any(word in x for word in quick_service_type)                                                                 else 'NA')

# Here we take care of the leftovers that weren't classified above
mask1 = ((building_pop['subgroup'] == 'NA') & (building_pop['SqFt'] <= 4000))
mask2 = ((building_pop['subgroup'] == 'NA') & (building_pop['SqFt'] > 4000))
building_pop.loc[mask1, 'subgroup'] = 'Quick Service'
building_pop.loc[mask2, 'subgroup'] = 'Full Service'

# Plot 1

is_restaurant = building_pop[building_pop['subgroup'].isin(['Full Service', 'Quick Service'])]
counts = is_restaurant.groupby(['County_x', 'Year', 'subgroup'], as_index=False).count()

pln.options.figure_size = (12,5)
(ggplot(counts, aes(x='County_x', y='Month', fill='subgroup')) + 
 geom_col(position='dodge') + 
 facet_grid('. ~ Year', scales='free_x') + 
 labs(title = "Full-Service vs Quick Service by County and Year",
       x = "County",
       y = "Number of Projects",
       fill = 'Retaurant Type') +
 theme_bw() + 
 theme(axis_text_x=element_text(angle=45, hjust=1, vjust=1)))

# Plot 2

buildings['type_group'] = buildings['Type'].apply(lambda t: "Restaurant" if t == 'Food_Beverage_Service' else 'Commerical')
counts = buildings.groupby(['Year', 'type_group'], as_index=False).count()

pln.options.figure_size = (6,4)
(ggplot(counts, aes(x='Year', y='Month', fill='type_group')) +
 geom_col(position="dodge") +
 scale_x_continuous(breaks=[2008, 2009]) +
 labs(title = "Restaurant vs Other Commerical Construction",
      y = "Number of Construction Projects",
      fill = "Project Type") +
 theme_bw())

# Plot 3

ada_buildings = buildings[buildings['County'] == 'ADA, ID'].copy()
ada_buildings['type_group'] = ada_buildings['Type'].apply(lambda t: "Restaurant" if t == 'Food_Beverage_Service' else 'Commerical')
counts = ada_buildings.groupby(['Year', 'type_group'], as_index=False).count()

pln.options.figure_size = (6,4)
(ggplot(counts, aes(x='Year', y='Month', fill='type_group')) +
 geom_col(position="dodge") +
 scale_x_continuous(breaks=[2008, 2009]) +
 labs(title = "Ada County Restaurant vs Other Commerical Construction",
      y = "Number of Construction Projects",
      fill = "Project Type") +
 theme_bw())

