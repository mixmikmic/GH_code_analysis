# Imports
import xlrd
import numpy as np
import pandas as pd

# Import modules for plotting graphs
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

data_path = "data/FGYearbookTable02_Foreign_Coarse_Grains_Supply_and_Disappearance - FGYearbookTable02.csv"
foreign_grains_data = pd.read_csv(data_path, skiprows = 1, index_col = 2, thousands=',')
foreign_grains_data.head()

# Quick overview
foreign_grains_data.describe()

# Remove first two columns
foreign_grains_data = foreign_grains_data.drop('Commodity', 1)
foreign_grains_data = foreign_grains_data.drop('Market Year', 1)

# Create sub-dataframes for other grains
# Indices found by observation from spreadsheet
corn_data = foreign_grains_data.iloc[0:57]
sorghum_data = foreign_grains_data.iloc[59:115]
barley_data = foreign_grains_data.iloc[117:173]
coarse_grains_data = foreign_grains_data.iloc[175:231]
corn_data.head()

# Explore datasets

corn_data.describe()

def prod_two_uses_plot(data, data_name):
    plt.plot(data['Production'])
    plt.plot(data['Food, alcohol, and industrial use'])
    plt.plot(data['Feed use'])
    plt.plot(data['Ending Stocks'])
    plt.title(data_name + ', World excluding US')
    plt.ylabel('Millions of metric tons')
    plt.legend()
prod_two_uses_plot(corn_data, 'Corn Data')

prod_two_uses_plot(sorghum_data, 'Sorghum Data')

prod_two_uses_plot(barley_data, 'Barley Data')

prod_two_uses_plot(coarse_grains_data, 'Coarse Grains Data')

def usdaer_pie_chart_plot(category, year):
    
    # Gather data
    corn = float(corn_data.loc[year][category])
    sorghum = float(sorghum_data.loc[year][category])
    barley = float(barley_data.loc[year][category])
    coarse_grains = float(coarse_grains_data.loc[year][category])
    
    labels = 'Corn', 'Sorghum', 'Barley', 'Coarse Grains'
    colors = ['gold', 'yellowgreen', 'lightskyblue', 'lightcoral']
    sizes = [corn, sorghum, barley, coarse_grains]
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.title('Each grain as proportion of total ' + category + ', ' + str(year))

usdaer_pie_chart_plot('Total supply', 1961)

usdaer_pie_chart_plot('Total supply', 2015)

# Supply of corn as proportion of total has increased by c. 15%, 
# supply of sorghum, barley and coarse grains as prop of total have all dropped.

usdaer_pie_chart_plot('Feed use', 1961)
# Similar to 2006 total supply proportions

usdaer_pie_chart_plot('Feed use', 2016)

# Feed use of corn as proportion of total has increased by c. 15%, 
# Feed use of sorghum as prop of total constant.
# Feed use of barley and coarse grains as prop of total both dropped by c. 8%, 
# barley significant drop from 16% to 8%.

usdaer_pie_chart_plot('Exports', 2016)

usdaer_pie_chart_plot('Exports', 1961)

import pandas as pd

# Changed 'Year: 2006/07' to 'Start Year: 2006'

fao_data_path = "data/FAO_Cereal_Supply_and_Demand.csv"
fao_data = pd.read_csv(fao_data_path, skiprows = 1, index_col = 0)
fao_data.head()

# Remove NaN row
world_cereal_market = fao_data.iloc[1:12]
world_wheat_market = fao_data.iloc[18:29]
world_coarse_grain_market = fao_data.iloc[37:48]
world_rice_market = fao_data.iloc[53:64]

world_cereal_market

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns

def fao_total_plot(data, data_name):
    plt.plot(data['Production'])
    plt.plot(data['Supply'])
    plt.plot(data['Utilization'])
    plt.plot(data['Trade'])
    plt.title(data_name)
    plt.ylabel('Millions of tonnes')
    plt.legend()
    plt.axis(ymin=0)
    
fao_total_plot(world_cereal_market, 'world_cereal_market')

fao_total_plot(world_wheat_market, 'world_wheat_market')

fao_total_plot(world_coarse_grain_market, 'world_coarse_grain_market')

fao_total_plot(world_rice_market, 'world_rice_market')

# What is this 'trade' figure?

def fao_ratios_plot(data, data_name):
    plt.plot(data['World stock-to-use ratio'])
    plt.plot(data['Major exporters\' stock-to-disappearance ratio'])
    plt.title(data_name)
    plt.ylabel('Ratio')
    plt.legend()
    plt.axis(ymin=0, ymax=100)
    
fao_ratios_plot(world_cereal_market, 'world_cereal_market')

# Playing around with ymax in these plots. ymax = 100 gives a better sense of ratios 
# but makes the change seem insignificant.

# Note sharp increase during the 2007-8 food crisis.

fao_ratios_plot(world_coarse_grain_market, 'world_coarse_grain_market')

fao_ratios_plot(world_wheat_market, 'world_wheat_market')

fao_ratios_plot(world_rice_market, 'world_rice_market')

def fao_pie_chart_plot(category, year_as_string):
    
    # Gather data
    wheat = float(world_wheat_market.loc[year_as_string][category])
    coarse_grain = float(world_coarse_grain_market.loc[year_as_string][category])
    rice = float(world_rice_market.loc[year_as_string][category])
    total = float(world_cereal_market.loc[year_as_string][category])
    other = total - wheat - coarse_grain - rice
    
    labels = 'Wheat', 'Coarse Grains', 'Rice', 'Other'
    colors = ['gold', 'yellowgreen', 'lightskyblue', 'lightcoral']
    sizes = [wheat, coarse_grain, rice, other]
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    # Set aspect ratio to be equal so that pie is drawn as a circle.
    plt.axis('equal')
    plt.title('Each grain as proportion of total ' + category + ', ' + year_as_string)

fao_pie_chart_plot('Utilization', '2006')

# Note that 'Other' takes up 0.0% of the total.

fao_pie_chart_plot('Utilization', '2016')

