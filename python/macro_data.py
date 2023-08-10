# Setting up a custom stylesheet in IJulia
from IPython.core.display import HTML
from IPython import utils  
import urllib2
HTML(urllib2.urlopen('http://bit.ly/1Bf5Hft').read())
#HTML("""
#<style>
#open('style.css','r').read()
#</style>
#""")

get_ipython().magic('matplotlib inline')
from __future__ import print_function
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sympy as sym
import pandas as pd
sym.init_printing() 
import quantecon as qe
import solowpy





pwt_raw_data = pd.read_stata('http://www.rug.nl/ggdc/docs/pwt90.dta')

pwt_raw_data.head()

dep_rates_raw_data = pd.read_excel('http://www.rug.nl/ggdc/docs/depreciation_rates.xlsx', )

# merge the data
pwt_merged_data = pd.merge(pwt_raw_data, dep_rates_raw_data, how='outer',
                           on=['countrycode', 'year'])

# create the hierarchical index
pwt_merged_data.year = pd.to_datetime(pwt_raw_data.year, format='%Y')
pwt_merged_data.set_index(['countrycode', 'year'], inplace=True)

# coerce into a panel
pwt_panel_data = pwt_merged_data.to_panel()







pwt_panel_data.major_axis

pwt_panel_data.major_xs('USA')











usa_raw_data = pd.read_excel('http://www.stanford.edu/~chadj/snapshots/USA.xls' )





usa_raw_data = pd.read_excel('http://www.stanford.edu/~chadj/snapshots/USA.xls',skiprows = 9)

usa_raw_data.head(11)

usa_raw_data.set_index(['Year'], inplace=True)



can_raw_data = pd.read_excel('http://www.stanford.edu/~chadj/snapshots/CAN.xls',skiprows = 9 )

can_raw_data.set_index(['Year'], inplace=True)



can_raw_data.head()







get_ipython().run_cell_magic('stata', '-o pwt', 'close all\n\nuse "http://www.rug.nl/ggdc/docs/pwt90.dta"\n\nsummarize')

pwt.tail()

pwt.columns

#     country = name of country
#     isocode = 3-letter country code 
#     year
#     POP     = population in thousands 
#     XRAT    = exchange rate in lcu's per dollar
#     Currency_Unit = name
#     ppp     
#     tcgdp   = PPP adjusted GDP in current prices, millions of international dollars 
#     cgdp    = same by per capita 
#     cgdp2   = another GDP per capita number 
#     cda2    = absorption per capita, current prices 
#     cc      = consumption share at current prices 
#     cg      = government purchases share at current prices
#     ci      = investment share at current prices
#     p       = price of GDP, GK method 
#     p2      = price of GDP, avg of GEKS and CPDW 
#     pc      = price of consumption
#     pg      = price of government consumption
#     pi      = price of investment 
#     openc   = 
#     cgnp    = ratio of GNP to GDP 
#     y       = GDP per capita rel to US
#     y2      = ditto, diff method  
#     rgdpl   = GDP per capita, Laspeyres 
#     rgdpl2  = GDP per capita, based on absorption 
#     rgdpch  = GDP per capita, chain weighted 
#     kc      = consumption share at constant prices (Laspeyres)
#     kg      = government share 
#     ki      = investment share 
#     openk   =
#     rgdpeqa = GDP per adult equivalent, chain 
#     rgdpwok = GDP per worker, chain  
#     rgdpl2wok = GDP per worker, Laspeyres 
#     rgdpl2pe = 
#     rgdpl2te
#     rgdpl2th
#     rgdptt  = 
#  Written by:  Paul Backus, October 2012 
# http://www.rug.nl/ggdc/docs/user_guide_to_pwt90_data_files.pdf



get_ipython().run_cell_magic('stata', '-o USDA', 'close all\n\nuse "http://graduateinstitute.ch/files/live/sites/iheid/files/sites/md4stata/shared/databases/USDA_historical_data.dta"\n\nsummarize')

USDA.columns

USDA[USDA.wbcode == "CAN"].tail()









get_ipython().run_cell_magic('stata', '-o statca', '\nclear all\n\nfreduse CANCPIALLQINMEI CANCPIALLQINMEI IRLTLT01CAQ156N LRUNTTTTCAQ156S IRSTCB01CAQ156N MANMM101CAQ189S MABMM203CAQ189S CANEXPORTQDSMEI CCUSSP01CAQ650N IR3TIB01CAQ156N')

