import numpy as np
import scipy as scp
import pandas as pd 
import warnings 

warnings.filterwarnings("ignore")

yu = pd.read_csv('../data-visualization/API_ILO_country_YU.csv')

yu_sort_lastyr = yu.sort(['2014'],ascending = True)

yu_sort_lastyr.head(10)

import matplotlib.pyplot as plt
import matplotlib.cm 

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import seaborn as sns

get_ipython().magic('matplotlib inline')
get_ipython().magic('pylab inline')

#--------script to choose color---------#
# from tkinter import *
# from tkinter.colorchooser import *
# def getColor():
#     color = askcolor() 
#     print(color)
# Button(text='Select Color', command=getColor).pack()
# mainloop()

yu_top_10 = yu_sort_lastyr.head(10)
yu_bottom_10 = yu_sort_lastyr.tail(10)

country_top_10 = yu_top_10['Country Name']
country_bottom_10 = yu_bottom_10['Country Name']
country_top_10 = np.array(country_top_10)
country_bottom_10 = np.array(country_bottom_10)

yu_top_10 = yu_top_10.drop(['Country Name','Country Code'], axis = 1)
yu_bottom_10 = yu_bottom_10.drop(['Country Name','Country Code'], axis = 1)

yu_top_10 = yu_top_10.reset_index(drop=True)
yu_bottom_10 = yu_bottom_10.reset_index(drop=True)
labels = ['2010','2011','2012','2013','2014']

plt.figure()
x = np.arange(0,5)
for i in range(0,yu_top_10.shape[0]):
    y = yu_top_10.ix[i]
    y = np.array(y)
    legend_label = country_top_10[i]
    plt.plot(x,y,'-s', label = legend_label)
    
plt.xticks(x, labels)
plt.xlabel("Year",color = "k",size = '16')
plt.ylabel("Unemployment rate (%)",color = "k",size = '16')
plt.title('Youth Unemployment rate for top 10 countries over last 5 years')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.subplots_adjust(left=0, bottom=0, right=1.0, top=1, wspace=0, hspace=0)
plot_margin = 0.25

x0, x1, y0, y1 = plt.axis()
plt.axis((x0 - plot_margin,x1 + plot_margin,y0 - 0,y1 + 0))
plt.show()

plt.figure()
x = np.arange(0,5)
for i in range(0,yu_top_10.shape[0]):
    y = yu_bottom_10.ix[i]
    y = np.array(y)
    legend_label = country_bottom_10[i]
    plt.plot(x,y,'-^', label = legend_label)
    
plt.xticks(x, labels)
plt.xlabel("Year",color = "k",size = '16')
plt.ylabel("Unemployment rate (%)",color = "k",size = '16')
plt.title('Youth Unemployment rate for bottom 10 countries over last 5 years')
plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)
plt.subplots_adjust(left=0, bottom=0, right=1.0, top=1, wspace=0, hspace=0)
plot_margin = 0.25

x0, x1, y0, y1 = plt.axis()
plt.axis((x0 - plot_margin,x1 + plot_margin,y0 - 0,y1 + 0))
plt.show()

country = ['Afghanistan','Bangladesh','Bhutan','India','Pakistan','Nepal','Maldives','Sri Lanka']
years = ['2010','2011','2012','2013','2014']
columns = ['Country','Year','Unemployment']
country_sasia = pd.DataFrame(columns=columns) #empty dataframe
    
for country_name in country:
    curr_country = yu_sort_lastyr[yu_sort_lastyr['Country Name'] == country_name]
    for year in years:
        values = curr_country[year].values[0] #df.values computes the entries as a numpy array or list
        entry = pd.DataFrame([[country_name,year,values]],columns=columns)
        country_sasia = country_sasia.append(entry)
country_sasia.head(20)

table_sasia = pd.pivot_table(data = country_sasia,
                            index = ['Year'],
                            columns = ['Country'],
                            values = ['Unemployment'],
                            aggfunc = 'mean')
plt.figure(figsize=(10,5))
ax = sns.heatmap(data = table_sasia['Unemployment'],vmin=0,annot=True,fmt='2.2f',linewidth = 0.5,cmap="YlGnBu")
plt.title('Unemployment for Countries in South Asia')
ticks = plt.setp(ax.get_xticklabels(),rotation=45)

from geonamescache import GeonamesCache
gc = GeonamesCache()
iso3_codes = list(gc.get_dataset_by_key(gc.get_countries(), 'iso3').keys())

codes = yu['Country Code'].unique()
codes.tolist()
yu.set_index('Country Code', inplace=True)
yu = yu.ix[iso3_codes].dropna()

num_colors = 9

values = yu['2014']
cm = plt.get_cmap('Reds')
scheme = [cm(i / num_colors) for i in range(num_colors)]
bins = np.linspace(values.min(), values.max(), num_colors)
yu['bin'] = np.digitize(values, bins) - 1

fig = plt.figure(1,figsize = (15,15))
ax = fig.add_subplot(111, axisbg='w', frame_on=False)
map = Basemap(projection='merc',llcrnrlat=-60,urcrnrlat=80,            llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
map.drawmapboundary(fill_color="w")

shapefile = '../data-visualization/ne_10m_admin_0_countries/ne_10m_admin_0_countries'
map.readshapefile(shapefile, 'units', color='#444444', linewidth=.2)
for info, shape in zip(map.units_info, map.units):
    iso3 = info['ADM0_A3']
    if iso3 not in codes:
        color = '#dddddd'
    else:
        color = scheme[yu.ix[iso3]['bin']]

    patches = [Polygon(np.array(shape), True)]
    pc = PatchCollection(patches)
    pc.set_facecolor(color)
    ax.add_collection(pc)
        
ax_legend = fig.add_axes([0.35, 0.24, 0.3, 0.03], zorder=3)
cmap = mpl.colors.ListedColormap(scheme)
cb = mpl.colorbar.ColorbarBase(ax_legend, cmap=cmap, ticks=bins, boundaries=bins, orientation='horizontal')
cb.ax.set_xticklabels([str(round(i, 1)) for i in bins])
plt.show()

wb = pd.read_csv('../data-visualization/world-development-indicators/Indicators.csv')
wb_country  = wb.copy()
wb_country.set_index('CountryName',inplace = True)

wb.head(10)

wb_indicator = wb.copy()
indicator = wb_indicator['IndicatorName'].unique()
num_indicator = indicator.shape[0]

wb_indicator = wb_indicator.groupby(['IndicatorName'])

cols = ['IndicatorCode','IndicatorName','NumCountries','NumYears','FirstYear','LastYear']
table_indicators = pd.DataFrame(columns=cols)
for i in range(0,num_indicator):
    temp = wb_indicator.get_group(indicator[i])
    temp = temp.reset_index(drop=True)
    indicatorname = indicator[i]
    numcountries = temp['CountryName'].unique().shape[0]
    numyears = temp['Year'].unique().shape[0]
    firstyear = np.min(np.array(temp['Year']))
    lastyear = np.max(np.array(temp['Year']))
    indicatorcode = temp.ix[0]['IndicatorCode']
    entry = pd.DataFrame([[indicatorcode,indicatorname,numcountries,numyears,firstyear,lastyear]],columns=cols)
    table_indicators = table_indicators.append(entry)

table_indicators[['NumCountries','NumYears','FirstYear','LastYear']] = table_indicators[['NumCountries','NumYears','FirstYear','LastYear']].astype('int')

table_indicators = table_indicators.reset_index(drop=True)
table_indicators.head(10)



