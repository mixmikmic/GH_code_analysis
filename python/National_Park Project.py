import pandas as pd
import numpy as np

parkList = pd.read_csv('Nationalpark__List.csv')

parkList = parkList.sort_values(by = 'Number_of_Visitors', ascending = False)

top10 = parkList.head(15)

from bokeh.plotting import figure
from bokeh.charts import Bar
from bokeh.charts.attributes import CatAttr
from bokeh.models import HoverTool, NumeralTickFormatter
from bokeh.io import output_notebook, show
output_notebook()

# create a new plot using figure
p = figure(plot_width=1000, plot_height=400)

visitors = parkList.groupby('ParkName')[['Number_of_Visitors']].mean().sort_values('Number_of_Visitors', ascending=False).head(15)
p = Bar(visitors, label=CatAttr(columns=['ParkName'], sort=False), values='Number_of_Visitors', color='ParkName')

p.legend.location = None
p.xaxis.axis_label = "ParkName"
p.yaxis.axis_label = "Visitors"

# x and y here refer to the x-axis and y-axis in your graph.
hover = HoverTool(tooltips=[
    ("ParkName", "@x"),
    ("Number_of_Visitors", "@y"),
])
p.add_tools(hover)

show(p)

import matplotlib.pyplot as plt
import pandas as pd
reviews_1 = pd.read_csv('/Users/joshualee/Downloads/reviews_1.csv', header = None)
reviews_1.columns = ['ratings', 'reviews_index', 'reviews_topic']
reviews_1 = reviews_1.dropna(axis=0, how='any')
from os import path
from scipy.misc import imread
import random
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords

stopwords = set(STOPWORDS)
fig_cloud = plt.figure(figsize = (10,10))
wordcloud = WordCloud(font_path='/Library/Fonts/Verdana.ttf',
                      relative_scaling = 1.0,
                      stopwords = stopwords, # set or space-separated string
                      regexp = r"\w[^\d\W']+"
                      ).generate(str(reviews_1.reviews_topic))

plt.imshow(wordcloud)
plt.axis("off")
plt.show()

import datetime
from datetime import date
from fbprophet import Prophet

camping= pd.read_csv('National_park_camp.csv')

import matplotlib.pyplot as plt
camping_smoky = camping.loc[camping['ParkName'] == 'Great Smoky Mountain']
camping_smoky['y'] = np.log(camping_smoky['Recreation Visitors'])
camping_smoky['ds']=camping_smoky['Month']
visitors = camping_smoky[['ds', 'y']]
visitors


m = Prophet()
m.fit(visitors)
future = m.make_future_dataframe(freq = 'D', periods=14)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat']].tail()
forecast
m.plot(forecast)
plt.show()

import matplotlib.pyplot as plt
camping_yosemite = camping.loc[camping['ParkName'] == 'Yosemite']
camping_yosemite['y'] = np.log(camping_yosemite['Recreation Visitors'])
camping_yosemite['ds']=camping_yosemite['Month']
visitors = camping_yosemite[['ds', 'y']]
visitors


m = Prophet()
m.fit(visitors)
future = m.make_future_dataframe(freq = 'D', periods=14)
future.tail()
forecast = m.predict(future)
forecast[['ds', 'yhat']].tail()
forecast
m.plot(forecast)
plt.show()

import glob, os
path = r'/Users/joshualee/Desktop/National_Park_Project'                     # use your path
all_files = glob.glob(os.path.join(path, "*.csv"))    # advisable to use os.path.join as this makes concatenation OS independent
df_from_each_file = (pd.read_csv(f) for f in all_files)
campground   = pd.concat(df_from_each_file, ignore_index=True)

cols = ['Longtitude','Latitude','Reviews','Maximum_Number_of_People']
campground[cols] = campground[cols].apply(pd.to_numeric, errors='coerce', axis=1)

count_=campground.groupby(['ParkName'])['Maximum_Number_of_People'].sum()
count_ = count_.reset_index()

top10.ParkName=top10.ParkName.replace('Great Smoky Mountains National Park', 'Great Smoky Mountain')
top10.ParkName=top10.ParkName.replace('Grand Canyon National Park', 'GrandCanyon')
top10.ParkName=top10.ParkName.replace('Yosemite National Park', 'Yosemite')
top10.ParkName=top10.ParkName.replace('Rocky Mountain National Park', 'Rocky Mountain')
top10.ParkName=top10.ParkName.replace('Zion National Park', 'Zion')
top10.ParkName=top10.ParkName.replace('Glacier National Park', 'Glacier')
top10.ParkName=top10.ParkName.replace('Olympic National Park', 'Olympic')
top10.ParkName=top10.ParkName.replace('Acadia National Park', 'Acadia')
top10.ParkName=top10.ParkName.replace('Joshua Tree National Park', 'Joshua Tree')
top10.ParkName=top10.ParkName.replace('Cuyahoga Valley National Park', 'Cuyahoga Valley')
top10.ParkName=top10.ParkName.replace('Bryce Canyon National Park', 'Bryce Canyon')
top10.ParkName=top10.ParkName.replace('Arches National Park', 'Arches')
top10=top10[top10.ParkName != 'Yellow Stone National Park']
top10=top10[top10.ParkName != 'Grand Teton National Park']
top10=top10[top10.ParkName != 'Hawai Volcanoes National Park']
top10 = top10.drop('Location', 1)

total=pd.merge(top10, count_, how='inner', on ='ParkName')
total['August'] =[1183778,743158,692450,772849,477507,813267,735945,748565,148427,743158,365738,188340]
total['campers']=[3010,6650,11555,186,2458,128,2250,293,2403,365,791,317]
total['density']=total['campers']/total['August']
total['high_weather'] = [87,103,89,77,98,69,76,78,99,81,77,97]
total['low_weather'] = [60,75,56,45,68,51,56,46,68,62,50,66]
total['mean_weather']=[74,89,73,61,83,60,66,62,84,72,64,82]
total=total.sort_values('density')

p = figure(plot_width=1000, plot_height=400)

p = Bar(total, label=CatAttr(columns=['ParkName'], sort=False), values='density', color='ParkName')

p.legend.location = None
p.xaxis.axis_label = "ParkName"
p.yaxis.axis_label = "Density"

# x and y here refer to the x-axis and y-axis in your graph.
hover = HoverTool(tooltips=[
    ("ParkName", "@x"),
    ("Number_of_Visitors", "@y"),
])
p.add_tools(hover)

show(p)

import seaborn as sns

#fig = plt.figure(figsize = (15,10))
#_ = sns.countplot(y = campground.ParkName)
#plt.show()

fig = plt.figure(figsize = (15,10))
campground.groupby('ParkName')['ParkName'].count().sort_values().plot.barh()
plt.show()

from mpl_toolkits.basemap import Basemap

fig2 = plt.figure(figsize = (15,10))
m = Basemap(projection = 'mill',
            llcrnrlat = 24.0, llcrnrlon = -125.0, 
            urcrnrlat = 52.0, urcrnrlon = -65.0, resolution = 'h')
m.drawcoastlines()
m.drawcountries(linewidth = 2)
m.drawstates(color = 'b')
m.fillcontinents(color = '#f2f2f2', lake_color = 'aqua', zorder = 1)
                
ax = fig2.add_subplot(111)
x, y = m(np.array(campground['Longtitude']), np.array(campground['Latitude']))
scatter1 = ax.scatter(x, y, 300, 
                      c = np.array(campground['Stars']),
                      marker = 'x',zorder = 1.5, cmap = plt.cm.tab10)
plt.colorbar(scatter1)
plt.show()

from mpl_toolkits.basemap import Basemap

fig2 = plt.figure(figsize = (15,10))
m = Basemap(projection = 'mill',
            llcrnrlat = 24.0, llcrnrlon = -125.0, 
            urcrnrlat = 52.0, urcrnrlon = -65.0, resolution = 'h')
m.drawcoastlines()
m.drawcountries(linewidth = 2)
m.drawstates(color = 'b')
m.fillcontinents(color = '#f2f2f2', lake_color = 'aqua', zorder = 1)
                
ax = fig2.add_subplot(111)
x, y = m(np.array(campground['Longtitude']), np.array(campground['Latitude']))
scatter1 = ax.scatter(x, y, 300, 
                      c = np.array(campground['Reviews']),
                      marker = 'x',zorder = 1.5, cmap = plt.cm.tab10)
plt.colorbar(scatter1)
plt.show()



