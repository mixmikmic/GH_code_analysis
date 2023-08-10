get_ipython().magic('matplotlib inline')

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import psycopg2

pd.options.display.max_columns = 40

connection = psycopg2.connect('dbname = threeoneone user=threeoneoneadmin password=threeoneoneadmin')
cursor = connection.cursor()

cursor.execute('''SELECT createddate, borough FROM service;''')
borodata = cursor.fetchall()

borodata = pd.DataFrame(borodata)

borodata.columns = ['Date', 'Boro']

borobydate = borodata.groupby(by='Boro').count()

borobydate

boropop = {
    'MANHATTAN': 1636268,
    'BRONX': 1438159,
    'BROOKLYN': 2621793,
    'QUEENS': 2321580,
    'STATEN ISLAND': 473279,
    }

borobydate['Pop'] = [boropop.get(x) for x in borobydate.index]

borobydate['CompPerCap'] = borobydate['Date']/borobydate['Pop']

borobydate

borodata['Year'] = [x.year for x in borodata['Date']]
borodata['Month'] = [x.month for x in borodata['Date']]

borodata = borodata.loc[borodata['Boro'] != 'Unspecified']

boroplotdata = borodata.groupby(by=['Boro', 'Year','Month']).count()
boroplotdata

boros = borobydate.index
borodict = {x:[] for x in boros}
borodict.pop('Unspecified')

for boro in borodict:
    borodict[boro] = list(boroplotdata.xs(boro).Date)

plotdata = np.zeros(len(borodict['BROOKLYN']))
for boro in borodict:
    plotdata = np.row_stack((plotdata, borodict[boro]))

plotdata = np.delete(plotdata, (0), axis=0)

plotdata

from matplotlib import patches as mpatches

x = np.arange(len(plotdata[0]))

#crude xlabels
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016']
xlabels = []
for year in years:
    for month in months:
        xlabels.append("{0} {1}".format(month,year))

plotcolors = [(1,0,103),(213,255,0),(255,0,86),(158,0,142),(14,76,161),(255,229,2),(0,95,57),            (0,255,0),(149,0,58),(255,147,126),(164,36,0),(0,21,68),(145,208,203),(98,14,0)]

#rescaling rgb from 0-255 to 0 to 1
plotcolors = [(color[0]/float(255),color[1]/float(255),color[2]/float(255)) for color in plotcolors]
legendcolors = [mpatches.Patch(color = color) for color in plotcolors]

plt.figure(figsize = (15,10));
plt.stackplot(x,plotdata, colors = plotcolors);
plt.xticks(x,xlabels,rotation=90);
plt.xlim(0,76)
plt.legend(legendcolors,borodict.keys(), bbox_to_anchor=(0.2, 1));
plt.title('311 Complaints by Borough', size = 24)
plt.ylabel('Number of Complaints',size = 14)

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

borodata.groupby(by = ['Boro', 'Year']).count()

