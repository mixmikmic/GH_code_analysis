get_ipython().magic('matplotlib inline')

from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import pandas as pd
import numpy as np
import psycopg2
from datetime import datetime
from datetime import date


pd.options.display.max_columns = 40

connection = psycopg2.connect('dbname = threeoneone user=threeoneoneadmin password=threeoneoneadmin')
cursor = connection.cursor()

cursor.execute('''SELECT createddate, borough, complainttype FROM service;''')
borodata = cursor.fetchall()

borodata = pd.DataFrame(borodata)

borodata.columns = ['Date', 'Boro', 'Comptype']

borodata = borodata.loc[borodata['Date'] > date(2011,3,1)]

borodata = borodata.loc [borodata['Date'] <date(2016,6,1)]

borodata = borodata.loc[borodata['Boro'] != 'Unspecified']

borodata['Year'] = [x.year for x in borodata['Date']]
borodata['Month'] = [x.month for x in borodata['Date']]
borodata['Day'] = [x.day for x in borodata['Date']]

plotdata = borodata.groupby(by=['Boro', 'Year', 'Month']).count()
plotdata.head()

plotdict = {x:[] for x in borodata['Boro'].unique()}

for boro in plotdict:
    plotdict[boro] = list(plotdata.xs(boro).Date)

plotdata = np.zeros(len(plotdict['BROOKLYN']))
legendkey = []

for boro in plotdict.keys():
    plotdata = np.row_stack((plotdata, plotdict[boro]))
    legendkey.append(boro)
    
plotdata = np.delete(plotdata, (0), axis=0)

x = np.arange(len(plotdata[0]))

#crude xlabels
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
years = ['2011', '2012', '2013', '2014', '2015', '2016']
xlabels = []
for year in years:
    for month in months:
        xlabels.append("{0} {1}".format(month,year))

        
xlabels = xlabels[2:-7] #start from march 2011, end may2016

plotcolors = [(1,0,103),(213,255,0),(255,0,86),(158,0,142),(14,76,161),(255,229,2),(0,95,57),            (0,255,0),(149,0,58),(255,147,126),(164,36,0),(0,21,68),(145,208,203),(98,14,0)]

#rescaling rgb from 0-255 to 0 to 1
plotcolors = [(color[0]/float(255),color[1]/float(255),color[2]/float(255)) for color in plotcolors]
legendcolors = [mpatches.Patch(color = color) for color in plotcolors]

plt.figure(figsize = (15,10));
plt.stackplot(x,plotdata, colors = plotcolors);
plt.xticks(x,xlabels,rotation=90);
plt.xlim(0,len(xlabels))
plt.legend(legendcolors,legendkey, bbox_to_anchor=(0.2, 1));
plt.title('Heating Complaints by Borough', size = 24)
plt.ylabel('Number of Complaints',size = 14)

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

plotdatacsv = pd.DataFrame(plotdata)
plotdatacsv.index = legendkey
plotdatacsv.columns = xlabels
plotdatacsv.to_csv('totalcomplaintsbyboroughandmonth.csv')

normplotdata = np.ndarray(shape=np.shape(plotdata))

totalcomplaintsbymonth = borodata.groupby(by=['Year','Month']).count().Date

totalcomplaintsbymonth = totalcomplaintsbymonth.values

totalcomplaintsbymonth = np.tile(totalcomplaintsbymonth,(5,1))

normplotdata = plotdata/totalcomplaintsbymonth

plotcolors = [(1,0,103),(213,255,0),(255,0,86),(158,0,142),(14,76,161),(255,229,2),(0,95,57),            (0,255,0),(149,0,58),(255,147,126),(164,36,0),(0,21,68),(145,208,203),(98,14,0)]

#rescaling rgb from 0-255 to 0 to 1
plotcolors = [(color[0]/float(255),color[1]/float(255),color[2]/float(255)) for color in plotcolors]
legendcolors = [mpatches.Patch(color = color) for color in plotcolors]

plt.figure(figsize = (15,10));
plt.stackplot(x,normplotdata, colors = plotcolors);
plt.xticks(x,xlabels,rotation=90);
plt.xlim(0,len(xlabels))
plt.ylim(0,1)
plt.legend(legendcolors,legendkey, bbox_to_anchor=(0.2, 1));
plt.title('Heating Complaints by Borough', size = 24)
plt.ylabel('Number of Complaints',size = 14)

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().yaxis.set_ticks_position('left')
plt.gca().xaxis.set_ticks_position('bottom')

boropop = {
    'MANHATTAN': 1636268,
    'BRONX': 1438159,
    'BROOKLYN': 2621793,
    'QUEENS': 2321580,
    'STATEN ISLAND': 473279,
    }

totalnycpop = reduce(lambda x, y: x +y, boropop.values())
totalnycpop

complaintsbyyear = borodata.groupby(by=['Year']).count()
borocomplaintsbyyear = borodata.groupby(by=['Year','Boro']).count()

complaintsbyyear['Pop'] = [totalnycpop for x in complaintsbyyear.index]

borocomplaintsbyyear['Pop'] = [boropop.get(x[1]) for x in borocomplaintsbyyear.index]

complaintsbyyear['CompPerCap'] = complaintsbyyear['Day']/complaintsbyyear['Pop']

complaintsbyyear

borocomplaintsbyyear['CompPerCap'] = borocomplaintsbyyear['Day']/borocomplaintsbyyear['Pop']

borocomplaintsbyyear

