get_ipython().magic('matplotlib inline')

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

pd.options.display.max_columns = 40

hpdcompprob = pd.read_csv("Complaint_Problems.csv")

type(hpdcompprob.StatusDate[0])

def extract_dates(series, nameofdatecolumn):
    series['Year'] = series[nameofdatecolumn].str.extract(r'../../(.*)', expand=True)
    series['Month'] = series[nameofdatecolumn].str.extract(r'(.*)/../....', expand=True)
    series['Day'] = series[nameofdatecolumn].str.extract(r'../(.*)/....', expand=True)
    
extract_dates(hpdcompprob, 'StatusDate')

def count_problem_types(pdseries):
    problemtypes = pdseries.unique()
    problemcounts = {k:0 for k in problemtypes}
    problemcounts['TOTAL'] = 0
    
    for problem in pdseries:
        problemcounts[problem] += 1
    
    problemcounts['TOTAL'] = len(pdseries)
    print year, problemcounts #this is just to show us what those counts are
        
    return problemcounts

#years = hpdcompprob.Year.unique()
years = ['2014','2015','2016'] #We know that only these data are relevant (the other, older years, are remnants of old cases)
yearsdict = {k:{} for k in years}
for year in years:
    yearsdict[year] = count_problem_types(hpdcompprob.loc[hpdcompprob.Year == year].MajorCategory)

labels = sorted(yearsdict['2014'].keys())
labels.remove('CONSTRUCTION')
labels.remove('HEATING') #removed these because 2015 doesn't have an entry, and both the other years have negligible entries
labels.remove('TOTAL') #not interested in total in this plot

years = ['2014','2015','2016']
plotdict = {k:[] for k in years}
for year in years:
    for label in labels:
        plotdict[year].append(yearsdict[year][label])

len(labels)

x = np.arange(len(years))
y = np.column_stack((plotdict['2014'],plotdict['2015'],plotdict['2016']))
plt.figure(figsize = (15,10));
plt.stackplot(x,y);
plt.legend(labels)

from matplotlib import patches as mpatches

plotcolors = [(0,0,0),(1,0,103),(213,255,0),(255,0,86),(158,0,142),(14,76,161),(255,229,2),(0,95,57),            (0,255,0),(149,0,58),(255,147,126),(164,36,0),(0,21,68),(145,208,203),(98,14,0)]

#rescaling rgb from 0-255 to 0 to 1
plotcolors = [(x[0]/float(255),x[1]/float(255),x[2]/float(255)) for x in plotcolors]
legendcolors = [mpatches.Patch(color = x) for x in plotcolors]

x = np.arange(len(years))
y = np.column_stack((plotdict['2014'],plotdict['2015'],plotdict['2016']))
plt.figure(figsize = (15,10));
plt.stackplot(x,y,colors = plotcolors);
plt.legend(legendcolors, labels);

#some labels for the x axis
xlabels = ['2014', '2015', '2016']
plt.xticks(x,xlabels, size = 18);
plt.yticks(size = 18);
plt.title('HPD Complaints by type over time', size = 24);

