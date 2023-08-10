get_ipython().magic('matplotlib inline')

import pandas as pd

def read_table(filename):
    fp = open(filename)
    t = pd.read_html(fp)
    table = t[5]
    return table

table1 = read_table('blogger1.html')
table1.shape

table2 = read_table('blogger2.html')
table2.shape

table = pd.concat([table1, table2], ignore_index=True)
table.shape

import string
chars = string.ascii_letters + ' '

def convert(s):
    return (int(s.rstrip(chars)))

def clean(s):
    i = s.find('Edit')
    return s[:i]

table['title'] = table[1].apply(clean)
table.title

table['plusses'] = table[4].fillna(0)
table.plusses.head()

table['comments'] = table[5].apply(convert)
table.comments.head()

table['views'] = table[6].apply(convert)
table.views

table['date'] = pd.to_datetime(table[7])
table.date.head()

table = table[table.views > 0]
table.shape

table.index = range(115, 0, -1)
table.title



dates = table.date.sort_values()
diffs = dates.diff()
diffs.head()

diffs.dropna().describe()

table.sort_values(by=['views'], ascending=False)[['title', 'views', 'date']].head(20)

table.sort_values(by=['views'], ascending=True)[['title', 'views', 'date']].head(20)

import thinkstats2
import thinkplot

cdf = thinkstats2.Cdf(table.views)

thinkplot.PrePlot(1)
thinkplot.Cdf(cdf, complement=True)
thinkplot.Config(xlabel ='Number of page views', xscale='log', 
                 ylabel='CCDF', yscale='log', 
                 legend=False)

table.sort_values(by=['comments'], ascending=False)[['title', 'comments', 'date']].head(5)

table.sort_values(by=['plusses'], ascending=False)[['title', 'plusses', 'date']].head(5)





