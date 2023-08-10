import os
import glob
import pandas as pd
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from dateutil.relativedelta import relativedelta
from datetime import datetime
from mwviews.api import PageviewsClient
from collections import defaultdict

def getViews(language, article, start=None, end=None):
    """
    Returns list of tuples of the form (date, pageViews)
    start, end: YYYYMMDDHH
    """
    pvc = PageviewsClient()
    project = language + ".wikipedia"
    data = pvc.article_views(project, [article], start=start, end=end)
    # put the views into a list of tuples, (date, views)
    tuples = [(d, data[d][article]) for d in data]
    # sort the tuples by date
    tuples.sort(key=lambda t: t[0])
    return tuples

def plotViews(language, article, start=None, end=None):    
    # pull the data from the mwviews API, and plot
    startdt, enddt = datetime.strptime(start, '%Y%m%d%H'), datetime.strptime(end, '%Y%m%d%H') 
    numMonths = (enddt.year - startdt.year)*12 + enddt.month - startdt.month
    data = getViews(language, article, start=start, end=end)
    
    months = MonthLocator(range(1,numMonths + 1), bymonthday=1, interval=1)
    monthsFmt = DateFormatter("%b `%y")

    dates = [t[0] for t in data]
    views = [t[1] for t in data]

    fig, ax = plt.subplots()
    ax.plot_date(dates, views, '-')
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.autoscale_view()
    fig.autofmt_xdate()
    plt.show()

plotViews('en', 'Zika', start='2015010100', end='2016123123')

pd.read_csv("./data/zika/CDC_Report-2016-02-24.csv")[:3]

path = '/home/william/wikidemics/data/zika'
files = glob.glob(path + "/*.csv")
zika = pd.DataFrame()
for f in sorted(files):
    week = pd.read_csv(f)
    zika = zika.append(week)
weekly = zika.groupby('report_date')['value'].sum()

# aggregated weekly sums of incidence
weekly.index = weekly.index.to_datetime()
weekly.plot(title="USA Zika Reported Cases")

