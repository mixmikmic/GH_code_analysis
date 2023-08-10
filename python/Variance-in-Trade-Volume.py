# import necessary libraries
import datetime
import numpy
import matplotlib
from matplotlib import pyplot
import math
from collections import Counter
from scipy import stats

# initialize fundamentals in the notebook
# fundamentals: https://www.quantopian.com/help#ide-fundamentals
fundamentals = init_fundamentals()

# helper function
def addDays(s,n):
    """ takes a date in string format and adds n days to it"""
    end = datetime.datetime.strptime(s,'%Y-%m-%d') + datetime.timedelta(days=n)
    return end.strftime('%Y-%m-%d')

# grab appropriate data, organize by year
dates = ['2002-01-03', '2003-01-03', '2004-01-03', '2005-01-03', '2006-01-03', '2007-01-03', 
        '2008-01-03', '2009-01-03', '2010-01-03', '2011-01-03', '2012-01-03', '2013-01-03', '2014-01-03']
years = range(2002,2015)

# create a dictionary that maps every year in the time period to a list of equities (by sid)
# under consideration for that year
tablesPerYear = {}

for i in range(13):
    # query the fundamentals dataset to get equities fitting our criteria
    tablesPerYear[years[i]] = [x.sid for x in list(get_fundamentals(query()
                            .filter(fundamentals.asset_classification.morningstar_sector_code == 311)
                            .filter(fundamentals.valuation.market_cap > 1000000000)
                           , dates[i]))]

# go through the dictionary and get data on trade volumes and variances for a given period

def getMultiDayData(periods,n):
    """where n = number of days, periods = list of dates to start at"""
    # dicts mapping years to lists of trade volumes / lists of variances
    periodicTradeVolumes = {}    
    periodicVariances = {}    

    for i in range(13):
        year = years[i]
        start = periods[i]
        end = addDays(periods[i],n)
        # collect volumes and variances for that year
        tradeVol = []
        variances = []
        for sid in tablesPerYear[year]:
            prices = get_pricing(sid, 
                                 start_date = start, 
                                 end_date = end,
                                 # note: only grabbing volume
                                 fields = 'volume',
                                 frequency = 'daily')
            # get a list of the volumes
            numerical = [x for x in prices if not numpy.isnan(x)]
            # compute variance, add to list of variances for year
            variance = numpy.var(numerical)
            variances.append(variance)
            # concatenate the list of trade volumes with the list of volumes from this sid
            tradeVol += numerical
        periodicTradeVolumes[year] = [z for z in tradeVol]
        periodicVariances[year] = variances
    
    return periodicTradeVolumes, periodicVariances

annualTradeVolumes, annualVariances = getMultiDayData(dates,365)

# To screen for anomalies and get a better feel for the data,
# I took a look at the trade-volume distributions by year.

fig = pyplot.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
# make a colormap
cmap = pyplot.get_cmap('jet')
# 13 is the number of different colors we want
colors = cmap(numpy.linspace(0.1,0.9,13))
for i in range(13):
    # where 900m is roughly the global max in variance over the time period
    n, bins, patches = ax1.hist(annualTradeVolumes[years[i]],bins=50, range=(0,900000000))
    bins_mean = [0.5 * (bins[j] + bins[j+1]) for j in range(len(n))]
    ax2.scatter(bins_mean, map(lambda x: math.log(x) if x>1 else 0, n), c=colors[i], label=years[i], alpha = 0.75)
    ax2.legend()
    pyplot.xlabel("Annual Trade Volume")
    pyplot.ylabel("Log-Number of Companies")
# delete the first subplot -- we don't need it, we only had to construct it to derive the second
fig.delaxes(ax1)
pyplot.draw()

# For every year: plot the log of the variances in trade volume, and fit a normal distribution to it

for i in range(13):
    year = years[i]
    variances = sorted([math.log(x) for x in annualVariances[year] if not (numpy.isnan(x) or x==0)])
    # fitting the histogram
    n, bins, patches = pyplot.hist(variances,bins=25,normed=True)
    mu = numpy.mean(variances)
    sigma = numpy.std(variances)
    # fit a normal probability distribution to the data
    pdf = stats.norm.pdf(bins, mu, sigma)
    pyplot.title(str(year))
    pyplot.xlabel("Log-Variance")
    pyplot.ylabel("Normed Frequency")
    pyplot.plot(bins,pdf, '-o')
    pyplot.show()

aggregateVariances = []

for i in range(13):
    aggregateVariances += [math.log(x) for x in annualVariances[years[i]] if not (numpy.isnan(x) or x==0)]
aggregateVariances.sort()

n, bins, patches = pyplot.hist(aggregateVariances,bins=25,normed=True)
mu = numpy.mean(aggregateVariances)
sigma = numpy.std(aggregateVariances)
# fit a normal probability distribution to the data
pdf = stats.norm.pdf(bins, mu, sigma)
pyplot.title("2002-2015 Variances")
pyplot.xlabel("Log-Variance")
pyplot.ylabel("Normed Frequency Count")
pyplot.plot(bins,pdf, '-o')
pyplot.show()

fig = pyplot.figure()
ax = fig.add_subplot(111)
stats.probplot(aggregateVariances, plot=ax)
pyplot.show()

randomDates1 = ['2002-08-24', '2003-03-01', '2004-12-23', '2005-06-14', '2006-08-31', '2007-08-12',
              '2008-01-25','2009-02-16','2010-11-09','2011-05-28','2012-01-12','2013-05-10','2014-10-02']

randomDates2 = ['2002-02-14', '2003-10-11', '2004-08-30', '2005-03-09', '2006-12-02', '2007-05-21',
              '2008-06-15','2009-06-09','2010-02-25','2011-05-22','2012-12-31','2013-07-09','2014-05-29']

# Note that we'll only need the variance data.
thirtyTradeVolumes, thirtyVariances = getMultiDayData(randomDates,30)

ninetyTradeVolumes, ninetyVariances = getMultiDayData(randomDates,90)

# To save space, for every year we'll display the plot for the 30-day data
# next to the plot of the 90-day data.

aggregateV30 = []
aggregateV90 = []

for i in range(13):
    year = years[i]
    variances30 = sorted([math.log(x) for x in thirtyVariances[year] if not (numpy.isnan(x) or x==0)])
    variances90 = sorted([math.log(x) for x in ninetyVariances[year] if not (numpy.isnan(x) or x==0)])
    
    # add to the aggregate variances for these shorter periods
    aggregateV30 += variances30
    aggregateV90 += variances90
    
    # make the left plot
    pyplot.subplot(121)
    n, bins, patches = pyplot.hist(variances30,bins=25,normed=True)
    mu = numpy.mean(variances)
    sigma = numpy.std(variances)
    pdf = stats.norm.pdf(bins, mu, sigma)
    pyplot.title(str(year)+ ": random 30 day period")
    pyplot.xlabel("Log-Variance")
    pyplot.ylabel("Normed Frequency Count")
    pyplot.plot(bins,pdf, '-o')

    # right plot
    pyplot.subplot(122)
    n, bins, patches = pyplot.hist(variances90,bins=25,normed=True)
    mu = numpy.mean(variances)
    sigma = numpy.std(variances)
    pdf = stats.norm.pdf(bins, mu, sigma)
    pyplot.title(str(year) + ": random 90 day period")
    pyplot.xlabel("Log-Variance")
    pyplot.ylabel("Normed Frequency Count")
    pyplot.plot(bins,pdf, '-o')
    
    pyplot.show()

# make the left plot
pyplot.subplot(121)
n, bins, patches = pyplot.hist(aggregateV30,bins=25,normed=True)
mu = numpy.mean(variances)
sigma = numpy.std(variances)
pdf = stats.norm.pdf(bins, mu, sigma)
pyplot.title("Aggregate plot of the 13 random 30-day periods")
pyplot.xlabel("Log-Variance")
pyplot.ylabel("Normed Frequency Count")
pyplot.plot(bins,pdf, '-o')

# right plot
pyplot.subplot(122)
n, bins, patches = pyplot.hist(aggregateV90,bins=25,normed=True)
mu = numpy.mean(variances)
sigma = numpy.std(variances)
pdf = stats.norm.pdf(bins, mu, sigma)
pyplot.title("Aggregate plot of the 13 random 90-day periods")
pyplot.xlabel("Log-Variance")
pyplot.ylabel("Normed Frequency Count")
pyplot.plot(bins,pdf, '-o')

pyplot.show()

pyplot.subplot(121)
stats.probplot(aggregateV30, plot=pyplot)
pyplot.title("Aggregate 30-day Probability Plot")

pyplot.subplot(122)
stats.probplot(aggregateV90, plot=pyplot)
pyplot.title("Aggregate 90-day Probability Plot")

pyplot.show()



