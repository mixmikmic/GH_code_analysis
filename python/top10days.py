# Imports
get_ipython().run_line_magic('matplotlib', 'inline')
import quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Get price data from Quandl
data = quandl.get("BCHAIN/MKPRU")
data = data.shift(-1) # data set has daily open, we want daily close

# Take a look
data.head()

# See where the 0's end
data.loc[data['Value'] > 0].head()

# Remove the 0's
data = data.loc['2010-08-17':]

# Look again
data.head()

# Summary, no. of rows looks correct
data.info()

# Visual check of price data
data['Value'].plot(figsize=(12,8))

# Using a log scale
fig , ax = plt.subplots(figsize=(12,10))
y = data['Value']
y.plot()
plt.yscale('log')
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
#ax.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%.1f'))

# Freq plot by year showing distribution of returns
(data.pct_change() * 100).hist(bins=100, figsize = (12,10), by = lambda x: x.year)

# Best and worst days from entire data set
print "Worst" + "-" * 20
print((data['Value'].pct_change() * 100).dropna().sort_values().head(10).sort_values())
print "Best" + "-" * 20
print((data['Value'].pct_change() * 100).dropna().sort_values().tail(10).sort_values(ascending = False))

# Group by year, calculate yearly returns, drop years without full year of data
data_annual = data.asfreq("Y")
data_annual['Returns'] = data_annual['Value'].pct_change() * 100
data_annual.index = data_annual.index.year
data_annual.dropna(inplace=True)

# Take a look at results
data_annual['Returns'].head(10)

# Chart the results
colors = np.sign(data_annual['Returns']).map({1 : 'g', -1 : 'r'}).values
data_annual.index.name = '' # Don't want to show 'Date' as a label on chart
ax = data_annual['Returns'].plot(kind='bar', figsize=(12,8),color = colors)
ax.legend(["Fully invested"]);
for p in ax.patches:
    b = p.get_bbox()
    val = "{:,.0f}%".format(b.y1 + b.y0)        
    ax.annotate(val,((b.x0 + b.x1) / 2 + len(val) * -0.03, b.y1 + 110))

# Back to daily data, calculate returns from prices and then group by year
data['Returns'] = data['Value'].pct_change()
data.dropna(inplace = True)
returns_by_year = data.groupby(lambda x: x.year)

# Go through each year, find the top 10 days, then calculate returns with and without

dates = []

data_annual['MissTop10'] = 0.0
data_annual['MissTop5'] = 0.0

for year, frame in returns_by_year:
    
    full_return = (frame['Returns'] + 1).cumprod()[-1] - 1
    top10 = frame['Returns'].sort_values()[-10:]
    top5 = top10[-5:]
    top10_return = (top10 + 1).cumprod()[-1] - 1
    top5_return = (top5 + 1).cumprod()[-1] - 1
    full_less_top10_return = ((1 + full_return) / (1 + top10_return)) - 1
    full_less_top5_return = ((1 + full_return) / (1 + top5_return)) - 1
    
    data_annual.loc[year, 'MissTop10'] = full_less_top10_return * 100
    data_annual.loc[year, 'MissTop5'] = full_less_top5_return * 100
    
    print year, '-' * 30 
    # print top10.index
    if year >= 2012:
        dates += list(top10.index.astype(str))
    print "Full year return {:,.0f}%".format(full_return * 100 )
    print "top 10 days return {:,.0f}%".format(top10_return * 100)
    print "Return if miss 10 top days {:,.0f}%".format(full_less_top10_return * 100)
    #if year in [2017]:
    print "10 best days"
    print top10[::-1] * 100

print(dates)

# View the results
data_annual.dropna(inplace = True)
data_annual.head(10)

# Chart the results
ax = data_annual[['Returns','MissTop10']].plot(kind='bar', figsize=(12,8), width = 0.8)
ax.legend(["Fully invested", "Miss Top 10"]);
for p in ax.patches:
    b = p.get_bbox()
    val = "{:,.0f}%".format(b.y1 + b.y0)        
    ax.annotate(val,((b.x0 + b.x1) / 2 + len(val) * -0.03, b.y1 + 110))

# Calculate total returns for period and compound annual returns
print "Total returns"
print "-" * 25
print "Fully invested since start 2011 {:,.0f}%".format((((data_annual['Returns'] / 100 + 1).cumprod().iloc[-1])- 1) * 100)
print "Missed top 10 each year {:,.0f}%".format((((data_annual['MissTop10'] / 100 + 1).cumprod().iloc[-1]) - 1) * 100)
print "Missed top 5 each year {:,.0f}%".format((((data_annual['MissTop5'] / 100 + 1).cumprod().iloc[-1])- 1) * 100)
print
print "Compound annual returns"
print "-" * 25
print "Fully invested since start 2011 {:,.1f}%".format((((data_annual['Returns'] / 100 + 1).cumprod().iloc[-1]) ** (1.0/len(data_annual)) - 1) * 100)
print "Missed top 10 each year {:,.1f}%".format((((data_annual['MissTop10'] / 100 + 1).cumprod().iloc[-1]) ** (1.0/len(data_annual)) - 1) * 100)
print "Missed top 5 each year {:,.1f}%".format((((data_annual['MissTop5'] / 100 + 1).cumprod().iloc[-1]) ** (1.0/len(data_annual)) - 1) * 100)

# sanity check on 2013 returns
start, end = data.loc['2012-12-31','Value'], data.loc['2013-12-31','Value']
print start, end
print (end / start - 1) * 100
print ((data.loc['2013-1-1':'2013-12-31','Returns'] + 1).cumprod()[-1] - 1) * 100

# check prices a few days before and after
print data.loc['2012-12-25':'2013-01-5','Value']
print data.loc['2013-12-25':'2014-01-5','Value']

# sanity check on total returns
start, end = data.loc['2010-12-31','Value'], data.loc['2017-12-31','Value']
print start, end
print (end / start - 1) * 100
print ((data.loc['2011-1-1':'2017-12-31','Returns'] + 1).cumprod()[-1] - 1) * 100

stats = returns_by_year.describe().drop(columns=['Value'], axis = 1)

stats.columns = [u'count', u'mean', u'std', u'min', u'25%', u'50%', u'75%', u'max']

stats['skew'] = 3 * (stats['mean'] - stats['50%']) / stats['std'] # Pearson Median Skewness
stats

# skew by year (3rd standardised moment)
[(year, frame['Returns'].skew()) for year, frame in returns_by_year]

data.loc['2017-12-06':].head(10)

