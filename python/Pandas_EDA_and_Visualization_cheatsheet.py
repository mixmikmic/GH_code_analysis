import pandas as pd
import matplotlib.pyplot as plt

# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')
# increase default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

# read in the drinks data
drink_cols = ['country', 'beer', 'spirit', 'wine', 'liters', 'continent']
url = 'https://github.com/JamesByers/Datasets/raw/master/drinks.csv'
drinks = pd.read_csv(url, header=0, names=drink_cols, na_filter=False)

# sort the beer column and mentally split it into 3 groups
drinks.beer.order().values

# compare with histogram
drinks.beer.plot(kind='hist', bins=3)

# try more bins
drinks.beer.plot(kind='hist', bins=20)

# add title and labels
drinks.beer.plot(kind='hist', bins=20, title='Histogram of Beer Servings')
plt.xlabel('Beer Servings')
plt.ylabel('Frequency')

# compare with density plot (smooth version of a histogram)
drinks.beer.plot(kind='density', xlim=(0, 500))

# select the beer and wine columns and sort by beer
drinks[['beer', 'wine']].sort('beer').values

# compare with scatter plot
drinks.plot(kind='scatter', x='beer', y='wine')

# add transparency
drinks.plot(kind='scatter', x='beer', y='wine', alpha=0.3)

# vary point color by spirit servings
drinks.plot(kind='scatter', x='beer', y='wine', c='spirit', colormap='Blues')

# scatter matrix of three numerical columns
pd.scatter_matrix(drinks[['beer', 'spirit', 'wine']])

# increase figure size
pd.scatter_matrix(drinks[['beer', 'spirit', 'wine']], figsize=(10, 8))

# count the number of countries in each continent
drinks.continent.value_counts()

# compare with bar plot
drinks.continent.value_counts().plot(kind='bar')

# calculate the mean alcohol amounts for each continent
drinks.groupby('continent').mean()

# side-by-side bar plots
drinks.groupby('continent').mean().plot(kind='bar')

# drop the liters column
drinks.groupby('continent').mean().drop('liters', axis=1).plot(kind='bar')

# stacked bar plots
drinks.groupby('continent').mean().drop('liters', axis=1).plot(kind='bar', stacked=True)

# sort the spirit column
drinks.spirit.order().values

# show "five-number summary" for spirit
drinks.spirit.describe()

# compare with box plot
drinks.spirit.plot(kind='box')

# include multiple variables
drinks.drop('liters', axis=1).plot(kind='box')

# read in the ufo data
url = 'https://github.com/JamesByers/Datasets/raw/master/ufo.csv'

ufo = pd.read_csv(url)
ufo['Time'] = pd.to_datetime(ufo.Time)
ufo['Year'] = ufo.Time.dt.year


# count the number of ufo reports each year (and sort by year)
ufo.Year.value_counts().sort_index()

# compare with line plot
ufo.Year.value_counts().sort_index().plot()

# don't use a line plot when there is no logical ordering
drinks.continent.value_counts().plot()

### Grouped Box Plots: show one box plot for each group

# reminder: box plot of beer servings
drinks.beer.plot(kind='box')

# box plot of beer servings grouped by continent
drinks.boxplot(column='beer', by='continent')

# box plot of all numeric columns grouped by continent
drinks.boxplot(by='continent')

# reminder: histogram of beer servings
drinks.beer.plot(kind='hist')

# histogram of beer servings grouped by continent
drinks.hist(column='beer', by='continent')

# share the x axes
drinks.hist(column='beer', by='continent', sharex=True)

# share the x and y axes
drinks.hist(column='beer', by='continent', sharex=True, sharey=True)

# change the layout
drinks.hist(column='beer', by='continent', sharex=True, layout=(2, 3))

# saving a plot to a file
drinks.beer.plot(kind='hist', bins=20, title='Histogram of Beer Servings')
plt.xlabel('Beer Servings')
plt.ylabel('Frequency')
plt.savefig('assets/beer_histogram.png')

# list available plot styles
plt.style.available

# change to a different style
drinks.beer.plot(kind='hist', bins=20, title='Histogram of Beer Servings')
plt.xlabel('Beer Servings')
plt.ylabel('Frequency')
plt.savefig('assets/beer_histogram.png')
plt.style.use('fivethirtyeight')



