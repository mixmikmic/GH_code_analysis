import pandas as pd
import matplotlib.pyplot as plt

# display plots in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

# increase default figure and font sizes for easier viewing
plt.rcParams['figure.figsize'] = (8, 6)
plt.rcParams['font.size'] = 14

# read in the drinks data
drink_cols = ['country', 'beer', 'spirit', 'wine', 'liters', 'continent']
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/drinks.csv'
drinks = pd.read_csv(url, header=0, names=drink_cols, na_filter=False)

# sort the beer column and mentally split it into 3 groups
drinks['beer'].sort_values().values

# compare with histogram

# try more bins

# add title and labels

# compare with density plot (smooth version of a histogram)

# select the beer and wine columns and sort by beer

# compare with scatter plot

# add transparency

# vary point color by spirit servings

# scatter matrix of three numerical columns

# increase figure size

# count the number of countries in each continent

# compare with bar plot

# calculate the mean alcohol amounts for each continent

# side-by-side bar plots

# drop the liters column

# stacked bar plots

# sort the spirit column

# show "five-number summary" for spirit

# compare with box plot

# include multiple variables

# read in the ufo data
url = 'https://raw.githubusercontent.com/justmarkham/DAT8/master/data/ufo.csv'
ufo = pd.read_csv(url)
ufo['Time'] = pd.to_datetime(ufo.Time)
ufo['Year'] = ufo.Time.dt.year

# count the number of ufo reports each year (and sort by year)

# compare with line plot

# don't use a line plot when there is no logical ordering

# reminder: box plot of beer servings

# box plot of beer servings grouped by continent

# box plot of all numeric columns grouped by continent

# reminder: histogram of beer servings

# histogram of beer servings grouped by continent

# share the x axes

# share the x and y axes

# change the layout

# saving a plot to a file
drinks.beer.plot(kind='hist', bins=20, title='Histogram of Beer Servings')
plt.xlabel('Beer Servings')
plt.ylabel('Frequency')
plt.savefig('beer_histogram.png')

# list available plot styles
plt.style.available

# change to a different style
plt.style.use('ggplot')

# TODO

# - find csv data of interest to your project
# - explore it using pandas with some visualisation

