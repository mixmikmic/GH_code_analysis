get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Following is optional: set plotting styles
import seaborn as sns

get_ipython().system("curl -O 'https://www.ssa.gov/oact/babynames/names.zip'")

get_ipython().system('mkdir -p data/names')
get_ipython().system('mv names.zip data/names/')
get_ipython().system('cd data/names/ && unzip names.zip')

get_ipython().system('ls data/names')

get_ipython().system('head data/names/yob1880.txt')

names1880 = pd.read_csv('data/names/yob1880.txt',
                        names=['name', 'gender', 'births'])
names1880.head()

# Pedestrian counting of number of male and female births
males = names1880[names1880.gender == 'M']
females = names1880[names1880.gender == 'F']
males.births.sum(), females.births.sum()

# Pandas provides a better way
grouped = names1880.groupby('gender')
grouped

grouped.sum()

grouped.size()

grouped.mean()

grouped.describe()

def load_year(year):
    data = pd.read_csv('data/names/yob{0}.txt'.format(year),
                       names=['name', 'gender', 'births'])
    data['year'] = year
    return data

names = pd.concat([load_year(year) for year in range(1880, 2014)])
names.head()

births = names.groupby('year').births.sum()
births.head()

births.plot();

names.groupby('year').births.count().plot();

def add_frequency(group):
    group['birth_freq'] = group.births / group.births.sum()
    return group

names = names.groupby(['year', 'gender']).apply(add_frequency)
names.head()

men = names[names.gender == 'M']
women = names[names.gender == 'W']

births = names.pivot_table('births',
                           index='year', columns='gender',
                           aggfunc=sum)
births.head()

births.plot(title='Total Births');

names_to_check = ['Allison', 'Alison']

# filter on just the names we're interested in
births = names[names.name.isin(names_to_check)]

# pivot table to get year vs. gender
births = births.pivot_table('births', index='year', columns='gender')

# fill all NaNs with zeros
births = births.fillna(0)

# normalize along columns
births = births.div(births.sum(1), axis=0)

births.plot(title='Fraction of babies named Allison');

births.rolling(5).mean().plot(title="Allisons: 5-year moving average");

