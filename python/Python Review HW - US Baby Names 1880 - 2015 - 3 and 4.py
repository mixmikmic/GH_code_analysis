get_ipython().magic('matplotlib inline')

from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Look pretty!
matplotlib.style.use('ggplot')

# The data is stored as CSV's in individual .txt files, one for each year from 1880-2015.
# Each file contains a list of names, with the sex and number of births.

years = range(1880, 2016)

a = []

for year in years:
    path = 'names/yob%d.txt' %year
    df = pd.read_csv(path, names = ['names', 'sex', 'births'])
    df['year'] = year
    a.append(df)

# a is now a list of DataFrames which need to be merged into a single giant DataFrame.
names_data = pd.concat(a, ignore_index = True)
#print names_data

print "Most popular female name in 1986:"
print names_data[(names_data.year == 1986) & (names_data.sex == 'F')].sort_values('births', ascending = False)[:1]
print "---------------------------------"
print "Most popular male name in 1986:"
print names_data[(names_data.year == 1986) & (names_data.sex == 'M')].sort_values('births', ascending = False)[:1]

print "Number of people born with name 'Arjun' in 1986:"
print names_data[(names_data.year == 1986) & (names_data.names == 'Arjun')]

# A plot of the number of instances of your name over time.
arjun_names = names_data[(names_data.names == 'Arjun')]
arjun_names.plot(x = 'year', y = 'births')
plt.ylabel("Number of births")
plt.title("Arjuns!")

# A plot of the number of the total boy names and the number of girls names each year.
# In other words, the number of female and male births per year.

total_fm_births = names_data.pivot_table('births', index = 'year', columns = 'sex', aggfunc = 'sum')
plt.plot(total_fm_births.index, total_fm_births.F, label = 'Female')
plt.plot(total_fm_births.index, total_fm_births.M, label = 'Male')
plt.title("Male and Female births per year")
plt.legend(loc = 'lower right')
plt.xlabel("year")
plt.ylabel("Number of births")
plt.show()

# A plot showing the fraction of male and female babies given a name similar to Lesley. By similar I mean the
# name starts with ‘lesl’ (make sure you make the name lowercase).

unique_names = names_data.names.unique()
lesl_mask = np.array([x.lower().startswith('lesl') for x in unique_names])
like_lesl = unique_names[lesl_mask]
print like_lesl # List of male and female names starting with "Lesl" (not just containing "Lesl"!!)

filtered_lesl = names_data[names_data.names.isin(like_lesl)]
lesl_table = filtered_lesl.pivot_table('births', index = 'year', columns = 'sex', aggfunc = 'sum')
lesl_table = lesl_table.div(lesl_table.sum(1), axis=0)
#print lesl_table.head(8)
lesl_table.plot(style={'M': 'k-', 'F': 'k--'})
plt.title("Proportion of names beginning with 'Lesl' over time")



