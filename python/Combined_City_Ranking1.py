import os
import pandas as pd

# filenames
housing_rank_file = os.path.join("Results","house_rent_ranking.csv")
school_rank_file = os.path.join("Results","school_ranking.csv")

housing_rank = pd.read_csv(housing_rank_file)
school_rank = pd.read_csv(school_rank_file)

housing_rank

school_rank

# INSERT HERE other ranking files 

# combine ranking into one file
combined_rank = housing_rank.merge(school_rank, on='City')

#INSERT HERE
# repeat merging (above cell) with additional csv that have been read
# sample combined_rank = combined_rank.merge(other_rank, on='City')

c = combined_rank.set_index("City")
c

## Write to CSV fike the combined rank

combined_rank_file = os.path.join("Results","house_afford_school_ranking.csv")
c.to_csv(combined_rank_file)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# read input from CSV
c = pd.read_csv(combined_rank_file)
c

# make the city the index
c = c.set_index('City')

# ensure valus are integers and not strings
c = c.apply(pd.to_numeric)

c

# prefix all column names with same string
for col in c.columns: 
    c=c.rename(columns={col:'rating_'+col})
c    

# prepare for visual
c = c.reset_index()
c = c.melt('City', var_name='rating', value_name='vals')

g = sns.factorplot(kind='bar',x="vals", y='City',hue='rating', data=c, size=10)

combined_ranking_visual = os.path.join("Results","combined_ranking_visual.png")
plt.savefig(combined_ranking_visual)
plt.show()

# Read combined ranking dataframe
c = combined_rank.set_index("City")

# ensure all values are integers, note the City is intentionally the index so that it does not 
# attempt to convert a city name into integer
c= c.apply(pd.to_numeric)

# get names for all the rankings
columns = c.columns
columns

c

# Compute totals for each city

c['total']=0
for col in columns:
    c['total']=c['total']+c[col]
c    

# Drop all columns, only need the totals
total = c[['total']]
total = total.sort_values('total', ascending=False)
total = total.reset_index()

total

# plot the results
plt.figure(figsize=(20,3))
sns.barplot(x='City',y='total',data=total)
plt.show()



