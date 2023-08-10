import pandas as pd #for building dataframes from CSV files
import glob, os #for reading file names
import seaborn as sns #for fancy charts
import numpy as np #for np.nan
from scipy import stats #for statistical analysis
from scipy.stats import norm #for statistical analysis
from datetime import datetime #for time-series plots
import statsmodels #for integration with pandas and analysis
import matplotlib.pyplot as plt 
get_ipython().magic('matplotlib inline')

#Initialize dataframes given API data
my_beer_df = pd.read_csv('../data/my-final-beer-data.csv')

my_beer_df.info()

#Let's check the statistics around the quantitative factors
my_beer_df.describe()

#This looks to be of the most interest, given I am curious how different my ratings are compared to the avg user
my_beer_df[['beer.auth_rating', 'beer.rating_score']].describe()

#let's visualize my ratings vs the avg user
sns.jointplot(x="beer.auth_rating", y="beer.rating_score", data=my_beer_df, kind='reg', color='green', size=8);

#Need to create a new dataframe and column that compares my rating vs the avg user rating
beer_diff = my_beer_df.groupby(['beer.beer_style'], as_index=False)[('beer.auth_rating', 'beer.rating_score')].mean()

#Here is the new column, named 'diff'
beer_diff['diff'] = beer_diff['beer.auth_rating'] - beer_diff['beer.rating_score']

#Let's isolate the top 15 styles for visualization
top_diff = beer_diff.sort_values(by='diff', ascending=False).head(15)

#Märzen causes an ASCII issue with plotting, need to research this further
top_diff = top_diff[top_diff['beer.beer_style'] != 'Märzen']

#Factorplot showing the beer styles with ratings that differ the most between myself and the avg user rating
#This factorplot shows the styles that I tend to rate HIGHER than the average user
t = sns.factorplot(x='diff', y='beer.beer_style', data=top_diff, size=8)
t.set_axis_labels('Diff: My Rating - Avg User Rating','Beer Style')

#Let's do a similar analysis for the beers that I like less than the average user
bottom_diff = beer_diff.sort_values(by='diff', ascending=True).head(14)

#Factorplot showing the beer styles with ratings that differ the most between myself and the avg user rating
#This factorplot shows the styles that I tend to rate LOWER than the average user
b = sns.factorplot(x='diff', y='beer.beer_style', data=bottom_diff, size=8)
b.set_axis_labels('Diff: My Rating - Avg User Rating','Beer Style')

#Now, a boxplot showing ALL beer styles
filtered_my_beer_df = my_beer_df[my_beer_df['beer.beer_style'] !='Märzen']

#Massive boxplot by beer style
s = sns.FacetGrid(filtered_my_beer_df, size=12, aspect=1.0)
s.map(sns.boxplot, "beer.auth_rating", "beer.beer_style")

#Boxplot by brewery country of origin
c = sns.FacetGrid(filtered_my_beer_df, size=10, aspect=1.0)
c.map(sns.boxplot, "beer.auth_rating", "brewery.country_name")

#Manually manipulating the date for easier use and plotting visuals
new_date_list = []
for i in filtered_my_beer_df['first_created_at'] :
    if len(new_date_list) == 0 :
        new_date_list = [str(i)]
    else :
        new_date_list.append(str(i))

#Manually manipulating the date for easier use and plotting visuals
final_date = []
for date in new_date_list :
    if len(final_date) == 0 :
        final_date = [date[:-6]]
    else :
        final_date.append(date[:-6])

#create dateframe object, need specific formatting based on the results returned from the API call
filtered_my_beer_df['create_date'] = filtered_my_beer_df['first_created_at'].apply(lambda x: 
                                    datetime.strptime(x, '%a, %d %b %Y %X %z'))

#Converting dataframe date
beer_index = 0
for date in range(len(final_date)) :
    filtered_my_beer_df.iloc[beer_index, 35] = final_date[beer_index]
    beer_index+=1

#Create new column 'create_date' for analysis and visualizaitons
filtered_my_beer_df['create_date'] = pd.to_datetime(filtered_my_beer_df['create_date'], format='%a, %d %b %Y %X')

# Slight configuration of visual parameters
# Get current size
fig_size = plt.rcParams["figure.figsize"]
 
# Prints: [8.0, 6.0]
print("Current size:", fig_size)
 
# Set figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
print("New size:", fig_size)

# Use GGplot
plt.style.use('ggplot')

#Look at my ratings over time, any trends?
ax = filtered_my_beer_df.plot(x="create_date", y="beer.auth_rating")
ax.set_title('My Beer Ratings vs Time')
ax.set_xlabel('Time (Date)')
ax.set_ylabel('My Beer Ratings')
plt.show()

#Let's add a hue=brewery.country_name to see the ratings by country
gc = sns.factorplot(x='create_date', y='beer.auth_rating', hue='brewery.country_name', data=filtered_my_beer_df, 
               size=10, legend_out=True)
(gc.set_axis_labels("Beer Check-in Date", "My Beer Ratings")
   .fig.suptitle("Beer Ratings Over Time (Color by Country)"))  

#Let's add a hue=beer.beer_style to see the ratings by style
gc = sns.factorplot(x='create_date', y='beer.auth_rating', hue='beer.beer_style', data=filtered_my_beer_df, 
               size=10)
(gc.set_axis_labels("Beer Check-in Date", "My Beer Ratings")
   .fig.suptitle("Beer Ratings Over Time (Color by Beer Style)"))





