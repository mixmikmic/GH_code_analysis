# INCLUDE ALL IMPORTS HERE (REMOVE THIS COMMENT WHEN FINISHED, but leave the one right below it)

# to prettify:
get_ipython().magic('reset')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")
get_ipython().magic('matplotlib inline')

# data processing & analysis packages: 
import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm
from scipy import stats

# data visualization packages: for plotting, geospatial analyzes, and more...
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import geocoder
import folium

# also to prettify:
rcParams['figure.figsize'] = [15, 15]
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
sns.set(style='whitegrid', context ='paper')

# our five-year crime dataset --> into a dataframe: df_crime
# the boose data --> load that badboy into: df_alcohol

df_crime = pd.read_csv("incidents-5y.csv")
df_alcohol = pd.read_csv("abc_licenses_sdcounty.csv")
aggregate_df = pd.read_csv("aggregate_dataset_5y.csv")

#get a glimpse of the crime data
df_crime.head()

# our data is very high dimensional, as can be seen above
# here is an overview of the number of cities represented by the dataset 
# and the relative frequencies of crimes in each
df_crime['city'].value_counts()

# SndCHU sure looks a lot like Chula Vista, but is it? Same goes with SndSOL and Solana Beach.
# we will use a small dataset from SANDAG (the same resource used for the crime dataset) to get the city names
df_code = pd.read_csv('city_codes.csv')
df_code.head()

# some code to go from the city code to the city name
# the general logic is to get the city code for each row, find its name in the df_code dataset, and return it!

# a function defintion- here is where the magic happens
def get_city_name(row):
    city_code = row['city']
    df_city_name = df_code.loc[df_code['code'] == city_code, 'name']
    city_name = df_city_name.to_string(index=False)
    return city_name

# we create a new column in our original dataframe that contains the full city name for each row.
#df_crime['city_name'] = df_crime.apply(get_city_name, axis=1)

# N.B. this step is very time expensive (about 30 minutes), for we are adding a new entry for over 750,000 rows
# we therefore have exported df_crime into a .csv file, to avoid having to recompute the dataframe
df_crime = pd.read_csv('crime_data.csv')

# we now admire our new column in df_crime
df_crime.head()

# we now turn to fixing up the ABC licenses dataset
# our general goal is once again to determine the full city name for each ABC license location
# that way, we will have a common standard by which to compare the crime and the ABC license data

# we first preview our ABC license data
df_alcohol.head()

# how unfortunate- the "neighborhood" and "community" columns, seemingly so helpful, are almost all NONE :/
# we quickly check whether the same is true for the "city" column
print( 'Ratio of no. of NONE entries to no. of ABC licenses' )
len( df_alcohol[df_alcohol['city'] == 'NONE'] ) / len( df_alcohol )

# alas, we need to find another way.
# our solution: extract the city name from the address!

# the "city" column is essentially useless, so we drop it to make it unhelpfulness official
df_alcohol = df_alcohol.drop(['city'], 1)

# a function definition- here is where the hard work happens
# the general logic is to get the address for each row and do some pythonic string operations to return the city name
import copy
def get_city(row):
    address = row['premisesaddress']
    address_components = address.split(", ")
    string = address_components[-2].title()
    city_name = copy.deepcopy(string)
    return city_name

# we dedicate a column in the df_alcohol dataframe to storing the full city name
df_alcohol['city_name'] = df_alcohol.apply(get_city, axis=1)

# we can see our new column below
df_alcohol.head()

# we list out the cities in both dataframes to see whether they are in fact identical
print("cities in crime dataset")
list_crimecities = df_crime['city_name'].unique().tolist()
print("cities in alcohol dataset")
list_alcoholcities = df_alcohol['city_name'].unique().tolist()

print("Are the cities in both dataframes identical?")
print( "That is %s" % list_crimecities == list_alcoholcities )
print( "Number of cities in crime df: %d" % len(list_crimecities) )
print( "Number of cities in ABC licenses df: %d" % len(list_alcoholcities))

# we display the list of cities to do a visual comparison
print("cities in crime dataset")
print(list_crimecities) 
print("cities in alcohol dataset")
print(list_alcoholcities)

# we remove S.D. County from dataframe as is not represented in the ABC licenses data
df_crime = df_crime[df_crime['city_name'] != 'S.D. County']

# we confirm the list of cities is the same now
list_crimecities = df_crime['city_name'].unique().tolist()
list_alcoholcities = df_alcohol['city_name'].unique().tolist()
print("Are the cities in both dataframes identical?")
print( "That is %s" % list_crimecities == list_alcoholcities )

# now we are ready to merge our data for both crime and alcohol together!

# we determine the number of crimes and ABC licneses that occur for each city
crime_counts = df_crime['city_name'].value_counts()
alcohol_counts = df_alcohol['city_name'].value_counts()

# function definition- here is where we get the total number of crimes for each city
def get_total_crime(row):
    city_name = row['city_name']
    return crime_counts[city_name]

# function definition- here is where we get the total number of ABC licenses for each city
def get_total_alcohol(row):
    city_name = row['city_name']
    return alcohol_counts[city_name]

# we establish our first column in the aggregate dataframe
# the city names will be unique so there will be 18 rows total
df = df_alcohol.loc[:, ['city_name']].drop_duplicates()

# we determine the number of crimes and ABC licenses for each city
df.loc[:,'total_crime'] = df.apply(get_total_crime, axis=1)
df.loc[:, 'total_alcohol'] = df.apply(get_total_alcohol, axis=1)

# we can see our new dataframe coming together!
df

# now we refine our data further!
# we determine the population for each city to normalize our alcohol and crime counts

# because of the lack of a clean dataset and the relatively small number of cities, we decided to hard code 
# the values by hand. long-term that is not the best practice, but for the short-term it worked for us
# source: U.S. Census Bureau 2010- http://www.togetherweteach.com/TWTIC/uscityinfo/05ca/capopr/05capr.html
df['city_pop'] = [1307402, 58582, 4161, 143191, 59518, 26324, 167086, 25320, 
                 105328, 243916, 99478, 93834, 57065, 53413, 47811, 18912,
                12867, 83781]

# now we fine-tune our crime counts even more, to differentiate them by type along with sheer number!

# we collect relevant data for the types of crimes and the city names from the original dataset
df_crime_type = df_crime[['city_name', 'type']]

# here is a preview of the types of crimes and their frequency 
# counts are not segregated by city yet. that will be our next job
df_stat = df_crime_type['type'].value_counts()
df_stat

# first, we convert 'type' column in df_crime to lower case
# there is no real reason, just for aesthetic purposes
df_crime['type'] = df_crime['type'].str.lower()

# next, we get a list of the unique crime type names. this will be useful later on.
list_crimetypes = df_crime['type'].drop_duplicates().tolist()

# function definition that returns the number of crimes with a given crimetype that occur for a given city
def get_crimetype_counts(row, crimetype):
    df_crimetype_counts = df_crime.loc[(df_crime['city_name'] == row['city_name']) & (df_crime['type'] == crimetype)]
    return len(df_crimetype_counts)

# we iterate over all the different crime types
for crimetype in list_crimetypes:
    # we determine the total number of crimes with a given crime type that occurs per city 
    # we save the results in a column called crimetype
    df.loc[:, crimetype] = df.apply(get_crimetype_counts, args=(crimetype,), axis=1)

# the data cleaning steps are complete, and we are now ready to admire our finished dataframe!
# for fun, just to make it official, we renname our dataframe as aggregate_df
aggregate_df = df
aggregate_df

# an enormous scatter matrix of crimerates by type

fig = pd.scatter_matrix(aggregate_df, figsize=(20, 16))

# let's normalize that crime and alcohol data! 

aggregate_df['norm_crime'] = aggregate_df['total_crime'] / aggregate_df['city_pop']
aggregate_df['norm_alcohol'] = aggregate_df['total_alcohol'] / aggregate_df['city_pop']

# a barplot of the normalized (by population) number of ABC licenses by city

axes = aggregate_df[['city_name', 'norm_alcohol']].plot.bar(figsize=(15,8), color="green")
axes.set_xticklabels(labels=aggregate_df.city_name, rotation=55)
axes.set(xlabel="City", ylabel="Alcohol License Counts", title="Normalized ABC Licenses by City")

# a barplot of the per capita crime rate against city population by city

axes = aggregate_df[['city_name', 'norm_crime']].plot.bar(figsize=(15,8))
axes.set_xticklabels(labels=aggregate_df.city_name, rotation=55)
axes.set(xlabel="City", ylabel="Crime Rate", title="Normalized Crime Rates by City")

# an enormous collection of pairplots: population to number of crime incidents

dims = (30,30)
pairplots = sns.pairplot(aggregate_df)
pairplots.set(xticklabels=[])

# yet another exciting barplot! normalized crime rates & ABC permits by city

axes = aggregate_df[['city_name', 'norm_alcohol', 'norm_crime']].plot.bar(figsize=(20,8))
axes.set_xticklabels(labels=aggregate_df.city_name, rotation=55)
axes.set(xlabel="City", ylabel="Crime and Alcohol License Counts", title="Normalized Crime Rates and ABC Licenses by City")

# let's see how the number of alcohol permits affect the rate of crime in a city! correlation table (normalized data)

x = aggregate_df['norm_alcohol']
y = aggregate_df['norm_crime']
aggregate_df.corr()

# check out the normalized alcohol license counts & normalized crime rates - each point represents a single city

plt.figure(figsize=(12,8.4))
plt.plot(x, y, "o")

# compute correlation and linear regression model WITH DATA AS GIVEN (includes an outlier - can you tell which one?)

# checking the distribution/normality of the residuals
model = sm.OLS(x, y)
fit_to_get_residuals = model.fit()

# histogram of the normalized residuals: how's our distribution?
plt.hist(fit_to_get_residuals.resid)
plt.ylabel('Normalized ABC counts')
plt.xlabel('Normalized residuals')
plt.title('Histogram for Residuals')

# we would like to grow up to be good [statisticians/researchers/data scientists/less clueless folks]:
# check out the plot of our residuals in a different form

sns.residplot(x, y)

# fit the linear regression for crime to community population (still got a peculiar data point in the mix)

slope, intercept, corr_value, p_value, std_err = stats.linregress(x,y)

predicted_y = intercept + slope * x
predicted_error = y - predicted_y
degr_freedom = len(x) - 2
residual_std_error = np.sqrt(np.sum(predicted_error ** 2) / degr_freedom)

# display the slope, intercept, correlation coefficient, p value, and standard error
display = "slope: " + repr(slope) + " | intercept: " + repr(intercept) + " | correlation coefficient: " + repr(corr_value)
display2 = " | p_value: " + repr(p_value) + " | standard error: " + repr(std_err)
print(display, display2)

# plotting linear regression/least squares
plt.figure(figsize=(12,8.4))
plt.plot(x, y, 'o')
plt.plot(x, predicted_y, 'k-')
plt.show()

# now, let's repeat the process AFTER removing the outlier - pesky Del Mar...

sans_outlier = aggregate_df[aggregate_df['city_name'] != 'Del Mar']

# plot the normalized alcohol license counts & normalized crime rates
x2 = sans_outlier['norm_alcohol']
y2 = sans_outlier['norm_crime']

# the correlation matrix
sans_outlier.corr()

# check out the new plot of normalized alcohol license counts & normalized crime rates (points are cities)

plt.figure(figsize=(12,8.4))
plt.plot(x2, y2, "o")

# hmmm, not as pretty as the last one! let's fit the linear regression without outlier anyway

slope2, intercept2, corr_value2, p_value2, std_err2 = stats.linregress(x2,y2)

predicted_y2 = intercept2 + slope2 * x2
predicted_error2 = y2 - predicted_y2
degr_freedom2 = len(x2) - 2
residual_std_error2 = np.sqrt(np.sum(predicted_error2 ** 2) / degr_freedom2)

display3 = "slope: " + repr(slope2) + " | intercept: " + repr(intercept2) + " | correlation coefficient: " + repr(corr_value2)
display4 = " | p_value: " + repr(p_value2) + " | standard error: " + repr(std_err2)
print(display3, display4)

# plotting linear regression/least squares 
plt.figure(figsize=(12,8.4))
plt.plot(x2, y2, 'o')
plt.plot(x2, predicted_y2, 'k-')
plt.show()

# computing Pearson correlation and linear regression model WITHOUT OUTLIER DATA POINT

# checking the distribution/normality of the residuals
model2 = sm.OLS(x2, y2)
fit_to_get_residuals2 = model2.fit()

# histogram of the normalized residuals: how's our distribution?
plt.hist(fit_to_get_residuals2.resid)
plt.ylabel('Normalized ABC counts')
plt.xlabel('Normalized residuals')
plt.title('Histogram for Residuals')

# correlation matrix (Pearson) of crime to population by community

plt.subplots(figsize=(10,10))
crime_corr = sans_outlier.corr()
crime_heatmap = sns.heatmap(crime_corr, 
            xticklabels=crime_corr.columns.values,
            yticklabels=crime_corr.columns.values)
crime_heatmap.set(title='Heat Map Correlation Matrix')

# let's look at violent crimes: assault, robbery, weapons, sex crimes, arson, homicide

violent_df = sans_outlier.loc[:, ['city_name', 'city_pop', 'norm_alcohol', 'assault', 'robbery', 'weapons', 'sex crimes', 'arson', 'homicide']]
# print(violent_df)

# normalize all crime and multiply by a factor of 100 to maintain precision 

col_names = list()
col_names = ('assault', 'robbery', 'weapons', 'sex crimes', 'arson', 'homicide')
for name in col_names:
    violent_df.loc[:, name] = (violent_df[name] / violent_df['city_pop']) * 100
# print(violent_df)

# pretty pairplots for violent crimes

dims = (20,20)
pairplots = sns.pairplot(violent_df)
pairplots.set(xticklabels=[])

# violent crimes analysis on normalized data - Spearman correlation method

# plot the normalized alcohol license counts & normalized crime rates
x3 = violent_df['norm_alcohol']
y3 = violent_df['assault']

violent_df.corr(method="spearman", min_periods=5)

# plot of the cities pre-linear regression computation

plt.figure(figsize=(12,8.4))
plt.plot(x3, y3, "o")

# computing correlation and linear regression model WITHOUT OUTLIER DATA POINT (still Del Mar!)

# checking the distribution/normality of the residuals
model3 = sm.OLS(x3, y3)
fit_to_get_residuals3 = model3.fit()

# histogram of the normalized residuals: how's our distribution?

plt.subplots(figsize=(10,10))
plt.ylabel('Normalized ABC counts')
plt.xlabel('Normalized residuals for Assault Crimes')
plt.title('Histogram of Residuals between Assault Crimes and ABC licenses')
plt.hist(fit_to_get_residuals3.resid)

# fit the linear regression without 

slope3, intercept3, corr_value3, p_value3, std_err3 = stats.linregress(x3,y3)

predicted_y3 = intercept3 + slope3 * x3
predicted_error3 = y3 - predicted_y3
degr_freedom3 = len(x3) - 2
residual_std_error3 = np.sqrt(np.sum(predicted_error3 ** 2) / degr_freedom3)

# display the slope, intercept, correlation coefficient, p value, and standard error
display5 = "slope: " + repr(slope3) + " | intercept: " + repr(intercept3) + " | correlation coefficient: " + repr(corr_value3)
display6 = " | p_value: " + repr(p_value3) + " | standard error: " + repr(std_err3)
print(display5, display6)

# plotting linear regression/least squares 
plt.figure(figsize=(12,8.4))
plt.xlabel("Normalized Rate of Assault Crimes")
plt.ylabel("Normalized Alcohol Licenses")
plt.title("Linear Regression of Assault Crimes and Alcohol Licenses")
plt.plot(x3, y3, 'o')
plt.plot(x3, predicted_y3, 'k-')
plt.show()



df = pd.read_csv("incidents-100k.csv")

# Let's create a list of lat and lon points from the aggregated data frame
j = 0
points = []
for i in range (len(aggregate_df)):
    latitude = df['lat'][j]
    longitude = df['lon'][j]
    points.append(tuple([latitude,longitude]))
    j = j+1

# Now, let's create a list of lat and lon points for the abc licenses
# As well only keeping points that are inside of the San Diego area
points2 = []
j = 0
df_alcohol = df_alcohol.loc[(df_alcohol["premisesaddress"].str.contains(', SAN DIEGO, CA'))] 

for j, row in df_alcohol.iterrows():
    latitude = row['lat']
    longitude = row['lon']
    points2.append(tuple([latitude,longitude]))

# Getting certain communities  with a similar population
df_Navajo = df.loc[df['community']=='SanNAV']

df_Encanto = df.loc[df['community']=='SanENC']

df_Encanto['comm_pop'].value_counts()

# Getting the Navajo population
San_NAV_Popu = df_Navajo['comm_pop'].value_counts()


zip_dict = {}
zip_length = len(San_NAV_Popu)
i=0
count = 0
total_population = 0
diff_populations = 0
for i in df_Navajo.index:
    get_NAV_popu=df_Navajo.ix[i, 'comm_pop'] 
    if get_NAV_popu in zip_dict:
        count = (zip_dict.get(get_NAV_popu))+1
        zip_dict.update({get_NAV_popu:count})
    else:
        count = 0
        zip_dict.update({get_NAV_popu:1})
        
for x,y in zip_dict.items():
    total_population = total_population+x
    if x!=0:
        diff_populations +=1
    

final_population = (total_population // diff_populations)

Navajo_population = final_population

# Getting the Encanto population
Encanto_Popu = df_Encanto['comm_pop'].value_counts()


zip_dict2 = {}
zip_length = len(Encanto_Popu)
i=0
count = 0
total_population = 0
diff_populations = 0
for i in df_Encanto.index:
    get_Encanto_popu=df_Encanto.ix[i, 'comm_pop'] 
    if get_Encanto_popu in zip_dict2:
        count = (zip_dict2.get(get_Encanto_popu))+1
        zip_dict2.update({get_Encanto_popu:count})
    else:
        count = 0
        zip_dict2.update({get_Encanto_popu:1})
        
for x,y in zip_dict2.items():
    total_population = total_population+x
    if x!=0:
        diff_populations +=1
    

final_population = (total_population // diff_populations) 

Encanto_population = final_population

# Looking at different crimes in two communities of similar population.
print ('Encanto population:', Encanto_population)
print  ('Navajo population:', Navajo_population)

# Now we can look at two communities of similar population
# And how alcohol lead to different crimes in these two areas, and get a better understanding of why that could be
diff_crime_Encanto = df_Encanto['type'].value_counts()
diff_crime_Encanto.plot.bar()
plt.ylabel('counts')
plt.xlabel('types of crime')
plt.title('Encanto')
plt.show()

diff_crime_Navajo = df_Navajo['type'].value_counts()
diff_crime_Navajo.plot.bar()
plt.ylabel('counts')
plt.xlabel('types of crime')
plt.title('Navajo')
plt.show()

# read in the dbf (metadata) file and list all the methods associated with it
import shapefile
sf = shapefile.Reader("ZillowNeighborhoods-CA.dbf")

# Get the metadata for every entry in the dbf file
metadata = sf.shapeRecords()

# This outputs the information that the 38th entry holds 
# (Just to get an understanding of how the file works)
#metadata[38].record

# Find indices of all San Diego neighborhoods
# And insert append them to sd_list
sd_list = []
counter = 0

for i in range(len(metadata)):
    if metadata[i].record[2] == 'San Diego':
        sd_list.append(i)
        counter += 1

# Create a list and append all shape points to it, this holds the outline/ boundary of California
shapes = sf.shapes()
sd_shapes = []

for i in range(len(sd_list)):
    sd_shapes.append(shapes[sd_list[i]].points)

for i in range(len(sd_shapes)):
    for j in range(len(sd_shapes[i])):
        sd_shapes[i][j] = sd_shapes[i][j][::-1]

read_shapemeta=shapefile.Reader('ZillowNeighborhoods-CA.dbf')
shapemeta = read_shapemeta.shapeRecords()


sorted_SD_list = []


# metadata[i].record[2] holds the city name, thus we filter through the whole California shape file for San Diego cities only
for i in range(len(shapemeta)):
     if metadata[i].record[2] == 'San Diego':
        sorted_SD_list.append(shapemeta[i].record[3])
sorted_SD_list = sorted(sorted_SD_list)   

# determine the population rate per city and save in a new dataframe df_stats
df_stats = df[['community','comm_pop']].drop_duplicates()

# add a column 'crime' to df_stats that represents the raw count of crimes for that community 
# (raw counts are determined by the number of rows containing the community )
df_stats['crime'] = df.groupby('community')['community'].transform('count')

# add a column 'crimerate' to df_stats to normalize for the population
df_stats['crimerate'] = df_stats['crime'] / df_stats['comm_pop']

#use data from df_stats to get a dictionary where the keys are the community names and the values are the crime rates
dict_crimerate = df_stats[['community', 'crimerate']].set_index('community')['crimerate'].to_dict()

# remove outlier: community with lower population than crime
df_stats = df_stats[df_stats['comm_pop'] > df_stats['crime']]

# Get max and min ratio of the 
max_ratio = (max(df_stats['crimerate']))
min_ratio = (min(df_stats['crimerate']))

# We will use this as incremental values
one_fifth = (max_ratio/5)

sorted_df_stats = sorted(df_stats['crimerate'])

# lat and lon, along with zoom will help us close in on San Diego on the map
j = 0
lat = 32.7157
lon = -117.1611
zoom_start = 11

# This initializes our map start location, zoom, and type of map
m = folium.Map(location=[lat, lon], zoom_start=zoom_start,tiles='Stamen Toner')

# We will keep plotting points based on the smaller dataset
if (len(points)<(len(points2))):
    length=(len(points))
else:
    length=(len(points2))
for i in range(length):
    coord_2 = points2[j]
    
    # Plot points of the abc licenses
    kw2 = dict(fill_color='black' ,radius=4)
    c1 = folium.CircleMarker(coord_2, **kw2)
    for c in [c1]:
        m.add_child(c)
    j=j+1

    

# Using the shape file as a boundary for San Diego
# We broke up the ratio into five portions, and if it fell into a certain range
# it would be a certain color, and the higher the ratio, the darker the color
# We multiplied the ratios by 100, to help with incrementation, and increased by the previous (1/5) incremental value
j=0
df_stats_length = len(df_stats)
for c in range(len(sd_shapes)):  
    
    
    if c <=(sorted_df_stats[10]*100):
        colour = '#f0f9e8'        #Transparent color
    if c >(sorted_df_stats[10]*100) and c<=(sorted_df_stats[20]*100):
        colour = '#bae4bc'         # green color
    if c >(sorted_df_stats[20]*100) and c<=(sorted_df_stats[30]*100):
        colour = '#7bccc4'          # Cyan color
    if c >(sorted_df_stats[30]*100) and c<=(sorted_df_stats[40]*100):
        colour = '#43a2ca'           # Blue color
    if c >(sorted_df_stats[40]*100) and c<=(sorted_df_stats[df_stats_length-1]*100):
        colour = '#0868ac'           # Dark blue
    if c>(df_stats_length) and c<=(len(sd_shapes)):
        colour = 'white'

    j=j+1
    # hood_line will draw the boundary based on the shape file of San Diego
    hood_line = folium.features.PolygonMarker(locations=sd_shapes[c], color='pink', fill_color=colour, weight=3)
    m.add_child(hood_line)

# A key for our map
import matplotlib.patches as mpatches

fig = plt.figure(figsize=(10,10))
colour_1 = mpatches.Patch(color = 'white', label = 'Out of ratio')
colour_2 = mpatches.Patch(color='#f0f9e8', label='0.028075 - 0.106897')
colour_3 = mpatches.Patch(color='#bae4bc', label='0.033225 - 0.306056')
colour_4 = mpatches.Patch(color='#7bccc4', label='0.063470 - 0.085520')
colour_5 = mpatches.Patch(color='#43a2ca', label='0.018914 - 0.029579')
colour_6 = mpatches.Patch(color='#0868ac', label='0.022333 - 0.021948 ')
plt.title("Key for choropleth map")
plt.legend(handles=[colour_1,colour_2,colour_3,colour_4,colour_5])

plt.show()



# 100k random incidents? put'er in: df

df = pd.read_csv("incidents-100k.csv")

# first, clean that data: remove any rows that do not have a community specified, or community population

# print(df.shape)
df = df[df['community'] != 'NONE']
# print(df.shape)
df = df[df['comm_pop'] != 0]
# print(df.shape)

# get the community populations and rates of crimes (normalized by pop)

counts_by_community = df['community'].value_counts()
counts_by_pops = df['comm_pop'].value_counts()

comm_dict = counts_by_community.to_dict()
# print(comm_dict)
pops_dict = counts_by_pops.to_dict()
# print(pops_dict)

# a lovely plot o' crime - by community 

dims = (8, 6)
fig, ax = plt.subplots(figsize=dims)
axes = sns.barplot(ax=ax, x=counts_by_community, y=counts_by_community.index)
axes.set(xlabel="Number of Crimes", ylabel="Community", title="Crime Rates by Community")
sns.plt.show()

# bar plot of the number of crimes by population counts

dims = (8, 6)
fig, ax = plt.subplots(figsize=dims)
bargraph = sns.barplot(x=counts_by_pops.index, y=counts_by_pops)
bargraph.set_xticklabels(labels=counts_by_pops.index, rotation=55)
bargraph.set(xlabel="Population", ylabel="Number of Crimes", title="The Number of Crimes by Community Population Size")
sns.plt.show()

# plot of incident counts by community population (histogram)

counts_by_community.plot(kind="hist")
f2 = plt.show()
f2 = plt.gcf()

# determine the population rate per city and save in a new dataframe df_stats
df_stats = df[['community','comm_pop']].drop_duplicates()

# add a column 'crime' to df_stats that represents the raw count of crimes for that community 
df_stats['crime'] = df.groupby('community')['community'].transform('count')

# add a column 'crimerate' to df_stats to normalize for the population
df_stats['crimerate'] = df_stats['crime'] / df_stats['comm_pop']

#use data from df_stats to get a dictionary where the keys are the community names and the values are the crime rates
dict_crimerate = df_stats[['community', 'crimerate']].set_index('community')['crimerate'].to_dict()

# remove the outlier: community with lower population than crime
df_stats = df_stats[df_stats['comm_pop'] > df_stats['crime']]
# print(df_stats)

# a scatter matrix of the normalized crime rates (by population)

fig = pd.scatter_matrix(df_stats)

# plot the total crime and population counts per community

axes = df_stats[['community', 'comm_pop', 'crime']].plot.bar(figsize=(9,6))
axes.set_xticklabels(labels=df_stats.community, rotation=55)
axes.set(xlabel="Community", ylabel="Crime and Population Counts", title="Crime Rates Compared to Populations by Community")

# normalized crime rates by city

axes = df_stats[['community', 'crimerate']].plot.bar(figsize=(9,6))
axes.set_xticklabels(labels=df_stats.community, rotation=55)
axes.set(xlabel="Community", ylabel="Crime Rate", title="Normalized Crime Rates by Community")

# pairplots of population size to number of crime incidents

pairplots = sns.pairplot(df_stats)
pairplots.set(xticklabels=[])

# does population increase the rate of crime in a community? let's find out

x = df_stats['comm_pop']
y = df_stats['crime']
plt.plot(x, y, "o")
df_stats.corr()

# checking the distribution/normality of the residuals
model = sm.OLS(x, y)
fit_to_get_residuals = model.fit()

# histogram of the normalized residuals: how's our distribution?
plt.hist(fit_to_get_residuals.resid)
plt.ylabel('Population Count')
plt.xlabel('Normalized residuals')
plt.title('Histogram for Residuals')

# well, looks like it does! but not by as much as one might expect...
# let's check out a heatmap - correlation matrix - of crime to population by community

crime_corr = df_stats.corr()
crime_heatmap = sns.heatmap(crime_corr, 
            xticklabels=crime_corr.columns.values,
            yticklabels=crime_corr.columns.values)
crime_heatmap.set(title='Heat Map Correlation Matrix of Crime & Population by Community')



