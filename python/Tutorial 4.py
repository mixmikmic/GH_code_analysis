import numpy as np
import pandas as pd
import matplotlib.pyplot as ply
get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('pokemon.csv')

# data frames from dictionary
country = ["Spain","France"]
population = ["11","12"]
list_label = ["country","population"]
list_col = [country,population]
zipped = list(zip(list_label,list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df

# Add new columns
df["capital"] = ["madrid","paris"]
df

# Broadcasting
df["income"] = 0 #Broadcasting entire column
df

# Plotting all data 
data1 = data.loc[:,["Attack","Defense","Speed"]]
data1.plot()

# subplots
data1.plot(subplots = True)

# scatter plot  
data1.plot(kind = "scatter",x="Attack",y = "Defense")

# histogram plot  
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True)

# histogram subplot with non cumulative and cumulative
fig, axes = ply.subplots(nrows=1,ncols=2)
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[0])
data1.plot(kind = "hist",y = "Defense",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

data.describe()

time_list = ["1992-03-08","1992-04-12"]
print type(time_list[1]) # As you can see date is string
# however we want it to be datetime object
datetime_object = pd.to_datetime(time_list)
print type(datetime_object)

# In order to practice lets take head of pokemon data and add it a time list
data2 = data.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
# lets make date as index
data2= data2.set_index("date")
data2 

# Now we can select according to our date index
print data2.loc["1993-03-16"]
print data2.loc["1992-03-10":"1993-03-16"]

# We will use data2 that we create at previous part
data2.resample("A").mean()

# Lets resample with month
data2.resample("M").mean()
# As you can see there are a lot of nan because data2 does not include all months

# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate
# We can interpolete from first value
data2.resample("M").first().interpolate("linear")

# read data
data = pd.read_csv('pokemon.csv')
data= data.set_index("#")
data.head()

# indexing using square brackets
data["HP"][1]

# using column attribute and row label
data.HP[1]

# using loc accessor
data.loc[1,["HP"]]

# Selecting only some columns
data[["HP","Attack"]]

# Difference between selecting columns: series and dataframes
print type(data["HP"])    # series
print type(data[["HP"]])   # data frames

# Slicing and indexing series
data.loc[10:20,"HP":"Defense"]   # 10 and "Defense" are inclusive

# Reverse slicing 
data.loc[10:1:-1,"HP":"Defense"] 

# Creating boolean series
boolean = data.HP > 200
data[boolean]

# Combining filters
first_filter = data.HP > 150
second_filter = data.Speed > 35
data[first_filter & second_filter]

data.HP[data.Speed<15] # Filtering column based others

# Plain python functions
def div(n):
    return n/2
data.HP.apply(div)

#We can use lambda function for the same
data.HP.apply(lambda n : n/2)

# Defining column using other columns
data["total_power"] = data.Attack + data.Defense
data.head()



