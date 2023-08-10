# We're importing the Plotly libraries for the first time
import pandas as pd

import plotly.offline as offline
import plotly.graph_objs as go

# Run this in offline mode
offline.init_notebook_mode()

# Make sure plotly is working
offline.iplot({'data': [{'y': [4, 2, 3, 4]}], 
               'layout': {'title': 'Test Plot', 
                          'font': dict(size=16)}})

# Run the transformations from the the first (01) file to make sure everyone is 
# at the same place.
url = "https://data.delaware.gov/api/views/bxyv-7mgn/rows.csv?accessType=DOWNLOAD"
waterfowl_df = pd.read_csv(url)
waterfowl_df_january = waterfowl_df[waterfowl_df['Month']=='January']
waterfowl_df_january_sub = waterfowl_df_january[waterfowl_df_january['Time Period']!='Late']

# Note: You can also use dot notation. For intance, this would have worked:
#waterfowl_df_january = waterfowl_df[waterfowl_df.Month=='January']
# BUT, dot notation does not work for 'Time Period' because of the space in the column name!

# Once again, look at the first few rows of data.
waterfowl_df_january_sub.head()

# Look at the last few rows:
waterfowl_df_january_sub.tail()

# Take a look at the index (far left) column. 
# Note how it maintains the same index numbers as the full data set!

# Pandas has a handy describe() function
# count tells the number of values that column has (some columns can be NaN (Not a Number))
# Look at the mean, median (50%) and max
waterfowl_df_january_sub.describe()

# Check the sums again. Remember, this will be for january of each year
waterfowl_df_january_sub.groupby('Year').sum().head()

# Let's look at just 1979
waterfowl_df_january_sub[waterfowl_df_january_sub.Year==1979].groupby('Year').sum()

# Compare the above output to this one.
waterfowl_df_january_sub.groupby('Year').sum()[4:5]

# ***** This cell requires you to fill something in! *****

# Why do you think they are the same? (Hint: Look at the table that sums all years.)

# The answer is that [4:5] is an example of slicing. This means start at 4 (where the first element is 0!) and 
# end before you get to 5. 

# Which do you think is eaiser to read?

# Copy and paste your favorite of the two and assign to the variable below
waterfowl_1979 = waterfowl_df_january_sub.groupby('Year').sum()[4:5]

# Print the variable
waterfowl_1979

# We need just the bird column names. First, print all
waterfowl_1979.columns

# Why isn't 'Year' a column?
# Remember the groupBy we used?

# Check the dataframe's index:
waterfowl_1979.index

# There's the year!

# To get bird names we only need to skip the first column (remember 0 is first!)
# Let's slice!
# Note: If the second number is not given, it will continue to the end.
birds = waterfowl_1979.columns[1:]

# Explore the first bird in the list
bird = birds[0]
print('bird:', bird)
print('full:', waterfowl_1979[bird])
print('values (list):', waterfowl_1979[bird].values)
print('first value in list:', waterfowl_1979[bird].values[0])
print('Set as an integer:', int(waterfowl_1979[bird]))

# There are multiple ways to get the number, I will be using: int(waterfowl_1979[bird])

# The line that prints 'full' above is outputted on multipole lines below. This next line might be
# easier for you to see the full record. Note the semicolon -- That means end the prevous instruction
# and start a new instruciton. They are rare in Python, but permitted!
#print('full:'); print(waterfowl_1979[birds[0]])

# Get the bird counts into a list
# Use a comprehension!
bird_counts = [int(waterfowl_1979[bird]) for bird in birds]



"""
# Long way:
bird_counts = []
for bird in birds:
    bird_counts.append(int(waterfowl_1979[bird]))
"""

bird_counts

# Uh oh, no need to chart the birds that weren't counted

# Only include birds that were counted!
birds = [bird for bird in waterfowl_1979.columns[1:] if int(waterfowl_1979[bird]) > 0]

bird_counts = [int(waterfowl_1979[bird]) for bird in birds]
bird_counts

# The zip() function can be handy. It combines two lists of equal length into a list of tuples.
z = zip(birds, bird_counts)

for i in z:
    print(i)

# Let's use Plotly to make a bar chart!

data = [go.Bar(x=birds,
            y=bird_counts)]

offline.iplot(data)

# ***** This cell requires you to fill something in! *****

# Still too many birds, and it would look better ordered.
# First, return to our completion, edit below to get birds with at least 1000 views

birds = [bird for bird in waterfowl_1979.columns[1:] if int(waterfowl_1979[bird]) > 1000]

bird_counts = [int(waterfowl_1979[bird]) for bird in birds]
bird_counts

# waterfowl_1979.columns[1:]
#pd.__version__

# Use zip to make a list of tuples
bird_tuples = [tuple(i) for i in zip(birds, bird_counts)]
bird_tuples

# Now sort!
# This will look confusing for new programmers. The word 'lambda' basically means 'create an annonymous function'
# Since counting starts at 0 in Python, returning tup[1] will return the count.
# Lastly, 'tup' is simply the variable we create for the annonymous function. You could replace the two 
# instances of 'tup' with 'mytuple' and it would work the same!
bird_tuples.sort(key=lambda tup: tup[1], reverse=True)

"""
This is a multi-line comment! Previously I was only using single line comments that begin with a hash.
The single line above is the equivlant of this multi-line code block, where we define a function to 
return the second ([1]) element in a tuple, and pass the function name to the sort() method

def return_second_tuple_element(mytuple):
    return mytuple[1]
    
bird_tuples.sort(key=return_second_tuple_element, reverse=True)
"""

bird_tuples

# And try the graph again. This time, pay attention to the x and y values. Rather than saying that the 
# value of 'Canada Goose' is 47,677 and the value of 'Mallard' is 16,390, you assign the x values as a 
# list of bird names and the y values as a list of counts. Make sure they line up!

data = [go.Bar(x=[b[0] for b in bird_tuples],
            y=[b[1] for b in bird_tuples])]

offline.iplot(data)

# Let's make it a horizontal bar chart. Don't forget to swap the x and y values!
# We also need to reverse the list.
# Note that the reverse() method alters the list! If you run this cell multiple times, the order of 
# the horizontal bar chart swaps between greatest-to-least and least-to-greatest!
# More bar chart options: https://plot.ly/python/bar-charts/

bird_tuples.reverse()

data = [go.Bar(orientation='h', x=[b[1] for b in bird_tuples],
             y=[b[0] for b in bird_tuples])]

offline.iplot(data)

# Now let's chart the population of a bird over the years. First, let's create a dataframe
# of sums by year, similar to what we did for just 1979
waterfowl_df_january_sub_by_year = waterfowl_df_january_sub.groupby('Year').sum()

waterfowl_df_january_sub_by_year.head()

# Let's chart just 'Canada Goose'
bird_name = 'Canada Goose'

# You could use the line below if you wanted to chart 'Mallard'
#bird_name = 'Mallard'

single_bird = waterfowl_df_january_sub_by_year[bird_name]

single_bird.head()

# Some more data exploring

print(single_bird.index)
print('first:', single_bird.index[0])

#single_bird[bird_name]

years = [str(year) for year in single_bird.index]
years
    

# 

bird_counts = [int(total) for total in single_bird]
bird_counts

# Make a line chart (scatter)

trace1 = go.Scatter(x=years, y=bird_counts, mode="markers+lines")
                                               
data=go.Data([trace1])

layout=go.Layout(title="First Plot", xaxis={'title':'Year'}, yaxis={'title':bird_name})

figure=go.Figure(data=data,layout=layout)

offline.iplot(figure, filename='pyguide_1')

# Now let's chart multiple birds!

# Plot the top three

bird_names = ['Canada Goose', 'American Black Duck', 'Mallard']

data = []

for bird_name in bird_names:
    
    single_bird = waterfowl_df_january_sub[['Year', bird_name]].groupby('Year').sum()

    bird_counts = [int(total) for total in single_bird[bird_name]]
    
    # Cheat and re-usse the years variable from before
    data.append(go.Scatter(x=years, y=bird_counts, mode="markers+lines", name=bird_name))

layout=go.Layout(title="Top three birds", xaxis={'title':'Year'}, yaxis={'title':'Number counted'})

figure=go.Figure(data=data,layout=layout)

offline.iplot(figure, filename='top_three')



