# Pandas for DataFrames
import pandas as pd 

# Matplotlib for style settings
import matplotlib   

# Show plots in-place
get_ipython().magic('matplotlib inline')

# Check available styles
matplotlib.style.available

# Use an attractive default
matplotlib.style.use('ggplot')

# Shell line magic for quick inspection
get_ipython().system('ls data')

# Read the data from CSV with pandas
interests = pd.read_csv("data/interests.csv")

# Get the first five entries
interests.head()

# Pandas transforming functions have `inplace`
# Set "Category" as the index - row name equivalent
interests.set_index("Category", inplace=True)

interests.Votes

interests.index

# Show the whole table
interests

# Quick plotting
interests.plot(kind='bar')

# Sort index alphabetically
interests.sort_index(inplace=True)
# Use convenience wrapper to plot functions
interests.plot.bar()

# Sort the DataFrame by Votes
interests.sort_values(by="Votes", inplace=True)
# Horizontal bar plot
interests.plot.barh()

# plotly could be installed easily with `pip install plotly`
# Use plotly in offline mode - no user account necessary
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode()

# `data` is a list of datasets
# `go.Bar` is one of the plotly 'geoms'
# `x` and `y` are columns of the DataFrame
data = [go.Bar(
        x=interests.index,
        y=interests.Votes)]

# Render new figure with data
iplot(go.Figure(data=data))

# Convert counts into percentages
# Apply vecotrized operations on the columns directly
percentages = interests.Votes/30*100
percentages

# Round all the numbers
rounded_percentages = round(percentages, 2)
rounded_percentages

# Convert to strings
string_percentages = rounded_percentages.apply(str)
string_percentages

# Add percent sign to each of them
percentage_labels = string_percentages + "%"
percentage_labels

# All of the above in one go
percentage_labels = round((interests.Votes/30*100),2).apply(str) + "%"
percentage_labels

# Create plotly `data` with `text` element - revealed on hover
data = [go.Bar(
        x=interests.index,
        y=interests.Votes,
        text=percentage_labels)]

# Add elements to layout
# each sub-category is a dict
layout = go.Layout(height=600,
                   xaxis=dict(
                        tickangle=-90),
                   margin=dict(
                        b=200))

# Same as above
layout = go.Layout(height=600,
                   xaxis={'tickangle':-90},
                   margin={'b':200})

iplot(go.Figure(data=data,layout=layout))



