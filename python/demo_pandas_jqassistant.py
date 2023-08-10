import py2neo
import pandas as pd

graph = py2neo.Graph()

query = "MATCH (a:Method) RETURN a"
result = graph.data(query)
result[0:3]

df = pd.DataFrame.from_dict([data['a'] for data in result]).dropna(subset=['name'])
df.head()

# filter out all the constructor "methods"
df = df[df['name'] != "<init>"]
# assumption 1: getter start with "get"
df.loc[df['name'].str.startswith("get"), "method_type"] = "Getter"
# assumption 2: "is" is just the same as a getter, just for boolean values
df.loc[df['name'].str.startswith("is"), "method_type"] = "Getter"
# assumption 3: setter start with "set"
df.loc[df['name'].str.startswith("set"), "method_type"] = "Setter"
# assumption 4: all other methods are "Business Methods"
df['method_type'] = df['method_type'].fillna('Business Methods')
df[['name', 'signature', 'visibility', 'method_type']][20:30]

grouped_data = df.groupby('method_type').count()['name']
grouped_data

import matplotlib.pyplot as plt
# some configuration for displaying nice diagrams directly in the notebook
get_ipython().magic('matplotlib inline')
plt.style.use('fivethirtyeight')
# apply additional style for getting a blank background
plt.style.use('seaborn-white')

# plot a nice business people compatible pie chart
ax = grouped_data.plot(kind='pie', figsize=(5,5), title="Business methods or just Getters or Setters?")
# get rid of the distracting label for the y-axis
ax.set_ylabel("")

