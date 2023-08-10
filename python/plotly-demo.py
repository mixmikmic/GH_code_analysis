import plotly.plotly as py        # plotly.plotly handles all connections to Plotly servers.
import plotly.offline as pyo      # plotly.offline handles offline plotting requests.
from plotly.graph_objs import *   # plotly.graph_objs holds components used for construcing plots.

pyo.init_notebook_mode()          # initialize ipython plotting

import pandas as pd
from math import log10

# Load Data & Extract Desired Info

# In this example, all we are intersted in is the Ks values, which are stored in the first column. Unfortunately, 
# our data is not really organized into columns, so we need to explicitly parse the lines to extract it.
ks_values = []
with open('mutation-data.txt', 'r') as mut_in:
    for line in mut_in:
        line = line.strip()          # remove leading/trailing white space & new line charactars.
        if len(line) == 0:           # skip empty lines.
            pass
        elif line[0] == '#':         # skip comment lines.
            pass
        else:
            line = line.split()      # split line on white space.
            try:
                ks = float(line[0])  # get Ks data by index (see mutation-data.txt headers), convert to float.
                ks_values.append(ks) # add to our values list.
            except ValueError:    
                pass                 # skip if value is 'NA'.

# Define Plotly Data() object
ks_data = Histogram(x=ks_values,        # populate Histogram() X values with our Ks values.
                   histnorm='percent')  # normalize so y axis shows bin's percentage of total.
ks_data = Data([ks_data])               # compile into a data object.

# Define Plotly Layout() object
ks_layout = Layout(title="Synonymous Mutations in Syntenic Gene Pairs<br>Maize & Sorghum",  # set plot title.
                  xaxis=dict(title="Synonymous Mutations/Synonymous Site"),                 # set x-axis label.
                  yaxis=dict(title="% of Syntenic Gene Pairs"))                             # set y-axis label.

# Assemble Data() and Layout() into a Figure()
ks_figure = Figure(data=ks_data, layout=ks_layout)

# Create an offline plot
pyo.plot(ks_figure, filename="ks-plot.html")

# Log10-transform our data
logks_values = map(log10, [k for k in ks_values if k != 0])  # skip zero values

# Build our Data(), Layout(), and Figure() objects
logks_data = Histogram(x=logks_values,       # populate data with our Ks values.
                       histnorm='percent')   # display y axis as percentage of total filling bin.
logks_data = Data([logks_data])              # compile into a data object

logks_layout = Layout(title="Synonymous Mutations (Log10 Transformed) in Syntenic Gene Pairs<br>Maize & Sorghum",
                      xaxis=dict(title="Log10(Ks)"),
                      yaxis=dict(title="% of Syntenic Gene Pairs"))

logks_figure = Figure(data=logks_data, layout=logks_layout)

# Create an offline iplot
pyo.iplot(logks_figure, filename="logks-iplot", show_link=False)

# Load data.
fractbias_data = pd.read_csv('fractbias-data.csv', header=4)  # skip the first 4 rows because they are comments.
print("Shape: " + str(fractbias_data.shape))                  # print shape (see how much data we have!)
fractbias_data.head()

# Select a subset of the data for plotting.

# In this case, we are going to select only target chromosome 1. To make the plot more clear, we will also remove
# any query chromosomes that have no gene retention across the entire sliding window analysis.

target_1 = fractbias_data[(fractbias_data["Target Chr"] == 1)]  # select only rows for first target chromosome.
target_1 = target_1.ix[:, target_1.sum(axis=0) > 0]             # select only columns with data.
target_1.head()

# Define Plotly Graph Object
fractbias_data = []
queries = target_1.columns[2:]         # get a list of the query chromosomes.
for q in queries:                      # for each query, build a trace & add to our collection of data.
    trace = Scatter(
        x=target_1['Sliding Window (Count)'], 
        y=target_1[q],
        name=q
    )
    fractbias_data.append(trace)
fractbias_data = Data(fractbias_data)  # compile our traces into a Data() object.

# Define Plotly Layout Object
fractbias_layout = Layout(title = "Example Fractionation Bias<br>Target Chromosome 1",       # set plot title
                         xaxis = {"title": "Sliding Window Position"},                       # set x-axis label
                         yaxis = {"title": "% Retention<br>(# Syntenic Genes/Window Size)"}) # set y-axis label

# Assemble Data() and Layout() into a Figure()
fractbias_fig = Figure(data=fractbias_data, layout=fractbias_layout)

# Create an online plot on Plot.ly
fractbias_plot_link = py.plot(fractbias_fig, filename='fractbias-plot')
print(fractbias_plot_link)

# Create an online, but inline 'iplot'
py.iplot(fractbias_fig, filename='fractbias-iplot')



