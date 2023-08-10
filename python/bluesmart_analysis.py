get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pprint import pprint as pp
from pylab import figure, show
import math
import matplotlib.ticker as ticker


data = []
with open("loadCellData.csv", "r") as f:
    column_names = f.readline().strip().split(",")
    for i, line in enumerate(f):
        data.append(line.strip().split(","))
        
pp (list(enumerate(column_names)))
print "total rows in the data: %s"%len(data)
pp(data[:1])

data[0][3:]

inspec_data = []
outspec_data = []

for row in data:
    if row[1] == "In Spec":
        inspec_data.append([float(i) for i in row[3:]])
    else:
        outspec_data.append([float(i) for i in row[3:]])
        


def make_plots(px, py, title='title', xlabel='xlabel', ylabel='ylabel', legend='legend'):
    fig, ax = plt.subplots(figsize=(16, 8))
    colors = ['k', 'r', 'y', 'g', 'm', 'b']
    for i in xrange(len(py)):
        ax.plot(px, py[i], '-', color=colors[i%6], linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax.grid(True)
    plt.show()

def make_plot(px, py, title='title', xlabel='xlabel', ylabel='ylabel', legend='legend', color = 'r', ylim=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(px, py, '-', color=color, alpha = 0.5, label=legend, linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    ax.legend()
    ax.grid(True)
    plt.show()

# plotting in-spec 
make_plots(range(11), inspec_data,
          "In Spec Data",
          "Weight in Kg",
          "Reading in mv",
          ""
         )    

# plotting out of spec 
make_plots(range(11), outspec_data,
          "Out of Spec Data",
          "Weight in Kg",
          "Reading in mv",
         ) 

#Calculating least squared errors
def get_least_sqrd_err(slope, intercept, data):
    errs = 0.0
    for i in range(len(data)):
        errs += (data[i]-((slope*i)+intercept))**2
    return math.sqrt(errs)    
        

outspec_data_errors = []
initial_values = []
slopes = []
for row in outspec_data:
    slope, intercept, r_value, p_value, std_err = stats.linregress(range(11),row)
    outspec_data_errors.append(get_least_sqrd_err(slope, intercept, row))
    initial_values.append(row[0])
    slopes.append(slope)
    

initial_values,outspec_data_errors = zip(*sorted(zip(initial_values,outspec_data_errors)))

#Plot Initial value vs Error
make_plot(initial_values, outspec_data_errors,
          "Initial value vs Non Linear Error ",
          "Initial Values",
          "Non Linear Error",
          "No Specific relation between initial error and non linearity"
         )    

# Correcting zero error for out of spec data
new_outspec_data = []
for row in outspec_data:
    zero_err = row[0]
    new_outspec_data.append([i-zero_err for i in row])

make_plots(range(11), new_outspec_data,
          "Out of Spec data after correcting Zero Error",
          "Weight in Kg",
          "Reading in mv",
          ""
         )  

print "Out of Spec errors:"
print sorted([round(i,3) for i in outspec_data_errors])

linearity_threshold = 0.17
error, slopes = zip(*([(x,y) for x,y in sorted(zip(outspec_data_errors, slopes)) if x < linearity_threshold]))

# variance of the Linear Slopes 
np.var(slopes)
# = 0.00023621230137282599

#Range of error in reading in mv
[0.00023621230137282599*i for i in xrange(11)]
#0.0 to 0.0024



