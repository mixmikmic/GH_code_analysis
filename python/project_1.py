# Import Packages
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# Open file
file_loc = 'data/sat_scores.csv'
with open(file_loc, 'r') as f:
    lines = f.readlines()

# Replace '\n' and ''
cleaned_data = []
for line in lines:
    cleaned_data.append(line.replace('\n',''))

# Split the data strings on commas
split_data = []
for datum in cleaned_data:
    split_data.append(datum.split(','))

print split_data

# create labels and data variables
labels = split_data[0]
data = split_data[1:]

print labels
print ''
print data

states = []
for row in data:
    states.append(row[0:1])
    
print states

print(type(data))
print(type(states))

rate = labels.index('Rate')
math = labels.index('Math')
verbal = labels.index('Verbal')

for row in data:
    row[rate] = int(row[rate])
    row[math] = int(row[math])
    row[verbal] = int(row[verbal])
[type(item) for item in data[0]]

state_dictionary = {}
for state_entries in data:
    # create new list for just scores
    list_scores = []
    for index, element in enumerate(state_entries):
        # index 0 refers to the states, so the for loop will pass
        if index == 0:
            pass
        else:
            list_scores.append(element)
    # create state dictionary        
    state_dictionary[state_entries[0]] = list_scores 
      
print state_dictionary

# create new list for just numerical data
num_data = []
for row in data:
    num_data.append(row[1:])

# create separate list for rate values to add to rate dictionary
rate_values = []
for row in num_data:
    rate_values.append(row[0])

# create rate dictionary
rate_dict = {}
rate_dict['Rate'] = rate_values
print rate_dict

# create separate list for verbal values to add to verbal dictionary
verbal_values = []
for row in num_data:
    verbal_values.append(row[1])

# create verbal dictionary
verbal_dict = {}
verbal_dict['Verbal'] = verbal_values
print ''
print verbal_dict

# create separate list for math values to add to math dictionary
math_values = []
for row in num_data:
    math_values.append(row[2])

# create math dictionary
math_dict = {}
math_dict['Math'] = math_values
print ''
print math_dict

# convert each string in each dictionary into floats
for key, value in rate_dict.items():
    value = [float(x) for x in value]
    rate_dict[key] = value 

for key, value in verbal_dict.items():
    value = [float(x) for x in value]
    verbal_dict[key] = value
    
for key, value in math_dict.items():
    value = [float(x) for x in value]
    math_dict[key] = value

# print dictionaries
print rate_dict
print ''
print verbal_dict
print ''
print math_dict

# create dictionary for all numerical values
num_dict = dict(rate_dict)
num_dict.update(verbal_dict)
num_dict.update(math_dict)

print num_dict

def sum_statistics(key, value):
    for key, value in num_dict.items():
        print "Name:", key
        print "Mean:", np.mean(value)
        print "Median:", np.median(value)
        print "Mode:", stats.mode(value)[0]
        print "Variance:", np.var(value)
        print "Standard Deviation:", np.std(value)
        print ''

sum_statistics(key, value)

def min_max(key, value):
    for key, value in num_dict.items():
        print "Name:", key
        print "Min:", np.min(value)
        print "Max:", np.max(value)
        print ''

min_max(key, value)

def stan_dev():
    return [np.std(value) for key, value in num_dict.items()]

print "Standard Deviation(Rate, Math, Verbal):", stan_dev()

pandas_data = pd.read_csv(file_loc)
pandas_data.head()

pandas_data.hist('Rate')

pandas_data.hist('Math')

pandas_data.hist('Verbal')

pandas_data.plot(kind='scatter', x='Rate', y='Verbal', c='blue', title='Rate vs Verbal')
pandas_data.plot(kind='scatter', x='Rate', y='Math', c='red', title = 'Rate vs Math')

plt.scatter(pandas_data['Rate'],pandas_data['Math'], color = 'r', label='Math')
plt.scatter(pandas_data['Rate'],pandas_data['Verbal'], color = 'b', label='Verbal')
plt.xlabel('Rate')
plt.title('Correlation between Math and Verbal scores')
plt.grid(True)

plt.legend(loc='upper right')
plt.show()

fig = plt.figure(figsize=(12,4))

ax1 = fig.add_subplot(131)
plt.boxplot(pandas_data['Rate'])
ax1.set_title('Box plot for Rate')

ax2 = fig.add_subplot(132)
plt.boxplot(pandas_data['Verbal'])
ax2.set_title('Box plot for Verbal')

ax3 = fig.add_subplot(133)
plt.boxplot(pandas_data['Math'])
ax3.set_title('Box plot for Math')
plt.show()

from IPython.display import Image
Image(filename='assets/SAT Participation Rate by State.png') 

from IPython.display import Image
Image(filename='assets/SAT Math Scores by State.png') 

from IPython.display import Image
Image(filename='assets/SAT Verbal Scores by State.png') 



