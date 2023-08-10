import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv('flavors_of_cacao.csv')

data.columns

#line plot
#attributes: color= color, label= label, linewidth= width of line, alpha= opacity, grid= grid, linestyle= style of line

data.Rating.plot(kind='line', color='g', label='Rating', linewidth=1, alpha=0.5, grid= True, linestyle='-.')

plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')  # title = title of plot

#Histogram plot
# bins= number of bars in figure

data.Rating.plot(kind='hist', bins= 100, figsize=(20,20))

# create a dictionary and look for its keys and values

dictionary ={'India':'Delhi', 'France':'Paris', 'UK':'London'}
print dictionary.keys()
print dictionary.values()

# Keys have to be immutable objects like string, bool, integer or float
# List is not immutable
# Keys are unique

dictionary['India'] = "Kolkata"     # update existing entry
print dictionary
dictionary['Afghanistan'] = "Kabul"       # Add new entry
print dictionary
del dictionary['UK']              # remove entry with key 'UK'
print dictionary
print 'france' in dictionary        # check include or not
dictionary.clear()                   # remove all entries in dict
print dictionary

data1 = pd.read_csv('pokemon.csv')

series = data1['Defense']      # data['Defense'] = series
print type(series)
data_frame = data1[['Defense']]   # data[['Defense']] = data frame
print type(data_frame)

#Comaprison Operator

print 3>2
print 3!=2

#Boolean Operator

print True and False
print True or False

# 1 - Filtering Pandas data frame
x = data1['Defense']>200     # There are only 3 pokemons who have higher defense value than 200
data1[x]

# 2 - Filtering pandas with logical_and
# There are only 2 pokemons who have higher defence value than 2oo and higher attack value than 100
data1[np.logical_and(data1['Defense']>200, data1['Attack']>100 )]

# This is also same with previous code line. Therefore we can also use '&' for filtering.
data1[(data1['Defense']>200) & (data1['Attack']>100)]

# While loop. Loop terminates once condition is not satisfied

i = 1
while i != 11:
    print i
    i += 1
print 'i is now='+ str(i)

# Stay in loop if condition( i is not equal 5) is true
lis = [1,2,3,4,5]
for i in lis:
    print 'i is: ',i
print ''

# Enumerate index and value of list
# index : value = 0:1, 1:2, 2:3, 3:4, 4:5
for index, value in enumerate(lis):
    print index," : ",value
print ''

# For dictionaries
# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.
dictionary1 = {'spain':'madrid','france':'paris'}
for key,value in dictionary1.items():
    print key," : ",value
print ''

# For pandas we can achieve index and value
for index,value in data1[['Attack']][0:5].iterrows():
    print index," : ",value

