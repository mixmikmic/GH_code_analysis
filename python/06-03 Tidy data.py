import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# pd.melt is similar to tidyr's gather function

name = ['daniel','john','jane']
trA = [np.NaN, 12, 24]
trB = [42, 31, 27]

df = pd.DataFrame({'name': name, 'treatment A': trA, 'treatment B': trB})
print(df)

# we call pd.melt
melted = pd.melt(frame = df, id_vars='name', 
                 value_vars=['treatment A', 'treatment B'],
                 var_name='treatment', value_name='result') # you can provide column names
print(melted)

print(pd.melt(df, id_vars = 'name')) # if you only provide id_vars, then it will melt all other columns

# the df.pivot() method is similar to tidyr's spread function

melted.pivot(index='name', columns = 'treatment', values = 'result')   # note we call pivot on the data frame itself

# creating a new variable

print(melted)

# you can create a new variable by defining a new column
melted['gender'] = ['m','m','f','m','m','f']

print(melted)

melted.result.astype('str').str[-2:]  # .astype makes a series.
# .str allows for string operators

melted['trt'] = melted.treatment.str[-1]  # The last letter from the treatment column, as an abreviation for the treatment
print(melted)

melted.name.str[:] + " - " +  melted.trt.str[:]

# pd.concat to concatenate tables:
name2 = ['amy','betty','carl']
trA2 = [20, 18, 10]
trB2 = [30, 38, 28]
df2 = pd.DataFrame({'name': name2, 'treatment A': trA2, 'treatment B': trB2})
print(df2)

print(df)

pd.concat([df, df2])  # call pd.concat. provide it a *list* of data frames, not the dataframe directly
# similar to rbind in R

# when we use pd. concat, the original indexes are kept.
concatenated = pd.concat([df, df2]) 
concatenated.loc[0,]  # returns two rows

print(concatenated.iloc[3:5,])  # iloc is unaffected

print(concatenated.iloc[3,])  # iloc is unaffected, # returning only one row reduces to a series

print(concatenated.index)

# you can reset the index during the concatenation process with ignore_index = True
concat2 = pd.concat([df, df2], ignore_index = True)
print(concat2)

import glob  # import this module

filenames = glob.glob('exa*.csv')  # use to find filenames in your working directory that fit the pattern
print(filenames)

list_data = []  # create empty list. This will be a list of dataframes

for file in filenames:
    data = pd.read_csv(file, header = 0)
    data['month'] = file   # append a column with the filename
    list_data.append(data)

print(list_data)  # this is a list of data frames

print(list_data[0])
print(type(list_data[0]))

pd.concat(list_data, ignore_index = True)  # to make a single dataframe, we use pd.concat over the list of dataframes

name = ['daniel','john','jane']
trA = [np.NaN, 12, 24]
trB = [42, 31, 27]
dfleft = pd.DataFrame({'name': name, 'treatment A': trA, 'treatment B': trB})
print(dfleft)

name = ['john','max','jane']
trC = [0, 12, 24]
trD = [42, 31, 27]
dfright = pd.DataFrame({'patient': name, 'treatment C': trC, 'treatment D': trD})
print(dfright)

dfleft.merge(dfright, left_on = 'name', right_on = 'patient')  # inner join, will return only rows that exist in both

dfleft.merge(dfright, left_on = 'name', right_on = 'patient', how = 'left')  
# left join, will return all the rows in the left table and rows from the right table that matches

dfleft.merge(dfright, left_on = 'name', right_on = 'patient', how = 'right')  
# right join, will return all the rows in the right table and rows from the left table that matches

dfleft.merge(dfright, left_on = 'name', right_on = 'patient', how = 'outer')  
# all rows from both tables

merged = dfleft.merge(dfright, left_on = 'name', right_on = 'patient', how = 'outer')  
print(merged)

name = ['daniel','john','jane']
trA = [np.NaN, 12, 24]
trB = [42, 31, 27]
dfleft = pd.DataFrame({'name': name, 'treatment A': trA, 'treatment B': trB})
print(dfleft)

name = ['daniel','daniel','john','john']
parents = ['mom','dad','mama','papa']
dfright = pd.DataFrame({'name': name, 'parents': parents})
print(dfright)

dfleft.merge(dfright, on = 'name')  # a one-to-many inner join
# returns only rows that match in left and right
# for rows that appear multiple times in the right
# the contents in the left gets duplicated



