# Importing relevant modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
plt.rcParams.update({'font.size': 16})
import re

# Reading data
data = pd.read_csv("thanksgiving.csv", encoding="Latin-1")

# indices of rows for people who celebrate Thanksgiving
yes_celebrating = data['Do you celebrate Thanksgiving?']=='Yes'

# Keep the rows for which [Do you celebrate Thanksgiving?]= Yes
data = data[yes_celebrating] 

# Count how many times each category occurs 
dish_type = pd.value_counts(data['What is typically the main dish at your Thanksgiving dinner?'].values, sort=True)

#Now make a pie chart
plt.figure(figsize=(12,4))
dish_type.plot(kind='bar', color=['purple', 'violet'])
plt.ylabel('Frequency')
plt.yscale('log')
plt.xlabel('Dish')
plt.title('What do people eat for Thanksgiving')
plt.show()

#apple
apple_isnull = pd.isnull(data['Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Apple'])
apple_notnull = pd.notnull(data['Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Apple'])
#pumpkin
pumpkin_isnull = pd.isnull(data['Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pumpkin'])
pumpkin_notnull = pd.notnull(data['Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pumpkin'])
#pecan
pecan_isnull = pd.isnull(data['Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pecan'])
pecan_notnull = pd.notnull(data['Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pecan'])

no_pies = apple_isnull & pumpkin_isnull & pecan_isnull
only_apple_pies = apple_notnull & pumpkin_isnull & pecan_isnull
only_pumpkin_pies = apple_isnull & pumpkin_notnull & pecan_isnull
only_pecan_pies = apple_isnull & pumpkin_isnull & pecan_notnull
# create a dictionary with pie counts
pie_types = {}
pie_types['Apple'] = pd.value_counts(only_apple_pies)[1]
pie_types['Pumpkin'] = pd.value_counts(only_pumpkin_pies)[1]
pie_types['Pecan'] = pd.value_counts(only_pecan_pies)[1]
pie_types['None'] = pd.value_counts(no_pies)[1]
pie_types['Multiple'] = pd.value_counts(no_pies)[0] - pie_types['Apple'] - pie_types['Pumpkin'] - pie_types['Pecan'] 

# plot pie data in pie chart 
plt.figure(figsize=(7,7))
plt.pie([int(v) for v in pie_types.values()],labels=pie_types.keys(), autopct='%1.1f%%')
plt.title("Pie chart of pies consumed")
plt.show()

print(data["Age"].value_counts())
income_col = 'How much total combined money did all members of your HOUSEHOLD earn last year?'
print(data[income_col].value_counts()[:2])


def get_int_age(in_str):
    if pd.isnull(in_str):
        return None
    split_str = in_str.split(" ")
    age_str = re.sub('\+$', '', split_str[0])
    try:
        age_int = int(age_str)
    except Exception: 
        age_int = None
    return age_int

def get_int_income(in_str):
    if pd.isnull(in_str):
        return None
    first_str = in_str.split(" ")[0]
    if first_str== 'Prefer':
        return None
    
    income_str = re.sub('[\$\,]', '', first_str)
    try:
        income_int = int(income_str)
    except Exception: 
        income_int = None
    return income_int/1000

# Clean data 
data["int_age"] = data["Age"].apply(get_int_age)
data["int_income"] = data[income_col].apply(get_int_income)
# Fill missing data with median
data["int_age"] = data["int_age"].fillna(data["int_age"].median())
data["int_income"] = data["int_income"].fillna(data["int_income"].median())

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))

data.hist(column = 'int_age', ax=ax1, color='orange')
ax1.set_title("Distribution of Age")
ax1.set_xlabel("Age (years)")
data.hist(column = 'int_income', ax=ax2, color='green')
ax2.set_title("Distribution of Income")
ax2.set_xlabel("Income (in 1000$)")

plt.show()

#  low income results <150K
is_low_income = data['int_income'] < 150
dist_low_income = data['How far will you travel for Thanksgiving?'][is_low_income]
value_dist_low = dist_low_income.value_counts()

# high income results >150K
is_high_income = data['int_income'] > 150
dist_high_income = data['How far will you travel for Thanksgiving?'][is_high_income]
value_dist_high = dist_high_income.value_counts()

# pie plots
my_label = ["No travel", "Local","Few Hours","Out of Town"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

ax1.pie(value_dist_low, labels=my_label, autopct='%1.1f%%')
ax1.set_title("Low Income")

ax2.pie(value_dist_high, labels=my_label, autopct='%1.1f%%')
ax2.set_title("High Income")
fig.subplots_adjust(hspace=6)
plt.show()

icol = "Have you ever tried to meet up with hometown friends on Thanksgiving night?"
ccol = 'Have you ever attended a "Friendsgiving?"'
vcol = 'int_age'
pd.pivot_table(data, values=vcol, index=icol, columns=ccol, aggfunc='mean')

vcol2 = 'int_income'
pd.pivot_table(data, values=vcol, index=icol, columns=ccol, aggfunc='mean')



