# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import inspect

get_ipython().magic('matplotlib inline')

sns.set_style("whitegrid")
pd.set_option("display.max_columns",100)

# Read in the wrangled credit card client default data set.

df_wrangled = pd.read_csv('default of credit card clients - wrangled.csv', 
                          header=1, 
                          index_col=0)
#df_wrangled.head()

# Create a contingency table of 
# gender (male / female) and
# default status (default / non-default)

default_sex_crosstab = pd.crosstab(df_wrangled['default payment next month'], 
                                   df_wrangled['SEX'], 
                                   margins=True,
                                   normalize=False)

# default payment next month:
# 0 = non-default; 1 = default
new_index = {0: 'Non-default', 1: 'Default', }

# SEX: 
# 1 = male; 2 = female
new_columns = {1 : 'Male', 2 : 'Female'}

default_sex_crosstab.rename(index=new_index, columns=new_columns, inplace=True)
default_sex_crosstab

# Plot the number of males and females

fig, ax = plt.subplots(1, 1)

my_colors = ['0.6']

default_sex_crosstab.loc['All'][0:2].plot.bar(figsize=(5,4), 
                                              rot=0, 
                                              fontsize=14,
                                              color=my_colors, 
                                              ax=ax)

plt.title('Number of Males and Females in the Data Set', 
          fontsize=20)

def yaxis_formatter_fn(x, pos):
    return "{:,}".format(int(x))

formatter = FuncFormatter(yaxis_formatter_fn)
ax.yaxis.set_major_formatter(formatter)

ax.xaxis.label.set_visible(False)

plt.show()

# Normalize the contingency table columns
# by dividing each column by the column's total.

default_sex_crosstab_norm =  default_sex_crosstab / default_sex_crosstab.loc['All']
default_sex_crosstab_norm

# Plot the proportion of defaults by gender, showing:
#   1. the proportion of defaults for males;
#   2. the proportion of defaults for females; and
#   3. the proportion of defaults for the entire data set.

fig, ax = plt.subplots(1, 1)

my_colors = ['0.7','0.35',]
default_sex_crosstab_norm[0:2].T.plot.barh(stacked=True, 
                                           figsize=(10,4), 
                                           xticks=list(np.linspace(0, 1, 11)),
                                           rot=0, 
                                           fontsize=14,
                                           color=my_colors, 
                                           ax=ax)

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)

ax.xaxis.set_ticks(list(np.linspace(0.05, .95, 10)), minor=True)
ax.grid(b=True, which='minor', color='.8', linestyle='--')

ax.yaxis.label.set_visible(False)


plt.title('Proportion of Defaults by Gender', fontsize=20)
plt.xlabel('Proportion of Defaults', fontsize=14)

plt.show()

# Create a contingency table of 
# education (Graduate school / University / High school /Others) and
# default status (default / non-default)

default_edu_crosstab = pd.crosstab(df_wrangled['default payment next month'], 
                                   df_wrangled['EDUCATION'], 
                                   margins=True,
                                   normalize=False)

# default payment next month:
# 0 = non-default; 1 = default
new_index = {0: 'Non-default', 1: 'Default', }

# EDUCATION: 
# 1 = graduate school; 2 = university; 3 = high school; 4 = others.
new_columns = {1 : 'Graduate school', 
               2 : 'University', 
               3 : 'High school', 
               4 : 'Others'}

default_edu_crosstab.rename(index=new_index, columns=new_columns, inplace=True)
default_edu_crosstab

# Plot the number of individuals for 
# each level of educational attainment.

fig, ax = plt.subplots(1, 1)

my_colors = ['0.6']

default_edu_crosstab.loc['All'][0:4].plot.bar(figsize=(8,4), 
                                              rot=0, 
                                              fontsize=14,
                                              color=my_colors, 
                                              ax=ax)

plt.title('Number of Individuals for Each Level \nof Educational Attainment', 
          fontsize=20)

def yaxis_formatter_fn(x, pos):
    return "{:,}".format(int(x))

formatter = FuncFormatter(yaxis_formatter_fn)
ax.yaxis.set_major_formatter(formatter)

ax.xaxis.label.set_visible(False)

plt.show()

# Normalize the contingency table columns
# by dividing each column by the column's total.

default_edu_crosstab_norm =  default_edu_crosstab / default_edu_crosstab.loc['All']
default_edu_crosstab_norm

# Plot the proportion of defaults by level of education, showing:
#   1. the proportion of defaults for individuals with high school education;
#   2. the proportion of defaults for individuals with univeresity education;
#   3. the proportion of defaults for individuals with graduate school education;
#   4. the proportion of defaults for individuals categorized as 'others';
#   5. the proportion of defaults for the entire data set.

fig, ax = plt.subplots(1, 1)

my_colors = ['0.7','0.35',]
default_edu_crosstab_norm[0:2].T.plot.barh(stacked=True, 
                                           figsize=(10,4), 
                                           xticks=list(np.linspace(0, 1, 11)),
                                           rot=0, 
                                           fontsize=14,
                                           color=my_colors, 
                                           ax=ax)

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)

ax.xaxis.set_ticks(list(np.linspace(0.05, .95, 10)), minor=True)
ax.grid(b=True, which='minor', color='.8', linestyle='--')

ax.yaxis.label.set_visible(False)

plt.title('Proportion of Defaults by Education', fontsize=20)
plt.xlabel('Proportion of Defaults', fontsize=14)

plt.show()

# Create a contingency table of 
# marital status (Married / Single / Divorce / Others and
# default status (default / non-default)

default_mar_crosstab = pd.crosstab(df_wrangled['default payment next month'], 
                                   df_wrangled['MARRIAGE'], 
                                   margins=True,
                                   normalize=False)

# default payment next month:
# 0 = non-default; 1 = default
new_index = {0: 'Non-default', 1: 'Default', }

# MARRIAGE:
# 1 = married; 2 = single; 3 = divorce; 0=others.
new_columns = {1 : 'Married', 
               2 : 'Single', 
               3 : 'Divorce', 
               0 : 'Others'}

default_mar_crosstab.rename(index=new_index, columns=new_columns, inplace=True)
default_mar_crosstab

# Plot the number of individuals for 
# each category of marital status.

fig, ax = plt.subplots(1, 1)

my_colors = ['0.6']

default_mar_crosstab.loc['All'][0:4].plot.bar(figsize=(6,4), 
                                              rot=0, 
                                              fontsize=14,
                                              color=my_colors, 
                                              ax=ax)

plt.title('Number of Individuals for Each \nCategory of Marital Status', 
          fontsize=20)

def yaxis_formatter_fn(x, pos):
    return "{:,}".format(int(x))

formatter = FuncFormatter(yaxis_formatter_fn)
ax.yaxis.set_major_formatter(formatter)

ax.xaxis.label.set_visible(False)

plt.show()

# Normalize the contingency table columns
# by dividing each column by the column's total.

default_mar_crosstab_norm =  default_mar_crosstab / default_mar_crosstab.loc['All']
default_mar_crosstab_norm

# Plot the proportion of defaults by marital status, showing:
#   1. the proportion of defaults for single individuals;
#   2. the proportion of defaults for married individuals;
#   3. the proportion of defaults for divorced individuals;
#   4. the proportion of defaults for individuals categorized as 'others';
#   5. the proportion of defaults for the entire data set.

fig, ax = plt.subplots(1, 1)

my_colors = ['0.7','0.35',]
default_mar_crosstab_norm[0:2].T.plot.barh(stacked=True, 
                                           figsize=(10,4), 
                                           xticks=list(np.linspace(0, 1, 11)),
                                           rot=0, 
                                           fontsize=14,
                                           color=my_colors, 
                                           ax=ax)

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)

ax.xaxis.set_ticks(list(np.linspace(0.05, .95, 10)), minor=True)
ax.grid(b=True, which='minor', color='.8', linestyle='--')

ax.yaxis.label.set_visible(False)

plt.title('Proportion of Defaults by Marital Status', fontsize=20)
plt.xlabel('Proportion of Defaults', fontsize=14)

plt.show()

# Plot the number of individuals by age.

# Create series of the number of individuals by age.
age_count = df_wrangled['AGE'].value_counts().sort_index(ascending=True)

# Fill in missing age counts with zeros.
for i in list(range(age_count.index.min(), age_count.index.max()+1)):
    if i not in age_count.index:
        s = pd.Series([0], index=[i])
        age_count = age_count.append(s)

age_count.sort_index(ascending=True, inplace=True)

# Plot the Series

fig, ax = plt.subplots(1, 1)

my_colors = ['0.4']

age_count.plot.bar(figsize=(11,5), 
                   rot=90, 
                   fontsize=10,
                   color=my_colors, 
                   ax=ax)

def yaxis_formatter_fn(x, pos):
    return "{:,}".format(int(x))

formatter = FuncFormatter(yaxis_formatter_fn)
ax.yaxis.set_major_formatter(formatter)

ax.xaxis.label.set_visible(False)

plt.title('Number of Individuals by Age', 
          fontsize=20)
plt.xlabel('Age', fontsize=14)

plt.show()

# Create a contingency table of age and
# default status (default / non-default)

default_age_crosstab = pd.crosstab(df_wrangled['default payment next month'], 
                                   df_wrangled['AGE'], 
                                   margins=True,
                                   normalize=False)

# default payment next month:
# 0 = non-default; 1 = default
new_index = {0: 'Non-default', 1: 'Default', }

default_age_crosstab.rename(index=new_index, inplace=True)
default_age_crosstab

# Normalize the contingency table columns
# by dividing each column by the column's total.

default_age_crosstab_norm =  default_age_crosstab / default_age_crosstab.loc['All']
default_age_crosstab_norm

# Plot the proportion of defaults by age, but only
# for ages with at least 50 observations.

my_df = df_wrangled.groupby(['AGE', 
                             'default payment next month']).size().unstack()

# Filter out ages with fewer than 50 observations.
my_df = my_df[my_df.sum(axis=1)>50]

# Calculate proportions
my_df = my_df.div(my_df.sum(axis=1), axis='index')

# Plot
fig, ax = plt.subplots(1, 1, figsize=(10,5))

my_colors = ['0.3']

ax.scatter(x=my_df.index,
           y=my_df[1], 
           color=my_colors)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 

plt.title('Proportion of Defaults by Age', fontsize=20)
plt.ylabel('Proportion of Defaults', fontsize=14)
plt.xlabel('Age', fontsize=14)

plt.show()

# Calculate summary statistics for credit limit.

df_wrangled['LIMIT_BAL'].describe()

# Plot the a histogram of the number of individuals 
# in each credit limit bin.

fig, ax = plt.subplots(1, 1, figsize=(11,5))

my_colors = ['0.4']

df_wrangled['LIMIT_BAL'].hist(bins=50,
                              color=my_colors,
                              ax=ax, 
                              ec='k',
                              lw=1)

ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

def yaxis_formatter_fn(x, pos):
    return "{:,}".format(int(x))

formatter = FuncFormatter(yaxis_formatter_fn)
ax.yaxis.set_major_formatter(formatter)
ax.xaxis.set_major_formatter(formatter)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for spine in ax.spines.values():
    spine.set_edgecolor('k')

plt.title('Histogram of Credit Limit', 
      fontsize=20)
plt.xlabel('Credit Limit', fontsize=14)

plt.show()

# Define a generator that yields a range of floating point numbers.
# This will be used for binning below.

def float_range(start, stop, step=1.0):
    '''Generator that yields a range of floating point numbers'''
    while start < stop:
        yield start
        start +=step        

# Plot the proportion of defaults for each 
# credit limit bin.

# Create a list to bin credit limit observations
# The bin size is 10,000 Taiwan New Dollars.

step = 10000
start = round((df_wrangled['LIMIT_BAL'].min()), -4)
stop = round((df_wrangled['LIMIT_BAL'].max() + step), -4)
        
bins = []
for i in float_range(start, stop, step):
    bins.append(round(i,1))

group_names = bins[0:(len(bins)-1)]
group_names

# Bin the data

df_wrangled_copy = df_wrangled.copy()

categories = pd.cut(df_wrangled_copy['LIMIT_BAL'], 
                    bins=bins, 
                    labels=group_names)

df_wrangled_copy['LIMIT_BAL_BINNED'] = categories

my_df = df_wrangled_copy.groupby(['LIMIT_BAL_BINNED',
                                    'default payment next month']).size().unstack()

# Require at least 50 total observations per age
my_df = my_df[my_df.sum(axis=1)>50]

my_df = my_df.div(my_df.sum(axis=1), axis='index')
my_df.sort_index(ascending=False, inplace=True)

# Plot 

fig, ax = plt.subplots(1, 1, figsize=(11,5))

my_colors = ['0.4']

plt.scatter(x=my_df.index, y=my_df[1],
            c=my_colors,  
            edgecolors='k', 
            linewidths=1)

ax.set_ylim(bottom=0)
ax.set_xlim(left=0)

def yaxis_formatter_fn(x, pos):
    return "{:,}".format(int(x))

formatter = FuncFormatter(yaxis_formatter_fn)
ax.xaxis.set_major_formatter(formatter)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for spine in ax.spines.values():
    spine.set_edgecolor('k')

plt.title('Proportion of Defaults Versus Credit Limit', 
          fontsize=20)    
plt.ylabel('Proportion of Defaults', fontsize=14)
plt.xlabel('Credit Limit', fontsize=14)

plt.show()

pay_list =['PAY_1',  'PAY_2',  'PAY_3',  'PAY_4',  'PAY_5',  'PAY_6']


label_dict ={'PAY_1': 'September, 2005', 
             'PAY_2': 'August, 2005', 
             'PAY_3': 'July, 2005', 
             'PAY_4': 'June, 2005',  
             'PAY_5': 'May, 2005',  
             'PAY_6': 'April, 2005'}

pay_dfs = {}

for item in pay_list:
    pay_dfs[item] = df_wrangled.groupby([item, 'default payment next month']).size().unstack()
    # Require at least 50 total observations
    
    pay_dfs[item] = pay_dfs[item][pay_dfs[item].sum(axis=1)>25]
    
    # Calculate proportions
    pay_dfs[item] = pay_dfs[item].div(pay_dfs[item].sum(axis=1), axis='index')
    pay_dfs[item].sort_index(ascending=False, inplace=True)
        
# plot

sns.set_palette(sns.light_palette("navy", reverse=True))

fig, ax = plt.subplots(1, 1, figsize=(10,5))

for item in pay_list:
    ax.scatter(x=pay_dfs[item].index, 
               y=pay_dfs[item][1],
               label=label_dict.get(item),
               s=100, 
               edgecolor='k', 
               lw=1)          
    
ax.set_ylim([0, 1])

ticks = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]#, 9]
plt.xticks(ticks, rotation=0)
#plt.xticks(x, labels, rotation='vertical')

ax.xaxis.set_ticks(ticks=ticks, minor=False)

ax.grid(b=True, which='major', color='0.4', linestyle='--')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for spine in ax.spines.values():
    spine.set_edgecolor('k')
        
sns.set_palette(sns.light_palette("navy", reverse=True))
        
plt.title('Proportion of Defaults Versus Repayment Status', 
          fontsize=20)    
plt.ylabel('Proportion of Defaults', fontsize=14)
plt.xlabel('Repayment Status', fontsize=14)
    
plt.show()

bill_amt_list =['BILL_AMT1', 
                'BILL_AMT2', 
                'BILL_AMT3', 
                'BILL_AMT4', 
                'BILL_AMT5', 
                'BILL_AMT6']

label_dict ={'BILL_AMT1': 'September, 2005',  
             'BILL_AMT2': 'August, 2005',
             'BILL_AMT3': 'July, 2005', 
             'BILL_AMT4': 'June, 2005',  
             'BILL_AMT5': 'May, 2005', 
             'BILL_AMT6': 'April, 2005'}

c_dict ={'BILL_AMT1': 'BILL_AMT1_OVER_LIMIT_BAL',  
         'BILL_AMT2': 'BILL_AMT2_OVER_LIMIT_BAL', 
         'BILL_AMT3': 'BILL_AMT3_OVER_LIMIT_BAL',  
         'BILL_AMT4': 'BILL_AMT4_OVER_LIMIT_BAL', 
         'BILL_AMT5': 'BILL_AMT5_OVER_LIMIT_BAL',  
         'BILL_AMT6': 'BILL_AMT6_OVER_LIMIT_BAL'}

c_dict_binned ={'BILL_AMT1': 'BILL_AMT1_OVER_LIMIT_BAL_BINNED',  
                'BILL_AMT2': 'BILL_AMT2_OVER_LIMIT_BAL_BINNED', 
                'BILL_AMT3': 'BILL_AMT3_OVER_LIMIT_BAL_BINNED',  
                'BILL_AMT4': 'BILL_AMT4_OVER_LIMIT_BAL_BINNED', 
                'BILL_AMT5': 'BILL_AMT5_OVER_LIMIT_BAL_BINNED',  
                'BILL_AMT6': 'BILL_AMT6_OVER_LIMIT_BAL_BINNED'}

df_wrangled_copy = df_wrangled.copy()

for item in bill_amt_list:
    df_wrangled_copy[c_dict.get(item)] = df_wrangled_copy[item].div(df_wrangled_copy['LIMIT_BAL'],
                                                                              axis='index')

min_set = set()
max_set = set()

for item in bill_amt_list:
    min_set.add(df_wrangled_copy[c_dict.get(item)].min())
    max_set.add(df_wrangled_copy[c_dict.get(item)].max())
    
step = 0.1
    
start = round((min(min_set) - step), 1)
stop = round((max(max_set) + step), 1)

bins = []
for i in float_range(start, stop, step):
    bins.append(round(i,1))

group_names = bins[0:(len(bins)-1)]
group_names

for item in bill_amt_list:
    categories = pd.cut(df_wrangled_copy[c_dict.get(item)], 
                        bins=bins, 
                        labels=group_names)
    df_wrangled_copy[c_dict_binned.get(item)] = categories

bill_amt_dfs = {}

for item in bill_amt_list:
    bill_amt_dfs[item] = df_wrangled_copy.groupby([c_dict_binned.get(item), 
                                                   'default payment next month']).size().unstack()
    # Require at least 50 total observations
    
    bill_amt_dfs[item] = bill_amt_dfs[item][bill_amt_dfs[item].sum(axis=1)>25]
    
    # Calculate proportions
    bill_amt_dfs[item] = bill_amt_dfs[item].div(bill_amt_dfs[item].sum(axis=1), axis='index')
    bill_amt_dfs[item].sort_index(ascending=False, inplace=True)
        
# plot

sns.set_palette(sns.light_palette("navy", reverse=True))

fig, ax = plt.subplots(1, 1, figsize=(10,5))

for item in bill_amt_list:
    ax.scatter(x=bill_amt_dfs[item].index, 
               y=bill_amt_dfs[item][1],
               label=label_dict.get(item),
               s=100, 
               edgecolor='k', 
               lw=1)          
    
ax.set_ylim([0, 0.5])

xmin, xmax = ax.get_xlim()

step = 0.1
start = round((xmin), 1)
stop = round((xmax + step), 1)

ticks = []
for i in float_range(start, stop, step):
    ticks.append(round(i,1))

plt.xticks(ticks, rotation=90)

ax.grid(b=True, which='major', color='0.4', linestyle='--')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for spine in ax.spines.values():
    spine.set_edgecolor('k')
        
sns.set_palette(sns.light_palette("navy", reverse=True))
        
plt.title('Proportion of Defaults Versus \nthe Ratio of (Bill Amount / Credit Limit)', 
          fontsize=20)    
plt.ylabel('Proportion of Defaults', fontsize=14)
plt.xlabel('Ratio of (Bill Amount / Credit Limit)', fontsize=14)
    
plt.show()

bill_amt_list =['BILL_AMT1', 
                'BILL_AMT2', 
                'BILL_AMT3', 
                'BILL_AMT4', 
                'BILL_AMT5', 
                'BILL_AMT6']

bill_pay_dict ={'BILL_AMT1': 'PAY_AMT1',  
                'BILL_AMT2': 'PAY_AMT2', 
                'BILL_AMT3': 'PAY_AMT3',  
                'BILL_AMT4': 'PAY_AMT4',  
                'BILL_AMT5': 'PAY_AMT5', 
                'BILL_AMT6': 'PAY_AMT6'}

label_dict ={'BILL_AMT1': 'September, 2005',  
             'BILL_AMT2': 'August, 2005',
             'BILL_AMT3': 'July, 2005', 
             'BILL_AMT4': 'June, 2005',  
             'BILL_AMT5': 'May, 2005', 
             'BILL_AMT6': 'April, 2005'}

c_dict ={'BILL_AMT1': 'BILL_AMT1_OVER_LIMIT_BAL',  
         'BILL_AMT2': 'BILL_AMT2_OVER_LIMIT_BAL', 
         'BILL_AMT3': 'BILL_AMT3_OVER_LIMIT_BAL',  
         'BILL_AMT4': 'BILL_AMT4_OVER_LIMIT_BAL', 
         'BILL_AMT5': 'BILL_AMT5_OVER_LIMIT_BAL',  
         'BILL_AMT6': 'BILL_AMT6_OVER_LIMIT_BAL'}

c_dict_binned ={'BILL_AMT1': 'BILL_AMT1_OVER_LIMIT_BAL_BINNED',  
                'BILL_AMT2': 'BILL_AMT2_OVER_LIMIT_BAL_BINNED', 
                'BILL_AMT3': 'BILL_AMT3_OVER_LIMIT_BAL_BINNED',  
                'BILL_AMT4': 'BILL_AMT4_OVER_LIMIT_BAL_BINNED', 
                'BILL_AMT5': 'BILL_AMT5_OVER_LIMIT_BAL_BINNED',  
                'BILL_AMT6': 'BILL_AMT6_OVER_LIMIT_BAL_BINNED'}

df_wrangled_copy = df_wrangled.copy()

for item in bill_amt_list:
    df_wrangled_copy[c_dict.get(item)] = (df_wrangled_copy[item] - df_wrangled_copy[bill_pay_dict.get(item)]).div(df_wrangled_copy['LIMIT_BAL'],
                                                                                                                  axis='index')

min_set = set()
max_set = set()

for item in bill_amt_list:
    min_set.add(df_wrangled_copy[c_dict.get(item)].min())
    max_set.add(df_wrangled_copy[c_dict.get(item)].max())
    
step = 0.1
    
start = round((min(min_set) - step), 1)
stop = round((max(max_set) + step), 1)

bins = []
for i in float_range(start, stop, step):
    bins.append(round(i,1))

group_names = bins[0:(len(bins)-1)]
group_names

for item in bill_amt_list:
    categories = pd.cut(df_wrangled_copy[c_dict.get(item)], 
                        bins=bins, 
                        labels=group_names)
    df_wrangled_copy[c_dict_binned.get(item)] = categories

bill_amt_dfs = {}

for item in bill_amt_list:
    bill_amt_dfs[item] = df_wrangled_copy.groupby([c_dict_binned.get(item), 
                                                   'default payment next month']).size().unstack()
    # Require at least 50 total observations
    
    bill_amt_dfs[item] = bill_amt_dfs[item][bill_amt_dfs[item].sum(axis=1)>25]
    
    # Calculate proportions
    bill_amt_dfs[item] = bill_amt_dfs[item].div(bill_amt_dfs[item].sum(axis=1), axis='index')
    bill_amt_dfs[item].sort_index(ascending=False, inplace=True)
        
# plot

sns.set_palette(sns.light_palette("navy", reverse=True))

fig, ax = plt.subplots(1, 1, figsize=(10,5))

for item in bill_amt_list:
    ax.scatter(x=bill_amt_dfs[item].index, 
               y=bill_amt_dfs[item][1],
               label=label_dict.get(item),
               s=100, 
               edgecolor='k', 
               lw=1)          
    
ax.set_ylim([0, 0.5])

xmin, xmax = ax.get_xlim()

step = 0.1
start = round((xmin), 1)
stop = round((xmax + step), 1)

ticks = []
for i in float_range(start, stop, step):
    ticks.append(round(i,1))

plt.xticks(ticks, rotation=90)

ax.grid(b=True, which='major', color='0.4', linestyle='--')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), fontsize=14)

for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(14) 
for spine in ax.spines.values():
    spine.set_edgecolor('k')
        
sns.set_palette(sns.light_palette("navy", reverse=True))
        
plt.title('Proportion of Defaults Versus the Ratio of \n((Bill Amount - Pay Amount) / Credit Limit)', 
          fontsize=20)     
plt.ylabel('Proportion of Defaults', fontsize=14)
plt.xlabel('Ratio of ((Bill Amount - Pay Amount) / Credit Limit)', fontsize=14)
    
plt.show()

