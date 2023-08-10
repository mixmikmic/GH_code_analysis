"""
@Project: Connexin Group 

@FileName: normalize_income_zipcode

@Author：Zhejian Peng

@Create date: Feb. 8th, 2018

@description：Normalized income according to zipcode. 

@Update date：  

@Vindicator： xx@ic.net.cn  

"""  

get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import math
import seaborn as sns
#from termcolor import colored
from numbers import Number
from scipy import stats
from pandas import plotting

# read in load data
df = pd.read_csv("LoanStats_2017Q3.csv", low_memory=False)
loan_data = pd.read_csv("LoanStats_2017Q3.csv", low_memory=False)

# 我写了一堆code 然后发现其实简单一点就能弄出来，所以大家忽略后面的code！！！
# I have wrote a lot of code for this only to find out that I only need this simple function!!!
def norm_inc_by_zip(zipcode, income):
    '''
        @description: Use on a column of data; output a dictionary that returns mean and average in each zipcode area
        @zipcode： zipcode dataframe column
        @income: income df column 
        @return:      return a dictionary
    '''  
    # I try to replace nan with 0 for income, and nan in zipcode for "000xx"
    df["annual_inc"].fillna(0)
    df["zip_code"].fillna("000xx")
    
    mean_var = {}
    for idx, value in zipcode.iteritems():
        # calculate total income
        if value in mean_var:
            mean_var[value].append(income[idx])
        else:
            mean_var[value] = [income[idx]]

    
    #assert(len(zip_code) == len(mean_var))
    # compute the average income in each zip_code area
    for key, value in mean_var.items():
        # if there only one element, we set their variance to 1. This way when normalize, it will have a 0 z-score.
        if len(value) == 1:
            #print(value[0])
            mean_var[key] = [value[0], 1]
        else:
            mean_var[key] = [np.mean(value), np.std(value)]
        
    # first loop through every annual income by calculate its z score. (Income - mean_by_zipcode) / variance_by_zipcode
    for idx, value in df["zip_code"].iteritems():
        #inc_colnum = df.columns.get_loc("annual_inc")
        col_num_inc = df["annual_inc"]
        mean, std = mean_var[value]
        df.at[idx, "annual_inc"] = (df.at[idx, "annual_inc"] - mean) / std
    print("Income is successfually normalized")
    return mean_var

dic = norm_inc_by_zip(df["zip_code"], df["annual_inc"])



# Now I want to check if this normalized distribution is in fact mean = 0, var = 1
def mean_var(zipcode, income):
    '''
        @description: Use on a column of data; output a dictionary that returns mean and average in each zipcode area
        @zipcode： zipcode dataframe column
        @income: income df column 
        @return:      return a dictionary
    '''  
    income.fillna(0)
    zipcode.fillna("000xx")
    
    total = {}
    for idx, value in zipcode.iteritems():
        # calculate total income
        if value in total:
            total[value].append(income[idx])
        else:
            total[value] = [income[idx]]

    
    #assert(len(zip_code) == len(total))
    # compute the average income in each zip_code area

    for key, value in total.items():
        # if there only one element, we set their variance to 1. This way when normalize, it will have a 0 z-score.
        if len(value) == 1:
            #print(value[0])
            total[key] = [value[0], 1]
        else:
            total[key] = [np.mean(value), np.std(value)]
        
    return total

m_v = mean_var(df["zip_code"], df["annual_inc"])

m_v

# We can see after normalization the data appears to be mean = 0, std = 1, so our normalization is good.



LARGE_FILE = "loan_data_seperate_current.csv"
CHUNKSIZE = 100000 # processing 100,000 rows at a time
reader = pd.read_csv(LARGE_FILE, chunksize=CHUNKSIZE, low_memory=False)
frames = []

for df in reader:
    norm_inc_by_zip(zipcode=df["zip_code"], income=df["annual_inc"])
    frames.append(df)

#  combine normalized data and output it to a file
normalized_income_loan_data = pd.concat(frames)

# For normalized data, study income distribution vs zipcode
# classif zip_codes
def classify_zipcode(column):
    '''
        @description: Use on a column of data; output a dictionary that stores every keys in column and count the number
                      of times this key appear in this column; 
        @dictionary： takes a pandas dataframe column as input
        @... 
        @return:      return a dictionary counting the number of elements in the column
    '''  
    dic = {}
    for i in column:
        if i in dic:
            dic[i] +=1
        else:
            dic[i] = 1
    return dic

zipcode = classify_zipcode(normalized_income_loan_data["zip_code"])

len(zipcode)

counter = 0 
for key, value in zipcode.items():
    if value >= 30:
        counter +=1
print(counter)

y = normalized_income_loan_data["annual_inc"]
plt.hist(y, bins=100, range=(-3,9))
plt.title("normalized income distribution")

CHUNKSIZE = 100000 # processing 100,000 rows at a time
reader = pd.read_csv(LARGE_FILE, chunksize=CHUNKSIZE, low_memory=False)
frames = []
for df in reader:
    frames.append(df)
unnormalized_income = pd.concat(frames)

plt.hist(unnormalized_income["annual_inc"], bins=100, range=(-100000, 600000))
plt.title("unnormalized income distribution")

# conditional distribution: income distribution according to loan_status:
# organize income mapped to each loan_status, using normalized income
loan_status = {}
# set normalize_income_loan_data na value to 0. Which means use average income to replace n/a value.
normalized_income_loan_data["annual_inc"].fillna(0)
for idx, i in normalized_income_loan_data["loan_status"].iteritems():
    if i in loan_status:
        loan_status[i].append(normalized_income_loan_data["annual_inc"][idx])
    else:
        loan_status[i] = [normalized_income_loan_data["annual_inc"][idx]]
assert(len(loan_status) == 9)
#loan_status

# plot 9 income distribution graph according to each loan_status
def plot_inc_distribution(key, loan_status, data = "normalized"):
    plt.hist(loan_status[key], bins=100, range=(-3,9))
    title = data + " " + key + " income distribution"
    plt.title(title)
    plt.show()

for key in loan_status:
    if isinstance(key, str):
        plot_inc_distribution(key, loan_status=loan_status)



# conditional distribution: income distribution according to loan_status:
# organize income mapped to each loan_status, using normalized income
unnormalized_loan_status = {}
# set normalize_income_loan_data na value to 0. Which means use average income to replace n/a value.
unnormalized_income["annual_inc"].fillna(0, inplace = True)
for idx, i in unnormalized_income["loan_status"].iteritems():
    if i in unnormalized_loan_status:
        unnormalized_loan_status[i].append(unnormalized_income["annual_inc"][idx])
    else:
        unnormalized_loan_status[i] = [unnormalized_income["annual_inc"][idx]]
assert(len(unnormalized_loan_status) == 9)
#loan_status



def plot_inc_distribution(key, data, data_type = "normalized"):
    if data_type == "normalized":
        plt.hist(data[key], bins=100, range=(-3,9))
        title = data_type + " " + key + " income distribution"
        plt.title(title)
        plt.show()
    elif data_type == "unnormalized":
        plt.hist(data[key], bins=100, range=(0,300000))
        title = data_type + " " + key + " income distribution"
        plt.title(title)
        plt.show()

for key in unnormalized_loan_status:
    if isinstance(key, str):
        plot_inc_distribution(key, unnormalized_loan_status, data_type= "unnormalized")


for key, value in unnormalized_loan_status.items():
    print(key, np.mean(value), np.var(value)**0.5)





# Estimate how many different output in Data sets
zip_code = classify_zipcode(df["zip_code"])
print("There are ", len(zip_code) , " different number of zipcode in the dataset")

# Calculate total income in each zip_code area
def total_income(zipcode, income):
    '''
        @description: Use on a column of data; output a dictionary that stores the total amound of income in each zipcode
        @zipcode： zipcode dataframe column
        @income: income df column 
        @return:      return a dictionary
    '''  
    total = {}
    for idx, value in zipcode.iteritems():
        if value in total:
            total[value] += income[idx]
        else:
            total[value] = 0
    return total



total_inc = total_income(df["zip_code"], df["annual_inc"])
total_inc

# Check len(total_income) == len(zip_code)
print(len(total_inc) == len(zip_code))

def ave_income(zipcode, income):
    '''
        @description: Use on a column of data; output a dictionary that returns the average annual income in 
                      each zipcode. If income is nan, replace it with 0.
        @zipcode： zipcode dataframe column
        @income: income df column 
        @return:      return a dictionary
    '''  
    income.fillna(0)
    zipcode.fillna("000xx")
    
    total = {}
    zip_code = {}
    ave_income = {}

    for idx, value in zipcode.iteritems():
        # calculate total income
        if value in total:
            total[value] += income[idx]
        else:
            total[value] = 0
        # count the number of borrowers in each zipcode
        if value in zip_code:
            zip_code[value] += 1
        else:
            zip_code[value] = 1
    
    assert(len(zip_code) == len(total))
    # compute the average income in each zip_code area
    for key, value in zip_code.items():
        ave_income[key] = total[key] / float(value)
    assert(len(total) == len(ave_income))
    return ave_income
    

ave = ave_income(df["zip_code"].fillna("000xx"), df["annual_inc"].fillna(0))



# Now we need variance to compute normalized annual income.

var_income = np.array(df["annual_inc"].fillna(0))

np.var(var_income)

ave_inc = {}
for key, value in zip_code.items():
    ave_inc[key] = total_income[key] / float(value)

for i in df["annual_inc"]:
    if math.isnan(i):
        print(i)

def norm_inc_by_zip(zipcode, income):
    '''
        @description: Use on a column of data; output a dictionary that returns mean and average in each zipcode area
        @zipcode： zipcode dataframe column
        @income: income df column 
        @return:      return a dictionary
    '''  
#    income.fillna(0)
#    zipcode.fillna("000xx")
    
    total = {}
    for idx, value in zipcode.iteritems():
        # calculate total income
        if value in total:
            total[value].append(income[idx])
        else:
            total[value] = [income[idx]]

    
    assert(len(zip_code) == len(total))
    # compute the average income in each zip_code area
    return total

dic = norm_inc_by_zip(df["zip_code"].fillna("000xx"), df["annual_inc"].fillna(0))

dic_a = {}
for key, value in dic.items():
    dic_a[key] = [np.mean(value), np.std(value)]

dic_a



