#Importing the pandas package

import pandas as pd

#Reading in the file with the proper encoding

data = pd.read_csv("thanksgiving.csv", encoding = "Latin-1")

#Viewing the first few rows of the dataframe

data.head(5)

columns = data.columns
columns[:5]

#Finding out the counts of the number of people who celebrate thanksgiving

data['Do you celebrate Thanksgiving?'].value_counts()

#Filtering out the new dataframe where the respondants answered 'Yes'

data_yes = data[data['Do you celebrate Thanksgiving?'] == 'Yes']
data_yes['Do you celebrate Thanksgiving?'].count()

#Finding out the counts of the main dishes that people eat

data_yes['What is typically the main dish at your Thanksgiving dinner?'].value_counts()

#Filtering out the data that have the main dish as Tofurkey

data_tofurkey = data_yes[data_yes['What is typically the main dish at your Thanksgiving dinner?'] == 'Tofurkey']

#Getting the counts of the people who have gravy with Tofurkey

data_tofurkey['Do you typically have gravy?']

ate_pies = (pd.isnull(data_yes["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Apple"])
&
pd.isnull(data_yes["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pecan"])
 &
 pd.isnull(data_yes["Which type of pie is typically served at your Thanksgiving dinner? Please select all that apply. - Pumpkin"])
)

#Finding out the counts for for the number of people who eat desserts

ate_pies.value_counts()

data_yes["Age"].value_counts()

def extract_age(age_str):
    if pd.isnull(age_str):
        return None
    age_str = age_str.split(" ")[0]
    age_str = age_str.replace("+", "")
    return int(age_str)

#Applying the function we created

data_yes["Age_int"] = data_yes["Age"].apply(extract_age)

#Extracting the summary statistics for the age

data_yes["Age_int"].describe()

#Analyzing the column 
data_yes["How much total combined money did all members of your HOUSEHOLD earn last year?"].value_counts()

def extract_income(income_str):
    if pd.isnull(income_str):
        return None
    income_str = income_str.split(" ")[0]
    if income_str == "Prefer":
        return None
    income_str = income_str.replace(",", "")
    income_str = income_str.replace("$", "")
    return int(income_str)

data_yes["Money_int"] = data_yes["How much total combined money did all members of your HOUSEHOLD earn last year?"].apply(extract_income)

#Displaying the summary statistics of the incomes

data_yes["Money_int"].describe()

#Filtering out the data where the income is lesser than 35000

low_income = data_yes[data_yes['Money_int'] < 35000]

#Counting the different distances that people will travel in the low income bracket

low_income['How far will you travel for Thanksgiving?'].value_counts()

#Filtering out the data where the income is greater than 150,000

high_income = data_yes[data_yes['Money_int'] > 150000]

#Counting the different distances that people will travel in the high income bracket

high_income['How far will you travel for Thanksgiving?'].value_counts()

#Proportion for low income 

len(low_income[low_income["How far will you travel for Thanksgiving?"] == 
               "Thanksgiving is happening at my home--I won't travel at all"])/len(low_income)

#Proportion for high income

len(high_income[high_income["How far will you travel for Thanksgiving?"] == 
               "Thanksgiving is happening at my home--I won't travel at all"])/len(high_income)

