#importing required packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

filename = 'SP_500_close_2015.csv'
priceData = pd.read_csv(filename,index_col = 0)

#Read company names into a dictionary
def readNamesIntoDict():
    d = dict()
    input_file = csv.DictReader(open("SP_500_firms.csv"))
    for row in input_file:
        #print(row)
        d[row['Symbol']] = [row['Name'],row['Sector']]
    return d

companyNames = readNamesIntoDict()

#creating new dataframe for Daily Returns (dailyret)
dailyret = priceData.pct_change(1)

#creating dataframe for yearly change in shareprice
yearlyret = priceData.pct_change(len(priceData)-1).dropna()

#FUNCTIONS
#Finding 3 companies with biggest increase in the dataframe.
#result produces company name, date of occurence and change in value
def maxrise(retdf):
    maxname=retdf.max().sort_values()
    maxdate=(retdf.max(1).dropna()).sort_values()
    result = pd.DataFrame()
    if len(retdf)==1:   # this function checks whether the return relates to a year (len =1) or a daily 
        for i in range(-1, -4, -1):     
            detail = pd.DataFrame({'Percentage (%)': [maxname.iloc[i]*100],
                                   'Company Name': [companyNames[str(maxname.index[i])][0]],
                                   'Code': [maxname.index[i]]})
            result = result.append(detail, ignore_index = True)
    else:
        for i in range(-1, -4, -1):
            detail = pd.DataFrame({'Equity': [maxname.index[i]],
                                   'Firm Name': [companyNames[str(maxname.index[i])][0]],
                                   'Date' : [maxdate.index[i]],
                                   'Percentage (%)': [maxname.iloc[i]*100]})
            result = result.append(detail, ignore_index = True)
    return result

#biggest rise in a day
maxrise(dailyret)

#Finding 3 companies with biggest fall in the dataframe:
#result produces company name, change in value and the date
def maxdrop(retdf):
    minname=retdf.min().sort_values()
    mindate=(retdf.min(1).dropna()).sort_values()
    result = pd.DataFrame()
    if len(retdf)==1: #this function establishes whether the return is calculated for a year (=1) or a daily one
        for i in range(0, 3):
            detail = pd.DataFrame({'Code': [minname.index[i]],
                                   'Company Name': [companyNames[str(minname.index[i])]],
                                   'Percentage (%)': [minname.iloc[i]*100]})
            result = result.append(detail, ignore_index = True)
    else: 
        for i in range(0, 3):
            detail = pd.DataFrame({'Equity': [minname.index[i]],
                                   'Firm Name': [companyNames[str(minname.index[i])]],
                                   'Date' : [mindate.index[i]], 
                                   'Percentage (%)': [minname.iloc[i]*100]})
            result = result.append(detail, ignore_index = True)
    return result

#biggest fall in a day
maxdrop(dailyret)

#Calculate the correlations between all stocks in the data using returns. 
#Create an empty data frame n*n setting as columns and rows the stocks' names
col = dailyret.columns
col = col.tolist()
cor = pd.DataFrame(index=col, columns=col)

#Create a data frame 'cor' with the correlations between all stocks (less efficient way)
#Calculate correlations in a faster and easier way
def calcCorr():
    cor1 = np.corrcoef(dailyret[1: ],rowvar=0)
    cor = pd.DataFrame(cor1, columns = col, index = col)
    return cor

#This is how correlation calculation works in general (less efficient way)
def calcCorr2():
    for i in range(0, dailyret.shape[1]):
        for j in range(0, dailyret.shape[1]):
            dividend = (dailyret.shape[0]-1) * sum(dailyret.iloc[1:dailyret.shape[0]-1, i] * dailyret.iloc[1:dailyret.shape[0]-1, j]) - sum(dailyret.iloc[1:dailyret.shape[0]-1, i]) * sum(dailyret.iloc[1:dailyret.shape[0]-1, j])
            divisor1 =  (((dailyret.shape[0]-1) * sum((dailyret.iloc[1:dailyret.shape[0]-1, i])**2) - (sum(dailyret.iloc[1:dailyret.shape[0]-1, i]))**2))**(1/2)   
            divisor2 = (((dailyret.shape[0]-1) * sum((dailyret.iloc[1:dailyret.shape[0]-1, j])**2) - (sum(dailyret.iloc[1:dailyret.shape[0]-1, j]))**2))**(1/2)   
            cor.iloc[i,j] = dividend/(divisor1*divisor2)
    return cor



corTable = calcCorr()
cor = corTable

#Provide a convenient way for a user to print out two companies' full names 
#and a correlation between their returns 
def compandcor():
    name1 = input("Please enter stock symbol of FIRST company: ")  #the function asks for user's input
    name2 = input("Please enter stock symbol of SECOND company: ") #the function asks for user's input
    col = dailyret.columns.tolist()
    #function looks for correlation information for entered companies    
    for i in col:
        for j in col:
            if i == name1.upper() and j == name2.upper():
                return print("Company Name 1: " + str(companyNames[name1][0]) + " \n"+ "Company Name 2: " + str(companyNames[name2][0]) + " \n" + "Correlation: " + str(corTable.loc[i, j]))
    return ("Error. Use correct format (Example: AAPL, MSFT, fb, zts, etc. No spaces.). Please also ensure that the entered company symbol was in S&P 500 for 2015.") # if stock symbol is entered incorrectly OR not in the list- this statement is returned.

compandcor()

#Provide a convenient way for a user to input a stock's name and print out the full name of the two companies with wcich has 
#the highest and lowest correlation respectively.
def bestandworstcor(name):
    compName = companyNames[name][0]
    largest = -1
    lowest = 1
    for j in col:
        if cor.loc[name, j] > largest and j != name:
            largest = cor.loc[name, j]
            top = companyNames[j][0]
        if cor.loc[name, j] < lowest:
            lowest = cor.loc[name, j]
            bottom = companyNames[j][0]
    return print ("Highest Correlation Company with " + str(compName)+ " is " + str(top) + " with "+ str(largest) + " \n" + "Lowest Correlation Company with " + str(compName)+ " is "+ str(bottom) + " with "+ str(lowest))

bestandworstcor('AMZN')

bestandworstcor('MSFT')

bestandworstcor('FB')

bestandworstcor('AAPL')

bestandworstcor('GOOGL')

