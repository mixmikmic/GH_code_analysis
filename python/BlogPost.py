# This is where we will load the data set and produce some basic summary statistics.
# For example, we might be interested in:
# 1. Counties served by the SFDO
# 2. How many loans per county per year were given out
# 3. Break out loans by loan status (discharged, paid off, etc.)
# 4. Do some basic correlations with demographic characteristics from Census Data
# 5. What's the breakdown of loans by types of businesses (NAICS code)
# 6. What population do each of the counties serve? E.g. if we give businesses in areas where higher population, would mean higher impact.
# 7. How do we define "successful" businesses? Is it those with good yelp ratings? Can we break down characteristics of "successful" businesses
# 8. Which counties have had the highest rate of loans to successful businesses? 

# 1. Counties served by the SFDO
sba_loans.ProjectCounty.unique()
#How do we compare this list with the total number of counties in SF District? Are there counties in SF that are NOT on this list?
#Would be interesting to find labor data to understand the # of businesses in each of these counties. Is there an even proportion of businesses served in each county? Why are some counties getting higher proporitions? Does this signify underserved by SBA if certain counties see lower proportion of loan disbursment? 
#Other interesting q's: what population does each county serve? Wouldn't SBA have higher impact if served more populated communities (more population = higher demand)? 

# 3. Break out loans by loan status (discharged, paid off, etc.)
county = pd.crosstab(sba_loans["ProjectCounty"],columns = "percentage of loans").sort_values('percentage of loans', ascending=False)
county/county.sum()

# 2. How many loans per county per year were given out
pd.crosstab(sba_loans["ProjectCounty"],sba_loans["ApprovalFiscalYear"],margins=True)

#Percentage breakdown of loans per county per year
def percConvert(ser):
  return ser/float(ser[-1])
pd.crosstab(sba_loans["ProjectCounty"],sba_loans["ApprovalFiscalYear"],margins=True).apply(percConvert, axis=0)

# 3. Break out loans by loan status (discharged, paid off, etc.)
loans = pd.crosstab(sba_loans["LoanStatus"],columns = "count").sort_values('count', ascending=False)
loans/loans.sum()
#pd.crosstab(index=titanic_train["Survived"] columns="count"),  # Make a crosstab columns="count") 

import pandas as pd
sba_loans = pd.read_csv('./Data/SFDO_504_7A-clean.csv')

sba_loans.head()

sba_loans.columns

type = pd.crosstab(sba_loans["NaicsDescription"],columns = "count").sort_values('count', ascending=False)
print type

sba_loans.ProjectCounty.value_counts()
cabin_tab/cabin_tab.sum()
#I want to find the percentage break out

import pandas as pd
sba_loans = pd.read_csv('./Data/SFDO_504_7A-clean.csv')

#Number of loans per county per year
sba_loans.groupby(['ApprovalFiscalYear','ProjectCounty']).size()

#sba_loans.ProjectCounty.value_counts()
#.sort_values(ascending=False)
#I want to get the total number of loans that year in the last row
#I want to get the breakout in percentages

# This is where we might put some more advanced analysis and data visualizations
# For example, we might be interested in:
# 1. Doing some basic regression analysis on racial demographics? Evidence of discrimination?
# 2. Some nice maps

# What are some overall conclusions? What did we take away?

