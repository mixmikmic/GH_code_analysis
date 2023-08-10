# import dependencies
import pandas as pd
import numpy as np

# read Medicaire spending csv into DF
medicare_spending = "Data Files/Medicare_Drug_Spending_PartD_All_Drugs_YTD_2015_12_06_2016.csv"
medicare_spending_df = pd.read_csv(medicare_spending,header = 3)
medicare_spending_df = medicare_spending_df.fillna(value = 0)
# cast to string and remove whitespace to merge
medicare_spending_df['Generic Name'] = medicare_spending_df['Generic Name'].astype(str)
medicare_spending_df['Generic Name'] = medicare_spending_df['Generic Name'].str.strip()

# clean df to include relevant rows ( total spending, average cost, unit count)
medicare_spending_cleaned_df = medicare_spending_df.iloc[:,[1,3,6,7,
                                                           13,16,17,
                                                           23,26,27,
                                                           33,36,37,
                                                           43,46,47]]
medicare_spending_cleaned_df.head()

# read opioids csv into DF
opioids = "Data Files/opioids.csv"
opioids_df = pd.read_csv(opioids)
# cast to string and remove whitespace to merge
opioids_df['Generic Name'] = opioids_df['Generic Name'] 
opioids_df['Generic Name'] = opioids_df['Generic Name'].str.strip()
opioids_df.head()

# merge opiods df and medicare spending on generic name to only include spending data with opioids
opioids_medicare_spending_df = pd.merge(medicare_spending_cleaned_df,opioids_df,how = 'inner', on = 'Generic Name')
# remove commas and '$' to cast to number
opioids_medicare_spending_df = opioids_medicare_spending_df.replace({'\$': '', ',': ''}, regex=True)
# change columns to floats
opioids_medicare_spending_df['Total Spending, 2011'] = pd.to_numeric(opioids_medicare_spending_df['Total Spending, 2011'])
opioids_medicare_spending_df['Unit Count, 2011'] = pd.to_numeric(opioids_medicare_spending_df['Unit Count, 2011'])
opioids_medicare_spending_df['Average Cost Per Unit (Weighted), 2011'] = pd.to_numeric(opioids_medicare_spending_df['Average Cost Per Unit (Weighted), 2011'])

opioids_medicare_spending_df['Total Spending, 2012'] = pd.to_numeric(opioids_medicare_spending_df['Total Spending, 2012'])
opioids_medicare_spending_df['Unit Count, 2012'] = pd.to_numeric(opioids_medicare_spending_df['Unit Count, 2012'])
opioids_medicare_spending_df['Average Cost Per Unit (Weighted), 2012'] = pd.to_numeric(opioids_medicare_spending_df['Average Cost Per Unit (Weighted), 2012'])

opioids_medicare_spending_df['Total Spending, 2013'] = pd.to_numeric(opioids_medicare_spending_df['Total Spending, 2013'])
opioids_medicare_spending_df['Unit Count, 2013'] = pd.to_numeric(opioids_medicare_spending_df['Unit Count, 2013'])
opioids_medicare_spending_df['Average Cost Per Unit (Weighted), 2013'] = pd.to_numeric(opioids_medicare_spending_df['Average Cost Per Unit (Weighted), 2013'])

opioids_medicare_spending_df['Total Spending, 2014'] = pd.to_numeric(opioids_medicare_spending_df['Total Spending, 2014'])
opioids_medicare_spending_df['Unit Count, 2014'] = pd.to_numeric(opioids_medicare_spending_df['Unit Count, 2014'])
opioids_medicare_spending_df['Average Cost Per Unit (Weighted), 2014'] = pd.to_numeric(opioids_medicare_spending_df['Average Cost Per Unit (Weighted), 2014'])

opioids_medicare_spending_df['Total Spending, 2015'] = pd.to_numeric(opioids_medicare_spending_df['Total Spending, 2015'])
opioids_medicare_spending_df['Unit Count, 2015'] = pd.to_numeric(opioids_medicare_spending_df['Unit Count, 2015'])
opioids_medicare_spending_df['Average Cost Per Unit (Weighted), 2015'] = pd.to_numeric(opioids_medicare_spending_df['Average Cost Per Unit (Weighted), 2015'])
opioids_medicare_spending_df.head()

# drop Drug Name column to remove duplicates
opioids_medicare_spending_cleaned_df = opioids_medicare_spending_df.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]
# remove duplicates
opioids_medicare_spending_cleaned_unique_df = opioids_medicare_spending_cleaned_df.drop_duplicates()
opioids_medicare_spending_cleaned_unique_df.columns

# make new empty DF and append columns and sums
opioids_medicare_summary_df = pd.DataFrame({
    'Generic Name':['Opioids'],
    #2011
    'Total Spending, 2011':[opioids_medicare_spending_cleaned_unique_df['Total Spending, 2011'].sum()], 
    'Unit Count, 2011':[opioids_medicare_spending_cleaned_unique_df['Unit Count, 2011'].sum()],
    'Average Cost Per Unit (Weighted), 2011':[opioids_medicare_spending_cleaned_unique_df['Average Cost Per Unit (Weighted), 2011'].sum()], 
    #2012
    'Total Spending, 2012':[opioids_medicare_spending_cleaned_unique_df['Total Spending, 2012'].sum()], 
    'Unit Count, 2012':[opioids_medicare_spending_cleaned_unique_df['Unit Count, 2012'].sum()],
    'Average Cost Per Unit (Weighted), 2012':[opioids_medicare_spending_cleaned_unique_df['Average Cost Per Unit (Weighted), 2012'].sum()], 
    #2013
    'Total Spending, 2013':[opioids_medicare_spending_cleaned_unique_df['Total Spending, 2013'].sum()], 
    'Unit Count, 2013':[opioids_medicare_spending_cleaned_unique_df['Unit Count, 2013'].sum()],
    'Average Cost Per Unit (Weighted), 2013':[opioids_medicare_spending_cleaned_unique_df['Average Cost Per Unit (Weighted), 2013'].sum()], 
    #2014
    'Total Spending, 2014':[opioids_medicare_spending_cleaned_unique_df['Total Spending, 2014'].sum()], 
    'Unit Count, 2014':[opioids_medicare_spending_cleaned_unique_df['Unit Count, 2014'].sum()],
    'Average Cost Per Unit (Weighted), 2014':[opioids_medicare_spending_cleaned_unique_df['Average Cost Per Unit (Weighted), 2014'].sum()], 
    #2015
    'Total Spending, 2015':[opioids_medicare_spending_cleaned_unique_df['Total Spending, 2015'].sum()], 
    'Unit Count, 2015':[opioids_medicare_spending_cleaned_unique_df['Unit Count, 2015'].sum()],
    'Average Cost Per Unit (Weighted), 2015':[opioids_medicare_spending_cleaned_unique_df['Average Cost Per Unit (Weighted), 2015'].sum()] 
})
opioids_medicare_summary_df

# adjust column order





