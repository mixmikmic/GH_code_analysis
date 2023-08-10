import pandas as pd
import numpy as np

income = pd.read_csv('..\Data_vys217\Median Income\ACS_14_5YR_B19013_with_ann.csv', 
usecols = ['Id2', 'Estimate; Median household income in the past 12 months \
(in 2014 Inflation-adjusted dollars)' ], header = 1)

income.columns = ['zip_code', 'median_income']

for i in xrange(len(income.median_income)):
    try:
        income.iloc[i, 1] = int(filter(str.isdigit, income.median_income[i]))
    except ValueError:
        income.iloc[i, 1] = np.NaN

# total_income = pd.read_csv('C:\Users\Vishwajeet\Documents\Python Scripts\
# \Homeless\Data\Income\ACS_14_5YR_S1903_with_ann.csv', 
#                             usecols = ['Id2', 'Total; Estimate; Households', 
#                                        'Median income (dollars); Estimate; Households'], 
#                             header = 1)

income



