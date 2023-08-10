import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

#Data 
revenue = [14574.49, 7606.46, 8611.41, 9175.41, 8058.65, 8105.44, 11496.28, 9766.09, 10305.32, 14379.96, 10713.97, 15433.50]
expenses = [12051.82, 5695.07, 12319.20, 12089.72, 8658.57, 840.20, 3285.73, 5821.12, 6976.93, 16618.61, 10054.37, 3803.96]

# Calculating profit (revenue - expenses)
profit = list([])

for i in range(0,len(revenue)):
    profit.append(revenue[i] - expenses[i])
profit

# Calculating tax (30% of profit)
tax =[round(i*.30,2) for i in profit]

tax

# Profit after tax
profit_after_tax = list([])

for i in range(0,len(profit)):
    profit_after_tax.append(profit[i] - tax[i])
profit_after_tax

# Profit margin after tax
profit_margin = list([])

for i in range(0,len(profit)):
    profit_margin.append(profit_after_tax[i]/revenue[i])
profit_margin = [round((i*100),2) for i in profit_margin]
profit_margin

# Profit after tax mean
mean_pat = sum(profit_after_tax)/len(profit_after_tax)
mean_pat

# Good months
good_months = list([])
for i in range(0, len(profit)):
    good_months.append(profit_after_tax[i] > mean_pat)
good_months

# Bad months
bad_months = list([])
for i in range(0, len(profit)):
    bad_months.append(profit_after_tax[i] < mean_pat)
bad_months

# Best month
best_month = list([])
for i in range(0, len(profit)):
    best_month.append(profit_after_tax[i]==max(profit_after_tax))
best_month

# Worst month
worst_month = list([])
for i in range(0, len(profit)):
    worst_month.append(profit_after_tax[i]==min(profit_after_tax))
worst_month

# Covert all calculations to units of one thousand dollars
revenue_1000 = [round(i/1000,2) for i in revenue]
expenses_1000 = [round(i/1000,2) for i in expenses]
profit_1000 = [round(i/1000,2) for i in profit]
profit_after_tax_1000 = [round(i/1000,2) for i in profit_after_tax]

revenue_1000 = [int(i) for i in revenue_1000]
expenses_1000 = [int(i) for i in expenses_1000]
profit_1000 = [int(i) for i in profit_1000]
profit_after_tax_1000 = [int(i) for i in profit_after_tax_1000]

# Print results
print('Revenue:')
print(revenue_1000)
print('\n')
print('Expenses:')
print(expenses_1000)
print('\n')
print('Profit:')
print(profit_1000)
print('\n')
print('Profit After Tax:')
print(profit_after_tax_1000)
print('\n')
print('Profit Margin:')
print(profit_margin)
print('\n')
print('Good Months:')
print(good_months)
print('\n')
print('Bad_Months:')
print(bad_months)
print('\n')
print('Best Month:')
print(best_month)
print('\n')
print('Worst Month:')
print(worst_month)

