from statsmodels.tsa.stattools import coint, adfuller
import pandas as pd

fundamentals = init_fundamentals()
data = get_fundamentals(query(fundamentals.income_statement.total_revenue)
                        .filter((fundamentals.company_reference.primary_symbol == 'MCD') |
                                (fundamentals.company_reference.primary_symbol == 'MSFT') |
                                (fundamentals.company_reference.primary_symbol == 'KO')),
                        '2015-01-01', '30q')

# Get time series for each security individually
x0 = data.values[0].T[1]
x1 = data.values[0].T[2]
x2 = data.values[0].T[0]

print 'p-values of Dickey-Fuller statistic on total revenue data:'
print 'PEP:', adfuller(x0)[1]
print 'KO:', adfuller(x1)[1]
print 'MSFT:', adfuller(x2)[1]

# Compute the p-value for the cointegration of the two series
print 'p-values of cointegration statistic on total revenue data:'
print 'MCD and MSFT:', coint(x0, x1)[1]
print 'MCD and KO:', coint(x0, x2)[1]
print 'MSFT and KO:', coint(x1, x2)[1]

