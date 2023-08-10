import pandas as mypd
from scipy import stats as mystats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

myData=mypd.read_csv('C:/Users/jmo4cob/ETI/Data/Sales_Revenue_Anova.csv')

myData

sales=myData.Sales_Revenue

#sales

location=myData.Location

#computing ANOVA table
mymodel=ols('sales ~ C(location)',myData).fit()

anova_table=anova_lm(mymodel)

anova_table

#conclusion is that <.05 means on an average the revenue changes with location==> location matters
#additional analysis is required to find the change in revenue value depending on location

import matplotlib.pyplot as myplot
sales.groupby(location).mean()

myData.boxplot(column='Sales_Revenue',by='Location')
myplot.show()

#2nd location fetches the maximum selling
#no high difference between the 



