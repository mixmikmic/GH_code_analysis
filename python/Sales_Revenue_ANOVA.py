import pandas as mypd
from scipy import stats as mystats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

myData=mypd.read_csv('.\datasets\Sales_Revenue_Anova.csv')

myData

sales=myData.Sales_Revenue

#sales

location=myData.Location

#computing ANOVA table
mymodel=ols('sales ~ C(location)',myData).fit()

anova_table=anova_lm(mymodel)

anova_table

#conclusion is that p < 0.05 means on an average the revenue changes with location==> location matters



