import pandas as pd
from statsmodels.formula.api import ols

Circulatory_monthly = pd.read_csv(r'D:\Annies_Dissertation\Analysis\Regression\HES\Circul_AP_monthly.csv')

Circulatory_monthly[:5]

Circulatory_monthly.sort_values(by=['lsoa11', 'month'])

Circulatory_monthly = Circulatory_monthly[['Disease', 
                                           'year', 'month', 'lsoa11', 'n', 
                                           'DSR', 'lcl', 'ucl', 
                                           'score', 'rank', 'decile', 'PM25']]

Circ_month_av =  Circulatory_monthly.groupby(['lsoa11', 'month']).mean()

Circ_month_av[:5]

Circ_month_av.corr()

Respiratory_monthly = pd.read_csv(r'D:\Annies_Dissertation\Analysis\Regression\HES\Respir_AP_monthly.csv')

Respiratory_monthly[:5]

Respiratory_monthly.sort_values(by=['lsoa11', 'month'])

Respiratory_monthly = Respiratory_monthly[['Disease', 
                                           'year', 'month', 'lsoa11', 'n', 
                                           'DSR', 'lcl', 'ucl', 
                                           'score', 'rank', 'decile', 'PM25']]

Resp_month_av = Respiratory_monthly.groupby(['lsoa11', 'month']).mean()

Resp_month_av[:5]

Resp_month_av.corr()



