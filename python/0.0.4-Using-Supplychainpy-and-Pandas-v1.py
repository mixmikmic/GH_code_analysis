get_ipython().magic('matplotlib inline')

import matplotlib
import pandas as pd

from supplychainpy.model_inventory import analyse
from supplychainpy.model_demand import simple_exponential_smoothing_forecast
from supplychainpy.sample_data.config import ABS_FILE_PATH
from decimal import Decimal
raw_df =pd.read_csv(ABS_FILE_PATH['COMPLETE_CSV_SM'])

orders_df = raw_df[['Sku','jan','feb','mar','apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']]
#orders_df.set_index('Sku')
analyse_kv =dict(
    df=raw_df, 
    start=1, 
    interval_length=12, 
    interval_type='months',
    z_value=Decimal(1.28), 
    reorder_cost=Decimal(400), 
    retail_price=Decimal(455), 
    file_type='csv',
    currency='USD'
)
analysis_df = analyse( **analyse_kv)
print(analysis_df[['sku','quantity_on_hand', 'excess_stock', 'shortages', 'ABC_XYZ_Classification']])

print(analysis_df.T)

analysis_rev = analysis_df[['sku', 'revenue']]
print(analysis_rev.sort_values(by='revenue', ascending=True))

analysis_trans = analysis_df.T
print(analysis_trans[0])



row_ds = raw_df[raw_df['Sku']=='KR202-212'].squeeze()
print(row_ds[1:12])

ses_df = simple_exponential_smoothing_forecast(ds=row_ds[1:12], length=12, smoothing_level_constant=0.5)
print(ses_df)

print(ses_df.get('forecast', 'UNKNOWN'))

print(ses_df.get('statistics', 'UNKNOWN'),'\n mape: {}'.format(ses_df.get('mape', 'UNKNOWN')))

print(ses_df.get('forecast_breakdown', 'UNKNOWN'))

forecast_breakdown_df = pd.DataFrame(ses_df.get('forecast_breakdown', 'UNKNOWN'))
print(forecast_breakdown_df)

forecast_breakdown_df.plot(x='t', y=['one_step_forecast','demand'])

regression = {'regression': [(ses_df.get('statistics')['slope']* i ) + ses_df.get('statistics')['intercept'] for i in range(1,12)]}
print(regression)

forecast_breakdown_df['regression'] = regression.get('regression')
print(forecast_breakdown_df)

forecast_breakdown_df.plot(x='t', y=['one_step_forecast','demand', 'regression'])

opt_ses_df = simple_exponential_smoothing_forecast(ds=row_ds[1:12], length=12, smoothing_level_constant=0.4,optimise=True)
print(opt_ses_df)

print(opt_ses_df.get('statistics', 'UNKNOWN'),'\n mape: {}'.format(opt_ses_df.get('mape', 'UNKNOWN')))

print(opt_ses_df.get('forecast', 'UNKNOWN')) 

optimised_regression = {'regression': [(opt_ses_df.get('statistics')['slope']* i ) + opt_ses_df.get('statistics')['intercept'] for i in range(1,12)]}
print(optimised_regression)

opt_forecast_breakdown_df = pd.DataFrame(opt_ses_df.get('forecast_breakdown', 'UNKNOWN'))

opt_forecast_breakdown_df['regression'] = optimised_regression.get('regression')
print(opt_forecast_breakdown_df)

opt_forecast_breakdown_df.plot(x='t', y=['one_step_forecast','demand', 'regression'])

