get_ipython().magic('matplotlib inline')
import pandas as pd
import os

PARQA = os.getenv('PARQA')

calls = pd.read_csv(PARQA + 'data/311/MatchedCalls.cav', index_col=0)
calls['year'] = calls['Created Date'].apply(lambda x: int(x.split(' ')[0].split('/')[-1]))

ts = calls[['year','parkDistrict']].groupby(['parkDistrict','year']).size().unstack()
ts.shape

ts.to_csv(PARQA + '/parqa/311/TIMESERIES/311_timeseries.csv')

ts.head(3)

ts.transpose().plot( figsize=(18,7), legend=0, alpha=.5,logy=1);

ts_m = calls[calls['Complaint Type']=='Maintenance or Facility'][['year','parkDistrict']].groupby(['parkDistrict','year']).size().unstack()

ts_m.to_csv(PARQA + '/parqa/311/TIMESERIES/311_timeseries_maintenance.csv')
ts_m.head(2)

p_calls = calls[(calls.Descriptor.isin(['Garbage or Litter','Broken Glass','Graffiti or Vandalism','Snow or Ice']))]

ts_p = p_calls.groupby(['parkDistrict','year']).size().unstack()
ts_p.to_csv(PARQA + '/parqa/311/TIMESERIES/311_timeseries_precize.csv')
ts_p.head(2)

def saveTS(sdf,title='feature'):
    ts_l = sdf.groupby(['parkDistrict','year']).size().unstack()
    ts_l.to_csv(PARQA + '/parqa/311/TIMESERIES/311_timeseries_%s.csv' % title)
    ts_l.head(2)

l_calls = calls[calls.Descriptor=='Garbage or Litter']
saveTS(l_calls,title='litter')

g_calls = calls[calls.Descriptor=='Graffiti or Vandalism']
saveTS(g_calls,title='graphity')

g_calls = calls[calls.Descriptor=='Broken Glass']
saveTS(g_calls,title='glass')



