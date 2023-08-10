from IPython.display import HTML
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/wNVJEG9Utlo" frameborder="0" gesture="media" allow="encrypted-media" allowfullscreen></iframe>')

import eikon as ek
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ek.set_app_id('Your App ID here')

RICS = ['0#.FCHI']
fields =['TR.TRBCIndustryGroup','TR.CombinedAlphaCountryRank(SDate=0,EDate=-7,Frq=FQ)','TR.CombinedAlphaCountryRank(SDate=0,EDate=-7,Frq=FQ).Date',
         'TR.TotalReturn3Mo(SDate=0,EDate=-7,Frq=FQ)','TR.TotalReturn3Mo(SDate=0,EDate=-7,Frq=FQ).calcdate']

ids,err=ek.get_data(RICS,fields=fields)
ids.head(20)

ids.info()

ids['Date']=pd.to_datetime(ids['Date'])
ads=ids.set_index('Date')[['Instrument','TRBC Industry Group Name','Combined Alpha Model Country Rank','3 Month Total Return']]
ads['3 Month Total Return'] = pd.to_numeric(ads['3 Month Total Return'], errors='coerse')
ads['TRBC Industry Group Name'] = ads['TRBC Industry Group Name'].astype(str)
ads.head(10) 

ads.dtypes

ads.head()

ads1 = ads.replace('', np.nan, regex=True)

ads1['TRBC Industry Group Name'].fillna(method='ffill',limit=7, inplace=True)
ads1.head(15)

ads1['3 Month Total Return'] = ads1.groupby('Instrument')['3 Month Total Return'].shift()

ads1.head(15)

ads1.info()

ads1.dropna(axis=0, how='any', inplace=True)
ads1.head(20)

get_ipython().magic('matplotlib inline')
adsl = ads1[ads1['TRBC Industry Group Name'] =='Industrial Conglomerates']
adsl = adsl.groupby('Instrument')
ax = adsl.plot(x='Combined Alpha Model Country Rank', y='3 Month Total Return', kind='scatter')

ads1.isnull().any().count()

adsNN = ads1.fillna(method='ffill')

adsNN.isnull().any()

import sklearn
import scipy
from sklearn import linear_model
model = linear_model.LinearRegression()
for (group, adsNN_gp) in adsNN.groupby('Instrument'):
    X=adsNN_gp[['Combined Alpha Model Country Rank']]
    y=adsNN_gp[['3 Month Total Return']]
    model.fit(X,y)
    spearmans = scipy.stats.spearmanr(X,y)
    adsNN.loc[adsNN.Instrument == adsNN_gp.iloc[0].Instrument, 'slope'] = model.coef_
    adsNN.loc[adsNN.Instrument == adsNN_gp.iloc[0].Instrument, 'Rho'] = spearmans[0]
    adsNN.loc[adsNN.Instrument == adsNN_gp.iloc[0].Instrument, 'p'] = spearmans[1]

adsNN.head(15)

Averages = adsNN.groupby(['Instrument']).mean()
Averages

ax = Averages.plot(x='Combined Alpha Model Country Rank', y='3 Month Total Return', kind='scatter')

spearmans = scipy.stats.spearmanr(Averages['Combined Alpha Model Country Rank'],Averages['3 Month Total Return'])
Rho = spearmans[0]
p = spearmans[1]
print(Rho,p)

