import pandas as pd
import cufflinks as cf
import numpy as np

cf.set_config_file(sharing='public',theme='pearl',offline=False)

df=pd.DataFrame(np.random.randn(100,5),index=pd.date_range('1/1/15',periods=100),
                columns=['IBM','MSFT','GOOG','VERZ','APPL'])
df=df.cumsum()

help(df.iplot)

df.iplot(filename='Tutorial 1')

df[['APPL','IBM','VERZ']].iplot(theme='white',filename='Tutorial White')

df.iplot(theme='pearl',filename='Tutorial Metadata',title='Stock Returns',xTitle='Dates',yTitle='Returns')

df['IBM'].iplot(filename='IBM Returns',bestfit=True)

df['IBM'].iplot(filename='IBM Returns - colors',bestfit=True,colors=['pink'],bestfit_colors=['blue'])

df['IBM'].iplot(filename='Tutorial Microsoft',fill=True,colors=['green'])

df.sum().iplot(kind='bar',filename='Tutorial Barchart')

df.resample('M').iplot(kind='bar',barmode='stacked',filename='Tutorial Bar Stacked')

df[['VERZ','IBM']].iplot(filename='Tutorial Spread',kind='spread')

(df[['GOOG','MSFT']]+20).iplot(filename='Tutorial Ratio',kind='ratio',colors=['green','red'])

annotations={'2015-01-15':'Dividends','2015-03-31':'Split Announced'}
df['MSFT'].iplot(filename='Tutorial Annotations',annotations=annotations)

df[['VERZ','MSFT']].iplot(filename='Tutorial Image',theme='white',colors=['pink','blue'],asImage=True,dimensions=(800,500))

df['GOOG'].figure()

data=df.figure()['data']

data[0]['name']='My Custom Name'

df.iplot(data=data,filename='Tutorial Custom Name')

