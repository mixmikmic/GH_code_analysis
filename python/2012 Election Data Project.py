import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')
from __future__ import division

import requests
from io import StringIO

url = 'http://elections.huffingtonpost.com/pollster/2012-general-election-romney-vs-obama.csv'

source = requests.get(url).text

poll_data = StringIO(source)

poll_df = pd.read_csv(poll_data)

poll_df

poll_df.info()

sns.factorplot('Affiliation',data=poll_df,kind='count')

sns.factorplot('Affiliation',data=poll_df,kind='count', order=['Dem', 'Rep'])

sns.factorplot('Affiliation',data=poll_df,kind='count',hue='Population')

poll_df.head()

avg = pd.DataFrame(poll_df.mean())
avg.drop('Number of Observations',axis=0,inplace=True)

avg

std = pd.DataFrame(poll_df.std())
std.drop('Number of Observations', axis=0, inplace=True)

std.head()

avg.plot(yerr=std,kind='bar',legend=False)

poll_avg = pd.concat([avg,std],axis=1)
poll_avg.columns = ['Average','STD']

poll_avg

poll_df.plot(x='End Date',y = ['Obama','Romney','Undecided'],
             linestyle='',marker='o')

from datetime import datetime
poll_df['Difference']= (poll_df['Obama'] - poll_df['Romney'])/100
poll_df.head()

poll_df = poll_df.groupby(['Start Date'],as_index=False).mean()
poll_df.head()

poll_df.plot('Start Date','Difference',figsize=(12,4),marker='o',
             linestyle='-',color='purple')

poll_df[poll_df['Difference']==poll_df['Difference'].min()]

row_in = 0
xlimit = []

for date in poll_df['Start Date']:
    if date[0:7] == '2012-10':
        xlimit.append(row_in)
        row_in += 1
    else:
        row_in += 1
        
print (min(xlimit))
print (max(xlimit))

poll_df.plot('Start Date','Difference',figsize=(12,4),marker='o',
            linestyle='-',color='purple',xlim=(329,356))

# Oct 3rd, add two days from the starting date
plt.axvline(x=329+2,linewidth=4,color='grey') 

# Oct 11th
plt.axvline(x=329+10,linewidth=4,color='grey')

# Oct 22nd
plt.axvline(x=329+21,linewidth=4,color='grey')

donor_df = pd.read_csv('Election_Donor_Data.csv',low_memory=False)

donor_df.head()

donor_df.info()

donor_df['contb_receipt_amt'].value_counts()

don_mean = donor_df['contb_receipt_amt'].mean()

don_std = donor_df['contb_receipt_amt'].std()

print ('The average donation was %.2f with a std %.2f' %(don_mean,don_std))

#can't sort without making a copy
top_donor = donor_df['contb_receipt_amt'].copy()

top_donor.sort()

top_donor

top_donor = top_donor[top_donor>0]

top_donor.sort()

top_donor.value_counts().head(10)

com_don = top_donor[ top_donor < 2500]

com_don.hist(bins=100)

candidates = donor_df.cand_nm.unique()

candidates

# Dictionary of party affiliation
party_map = {'Bachmann, Michelle': 'Republican',
           'Cain, Herman': 'Republican',
           'Gingrich, Newt': 'Republican',
           'Huntsman, Jon': 'Republican',
           'Johnson, Gary Earl': 'Republican',
           'McCotter, Thaddeus G': 'Republican',
           'Obama, Barack': 'Democrat',
           'Paul, Ron': 'Republican',
           'Pawlenty, Timothy': 'Republican',
           'Perry, Rick': 'Republican',
           "Roemer, Charles E. 'Buddy' III": 'Republican',
           'Romney, Mitt': 'Republican',
           'Santorum, Rick': 'Republican'}

donor_df['Party'] = donor_df.cand_nm.map(party_map)

donor_df = donor_df[donor_df.contb_receipt_amt > 0]

donor_df.head()

donor_df.groupby('cand_nm')['contb_receipt_amt'].count()

donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()

cand_amount = donor_df.groupby('cand_nm')['contb_receipt_amt'].sum()

i = 0

for don in cand_amount:
    print ('The candidate %s raise %.0f dollars' %(cand_amount.index[i],don))
    i += 1

cand_amount.plot(kind='bar')

donor_df.groupby('Party')['contb_receipt_amt'].sum().plot(kind='bar')

occupation_df = donor_df.pivot_table('contb_receipt_amt',
                                    index='contbr_occupation',
                                    columns= 'Party',
                                    aggfunc='sum')

occupation_df.head()

occupation_df.tail()

occupation_df.shape

occupation_df = occupation_df[occupation_df.sum(1) > 1000000]

occupation_df.shape

occupation_df.plot(kind='bar')

occupation_df.plot(kind='barh',figsize=(10,12),cmap='seismic')

occupation_df.drop(['INFORMATION REQUESTED PER BEST EFFORTS','INFORMATION REQUESTED'],axis=0,inplace=True)

occupation_df.loc['CEO'] = occupation_df.loc['CEO'] + occupation_df.loc['C.E.O.']

occupation_df.drop('C.E.O.',inplace=True)

occupation_df.plot(kind='barh',figsize=(10,12),cmap='seismic')



