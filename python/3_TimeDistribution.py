__author__ = "me"
__date__ = "2015_10_13"

get_ipython().magic('pylab inline')
import pandas as pd
import numpy as np
import geopandas as gp

import pylab as plt
import os

import requests
s = requests.get("https://raw.githubusercontent.com/Casyfill/CUSP_templates/master/Py/fbMatplotlibrc.json").json()
plt.rcParams.update(s)

PARQA = os.getenv('PARQA')

insps = pd.read_csv(PARQA + "data/All_Inspections.csv",index_col=0)
litter = insps[insps.Litter==1]
nolitter = insps[insps.Litter==0]

litter.Date = pd.to_datetime(litter.Date, format='%Y-%m-%d')
nolitter.Date = pd.to_datetime(nolitter.Date, format='%Y-%m-%d')

litter.head(2)

def getBefore(x):
    '''find the timedelta between 
       the inspection and last call before it'''
    ID = x['Prop ID']
    if ID in pids.keys():
        calls = pids[ID]
        if len(calls[calls<x.Date])>0:
            delta = (x.Date - calls[calls<x.Date].iloc[-1]).days
            return delta
    
def getAfter(x): 
    '''find the timedelta between 
       the inspection and first call after it'''
    ID = x['Prop ID']
    if ID in pids.keys():
        calls = pids[ID]
        if len(calls[calls>x.Date])>0:
            delta = (calls[calls>x.Date].iloc[0] - x.Date).days
            return delta
    

calls = pd.read_csv(PARQA + 'data/311/311_rPID_litter.csv')

calls['Created Date'] = pd.to_datetime(calls['Created Date'])

pids = {}

c = calls.groupby('rParkID')
for name, df in c:
    x = df['Created Date'].sort_values()
    pids[name] = x

pids.values()[0]

litter['timeAfter'] = litter.apply(getAfter, axis=1)
litter['timeBefore'] = litter.apply(getBefore, axis=1)

nolitter['timeAfter'] =  nolitter.apply(getAfter, axis=1)
nolitter['timeBefore'] = nolitter.apply(getBefore, axis=1)

# litter[litter.timeAfter>4000]

fig, ax = plt.subplots(figsize=(12,12))

nolitter.plot(kind='scatter',x='timeBefore',y='timeAfter',c='r', s=5, ax=ax, label='litter passed')
litter.plot(kind='scatter',x='timeBefore',y='timeAfter',c='blue', s=10,ax=ax, label='litter failed')
plt.legend(scatterpoints=4)
plt.savefig(PARQA + "parqa/Inspections/img/days_before_after_scatter.png")

fig, ax = plt.subplots(2,2, figsize=(12,12))

nolitter.timeBefore[(nolitter.timeBefore<180)].plot(kind='hist',
                                                    color='blue', 
                                                    alpha=.6, 
                                                    ax=ax[0][0],
                                                    title='before passed litter inspection')

litter.timeBefore[(litter.timeBefore<180)].plot(kind='hist',
                                                    color='blue', 
                                                    alpha=.6, 
                                                    ax=ax[0][1],
                                                    title='before failed litter inspection')

nolitter.timeAfter[(nolitter.timeAfter<180)].plot(kind='hist',
                                                    color='blue', 
                                                    alpha=.6, 
                                                    ax=ax[1][0],
                                                    title='after passed litter inspection')

litter.timeAfter[(litter.timeAfter<180)].plot(kind='hist',
                                                    color='blue', 
                                                    alpha=.6, 
                                                    ax=ax[1][1],
                                                    title='after failed litter inspection')

plt.suptitle('Histograms of time (in days) to the first (last) litter complain after(before) failed/passed litter inspection',
            fontsize=14)
plt.savefig(PARQA + "parqa/Inspections/img/days_before_after_histogram.png")

fig, ax = plt.subplots(figsize=(12,12))

nolitter[(nolitter.timeBefore<180) & (nolitter.timeAfter<180)].plot(kind='scatter',x='timeBefore',y='timeAfter',c='r', s=20, ax=ax, label='litter passed')
litter[(litter.timeBefore<180) & (litter.timeAfter<180)].plot(kind='scatter',x='timeBefore',y='timeAfter',c='blue', s=40,ax=ax, label='litter failed')
plt.legend(scatterpoints=4)
plt.savefig(PARQA + "parqa/Inspections/img/days_before_after_scatter2.png")



