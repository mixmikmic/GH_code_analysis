import numpy as np
import pandas as pd
import os, re
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

ord_mdw = pd.read_csv('nexrad_data/ord-mdw-dbz.csv')
ord_mdw = ord_mdw[['timestamp', 'ORD', 'MDW']]
print(ord_mdw.shape)
print(ord_mdw.dtypes)
ord_mdw.head()

ord_mdw['timestamp'] = pd.to_datetime(ord_mdw['timestamp'])
ord_mdw.head()

ord_mdw = ord_mdw.set_index(pd.DatetimeIndex(ord_mdw['timestamp']))
ord_mdw = ord_mdw[['ORD', 'MDW']]
ord_mdw.plot()

# Marshall-Palmer
def precip_rate1(dbz):
    return pow(pow(10, dbz/10)/200, 0.625)

# Misc. version found online
def precip_rate2(dbz):
    return pow(pow(10, dbz/10)/250, (10/12))

# Version referenced for storms
def precip_rate3(dbz):
    return pow(pow(10, dbz/10)/300, (10/14))
# 325 825
def precip_rate_test(dbz):
    return pow(pow(10, dbz/10)/300, 0.85)

# Copy the main DataFrame, just look at a single day
converted_df = ord_mdw.copy()
day_df = converted_df['2016-08-12']

fig, axs = plt.subplots(1,2)
plt.rcParams["figure.figsize"] = [15, 5]

pr_df = precip_rate1(day_df)
pr_df[['ORD2','MDW2']] = precip_rate_test(day_df)

print('{} vs {}, % diff: {}'.format(pr_df['ORD'].sum(), pr_df['ORD2'].sum(), (abs(pr_df['ORD2'].sum() - pr_df['ORD'].sum())/pr_df['ORD'].sum())))
print('{} vs {}, % diff: {}'.format(pr_df['MDW'].sum(), pr_df['MDW2'].sum(), (abs(pr_df['MDW2'].sum() - pr_df['MDW'].sum())/pr_df['MDW'].sum())))
pr_df[['ORD','ORD2']].plot(title='ORD Comparison', ax=axs[0])
pr_df[['MDW','MDW2']].plot(title='mdW Comparison', ax=axs[1])

# fig, axs = plt.subplots(1,2)
# plt.rcParams["figure.figsize"] = [15, 5]

# # Converting to precipitation, will be substantially lower because not pulling from wider area

# pr_df = precip_rate1(day_df)
# pr_df[['ORD2','MDW2']] = pr_df[['ORD','MDW']].applymap(lambda x: precip_rate_test_tier(x))
# #pr_df = pr_df * 0.0833 / 25.4

# print('{} vs {}, % diff: {}'.format(pr_df['ORD'].sum(), pr_df['ORD2'].sum(), (abs(pr_df['ORD2'].sum() - pr_df['ORD'].sum())/pr_df['ORD'].sum())))
# print('{} vs {}, % diff: {}'.format(pr_df['MDW'].sum(), pr_df['MDW2'].sum(), (abs(pr_df['MDW2'].sum() - pr_df['MDW'].sum())/pr_df['MDW'].sum())))
# pr_df[['ORD','ORD2']].plot(title='ORD Comparison', ax=axs[0])
# pr_df[['MDW','MDW2']].plot(title='mdW Comparison', ax=axs[1])

fig, axs = plt.subplots(1,2)
plt.rcParams["figure.figsize"] = [15, 5]

# Converting to precipitation, will be substantially lower because not pulling from wider area

pr_df = precip_rate1(day_df)
pr_df[['ORD2','MDW2']] = precip_rate_test(day_df)
pr_df = pr_df * 0.0833 / 25.4

print('{} vs {}, % diff: {}'.format(pr_df['ORD'].sum(), pr_df['ORD2'].sum(), (abs(pr_df['ORD2'].sum() - pr_df['ORD'].sum())/pr_df['ORD'].sum())))
print('{} vs {}, % diff: {}'.format(pr_df['MDW'].sum(), pr_df['MDW2'].sum(), (abs(pr_df['MDW2'].sum() - pr_df['MDW'].sum())/pr_df['MDW'].sum())))
pr_df[['ORD','ORD2']].plot(title='ORD Comparison', ax=axs[0])
pr_df[['MDW','MDW2']].plot(title='mdW Comparison', ax=axs[1])



