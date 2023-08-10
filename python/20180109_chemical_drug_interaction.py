import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("ggplot")
# plot inline
get_ipython().magic('matplotlib inline')

df = pd.read_pickle('/home/hermuba/data/drug/tmacc_df')
df.drop('ACTIVITY', axis=1, inplace=True)
df.set_index('chem_name', drop=True, append=False, inplace=True)

new_df = df.fillna(0)

norm_tmacc = new_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
norm_tmacc.dropna(axis = 1, inplace = True) # some std = 0, provide no info therefore remove them; normalized showed best result in clustering

amr_data = pd.read_pickle('/home/hermuba/data/annotated_RIS/anno_sps_df')
cluster = pd.read_pickle('/home/hermuba/data/genePredicted/cdhit/ec0102_df')
card = pd.read_pickle('/home/hermuba/data/aro_pattern_df')

norm_tmacc.head()

amr_data['Antibiotic'] = amr_data['Antibiotic'].str.lower()
amr_data.head()

#show drug combination
amr_data.loc[amr_data['Antibiotic'].str.contains('/')] # not all of them with "/" in MIC

# joining dataframe
print(set(norm_tmacc.index).difference(set(amr_data['Antibiotic'].unique()))) # in TMACC but not in amr
print(set(amr_data['Antibiotic'].unique()).difference(set(norm_tmacc.index))) # in amr_data but not in TMACC

# join dataframe
tmacc_amr = norm_tmacc.merge(amr_data[['Genome ID', 'Measurement Value', 'Species', 'Antibiotic', 'Resistant Phenotype']] , right_on='Antibiotic', left_index = True, suffixes=('_x', '_y'))
tmacc_amr_num = tmacc_amr.loc[tmacc_amr['Measurement Value'] != 'nan']

# some numeric operation
import math
tmacc_amr_num['log_mic'] = pd.to_numeric(tmacc_amr_num['Measurement Value'])
tmacc_amr_num['log_mic'] = tmacc_amr_num['log_mic'].apply(lambda x:np.log(x)/np.log(2)) # log transformation

# join gene with tmacc
all_num = tmacc_amr_num.merge(cluster, left_on = 'Genome ID', right_index = True) #using Ecoli as a test
all_num_card = tmacc_amr_num.merge(card, left_on = 'Genome ID', right_index = True) #using Ecoli as a test

# indexing
tmacc_id = tmacc_amr.columns[:-5]
card_id = all_num_card.columns[-189:]
cluster_id = all_num.columns[-15950:]

all_num.shape

sim_corr = []
# simple regression
for x in tmacc_amr.columns[:-5]:
    
    corr = pd.to_numeric(tmacc_amr_num['log_mic']).corr(tmacc_amr_num[x])
    sim_corr.append(corr)

# WHAT ABOUT STRATIFIED WITH SPECIES
corr_d = {'all': np.array(sim_corr)}
for sps in tmacc_amr['Species'].unique():
    sub_df = tmacc_amr_num.loc[tmacc_amr['Species'] == sps]
    sps_corr = []
    for x in tmacc_amr.columns[:-5]:
    
        corr = pd.to_numeric(sub_df['log_mic']).corr(sub_df[x])
        sps_corr.append(corr)
    corr_d[sps]=np.nan_to_num(np.array(sps_corr)) # why is there nan
    

fig = plt.figure(1, figsize=(15, 6))

# Create an axes instance
ax = fig.add_subplot(111)

ax.boxplot(corr_d.values())
ax.set_xticklabels(corr_d.keys(), rotation = 90)

sim_corr = []
# simple regression
for x in cluster_id:
    
    corr = pd.to_numeric(all_num['log_mic']).corr(all_num[x])
    sim_corr.append(corr)
corr_d['Escherichia_cdhit'] = np.nan_to_num(np.array(sim_corr))

sim_corr = []
# simple regression
for x in card_id:
    
    corr = pd.to_numeric(all_num_card['log_mic']).corr(all_num_card[x])
    sim_corr.append(corr)
corr_d['Escherichia_card'] = np.nan_to_num(np.array(sim_corr))

itxn_corr = []
for x1, x2 in zip(cluster_id, tmacc_amr.columns[:-5]):
    itxn = all_num[x1].multiply(all_num[x2]) # interaction
    corr = pd.to_numeric(all_num['log_mic']).corr(itxn)
    itxn_corr.append(corr)
corr_d['Escherichia_itxn'] = np.nan_to_num(np.array(itxn_corr))

itxn_card_corr = []
for x1, x2 in zip(card_id, tmacc_amr.columns[:-5]):
    itxn = all_num_card[x1].multiply(all_num_card[x2]) # interaction
    corr = pd.to_numeric(all_num_card['log_mic']).corr(itxn)
    itxn_card_corr.append(corr)
corr_d['Escherichia_itxn_card'] = np.nan_to_num(np.array(itxn_card_corr))

corr_d['Escherichia_itxn_card'] = np.nan_to_num(np.array(itxn_card_corr))

fig = plt.figure(1, figsize=(5, 6))

# Create an axes instance
ax = fig.add_subplot(111)

ax.boxplot([corr_d[x] for x in ['all', 'Escherichia', 'Escherichia_itxn', 'Escherichia_itxn_card', 'Escherichia_cdhit', 'Escherichia_card']])
ax.set_xticklabels(['all', 'Escherichia', 'Escherichia itxn', 'Escherichia_itxn_card', 'Escherichia_cdhit', 'Escherichia_card'], rotation = 90)

## Without a constant

import statsmodels.api as sm

def try_regression(df,x):

    X = df[x]
    X = sm.add_constant(X)
    y = df['log_mic']

    # Note the difference in argument order
    model = sm.OLS(y, X.astype(float)).fit()
    predictions = model.predict(X) # make the predictions by the model
    print(model.summary())
    # Print out the statistics
    return(model.rsquared, model.tvalues, model.f_test)

r,t,f = try_regression(all_num, tmacc_id)
print(r)

r,t,f = try_regression(all_num, cluster_id)
print(t[1])

r,t,f = try_regression(all_num, tmacc_id.append(cluster_id))
print(r)

r,t,f = try_regression(all_num_card, card_id)

r,t,f = try_regression(all_num_card, card_id.append(tmacc_id))

itxn_card_corr = []
card_itxn = pd.DataFrame()
for x1 in card_id:
    for x2 in tmacc_id:
        itxn = all_num_card[x1].multiply(all_num_card[x2])
        corr = pd.to_numeric(all_num_card['log_mic']).corr(itxn)
        itxn_card_corr.append(corr)
        if corr > 0.6:
            card_itxn[str(x1)+str(x2)] = itxn
        else: 
            print("useless")
        # interaction
        

len(itxn_card_corr)

itxn_corr = []
cluster_itxn = pd.DataFrame()
for x1 in cluster_id:
    for x2 in tmacc_id:
        itxn = all_num[x1].multiply(all_num[x2])
        corr = pd.to_numeric(all_num['log_mic']).corr(itxn)
        itxn_corr.append(corr)
        if corr > 0.6:
            card_itxn[str(x1)+str(x2)] = itxn
      



