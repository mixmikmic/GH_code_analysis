import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon  # must import explicitly
import seaborn as sns
sns.set(style="whitegrid", palette="pastel", color_codes=True)
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

def query_data(df, Col, structure, metric):   # helper function to get data, Col = AAA, AXB or diff 
    return df[Col][(df['structure'] == structure) & (df['metric'] == metric)]

# v2 ot the plot, show the case number as marker

def bland_altman_plot2(df, structure, metric, *args, **kwargs):
    data = df[(df['structure'] == structure) & (df['metric'] == metric)]
    
    Dm_data     = np.asarray(data['Dm'])
    AAA_data     = np.asarray(data['AAA'])
    cases = [case.replace('Case', '') for case in data['Case'].values]
    
    z_stat, p_val = wilcoxon(Dm_data, AAA_data)
    mean      = np.mean([Dm_data, AAA_data], axis=0)
    diff      = Dm_data - AAA_data                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    # plt.scatter(mean, diff, *args, **kwargs)
    plt.scatter(AAA_data, diff, *args, **kwargs)
    for i, txt in enumerate(cases):
        plt.annotate(txt, (AAA_data[i],diff[i]))
    
    plt.axhline(md,           color='red', linestyle='-')
    plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
    plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
    plt.xlabel('AAA Dose (Gy)')
    plt.ylabel('AXB - AAA Difference (Gy)')
   # plt.title( str(np.round(md, decimals = 3)) + ' Gy (' + str(np.round(md_percent, decimals = 3)) +  ' %) difference with p = ' + str(np.round(p_val, decimals = 3))  + ' for ' + structure + ' and metric ' + metric)
    plt.title( str(np.round(md, decimals = 3)) + ' Gy difference with p = ' + str(np.round(p_val, decimals = 3))  + '\n for ' + structure + ' and metric ' + metric)
    #plt.savefig('BA.png')

data = pd.read_csv('metrics_all_7_2_2017.csv')  # import AAA data

data['Case'] = data['patID'].str.split('_').str.get(0)  # get the case ID e.g. case1
data['Algo'] = data['patID'].str.split('_').str.get(1)  # get the case ID e.g. case1

data = data.drop('patID', 1)  # drop the patID col

data = data.replace(to_replace='ribs', value='Ribs')    # Fix some lables
data = data.replace(to_replace='skin', value='Skin')

data['structure'].unique()

data.head()

data['Case'].unique()

data = data[data['Case'] != 'Case5']  # drop Case5
data = data[data['Case'] != 'Case14']  # drop Case5
data = data[data['Case'] != 'Case16']  # drop Case5
data = data[data['Case'] != 'Case24']  # drop Case5

print('number of cases is ' + str(len(data['Case'].unique())))
data['Case'].unique()

AAA_df = data[data['Algo'] == 'AAA']
AAA_df.rename(columns={'observed': 'AAA'}, inplace=True)    # rename 
AAA_df = AAA_df.drop('Algo', 1)
#AAA_df.head()

Dm_df = data[data['Algo'] == 'AXB']
Dm_df.rename(columns={'observed': 'Dm'}, inplace=True)    # rename 
Dm_df = Dm_df.drop('Algo', 1)
#Dm_df.head()

AAA_Dm_data = pd.merge(AAA_df, Dm_df, how='inner', on=['metric', 'Case', 'structure'])  # Get merged set
AAA_Dm_data = AAA_Dm_data[['Case', 'structure', 'metric', 'AAA', 'Dm']]   # rearrange  
AAA_Dm_data.head()

AAA_Dm_data['Dm-AAA'] = AAA_Dm_data.Dm - AAA_Dm_data.AAA

AAA_Dm_data.head()

AAA_Dm_data['Case'].unique()

AAA_Dm_data['structure'].unique()

# AAA_Dm_data[AAA_Dm_data['structure'] == 'Heart']

AAA_Dm_data['metric'].unique()

# Single metric

#structure = 'ITV'
#structure = 'Heart'
structure = structure = 'PTV CHEST'
#structure = structure = 'Foramen'
metric = 'D1CC'
plt.figure(figsize=(5, 4)) 
bland_altman_plot2(AAA_Dm_data, structure, metric)
plt.show()

df = AAA_Dm_data[(AAA_Dm_data['structure'] == structure) & (AAA_Dm_data['metric'] == metric)]
df 

mean_diff_table = AAA_Dm_data.groupby(['structure', 'metric'],as_index=False).mean().pivot(index='structure', columns='metric', values='Dm-AAA')

structures_of_interest = ['PTV CHEST','ITV','Foramen', 'Oesophagus', 'L Brachial plex', 'Rt Brachial plex', 'Heart', 'Trachea','Bronchial tree', 'Ribs', 'Liver', 'Skin']  # 'Oesophagus', 'Heart', 'GTV', 'Liver', 'Trachea', 'Bronchial tree',
metrics_of_interest  =  ['DMAX', 'D0.1CC', 'D1CC', 'D5%', 'D50%', 'D95%', 'D99%', 'DHI']

i = len(structures_of_interest)
j = len(metrics_of_interest)
wilcox_data = np.zeros((i,j))  # init an empty array

sub_diff_table = mean_diff_table[metrics_of_interest].loc[structures_of_interest]

sub_diff_table

d = AAA_Dm_data  # for convenience just copy

j = 0
for structure in structures_of_interest:
    i = 0
    for metric in metrics_of_interest:
        A =  d[(d['structure'] == structure) & (d['metric'] == metric)]
        D =  A['Dm-AAA']
        #wilcox_data[j][i] = my_wilcox(D.values)
        z_stat, p_val = wilcoxon(D.values)
        wilcox_data[j][i] = p_val
        i = i + 1
    j = j+ 1

wilcox_data_df = pd.DataFrame(data=wilcox_data,    # values
             index=structures_of_interest,    # 1st column as index
             columns=metrics_of_interest)  # 1st row as the column names

wilcox_data_df

plt.figure(figsize=(9, 5))  
data = sub_diff_table[wilcox_data_df<0.05]

ax3 = sns.heatmap(data, annot=True, linewidths=.5, center=0, vmin = -1.1, vmax = 1.1, cmap='PRGn') #  mask=mask,
ax3.set_title('Significant (p<0.05) mean dose differences AXB-AAA (Gy) by metric', size='large');
plt.savefig('Lung_wilcox.png')

structure1 = 'PTV CHEST'
metric = 'D50%'
data1 = query_data(AAA_Dm_data, 'Dm-AAA', structure1, metric).values


structure2 = 'Oesophagus'
data2 = query_data(AAA_Dm_data, 'Dm-AAA', structure2, metric).values

plt.scatter(data1, data2)
plt.xlabel(structure1)
plt.ylabel(structure2)

np.corrcoef(data1, data2)[0,1]  # print the correlation coeff

def corr_coeff(structure1, structure2, metric):
    data1 = query_data(AAA_Dm_data, 'Dm-AAA', structure1, metric).values
    data2 = query_data(AAA_Dm_data, 'Dm-AAA', structure2, metric).values
    return np.corrcoef(data1, data2)[0,1]  # print the correlation coeff

a = corr_coeff(structure1 = 'PTV CHEST', structure2 = 'Skin', metric = 'D50%')
a

structures_of_interest = ['PTV CHEST','ITV','Foramen', 'Oesophagus','Heart', 'Trachea','Bronchial tree', 'Ribs', 'Skin']  # Don't have complete data for all structures 

i = len(structures_of_interest)
j = i
corr_data = np.zeros((i,j))  # init an empty array
metric = 'DMAX'

j = 0
for structure1 in structures_of_interest:
    i = 0
    for structure2 in structures_of_interest:
        data1 = query_data(AAA_Dm_data, 'Dm-AAA', structure1, metric).values
        data2 = query_data(AAA_Dm_data, 'Dm-AAA', structure2, metric).values
        # print(str(coeff) + structure1 + ':' + str(len(data1)) + structure2 + ':' + str(len(data2)))
        coeff = np.corrcoef(data1, data2)[0,1] 
        corr_data[j][i] = coeff
       # print(structure1 + ':' + structure2)
    
        i = i + 1
    j = j+ 1
    
corr_data_df = pd.DataFrame(data=corr_data, index=structures_of_interest, columns=structures_of_interest)  # 1st row as the column names

# corr_data_df

mask = np.zeros_like(corr_data, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

ax3 = sns.heatmap(corr_data_df, mask=mask, annot=True, linewidths=.5, center=0, vmin = -1.0, vmax = 1.0)
ax3.set_title('Table of correlation coefficients for ' + metric)



