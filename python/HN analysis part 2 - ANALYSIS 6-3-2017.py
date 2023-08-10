import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from scipy import stats
from scipy.stats import wilcoxon  # must import explicitly
import seaborn as sns
sns.set(style="whitegrid", palette="pastel", color_codes=True)

def diff_percent(a,ref):
    return 100*((a-ref)/ref)

print(diff_percent(65,70.0))

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

HN_df = pd.read_csv('HN_df_clean_28_11.csv')  # read in the cleaned data
HN_df['Dm-AAA'] = HN_df['Dm'] - HN_df['AAA'] # get abs diff

to_exclude = ['Case3', 'Case4', 'Case5', 'Case7']  # has prescription of 54 Gy  
to_include = list(set(HN_df['Case'].unique()) - set(to_exclude))
HN_df = HN_df[HN_df['Case'].isin(to_include)]
HN_df['Case'].unique()

len(HN_df['Case'].unique())

HN_df['structure'].unique()

HN_df['metric'].unique()

HN_df.head()

mean_diff_table = HN_df.groupby(['structure', 'metric'],as_index=False).mean().pivot(index='structure', columns='metric', values='Dm-AAA')

structures_of_interest = ['PTV1', 'PTV2', 'L Parotid', 'R Parotid', 'Brain Stem', 'Spinal Cord']
metrics_of_interest  =  ['DMAX', 'D0.1CC', 'D1CC', 'D5%', 'D50%', 'D95%', 'DHI']

sub_diff_table = mean_diff_table[metrics_of_interest].loc[structures_of_interest]

i = len(structures_of_interest)
j = len(metrics_of_interest)
wilcox_data = np.zeros((i,j))  # init an empty array

d = HN_df  # for convenience just copy

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

plt.figure(figsize=(9, 5))   
data = sub_diff_table[wilcox_data_df<0.05]

ax3 = sns.heatmap(data, annot=True, linewidths=.5, center=0, vmin = -1.2, vmax = 1.2, cmap='PRGn') 
ax3.set_title('HN significant (p<0.05) mean dose differences AXB-AAA (Gy) by metric', size='large');
plt.savefig('HN_wilcox.png')

sub_diff_table[wilcox_data_df<0.05].mean().mean()

structure = 'PTV1'
structure = 'Brain Stem'
metric = 'D0.1CC'
plt.figure(figsize=(5, 4)) 
bland_altman_plot2(HN_df, structure, metric)
plt.xlim([45,55])
plt.show()



